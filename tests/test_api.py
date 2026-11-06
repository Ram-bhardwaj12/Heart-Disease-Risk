"""API endpoint integration tests using fastapi.testclient.TestClient."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
import engine.core as engine

REPO_ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


def test_health_endpoint():
    """Test GET /api/health returns status ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_schema_endpoint():
    """Test GET /api/schema returns 13 fields and correct metadata."""
    resp = client.get("/api/schema")
    assert resp.status_code == 200
    data = resp.json()

    assert data["product_name"] == "CardioLens"
    assert "tagline" in data
    assert "disclaimer" in data
    assert "cutoffs" in data
    assert data["cutoffs"]["low"] == 0.3
    assert data["cutoffs"]["high"] == 0.5
    assert data["inferred_codes"] == ["cp", "restecg", "slope", "thal"]
    assert "inferred_codes_note" in data
    assert "Cleveland" in data["inferred_codes_note"]
    assert len(data["fields"]) == 13

    # Check key features present and short_label format
    field_keys = [f["key"] for f in data["fields"]]
    assert "age" in field_keys
    assert "ca" in field_keys
    assert "thal" in field_keys

    for f in data["fields"]:
        assert "short_label" in f
        assert "(" not in f["short_label"] and ")" not in f["short_label"]

    ca_spec = next(f for f in data["fields"] if f["key"] == "ca")
    assert ca_spec["optional"] is True
    age_spec = next(f for f in data["fields"] if f["key"] == "age")
    assert age_spec["optional"] is False


def test_predict_endpoint_success_and_equivalence():
    """Test POST /api/predict for all 61 test rows matches engine and direct predict_proba within 1e-12."""
    import re
    test_df = engine.load_test_features()
    pipeline = engine.load_model_pipeline()
    expected_cols = engine.load_train_features().columns.tolist()

    for idx in range(len(test_df)):
        sample = test_df.iloc[idx].to_dict()
        clean_sample = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in sample.items()
        }
        resp = client.post("/api/predict", json={"inputs": clean_sample})
        assert resp.status_code == 200, f"Predict failed on row {idx}: {resp.text}"

        body = resp.json()
        assert set(body.keys()) == {
            "score",
            "score_percent",
            "tier",
            "cutoffs",
            "warnings",
            "missing_used",
            "summary",
            "contributions",
            "top_raising",
            "top_lowering",
            "base_value",
            "model_agreement",
            "disclaimer",
        }
        assert isinstance(body["score"], float)
        assert 0.0 <= body["score"] <= 1.0
        assert 0.0 <= body["score_percent"] <= 100.0
        assert set(body["tier"].keys()) == {"key", "label", "text"}
        assert body["tier"]["key"] in {"lower", "moderate", "higher"}
        assert set(body["cutoffs"].keys()) == {"low", "high"}
        assert isinstance(body["warnings"], list)
        assert isinstance(body["missing_used"], list)
        assert len(body["contributions"]) == 13
        assert len(body["top_raising"]) <= 3
        assert len(body["top_lowering"]) <= 3

        for item in body["contributions"]:
            assert set(item.keys()) == {"key", "label", "short_label", "value_display", "amount", "direction"}
            assert item["direction"] in {"raised", "lowered"}
            assert "(" not in item["short_label"] and ")" not in item["short_label"]

        # Contributions sorted descending by absolute amount
        abs_amounts = [abs(c["amount"]) for c in body["contributions"]]
        assert abs_amounts == sorted(abs_amounts, reverse=True)

        # Summary contains short labels, no consecutive brackets, and no text 'Option'
        summary = body["summary"]
        short_labels = [f["short_label"] for f in engine.get_feature_specs().values()]
        matched_labels = [lbl for lbl in short_labels if lbl in summary]
        assert len(matched_labels) >= 2, f"Summary did not contain >= 2 field labels: {summary}"

        # No two consecutive bracket groups
        assert not re.search(r"\)\s*\(", summary), f"Two consecutive bracket groups found in summary: {summary}"

        # Never print the text 'Option'
        assert "Option" not in summary, f"Text 'Option' found in summary: {summary}"

        # Model agreement
        assert set(body["model_agreement"].keys()) == {"tree_count", "share_at_or_above_half", "mean_score"}
        assert body["model_agreement"]["tree_count"] == len(pipeline.named_steps["classifier"].estimators_)
        assert body["model_agreement"]["tree_count"] == 400

        # Equivalence with engine and direct pipeline predict_proba
        engine_score = engine.predict_score(sample)
        assert abs(body["score"] - engine_score) < 1e-12, f"Row {idx} score mismatch with engine"

        row_df = pd.DataFrame([sample], columns=expected_cols, dtype=np.float64)
        direct_prob = float(pipeline.predict_proba(row_df)[0, 1])
        assert abs(body["score"] - direct_prob) < 1e-12, f"Row {idx} score mismatch with direct pipeline"


def test_predict_endpoint_optional_fields_and_warnings():
    """Test POST /api/predict with omitted ca/thal and out-of-range warnings."""
    test_df = engine.load_test_features()
    sample = test_df.iloc[0].to_dict()

    # Omit ca and thal
    sample_no_opt = {k: v for k, v in sample.items() if k not in ["ca", "thal"]}
    sample_no_opt["age"] = 25.0  # Trigger warning: within plausible (18-100), outside training (29-76)

    resp = client.post("/api/predict", json={"inputs": sample_no_opt})
    assert resp.status_code == 200
    body = resp.json()

    assert set(body["missing_used"]) == {"ca", "thal"}
    assert len(body["warnings"]) > 0
    assert any(w["field"] == "age" for w in body["warnings"])
    assert any("outside the range the model learned from" in w["message"] for w in body["warnings"])


def test_predict_endpoint_422_validations():
    """Test POST /api/predict returns 422 for invalid inputs, plausible limits, and extra fields."""
    test_df = engine.load_test_features()
    valid_sample = test_df.iloc[0].to_dict()

    # 1. Missing required field
    bad_sample_1 = {k: v for k, v in valid_sample.items() if k != "age"}
    resp = client.post("/api/predict", json={"inputs": bad_sample_1})
    assert resp.status_code == 422
    assert any(e["field"] == "age" for e in resp.json()["errors"])

    # 2. String in numeric field
    bad_sample_2 = dict(valid_sample, trestbps="one-twenty")
    resp = client.post("/api/predict", json={"inputs": bad_sample_2})
    assert resp.status_code == 422
    assert any(e["field"] == "trestbps" for e in resp.json()["errors"])

    # 3. Unseen categorical code
    bad_sample_3 = dict(valid_sample, cp=99)
    resp = client.post("/api/predict", json={"inputs": bad_sample_3})
    assert resp.status_code == 422
    assert any(e["field"] == "cp" for e in resp.json()["errors"])

    # 4. Extra payload field (forbid extra)
    resp = client.post("/api/predict", json={"inputs": valid_sample, "unrecognized_key": 123})
    assert resp.status_code == 422

    # 5. Plausible limit violations rejected with 422 and plain language messages
    resp_150 = client.post("/api/predict", json={"inputs": dict(valid_sample, age=150)})
    assert resp_150.status_code == 422
    assert any(e["field"] == "age" for e in resp_150.json()["errors"])
    assert any("Enter an age between 18 and 100" in e["message"] for e in resp_150.json()["errors"])

    resp_5000 = client.post("/api/predict", json={"inputs": dict(valid_sample, chol=5000)})
    assert resp_5000.status_code == 422
    assert any(e["field"] == "chol" for e in resp_5000.json()["errors"])
    assert any("Enter a cholesterol level between 80 and 700" in e["message"] for e in resp_5000.json()["errors"])

    resp_0 = client.post("/api/predict", json={"inputs": dict(valid_sample, trestbps=0)})
    assert resp_0.status_code == 422
    assert any(e["field"] == "trestbps" for e in resp_0.json()["errors"])
    assert any("Enter a resting blood pressure between 60 and 260" in e["message"] for e in resp_0.json()["errors"])

    # 6. Accepted within plausible limits with warning (HTTP 200)
    resp_25 = client.post("/api/predict", json={"inputs": dict(valid_sample, age=25)})
    assert resp_25.status_code == 200
    assert any(w["field"] == "age" for w in resp_25.json()["warnings"])
    assert any("outside the range the model learned from" in w["message"] for w in resp_25.json()["warnings"])

    resp_90 = client.post("/api/predict", json={"inputs": dict(valid_sample, age=90)})
    assert resp_90.status_code == 200
    assert any(w["field"] == "age" for w in resp_90.json()["warnings"])
    assert any("outside the range the model learned from" in w["message"] for w in resp_90.json()["warnings"])


def test_whatif_endpoint_success_and_rejection():
    """Test POST /api/whatif computes sensitivity curve and rejects categorical features."""
    test_df = engine.load_test_features()
    sample = test_df.iloc[0].to_dict()

    # 1. Valid continuous numeric feature
    resp = client.post(
        "/api/whatif",
        json={"inputs": sample, "feature": "thalach", "n_points": 25},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["feature"] == "thalach"
    assert len(data["curve"]) == 25
    assert "current_value" in data
    assert "current_score" in data
    assert "cutoffs" in data

    # 2. Reject categorical feature (cp) with 422
    resp_bad = client.post(
        "/api/whatif",
        json={"inputs": sample, "feature": "cp", "n_points": 25},
    )
    assert resp_bad.status_code == 422

    # 3. Reject discrete count feature (ca) with 422
    resp_ca = client.post(
        "/api/whatif",
        json={"inputs": sample, "feature": "ca", "n_points": 25},
    )
    assert resp_ca.status_code == 422


def test_model_info_endpoint():
    """Test GET /api/model-info retrieves reports without hard-coding."""
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.json()

    audit = engine.load_data_audit_summary()

    assert "dataset_facts" in data
    assert data["dataset_facts"]["raw_rows"] == audit["raw_rows"]
    assert data["dataset_facts"]["unique_rows"] == audit["unique_rows"]
    assert "cross_validation" in data
    assert "held_out_test" in data
    assert "confusion_matrices" in data
    assert "random_forest_minus_logistic_regression" in data
    assert "top_5_grouped_shap" in data
    assert len(data["top_5_grouped_shap"]) == 5
    assert "figure_urls" in data
    assert len(data["figure_urls"]) > 0


def test_figures_and_path_traversal_protection():
    """Test GET /figures allows legitimate figures and rejects path traversal."""
    # Legitimate figure request
    resp = client.get("/figures/confusion_matrix.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    # Path traversal attempts
    traversal_attacks = [
        "/figures/..%2fREADME.md",
        "/figures/%2e%2e/README.md",
        "/figures/..%2f..%2f..%2fetc/passwd",
        "/figures/../app/app.py",
        "/figures/sub/dir/test.png",
        "/figures/nonexistent.png",
    ]
    for attack in traversal_attacks:
        bad_resp = client.get(attack)
        assert bad_resp.status_code == 404, f"Attack did not yield 404: {attack}"


def test_request_body_size_limit():
    """Test request body size limit enforces 20 KB limit with HTTP 413."""
    # Valid payload size (< 20 KB)
    test_df = engine.load_test_features()
    sample = test_df.iloc[0].to_dict()
    clean_sample = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in sample.items()}
    resp_ok = client.post("/api/predict", json={"inputs": clean_sample})
    assert resp_ok.status_code == 200

    # Oversized payload (> 20 KB)
    large_garbage = "x" * (25 * 1024)
    resp_large = client.post(
        "/api/predict",
        content=large_garbage,
        headers={"Content-Type": "application/json", "Content-Length": str(len(large_garbage))},
    )
    assert resp_large.status_code == 413


def test_security_headers_and_cache_control():
    """Test required security headers and Cache-Control: no-store on API endpoints."""
    resp = client.get("/api/health")
    assert resp.status_code == 200

    assert resp.headers.get("Cache-Control") == "no-store"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("Referrer-Policy") == "no-referrer"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    expected_csp = (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
    )
    assert resp.headers.get("Content-Security-Policy") == expected_csp


def test_zero_disk_writes_during_requests():
    """Verify that 20 diverse API requests do not write or modify any files on disk."""
    ignored = {"venv", ".git", ".pytest_cache", "__pycache__", ".idea", ".vscode"}

    def get_repo_file_manifest():
        manifest = {}
        for child in REPO_ROOT.iterdir():
            if child.name in ignored:
                continue
            if child.is_file():
                manifest[child.name] = child.stat().st_mtime_ns
            elif child.is_dir():
                for p in child.rglob("*"):
                    if p.is_file() and not any(part in ignored for part in p.parts):
                        manifest[str(p.relative_to(REPO_ROOT))] = p.stat().st_mtime_ns
        return manifest

    before = get_repo_file_manifest()

    test_df = engine.load_test_features()
    sample = test_df.iloc[0].to_dict()
    clean_sample = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in sample.items()}

    # Execute 20 requests
    for _ in range(4):
        client.get("/api/health")
        client.get("/api/schema")
        client.post("/api/predict", json={"inputs": clean_sample})
        client.post("/api/whatif", json={"inputs": clean_sample, "feature": "age"})
        client.get("/api/model-info")

    after = get_repo_file_manifest()

    # Compare manifests
    new_files = set(after.keys()) - set(before.keys())
    assert len(new_files) == 0, f"New files created on disk during requests: {new_files}"

    modified_files = [
        f for f in before if f in after and before[f] != after[f]
    ]
    assert len(modified_files) == 0, f"Files modified on disk during requests: {modified_files}"
