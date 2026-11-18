"""Unit tests for engine package: equivalence, SHAP additivity, validation, and content integrity."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest

import app.utils as legacy_utils
import engine.content as content
import engine.core as engine

REPO_ROOT = Path(__file__).resolve().parent.parent


def generate_random_valid_inputs(n: int = 300, seed: int = 42) -> list[dict]:
    """Generate n random valid clinical inputs according to training specs."""
    rng = np.random.RandomState(seed)
    specs = engine.get_feature_specs()

    samples = []
    for _ in range(n):
        sample = {}
        for col, spec in specs.items():
            if spec["type"] == "choice":
                opts = [opt["value"] for opt in spec["options"]]
                sample[col] = int(rng.choice(opts))
            else:
                min_v = spec["min"]
                max_v = spec["max"]
                val = rng.uniform(min_v, max_v)
                if spec["step"] == 0.1:
                    sample[col] = round(val, 1)
                else:
                    sample[col] = round(val)
        samples.append(sample)
    return samples


def test_equivalence_61_test_rows_and_300_random_inputs():
    """(a) Test predict_score (< 1e-12) and explain_grouped SHAP sums (< 1e-9) equivalence with legacy app.utils."""
    test_df = engine.load_test_features()

    # 1. Test across all 61 test rows
    for idx in range(len(test_df)):
        row_dict = test_df.iloc[idx].to_dict()

        engine_score = engine.predict_score(row_dict)
        legacy_score = legacy_utils.predict_score(row_dict)
        assert abs(engine_score - legacy_score) < 1e-12, f"Score mismatch on test row {idx}"

        engine_grouped = engine.explain_grouped(row_dict)
        legacy_grouped = legacy_utils.explain_grouped(row_dict)

        engine_sum = float(engine_grouped["shap_amount"].sum())
        legacy_sum = float(legacy_grouped["shap_amount"].sum())
        assert abs(engine_sum - legacy_sum) < 1e-9, f"SHAP sum mismatch on test row {idx}"

    # 2. Test across 300 random valid inputs
    random_samples = generate_random_valid_inputs(n=300, seed=42)
    for idx, sample in enumerate(random_samples):
        engine_score = engine.predict_score(sample)
        legacy_score = legacy_utils.predict_score(sample)
        assert abs(engine_score - legacy_score) < 1e-12, f"Score mismatch on random sample {idx}"

        engine_grouped = engine.explain_grouped(sample)
        legacy_grouped = legacy_utils.explain_grouped(sample)

        engine_sum = float(engine_grouped["shap_amount"].sum())
        legacy_sum = float(legacy_grouped["shap_amount"].sum())
        assert abs(engine_sum - legacy_sum) < 1e-9, f"SHAP sum mismatch on random sample {idx}"


def test_explain_grouped_13_rows_and_thal_composition():
    """(b) Test explain_grouped has 13 rows, includes short_label, and thal amount equals the sum of the three nom__thal columns."""
    test_df = engine.load_test_features()
    sample = test_df.iloc[0].to_dict()

    grouped_df = engine.explain_grouped(sample)
    assert len(grouped_df) == 13
    assert "short_label" in grouped_df.columns
    for sl in grouped_df["short_label"]:
        assert "(" not in sl and ")" not in sl

    thal_row = grouped_df[grouped_df["key"] == "thal"].iloc[0]
    thal_amount = float(thal_row["amount"])

    cols_df = engine.explain_columns(sample)
    assert len(cols_df) == 22
    nom_thal_cols = cols_df[cols_df["key"].str.startswith("nom__thal_")]
    assert len(nom_thal_cols) == 3

    sum_nom_thal = float(nom_thal_cols["amount"].sum())
    assert abs(thal_amount - sum_nom_thal) < 1e-12


def test_shap_additivity_300_random_inputs():
    """(c) Test TreeSHAP additivity: base_value + sum(contributions) == score within 1e-4 for 300 inputs."""
    random_samples = generate_random_valid_inputs(n=300, seed=123)
    base_val = engine.get_base_value()

    for idx, sample in enumerate(random_samples):
        score = engine.predict_score(sample)
        grouped_df = engine.explain_grouped(sample)
        shap_sum = float(grouped_df["amount"].sum())

        reconstructed = base_val + shap_sum
        diff = abs(reconstructed - score)
        assert diff < 1e-4, f"Additivity failure on sample {idx}: score={score}, reconstructed={reconstructed}, diff={diff}"


def test_validate_inputs_rules():
    """(d) Test validate_inputs rejects NaN required, strings, unseen categories, warns for bounds, accepts optional."""
    test_df = engine.load_test_features()
    valid_sample = test_df.iloc[0].to_dict()

    # Reject missing required key
    sample_missing_age = {k: v for k, v in valid_sample.items() if k != "age"}
    res = engine.validate_inputs(sample_missing_age)
    assert not res.is_valid
    assert any(e["field"] == "age" for e in res.errors)

    # Reject NaN for required field
    sample_nan_trestbps = dict(valid_sample, trestbps=np.nan)
    res = engine.validate_inputs(sample_nan_trestbps)
    assert not res.is_valid
    assert any(e["field"] == "trestbps" for e in res.errors)

    # Reject non-numeric string
    sample_str = dict(valid_sample, chol="invalid_chol")
    res = engine.validate_inputs(sample_str)
    assert not res.is_valid
    assert any(e["field"] == "chol" for e in res.errors)

    # Reject unseen category
    sample_bad_cat = dict(valid_sample, cp=9)
    res = engine.validate_inputs(sample_bad_cat)
    assert not res.is_valid
    assert any(e["field"] == "cp" for e in res.errors)

    # Reject out of plausible physiological limits (e.g. age=150, chol=5000, trestbps=0)
    res_age_150 = engine.validate_inputs(dict(valid_sample, age=150))
    assert not res_age_150.is_valid
    assert any(e["field"] == "age" for e in res_age_150.errors)
    assert any("Enter an age between 18 and 100" in e["message"] for e in res_age_150.errors)

    res_chol_5000 = engine.validate_inputs(dict(valid_sample, chol=5000))
    assert not res_chol_5000.is_valid
    assert any(e["field"] == "chol" for e in res_chol_5000.errors)
    assert any("Enter a cholesterol level between 80 and 700" in e["message"] for e in res_chol_5000.errors)

    res_trestbps_0 = engine.validate_inputs(dict(valid_sample, trestbps=0))
    assert not res_trestbps_0.is_valid
    assert any(e["field"] == "trestbps" for e in res_trestbps_0.errors)
    assert any("Enter a resting blood pressure between 60 and 260" in e["message"] for e in res_trestbps_0.errors)

    # Warn for out of training range but inside plausible limits (e.g. age=25, age=90)
    sample_out_of_train = dict(valid_sample, age=25)
    res_25 = engine.validate_inputs(sample_out_of_train)
    assert res_25.is_valid
    assert len(res_25.warnings) > 0
    assert any(w["field"] == "age" for w in res_25.warnings)
    assert any("The model learned from patients aged 29 to 77, so this score may be less reliable." in w["message"] for w in res_25.warnings)

    sample_out_of_train_90 = dict(valid_sample, age=90)
    res_90 = engine.validate_inputs(sample_out_of_train_90)
    assert res_90.is_valid
    assert len(res_90.warnings) > 0
    assert any(w["field"] == "age" for w in res_90.warnings)
    assert any("The model learned from patients aged 29 to 77, so this score may be less reliable." in w["message"] for w in res_90.warnings)

    # Accept missing ca and thal
    sample_no_ca_thal = {k: v for k, v in valid_sample.items() if k not in ["ca", "thal"]}
    res = engine.validate_inputs(sample_no_ca_thal)
    assert res.is_valid
    assert set(res.missing_used) == {"ca", "thal"}
    assert np.isnan(res.cleaned_inputs["ca"])
    assert np.isnan(res.cleaned_inputs["thal"])


def test_age_limits_acceptance_and_warnings():
    """Verify ages 18, 25, 28, 29, 77, 100 are accepted, outside 29-77 get warning, and 17, 101 are rejected."""
    test_df = engine.load_test_features()
    valid_sample = test_df.iloc[0].to_dict()
    warning_text = "The model learned from patients aged 29 to 77, so this score may be less reliable."

    # Ages 18, 25, 28: accepted with soft warning
    for age_val in [18, 25, 28]:
        res = engine.validate_inputs(dict(valid_sample, age=age_val))
        assert res.is_valid, f"Age {age_val} should be accepted"
        assert any(w["field"] == "age" and warning_text in w["message"] for w in res.warnings), (
            f"Age {age_val} must have warning: {warning_text}"
        )

    # Ages 29, 77: accepted with NO age warning (within training range)
    for age_val in [29, 77]:
        res = engine.validate_inputs(dict(valid_sample, age=age_val))
        assert res.is_valid, f"Age {age_val} should be accepted"
        age_warnings = [w for w in res.warnings if w["field"] == "age"]
        assert len(age_warnings) == 0, f"Age {age_val} inside training range should not produce age warning"

    # Age 100: accepted with soft warning
    res_100 = engine.validate_inputs(dict(valid_sample, age=100))
    assert res_100.is_valid, "Age 100 should be accepted"
    assert any(w["field"] == "age" and warning_text in w["message"] for w in res_100.warnings)

    # Ages 17, 101: rejected with clear error message
    for bad_age in [17, 101]:
        res_bad = engine.validate_inputs(dict(valid_sample, age=bad_age))
        assert not res_bad.is_valid, f"Age {bad_age} should be rejected"
        age_errors = [e for e in res_bad.errors if e["field"] == "age"]
        assert len(age_errors) > 0, f"Age {bad_age} should produce an error for age"
        assert "Enter an age between 18 and 100" in age_errors[0]["message"]


def test_optional_ca_thal_skipped_equals_pipeline_nan():
    """(e) A request with ca and thal skipped equals pipeline prediction on a row with those values set to NaN within 1e-12."""
    test_df = engine.load_test_features()
    pipeline = engine.load_model_pipeline()
    expected_cols = engine.load_train_features().columns.tolist()

    for idx in range(min(15, len(test_df))):
        orig_row = test_df.iloc[idx].to_dict()

        # Skipped dictionary (omitted keys)
        skipped_dict = {k: v for k, v in orig_row.items() if k not in ["ca", "thal"]}
        engine_score = engine.predict_score(skipped_dict)

        # Direct pipeline prediction with NaN
        nan_row_dict = dict(orig_row, ca=np.nan, thal=np.nan)
        nan_df = pd.DataFrame([nan_row_dict], columns=expected_cols, dtype=np.float64)
        pipeline_score = float(pipeline.predict_proba(nan_df)[0, 1])

        assert abs(engine_score - pipeline_score) < 1e-12, f"Row {idx} mismatch with pipeline NaN"


def test_risk_tier_boundaries():
    """(f) Test risk_tier boundary classifications at 0.299, 0.30, 0.499, and 0.50."""
    tier_299 = engine.risk_tier(0.299)
    assert tier_299["key"] == "lower"
    assert tier_299["label"] == "Lower"
    assert tier_299 == "Lower"

    tier_300 = engine.risk_tier(0.30)
    assert tier_300["key"] == "moderate"
    assert tier_300["label"] == "Moderate"
    assert tier_300 == "Moderate"

    tier_499 = engine.risk_tier(0.499)
    assert tier_499["key"] == "moderate"
    assert tier_499["label"] == "Moderate"
    assert tier_499 == "Moderate"

    tier_500 = engine.risk_tier(0.50)
    assert tier_500["key"] == "higher"
    assert tier_500["label"] == "Higher"
    assert tier_500 == "Higher"


def test_what_if_curve_properties():
    """(g) Test what_if_curve returns 25 rows in [0, 1] and matches predict_score at current value within 1e-9."""
    test_df = engine.load_test_features()
    sample = test_df.iloc[0].to_dict()

    for num_feat in engine.CONTINUOUS_NUMERIC_FEATURES:
        curve_df = engine.what_if_curve(sample, num_feat, n_points=25)
        assert len(curve_df) == 25
        assert (curve_df["score"] >= 0.0).all() and (curve_df["score"] <= 1.0).all()

        curr_val = float(sample[num_feat])
        matching_rows = curve_df[np.isclose(curve_df["value"], curr_val, atol=1e-5)]
        assert len(matching_rows) >= 1

        curve_score_at_curr = float(matching_rows.iloc[0]["score"])
        expected_score = engine.predict_score(sample)
        assert abs(curve_score_at_curr - expected_score) < 1e-9


def test_tier_plain_language_texts():
    """Verify exact plain-language text for tiers."""
    assert engine.risk_tier(0.20)["text"] == "Your answers match patterns the model more often saw in people without heart disease."
    assert engine.risk_tier(0.35)["text"] == "Your answers sit between the two cut-offs, so the model does not lean clearly either way."
    assert engine.risk_tier(0.60)["text"] == "Your answers match patterns the model more often saw in people with heart disease."


def test_option_display_order_and_labels():
    """Verify options are returned in preferred clinical display order."""
    specs = engine.get_feature_specs()
    assert [opt["value"] for opt in specs["cp"]["options"]] == [3, 1, 2, 0]
    assert specs["cp"]["options"][3]["label"] == "Asymptomatic (no chest pain symptoms)"
    assert [opt["value"] for opt in specs["restecg"]["options"]] == [1, 2, 0]
    assert [opt["value"] for opt in specs["slope"]["options"]] == [2, 1, 0]
    assert [opt["value"] for opt in specs["thal"]["options"]] == [2, 1, 3]
    assert [opt["value"] for opt in specs["sex"]["options"]] == [0, 1]


def test_summary_threshold_filtering():
    """Verify answers count as raised/lowered only if absolute amount is >= 0.01."""
    tier_info = engine.risk_tier(0.25)
    # Case 1: no contribution reaches 0.01
    tiny_contribs = [
        {"key": "age", "label": "Age", "short_label": "Age", "value_display": "50", "amount": 0.005, "direction": "raised"},
        {"key": "sex", "label": "Sex", "short_label": "Sex", "value_display": "Male", "amount": -0.008, "direction": "lowered"},
    ]
    summary1 = content.build_summary(0.25, tier_info, tiny_contribs)
    assert "No single answer moved the score by more than one percentage point." in summary1
    assert "Answers that raised" not in summary1
    assert "Answers that lowered" not in summary1

    # Case 2: only raising >= 0.01
    raising_only = [
        {"key": "age", "label": "Age", "short_label": "Age", "value_display": "50", "amount": 0.05, "direction": "raised"},
        {"key": "sex", "label": "Sex", "short_label": "Sex", "value_display": "Male", "amount": -0.005, "direction": "lowered"},
    ]
    summary2 = content.build_summary(0.25, tier_info, raising_only)
    assert "Answers that raised the score most:" in summary2
    assert "Answers that lowered it most:" not in summary2

    # Case 3: only lowering <= -0.01
    lowering_only = [
        {"key": "age", "label": "Age", "short_label": "Age", "value_display": "50", "amount": 0.005, "direction": "raised"},
        {"key": "sex", "label": "Sex", "short_label": "Sex", "value_display": "Male", "amount": -0.06, "direction": "lowered"},
    ]
    summary3 = content.build_summary(0.25, tier_info, lowering_only)
    assert "Answers that lowered it most:" in summary3
    assert "Answers that raised the score most:" not in summary3


def test_no_forbidden_wording_in_summaries():
    """(h) Test that no forbidden words ('you have', 'diagnosis of', 'you are at risk'), no double brackets, and no 'Option' appear."""
    test_df = engine.load_test_features()
    forbidden = ["you have", "diagnosis of", "diagnosis", "you are at risk", "at risk of"]

    for idx in range(len(test_df)):
        sample = test_df.iloc[idx].to_dict()
        score = engine.predict_score(sample)
        tier_info = engine.risk_tier(score)
        grouped_df = engine.explain_grouped(sample)

        summary = content.build_summary(score, tier_info, grouped_df)
        summary_lower = summary.lower()

        for f_phrase in forbidden:
            assert f_phrase not in summary_lower, f"Forbidden phrase '{f_phrase}' found in row {idx} summary: {summary}"

        # No two consecutive bracket groups
        assert not re.search(r"\)\s*\(", summary), f"Two consecutive bracket groups found in row {idx} summary: {summary}"

        # Never print the text 'Option'
        assert "Option" not in summary, f"Text 'Option' found in row {idx} summary: {summary}"


def test_no_emojis_in_engine_api_web():
    """(i) Scan engine/content.py, api/, and web/ for emojis and pictographs."""
    pattern = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")

    scan_paths = [
        REPO_ROOT / "engine" / "content.py",
        REPO_ROOT / "engine" / "core.py",
        REPO_ROOT / "api" / "main.py",
        REPO_ROOT / "web" / "index.html",
    ]

    for p in scan_paths:
        text = p.read_text(encoding="utf-8")
        matches = pattern.findall(text)
        assert len(matches) == 0, f"Emoji or pictograph detected in {p}: {matches}"

