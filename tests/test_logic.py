"""Automated test suite for application logic, inference safety, and UI consistency.

IMPORTANT: Strictly imports ONLY from app.utils and public libraries.
Never imports from src/.
"""

import copy
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app import utils

REPO_ROOT = utils.REPO_ROOT
X_test = pd.read_csv(REPO_ROOT / "data" / "processed" / "X_test.csv")
pipeline = utils.load_model_pipeline()
shap_summary = utils.load_shap_summary()
exemplars = shap_summary["selected_exemplar_rows"]


def test_a_predict_score_all_61_test_rows():
    """Verify predict_score equals pipeline.predict_proba within 1e-9 on all 61 test rows."""
    expected_cols = X_test.columns.tolist()
    for idx in range(len(X_test)):
        row_dict = X_test.iloc[idx].to_dict()
        score = utils.predict_score(row_dict)
        row_df = pd.DataFrame([{col: float(row_dict[col]) for col in expected_cols}], columns=expected_cols, dtype=np.float64)
        direct_prob = float(pipeline.predict_proba(row_df)[0, 1])
        assert abs(score - direct_prob) < 1e-9, f"Row {idx} score mismatch: {score} vs {direct_prob}"


def test_b_stored_exemplar_rows():
    """Verify predict_score matches probabilities in shap_summary.json within 1e-4."""
    for case_name, details in exemplars.items():
        idx = details["test_row_index"]
        expected_prob = details["predicted_probability"]
        row_dict = X_test.iloc[idx].to_dict()
        score = utils.predict_score(row_dict)
        assert abs(score - expected_prob) < 1e-4, f"Exemplar {case_name} (row {idx}): {score} vs {expected_prob}"


def test_c_300_random_valid_inputs():
    """Verify 300 random valid inputs: score in [0, 1], explain_patient works, additivity holds within 1e-4."""
    rng = np.random.default_rng(42)
    specs = utils.get_feature_specs()

    for i in range(300):
        sample_input = {}
        for col, spec in specs.items():
            if spec["widget"] == "selectbox":
                sample_input[col] = int(rng.choice(spec["options"]))
            elif spec.get("is_float", False):
                sample_input[col] = float(rng.uniform(spec["min"], spec["max"]))
            else:
                sample_input[col] = int(rng.integers(int(spec["min"]), int(spec["max"]) + 1))

        score = utils.predict_score(sample_input)
        assert not np.isnan(score), f"Sample {i} returned NaN score"
        assert 0.0 <= score <= 1.0, f"Sample {i} score out of bounds: {score}"

        exp = utils.explain_patient(sample_input)
        shap_sum = float(exp.base_values + np.sum(exp.values))
        assert abs(shap_sum - score) < 1e-4, f"Sample {i} additivity violated: base + sum(shap) = {shap_sum}, score = {score}"


def test_d_validate_inputs():
    """Verify validate_inputs rejects NaN, strings, unseen categories; warns for > max; accepts 61 test rows."""
    base_row = X_test.iloc[0].to_dict()

    # Rejects NaN
    row_nan = copy.deepcopy(base_row)
    row_nan["age"] = np.nan
    res_nan = utils.validate_inputs(row_nan)
    assert not res_nan.is_valid and any("missing" in err.lower() or "nan" in err.lower() for err in res_nan.errors)
    with pytest.raises(ValueError):
        utils.predict_score(row_nan)

    # Rejects String
    row_str = copy.deepcopy(base_row)
    row_str["trestbps"] = "high"
    res_str = utils.validate_inputs(row_str)
    assert not res_str.is_valid and any("numeric" in err.lower() for err in res_str.errors)
    with pytest.raises(ValueError):
        utils.predict_score(row_str)

    # Rejects Unseen Category
    row_cat = copy.deepcopy(base_row)
    row_cat["cp"] = 99
    res_cat = utils.validate_inputs(row_cat)
    assert not res_cat.is_valid and any("unseen category" in err.lower() for err in res_cat.errors)

    row_thal0 = copy.deepcopy(base_row)
    row_thal0["thal"] = 0
    res_thal0 = utils.validate_inputs(row_thal0)
    assert not res_thal0.is_valid and any("unseen category" in err.lower() for err in res_thal0.errors)

    row_ca4 = copy.deepcopy(base_row)
    row_ca4["ca"] = 4
    res_ca4 = utils.validate_inputs(row_ca4)
    assert not res_ca4.is_valid and any("unseen category" in err.lower() for err in res_ca4.errors)

    # Warns for numeric value above training maximum (does not error)
    row_high = copy.deepcopy(base_row)
    row_high["age"] = 120.0
    res_high = utils.validate_inputs(row_high)
    assert res_high.is_valid
    assert len(res_high.warnings) > 0 and any("age" in warn.lower() for warn in res_high.warnings)
    # predict_score should still succeed despite warning
    score_high = utils.predict_score(row_high)
    assert 0.0 <= score_high <= 1.0

    # Accepts all 61 test rows with no fatal errors
    for idx in range(len(X_test)):
        row = X_test.iloc[idx].to_dict()
        res_test = utils.validate_inputs(row)
        assert res_test.is_valid, f"Test row {idx} rejected: {res_test.errors}"


def test_e_risk_tier():
    """Verify risk_tier thresholds and error on missing threshold."""
    assert utils.risk_tier(0.299)[0] == "Low"
    assert utils.risk_tier(0.300)[0] == "Moderate"
    assert utils.risk_tier(0.499)[0] == "Moderate"
    assert utils.risk_tier(0.500)[0] == "High"

    # Raises clear error if threshold missing
    with mock.patch("app.utils.load_evaluation_summary", return_value={}):
        with pytest.raises(KeyError) as exc_info:
            utils.risk_tier(0.40)
        assert "Could not locate Random Forest recall-oriented operating threshold" in str(exc_info.value)


def test_f_predict_score_immutability_and_determinism():
    """Verify predict_score does not mutate input dictionary and is deterministic."""
    sample = X_test.iloc[10].to_dict()
    sample_snapshot = copy.deepcopy(sample)

    score1 = utils.predict_score(sample)
    assert sample == sample_snapshot, "predict_score mutated the input dictionary"

    score2 = utils.predict_score(sample)
    assert abs(score1 - score2) < 1e-9, f"predict_score was non-deterministic: {score1} vs {score2}"


def test_g_what_if_curve():
    """Verify what_if_curve returns 25 rows in [0, 1] and matches predict_score when value is included."""
    sample = X_test.iloc[5].to_dict()
    df_curve = utils.what_if_curve(sample, "chol", n_points=25)

    assert len(df_curve) == 25, f"Expected 25 points, got {len(df_curve)}"
    assert (df_curve["score"] >= 0.0).all() and (df_curve["score"] <= 1.0).all()

    # Current value is included in the grid
    curr_chol = float(sample["chol"])
    mask = np.isclose(df_curve["value"], curr_chol)
    assert mask.any(), f"Current chol ({curr_chol}) was not included in what-if grid"

    curve_score_at_curr = float(df_curve.loc[mask, "score"].iloc[0])
    direct_score = utils.predict_score(sample)
    assert abs(curve_score_at_curr - direct_score) < 1e-9, f"Mismatch at current input: {curve_score_at_curr} vs {direct_score}"


def test_h_apptest_pages_and_exemplars():
    """Verify AppTest runs all pages without exceptions and exemplar button clicks match predict_score."""
    pages = [
        "app/app.py",
        "app/pages/1_Model_Performance.py",
        "app/pages/2_Global_Explanations.py",
        "app/pages/3_About.py",
    ]

    for p in pages:
        at = AppTest.from_file(str(REPO_ROOT / p))
        at.run(timeout=30)
        assert len(at.exception) == 0, f"Exceptions on page {p}: {[e.value for e in at.exception]}"

    # Test main app exemplar buttons
    at_main = AppTest.from_file(str(REPO_ROOT / "app" / "app.py"))
    at_main.run(timeout=30)
    assert len(at_main.exception) == 0

    # 4 sidebar buttons: [0]: High score #41, [1]: Low score #6, [2]: Missed at 0.5 #18, [3]: Reset to typical
    # Button 0: High score (#41)
    at_main.button[0].click().run(timeout=30)
    assert len(at_main.exception) == 0
    score_41 = utils.predict_score(X_test.iloc[41].to_dict())
    assert at_main.metric[0].value == f"{score_41 * 100.0:.1f}%"

    # Button 1: Low score (#6)
    at_main.button[1].click().run(timeout=30)
    assert len(at_main.exception) == 0
    score_6 = utils.predict_score(X_test.iloc[6].to_dict())
    assert at_main.metric[0].value == f"{score_6 * 100.0:.1f}%"

    # Button 2: Missed at 0.5 (#18)
    at_main.button[2].click().run(timeout=30)
    assert len(at_main.exception) == 0
    score_18 = utils.predict_score(X_test.iloc[18].to_dict())
    assert at_main.metric[0].value == f"{score_18 * 100.0:.1f}%"

