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


def test_i_tree_agreement():
    """Verify mean over all trees equals pipeline.predict_proba[:, 1] within 1e-9 for all 61 test rows."""
    expected_cols = X_test.columns.tolist()
    for idx in range(len(X_test)):
        row_dict = X_test.iloc[idx].to_dict()
        tree_scores, share_ge_05, mean_tree_score = utils.tree_agreement(row_dict)
        row_df = pd.DataFrame([{col: float(row_dict[col]) for col in expected_cols}], columns=expected_cols, dtype=np.float64)
        direct_prob = float(pipeline.predict_proba(row_df)[0, 1])
        assert abs(mean_tree_score - direct_prob) < 1e-9, f"Row {idx} tree agreement mean mismatch: {mean_tree_score} vs {direct_prob}"
        assert len(tree_scores) == 400


def test_j_explain_grouped():
    """Verify explain_grouped returns 13 rows, sums to 22-column SHAP sum within 1e-9, and thal equals sum of nom__thal columns."""
    sample = X_test.iloc[0].to_dict()
    df_grouped = utils.explain_grouped(sample)
    assert len(df_grouped) == 13, f"Expected 13 rows, got {len(df_grouped)}"
    assert set(df_grouped["feature"]) == set(sample.keys())

    # Direct 22-column SHAP
    raw_exp = utils.explain_patient(sample)
    raw_shap_sum = float(np.sum(raw_exp.values))
    grouped_shap_sum = float(df_grouped["shap_amount"].sum())
    assert abs(grouped_shap_sum - raw_shap_sum) < 1e-9, f"Grouped SHAP sum mismatch: {grouped_shap_sum} vs {raw_shap_sum}"

    # thal row equals sum of the three nom__thal columns
    thal_grouped_val = float(df_grouped.loc[df_grouped["feature"] == "thal", "shap_amount"].iloc[0])
    thal_col_indices = [i for i, name in enumerate(raw_exp.feature_names) if "nom__thal" in name or name == "thal"]
    expected_thal_sum = float(np.sum(raw_exp.values[thal_col_indices]))
    assert abs(thal_grouped_val - expected_thal_sum) < 1e-9, f"thal sum mismatch: {thal_grouped_val} vs {expected_thal_sum}"


def test_k_risk_meter_boundaries():
    """Verify zone boundaries equal the cut-offs read from reports/evaluation_summary.json (0.30 and 0.50)."""
    t_opt = utils.get_recall_threshold()
    assert abs(t_opt - 0.30) < 1e-4, f"Operating threshold mismatch: {t_opt}"

    t_low = t_opt - 1e-4
    t_mod1 = t_opt
    t_mod2 = 0.50 - 1e-4
    t_high = 0.50

    assert utils.risk_tier(t_low)[0] == "Low"
    assert utils.risk_tier(t_mod1)[0] == "Moderate"
    assert utils.risk_tier(t_mod2)[0] == "Moderate"
    assert utils.risk_tier(t_high)[0] == "High"


def test_l_quick_start_buttons():
    """Verify displayed score after each Quick start button equals predict_score for that patient."""
    at = AppTest.from_file(str(REPO_ROOT / "app" / "app.py"))
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Button 0: High score (row 41)
    at.button[0].click().run(timeout=30)
    assert len(at.exception) == 0
    score_41 = utils.predict_score(X_test.iloc[41].to_dict())
    assert at.metric[0].value == f"{score_41 * 100.0:.1f}%"

    # Button 1: Low score (row 6)
    at.button[1].click().run(timeout=30)
    assert len(at.exception) == 0
    score_6 = utils.predict_score(X_test.iloc[6].to_dict())
    assert at.metric[0].value == f"{score_6 * 100.0:.1f}%"

    # Button 2: Borderline / Missed at 0.5 (row 18)
    at.button[2].click().run(timeout=30)
    assert len(at.exception) == 0
    score_18 = utils.predict_score(X_test.iloc[18].to_dict())
    assert at.metric[0].value == f"{score_18 * 100.0:.1f}%"

    # Button 3: Reset to typical values
    at.button[3].click().run(timeout=30)
    assert len(at.exception) == 0
    score_typical = utils.predict_score(utils.get_typical_values())
    assert at.metric[0].value == f"{score_typical * 100.0:.1f}%"


def test_m_html_and_csv_reports():
    """Verify HTML report contains disclaimer sentence and no unescaped '<script'; CSV contains inputs, score, tier, disclaimer."""
    from app.components import COPY
    disclaimer = COPY["DISCLAIMER"]
    assert disclaimer == "Educational project, not a medical diagnosis. Do not use this tool to make health decisions."

    sample = X_test.iloc[2].to_dict()
    score = utils.predict_score(sample)
    tier_name, badge_text, _ = utils.risk_tier(score)
    df_grouped = utils.explain_grouped(sample)
    tree_scores, share_ge_05, mean_score = utils.tree_agreement(sample)
    summary_sentence = utils.build_summary_sentence(score, tier_name, df_grouped)

    html_report = utils.generate_html_report(sample, score, tier_name, summary_sentence, df_grouped, share_ge_05)
    assert disclaimer in html_report, "HTML report missing verbatim medical disclaimer"
    assert "<script" not in html_report.lower(), "HTML report contains potentially unsafe <script tag"
    assert "<html" in html_report.lower() and "</html>" in html_report.lower()

    # CSV report verification
    import io
    csv_row = dict(sample)
    csv_row["predicted_score"] = round(score, 4)
    csv_row["risk_tier"] = tier_name
    csv_row["disclaimer"] = disclaimer
    df_csv = pd.DataFrame([csv_row])
    csv_buffer = io.StringIO()
    df_csv.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    assert disclaimer in csv_content
    read_df = pd.read_csv(io.StringIO(csv_content))
    expected_cols = set(sample.keys()) | {"predicted_score", "risk_tier", "disclaimer"}
    assert set(read_df.columns) == expected_cols
    assert read_df["risk_tier"].iloc[0] == tier_name


