"""Utility functions, cached loaders, and inference routines for the Streamlit application.

IMPORTANT: This module is strictly self-contained and never imports from src/.
All paths are dynamically resolved relative to the repository root.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Root Directory resolution
REPO_ROOT = Path(__file__).resolve().parent.parent

# Centralized Display Labels Dictionary
DISPLAY_LABELS = {
    "age": "Age (years)",
    "sex": "Sex code (0 or 1)",
    "cp": "Chest pain type code (cp)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar code (0 or 1)",
    "restecg": "Resting ECG result code (restecg)",
    "thalach": "Maximum Heart Rate Achieved (bpm)",
    "exang": "Exercise-induced angina code (0 or 1)",
    "oldpeak": "ST Depression (oldpeak)",
    "slope": "ST segment slope code (slope)",
    "ca": "Number of major vessels (ca)",
    "thal": "Thalassemia type code (thal)",
}


class ValidationResult(NamedTuple):
    """Container for input validation errors and out-of-range warnings."""

    errors: list[str]
    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        """Return True if there are zero fatal errors."""
        return len(self.errors) == 0


@st.cache_resource
def load_model_pipeline() -> Any:
    """Load serialized scikit-learn Pipeline (preprocessor + classifier)."""
    model_path = REPO_ROOT / "models" / "best_random_forest.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model pipeline artifact missing: {model_path}")
    return joblib.load(model_path)


@st.cache_resource
def load_shap_explainer() -> shap.TreeExplainer:
    """Load serialized SHAP TreeExplainer."""
    explainer_path = REPO_ROOT / "models" / "shap_explainer.joblib"
    if not explainer_path.exists():
        raise FileNotFoundError(f"TreeExplainer artifact missing: {explainer_path}")
    return joblib.load(explainer_path)


@st.cache_data
def load_evaluation_summary() -> dict:
    """Load evaluation summary JSON."""
    path = REPO_ROOT / "reports" / "evaluation_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation summary missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_shap_summary() -> dict:
    """Load SHAP summary JSON."""
    path = REPO_ROOT / "reports" / "shap_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"SHAP summary missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_training_history() -> dict:
    """Load training history JSON."""
    path = REPO_ROOT / "models" / "training_history.json"
    if not path.exists():
        raise FileNotFoundError(f"Training history missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data_audit_summary() -> dict:
    """Load data audit summary JSON."""
    path = REPO_ROOT / "reports" / "data_audit_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Data audit summary missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_preprocessing_summary() -> dict:
    """Load preprocessing summary JSON."""
    path = REPO_ROOT / "reports" / "preprocessing_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing summary missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_train_features() -> pd.DataFrame:
    """Load training feature matrix X_train.csv."""
    path = REPO_ROOT / "data" / "processed" / "X_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training features missing: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_test_features() -> pd.DataFrame:
    """Load held-out test feature matrix X_test.csv."""
    path = REPO_ROOT / "data" / "processed" / "X_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test features missing: {path}")
    return pd.read_csv(path)


@st.cache_data
def get_feature_specs() -> dict:
    """Derive dynamic widget boundaries and defaults from training data."""
    df_train = load_train_features()

    specs = {}
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # Help text definitions
    help_texts = {
        "age": "Patient age in years (Range observed: 29 to 76).",
        "sex": "Biological sex code (0 or 1). Raw code from dataset.",
        "cp": "Chest pain type code (0 to 3). Raw code from the dataset; its clinical meaning is not verified.",
        "trestbps": "Resting blood pressure in mm Hg upon hospital admission (Range: 94 to 192 mm Hg).",
        "chol": "Serum cholesterol level in mg/dl (Range: 126 to 564 mg/dl).",
        "fbs": "Fasting blood sugar > 120 mg/dl code (0 or 1). Raw code from dataset.",
        "restecg": "Resting electrocardiographic results code (0, 1, or 2). Raw code from dataset.",
        "thalach": "Maximum heart rate achieved during exercise test in bpm (Range: 71 to 202 bpm).",
        "exang": "Exercise-induced angina code (0 or 1). Raw code from dataset.",
        "oldpeak": "ST depression induced by exercise relative to rest (Range: 0.0 to 6.2).",
        "slope": "Slope of peak exercise ST segment code (0, 1, or 2). Raw code from dataset; clinical meaning not verified.",
        "ca": "Number of major vessels (0, 1, 2, or 3) colored by fluoroscopy. Discrete count code.",
        "thal": "Thalassemia type code (1, 2, or 3). Raw code from the dataset; its clinical meaning is not verified.",
    }

    # Grouping categories
    groups = {
        "age": "Patient",
        "sex": "Patient",
        "cp": "Exercise and symptoms",
        "trestbps": "Tests",
        "chol": "Tests",
        "fbs": "Tests",
        "restecg": "Tests",
        "thalach": "Exercise and symptoms",
        "exang": "Exercise and symptoms",
        "oldpeak": "Exercise and symptoms",
        "slope": "Exercise and symptoms",
        "ca": "Tests",
        "thal": "Tests",
    }

    for col in df_train.columns:
        valid_series = df_train[col].dropna()
        label = DISPLAY_LABELS.get(col, col)
        h_text = help_texts.get(col, f"Observed range: {valid_series.min()} to {valid_series.max()}")
        group = groups.get(col, "Tests")

        if col in categorical_cols:
            options = sorted([int(x) for x in valid_series.unique()])
            mode_val = int(valid_series.mode().iloc[0])
            specs[col] = {
                "label": label,
                "help": h_text,
                "widget": "selectbox",
                "options": options,
                "default": mode_val,
                "group": group,
            }
        else:
            min_val = float(valid_series.min())
            max_val = float(valid_series.max())
            median_val = float(valid_series.median())
            is_float = col == "oldpeak"
            specs[col] = {
                "label": label,
                "help": h_text,
                "widget": "slider" if not is_float else "number_input",
                "min": min_val,
                "max": max_val,
                "default": median_val,
                "step": 0.1 if is_float else 1.0,
                "is_float": is_float,
                "group": group,
            }

    return specs


@st.cache_data
def get_typical_values() -> dict:
    """Compute typical clinical inputs (medians for numeric, modes for categorical) from training data."""
    df_train = load_train_features()
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    typical = {}
    for col in df_train.columns:
        valid_s = df_train[col].dropna()
        if col in categorical_cols:
            typical[col] = int(valid_s.mode().iloc[0])
        else:
            is_float = col == "oldpeak"
            val = float(valid_s.median())
            typical[col] = round(val, 1) if is_float else int(round(val))
    return typical


def validate_inputs(inputs: Any) -> ValidationResult:
    """Validate 13 patient input features against training schema and boundaries.

    Returns: ValidationResult(errors, warnings).
    - Errors (fatal): missing keys, NaN/None, non-numeric values, or unseen categorical codes.
    - Warnings (non-fatal): numeric values outside training min/max bounds.
    """
    if isinstance(inputs, pd.DataFrame):
        inputs_dict = inputs.iloc[0].to_dict()
    elif isinstance(inputs, pd.Series):
        inputs_dict = inputs.to_dict()
    else:
        inputs_dict = dict(inputs)

    df_train = load_train_features()
    expected_cols = df_train.columns.tolist()

    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    errors: list[str] = []
    warnings: list[str] = []

    # Valid categories observed in training data (excluding NaN)
    valid_categories = {}
    for col in categorical_cols:
        valid_categories[col] = set(df_train[col].dropna().astype(int).unique())

    # Training ranges for continuous numeric features
    train_ranges = {}
    for col in numeric_cols:
        valid_s = df_train[col].dropna()
        train_ranges[col] = (float(valid_s.min()), float(valid_s.max()))

    for col in expected_cols:
        if col not in inputs_dict:
            errors.append(f"Missing required input feature: '{col}'")
            continue

        raw_val = inputs_dict[col]

        # Check for missing / None / NaN
        if raw_val is None or (isinstance(raw_val, (float, np.floating)) and np.isnan(raw_val)):
            # In Phase 3 preprocessing, ca and thal were the only features where invalid codes (ca=4, thal=0)
            # were converted to NaN and imputed by the fitted preprocessor imputer.
            if col in ["ca", "thal"]:
                continue
            errors.append(f"Feature '{col}' has a missing (NaN/None) value.")
            continue

        # Check for non-numeric (strings that cannot be parsed as float, complex types, etc.)
        try:
            val_float = float(raw_val)
            if np.isnan(val_float) or np.isinf(val_float):
                errors.append(f"Feature '{col}' has invalid non-finite value: {raw_val}")
                continue
        except (ValueError, TypeError):
            errors.append(f"Feature '{col}' must be numeric; got {type(raw_val).__name__} ({raw_val!r})")
            continue

        # Check categorical features
        if col in categorical_cols:
            if not float(val_float).is_integer():
                errors.append(f"Categorical feature '{col}' must be an integer code; got {val_float}")
            else:
                int_code = int(val_float)
                if int_code not in valid_categories[col]:
                    allowed_str = ", ".join(str(c) for c in sorted(valid_categories[col]))
                    errors.append(
                        f"Categorical feature '{col}' has unseen category {int_code}. Allowed values: [{allowed_str}]."
                    )
        elif col in numeric_cols:
            min_bound, max_bound = train_ranges[col]
            if val_float < min_bound or val_float > max_bound:
                disp = DISPLAY_LABELS.get(col, col)
                warnings.append(
                    f"{disp} ({val_float}) is outside training range [{min_bound:.1f}, {max_bound:.1f}]"
                )

    return ValidationResult(errors=errors, warnings=warnings)


def get_recall_threshold() -> float:
    """Retrieve Random Forest recall-oriented operating threshold from evaluation summary."""
    summary = load_evaluation_summary()
    try:
        threshold = summary["recall_oriented_operating_point"]["chosen_threshold"]["random_forest"]["threshold"]
        return float(threshold)
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "Could not locate Random Forest recall-oriented operating threshold in reports/evaluation_summary.json"
        ) from exc


def risk_tier(score: float) -> tuple[str, str, str]:
    """Classify model score into standardized risk tier.

    Returns: (tier_name, badge_text, color_hex)
    """
    threshold_t = get_recall_threshold()

    if score < threshold_t:
        return "Low", "🟢 Low Risk", "#2e7d32"
    elif score < 0.50:
        return "Moderate", "🟡 Moderate Risk", "#f57f17"
    else:
        return "High", "🔴 High Risk", "#c62828"


def predict_score(inputs: Any) -> float:
    """Predict heart disease model score for input dictionary or DataFrame/Series.

    Refuses to run on invalid inputs with a clear ValueError.
    Does not mutate the input dictionary.
    """
    if isinstance(inputs, pd.DataFrame):
        inputs_dict = inputs.iloc[0].to_dict()
    elif isinstance(inputs, pd.Series):
        inputs_dict = inputs.to_dict()
    else:
        inputs_dict = dict(inputs)

    val_res = validate_inputs(inputs_dict)
    if not val_res.is_valid:
        raise ValueError(f"Invalid inputs: {'; '.join(val_res.errors)}")

    df_train = load_train_features()
    expected_cols = df_train.columns.tolist()

    row_data = {col: float(inputs_dict[col]) for col in expected_cols}
    row_df = pd.DataFrame([row_data], columns=expected_cols, dtype=np.float64)

    pipeline = load_model_pipeline()
    prob = pipeline.predict_proba(row_df)[0, 1]
    return float(prob)


def what_if_curve(inputs: Any, feature: str, n_points: int = 25) -> pd.DataFrame:
    """Compute model score curve by varying one numeric feature while holding others fixed.

    Returns a DataFrame with columns ['value', 'score'].
    """
    if isinstance(inputs, pd.DataFrame):
        inputs_dict = inputs.iloc[0].to_dict()
    elif isinstance(inputs, pd.Series):
        inputs_dict = inputs.to_dict()
    else:
        inputs_dict = dict(inputs)

    val_res = validate_inputs(inputs_dict)
    if not val_res.is_valid:
        raise ValueError(f"Invalid inputs: {'; '.join(val_res.errors)}")

    df_train = load_train_features()
    if feature not in df_train.columns:
        raise ValueError(f"Feature '{feature}' not found in training columns.")

    valid_s = df_train[feature].dropna()
    min_val = float(valid_s.min())
    max_val = float(valid_s.max())

    grid_values = np.linspace(min_val, max_val, n_points)

    curr_val = float(inputs_dict[feature])
    if min_val <= curr_val <= max_val:
        closest_idx = int(np.argmin(np.abs(grid_values - curr_val)))
        grid_values[closest_idx] = curr_val
        grid_values = np.sort(grid_values)

    expected_cols = df_train.columns.tolist()
    pipeline = load_model_pipeline()

    rows = []
    for val in grid_values:
        row_dict = {c: float(inputs_dict[c]) for c in expected_cols}
        row_dict[feature] = float(val)
        rows.append(row_dict)

    batch_df = pd.DataFrame(rows, columns=expected_cols, dtype=np.float64)
    probs = pipeline.predict_proba(batch_df)[:, 1]

    return pd.DataFrame({
        "value": grid_values,
        "score": probs,
    })


def explain_patient(inputs: Any) -> shap.Explanation:
    """Generate SHAP Explanation for patient input with original display units.

    Refuses to run on invalid inputs with a clear ValueError.
    Does not mutate the input dictionary.
    """
    if isinstance(inputs, pd.DataFrame):
        inputs_dict = inputs.iloc[0].to_dict()
    elif isinstance(inputs, pd.Series):
        inputs_dict = inputs.to_dict()
    else:
        inputs_dict = dict(inputs)

    val_res = validate_inputs(inputs_dict)
    if not val_res.is_valid:
        raise ValueError(f"Invalid inputs: {'; '.join(val_res.errors)}")

    df_train = load_train_features()
    expected_cols = df_train.columns.tolist()

    row_data = {col: float(inputs_dict[col]) for col in expected_cols}
    row_df = pd.DataFrame([row_data], columns=expected_cols, dtype=np.float64)

    pipeline = load_model_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    explainer = load_shap_explainer()

    # Transform feature values
    row_trans = preprocessor.transform(row_df)
    feature_names = list(row_trans.columns)

    # Compute SHAP values
    shap_raw = explainer.shap_values(row_trans)

    if isinstance(shap_raw, list):
        shap_values_class1 = shap_raw[1][0]
        base_val = float(explainer.expected_value[1])
    elif isinstance(shap_raw, np.ndarray):
        if shap_raw.ndim == 3 and shap_raw.shape[2] == 2:
            shap_values_class1 = shap_raw[0, :, 1]
            base_val = float(explainer.expected_value[1])
        elif shap_raw.ndim == 2:
            shap_values_class1 = shap_raw[0]
            base_val = float(explainer.expected_value)
        else:
            raise ValueError(f"Unexpected SHAP array shape: {shap_raw.shape}")
    else:
        raise TypeError(f"Unexpected SHAP return type: {type(shap_raw)}")

    # Prepare display data in original units for numeric/count features
    display_data = np.zeros(len(feature_names))

    for j, col_name in enumerate(feature_names):
        if col_name.startswith("num__"):
            orig_col = col_name.replace("num__", "")
            display_data[j] = float(row_df[orig_col].iloc[0])
        else:
            display_data[j] = float(row_trans.iloc[0, j])

    return shap.Explanation(
        values=shap_values_class1,
        base_values=base_val,
        data=display_data,
        feature_names=feature_names,
    )


def tree_agreement(inputs: Any) -> tuple[np.ndarray, float, float]:
    """Compute per-tree predictions and agreement statistics across the Random Forest ensemble.

    Returns:
        (tree_scores, share_ge_05, mean_score)
        where:
        - tree_scores: 1D np.ndarray of length n_estimators with each tree's predicted probability for class 1.
        - share_ge_05: float in [0, 1] representing the fraction of trees with score >= 0.50.
        - mean_score: float arithmetic mean of all tree scores (identical to ensemble predict_proba).
    """
    if isinstance(inputs, pd.DataFrame):
        inputs_dict = inputs.iloc[0].to_dict()
    elif isinstance(inputs, pd.Series):
        inputs_dict = inputs.to_dict()
    else:
        inputs_dict = dict(inputs)

    val_res = validate_inputs(inputs_dict)
    if not val_res.is_valid:
        raise ValueError(f"Invalid inputs: {'; '.join(val_res.errors)}")

    df_train = load_train_features()
    expected_cols = df_train.columns.tolist()

    row_data = {col: float(inputs_dict[col]) for col in expected_cols}
    row_df = pd.DataFrame([row_data], columns=expected_cols, dtype=np.float64)

    pipeline = load_model_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    rf_classifier = pipeline.named_steps["classifier"]

    row_trans = preprocessor.transform(row_df)
    row_trans_arr = np.asarray(row_trans)

    tree_scores_list = []
    for tree in rf_classifier.estimators_:
        if tree.n_classes_ == 2:
            p1 = float(tree.predict_proba(row_trans_arr)[0, 1])
        else:
            p1 = 1.0 if tree.classes_[0] == 1 else 0.0
        tree_scores_list.append(p1)

    tree_scores = np.array(tree_scores_list, dtype=np.float64)
    share_ge_05 = float(np.mean(tree_scores >= 0.50))
    mean_score = float(np.mean(tree_scores))

    return tree_scores, share_ge_05, mean_score


def explain_grouped(inputs: Any) -> pd.DataFrame:
    """Compute patient SHAP values grouped back into the 13 original clinical features.

    Uses exact column-name matching (never prefix startswith) to prevent thal/thalach collisions.
    Returns:
        pd.DataFrame with columns ['feature', 'label', 'value_entered', 'shap_amount', 'abs_shap', 'direction', 'sign_str']
        sorted by abs_shap descending.
    """
    if isinstance(inputs, pd.DataFrame):
        inputs_dict = inputs.iloc[0].to_dict()
    elif isinstance(inputs, pd.Series):
        inputs_dict = inputs.to_dict()
    else:
        inputs_dict = dict(inputs)

    explanation = explain_patient(inputs_dict)
    feature_names = explanation.feature_names
    shap_vals = explanation.values

    orig_13 = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    grouped_records = []
    for f in orig_13:
        matching_cols = [
            c for c in feature_names
            if c in [f"num__{f}", f"bin__{f}"] or c.startswith(f"nom__{f}_")
        ]
        col_indices = [feature_names.index(c) for c in matching_cols]
        feature_shap = float(np.sum(shap_vals[col_indices]))
        val_entered = inputs_dict.get(f, np.nan)

        grouped_records.append({
            "feature": f,
            "label": DISPLAY_LABELS.get(f, f),
            "value_entered": val_entered,
            "shap_amount": feature_shap,
            "abs_shap": abs(feature_shap),
            "direction": "Raised score (+)" if feature_shap >= 0 else "Lowered score (−)",
            "sign_str": f"+{feature_shap:.4f}" if feature_shap >= 0 else f"{feature_shap:.4f}",
        })

    df_grouped = pd.DataFrame(grouped_records)
    df_grouped = df_grouped.sort_values(by="abs_shap", ascending=False).reset_index(drop=True)
    return df_grouped


def build_summary_sentence(score: float, tier_name: str, grouped_df: pd.DataFrame) -> str:
    """Build neutral, factual one-sentence summary of model score and top local contributors."""
    raising = [r for _, r in grouped_df.iterrows() if r["shap_amount"] > 0]
    lowering = [r for _, r in grouped_df.iterrows() if r["shap_amount"] < 0]

    raising_parts = []
    for r in raising[:2]:
        raising_parts.append(f"{r['label']} (entered: {r['value_entered']}, contribution: +{r['shap_amount']:.4f})")
    raising_str = " and ".join(raising_parts) if raising_parts else "none"

    if lowering:
        low = lowering[0]
        lowering_str = f"{low['label']} (entered: {low['value_entered']}, contribution: {low['shap_amount']:.4f})"
    else:
        lowering_str = "no inputs"

    if raising_str != "none":
        return (
            f"The model estimated a {score*100.0:.1f}% risk score ({tier_name} Risk), with {raising_str} "
            f"contributing most to raising the score, while {lowering_str} contributed most to lowering the score."
        )
    else:
        return (
            f"The model estimated a {score*100.0:.1f}% risk score ({tier_name} Risk), with {lowering_str} "
            f"contributing to lowering the score."
        )


def generate_html_report(
    inputs: dict,
    score: float,
    tier_name: str,
    summary_sentence: str,
    grouped_df: pd.DataFrame,
    share_ge_05: float,
) -> str:
    """Generate in-memory, self-contained single-page HTML clinical risk report."""
    import datetime
    import html as html_lib

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_threshold = get_recall_threshold()

    table_rows = []
    for _, row in grouped_df.iterrows():
        f_label = html_lib.escape(str(row["label"]))
        val = html_lib.escape(str(row["value_entered"]))
        amt = html_lib.escape(f"{row['shap_amount']:+.4f}")
        dir_color = "#C8423B" if row["shap_amount"] >= 0 else "#1E8E5A"
        dir_text = "Raised (+)" if row["shap_amount"] >= 0 else "Lowered (−)"
        table_rows.append(
            f"<tr><td>{f_label}</td><td><code>{val}</code></td>"
            f"<td style='color:{dir_color}; font-weight:600;'>{amt}</td><td>{dir_text}</td></tr>"
        )
    table_html = "\n".join(table_rows)

    disclaimer_text = html_lib.escape(
        "Educational project, not a medical diagnosis. Do not use this tool to make health decisions."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Heart Disease Risk Assessment Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #F6F8FB; color: #0F1B2D; margin: 0; padding: 30px; }}
  .container {{ max-width: 820px; margin: 0 auto; background: #FFFFFF; border: 1px solid #E3E8EF; border-radius: 12px; padding: 32px; box-shadow: 0 2px 10px rgba(0,0,0,0.04); }}
  h1 {{ font-size: 24px; margin-top: 0; color: #0F1B2D; }}
  .meta {{ color: #5B6B7F; font-size: 13px; margin-bottom: 24px; }}
  .kpi-row {{ display: flex; gap: 20px; margin-bottom: 24px; }}
  .kpi-card {{ flex: 1; background: #F8FAFC; border: 1px solid #E3E8EF; border-radius: 8px; padding: 16px; text-align: center; }}
  .kpi-num {{ font-size: 32px; font-weight: 700; color: #0F1B2D; }}
  .kpi-label {{ font-size: 13px; color: #5B6B7F; text-transform: uppercase; letter-spacing: 0.05em; }}
  .summary-box {{ background: #F0F9FF; border-left: 4px solid #0E7C86; padding: 14px 18px; border-radius: 4px; font-size: 14px; margin-bottom: 24px; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 24px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #E3E8EF; }}
  th {{ background: #F8FAFC; color: #5B6B7F; font-weight: 600; }}
  .disclaimer {{ background: #FFF8E1; border: 1px solid #FFE082; padding: 14px; border-radius: 6px; font-size: 13px; color: #5D4037; text-align: center; font-weight: 600; margin-top: 24px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🫀 Patient Risk Assessment Report</h1>
  <div class="meta">Generated: {now_str} | Screening Threshold: {t_threshold:.2f} | High Risk Threshold: 0.50</div>
  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-num">{score*100.0:.1f}%</div>
      <div class="kpi-label">Predicted Model Score</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num">{html_lib.escape(tier_name)}</div>
      <div class="kpi-label">Assigned Risk Tier</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num">{share_ge_05*100.0:.1f}%</div>
      <div class="kpi-label">Ensemble Agreement (≥0.50)</div>
    </div>
  </div>
  <div class="summary-box">
    <strong>Case Assessment:</strong> {html_lib.escape(summary_sentence)}
  </div>
  <h3>Clinical Attribute Contributions (TreeSHAP Grouped)</h3>
  <table>
    <thead>
      <tr><th>Clinical Indicator</th><th>Entered Value</th><th>SHAP Impact</th><th>Direction</th></tr>
    </thead>
    <tbody>
      {table_html}
    </tbody>
  </table>
  <div class="disclaimer">
    ⚠️ {disclaimer_text}
  </div>
</div>
</body>
</html>"""
