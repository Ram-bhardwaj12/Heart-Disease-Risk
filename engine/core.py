"""Framework-independent inference, validation, and explainability engine for CardioLens.

IMPORTANT: Strictly zero imports of streamlit, src, or app.
All paths are resolved relative to the repository root.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd
import shap

from engine.content import (
    CODE_LABELS,
    DISCLAIMER,
    FIELD_GROUPS,
    FIELD_METADATA,
    OPTIONAL_FIELDS,
    OPTION_DISPLAY_ORDER,
    PRODUCT_NAME,
    REQUIRED_FIELDS,
    TAGLINE,
    TIERS,
    UNVERIFIED_CODES,
    build_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
CONTINUOUS_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

PLAUSIBLE_LIMITS: dict[str, tuple[float, float]] = {
    "age": (18.0, 100.0),
    "trestbps": (60.0, 260.0),
    "chol": (80.0, 700.0),
    "thalach": (40.0, 230.0),
    "oldpeak": (0.0, 10.0),
}


class ValidationResult(NamedTuple):
    """Validation outcome holding status, errors, warnings, cleaned inputs, and missing keys."""

    is_valid: bool
    errors: list[dict[str, str]]
    warnings: list[dict[str, str]]
    cleaned_inputs: dict[str, Any]
    missing_used: list[str]


class RiskTier(dict):
    """Risk tier descriptor with both dictionary and attribute accessibility."""

    def __init__(self, key: str, label: str, text: str):
        super().__init__(key=key, label=label, text=text)
        self.key = key
        self.label = label
        self.text = text

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.key == other.lower() or self.label.lower() == other.lower()
        return super().__eq__(other)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, int):
            return [self.key, self.label, self.text][item]
        return super().__getitem__(item)


# -----------------------------------------------------------------------------
# Lazy Cached Loaders
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_train_features() -> pd.DataFrame:
    """Load training features from data/processed/X_train.csv."""
    path = REPO_ROOT / "data" / "processed" / "X_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training features missing: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_test_features() -> pd.DataFrame:
    """Load test features from data/processed/X_test.csv."""
    path = REPO_ROOT / "data" / "processed" / "X_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test features missing: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_model_pipeline() -> Any:
    """Load trained scikit-learn Pipeline (preprocessor + classifier)."""
    path = REPO_ROOT / "models" / "best_random_forest.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model pipeline missing: {path}")
    pipeline = joblib.load(path)
    if hasattr(pipeline, "named_steps") and "classifier" in pipeline.named_steps:
        pipeline.named_steps["classifier"].n_jobs = 1
    return pipeline


@lru_cache(maxsize=1)
def load_shap_explainer() -> shap.TreeExplainer:
    """Load trained SHAP TreeExplainer."""
    path = REPO_ROOT / "models" / "shap_explainer.joblib"
    if not path.exists():
        raise FileNotFoundError(f"SHAP explainer missing: {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_evaluation_summary() -> dict:
    """Load evaluation summary report."""
    path = REPO_ROOT / "reports" / "evaluation_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation summary missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_shap_summary() -> dict:
    """Load SHAP summary report."""
    path = REPO_ROOT / "reports" / "shap_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"SHAP summary missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_data_audit_summary() -> dict:
    """Load data audit summary report."""
    path = REPO_ROOT / "reports" / "data_audit_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Data audit summary missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_cutoffs() -> tuple[float, float]:
    """Retrieve decision cut-offs (low_threshold, high_threshold=0.50)."""
    summary = load_evaluation_summary()
    try:
        low_t = float(
            summary["recall_oriented_operating_point"]["chosen_threshold"]["random_forest"]["threshold"]
        )
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "Could not locate Random Forest operating threshold in reports/evaluation_summary.json"
        ) from exc
    return low_t, 0.50


@lru_cache(maxsize=1)
def get_base_value() -> float:
    """Return expected value (base value) for class 1 from the fitted explainer."""
    explainer = load_shap_explainer()
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1])
    return float(ev)


# -----------------------------------------------------------------------------
# Dynamic Feature Specifications
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_feature_specs() -> dict[str, dict[str, Any]]:
    """Derive ranges, medians, modes, and allowed options from training data at load time."""
    df_train = load_train_features()
    specs: dict[str, dict[str, Any]] = {}

    for col in df_train.columns:
        valid_series = df_train[col].dropna()
        meta = FIELD_METADATA.get(col, {})
        label = meta.get("label", col)
        help_text = meta.get("help", "")
        group = meta.get("group", "report")
        field_type = meta.get("type", "number")
        unit = meta.get("unit", "")
        optional = meta.get("optional", False)

        if col in CATEGORICAL_FEATURES:
            raw_options = sorted([int(x) for x in valid_series.unique()])
            preferred_order = OPTION_DISPLAY_ORDER.get(col)
            if preferred_order:
                order_map = {val: idx for idx, val in enumerate(preferred_order)}
                raw_options.sort(key=lambda x: order_map.get(x, 999))
            mode_val = int(valid_series.mode().iloc[0])
            options_list = [
                {"value": opt, "label": CODE_LABELS.get(col, {}).get(opt, str(opt))}
                for opt in raw_options
            ]
            specs[col] = {
                "key": col,
                "label": label,
                "short_label": meta.get("short_label", label),
                "help": help_text,
                "group": group,
                "type": "choice",
                "unit": unit,
                "min": None,
                "max": None,
                "step": None,
                "default": mode_val,
                "options": options_list,
                "optional": optional,
            }
        else:
            plausible_min, plausible_max = PLAUSIBLE_LIMITS.get(col, (float(valid_series.min()), float(valid_series.max())))
            train_min = float(valid_series.min())
            train_max = float(valid_series.max())
            median_val = float(valid_series.median())
            step_val = meta.get("step", 1.0)
            specs[col] = {
                "key": col,
                "label": label,
                "short_label": meta.get("short_label", label),
                "help": help_text,
                "group": group,
                "type": field_type,
                "unit": unit,
                "min": plausible_min,
                "max": plausible_max,
                "training_min": train_min,
                "training_max": train_max,
                "step": step_val,
                "default": round(median_val, 1) if step_val == 0.1 else round(median_val),
                "options": [],
                "optional": optional,
            }

    return specs


# -----------------------------------------------------------------------------
# Input Validation
# -----------------------------------------------------------------------------


def validate_inputs(inputs: Any) -> ValidationResult:
    """Validate 13 patient input features against training schema and boundaries.

    - Rejects missing keys for required fields, NaN/None for required fields, non-numeric strings,
      and unseen categorical codes.
    - Accepts missing/None/NaN for optional fields ('ca', 'thal').
    - Emits warnings (non-fatal) when numeric features exceed training min/max bounds.
    """
    if isinstance(inputs, pd.DataFrame):
        inputs_dict = inputs.iloc[0].to_dict()
    elif isinstance(inputs, pd.Series):
        inputs_dict = inputs.to_dict()
    else:
        inputs_dict = dict(inputs)

    df_train = load_train_features()
    expected_cols = df_train.columns.tolist()

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    cleaned: dict[str, Any] = {}
    missing_used: list[str] = []

    valid_categories = {
        col: set(df_train[col].dropna().astype(int).unique()) for col in CATEGORICAL_FEATURES
    }

    train_ranges = {
        col: (float(df_train[col].dropna().min()), float(df_train[col].dropna().max()))
        for col in CONTINUOUS_NUMERIC_FEATURES
    }

    for col in expected_cols:
        is_optional = col in OPTIONAL_FIELDS
        short_lbl = FIELD_METADATA.get(col, {}).get("short_label", col).lower()

        # 1. Missing key check
        if col not in inputs_dict:
            if is_optional:
                missing_used.append(col)
                cleaned[col] = np.nan
                continue
            errors.append({
                "field": col,
                "message": f"Please enter a value for {short_lbl}.",
            })
            continue

        raw_val = inputs_dict[col]

        # 2. None / NaN / Empty check
        is_missing_val = (
            raw_val is None
            or raw_val == ""
            or (isinstance(raw_val, (float, np.floating)) and np.isnan(raw_val))
        )
        if is_missing_val:
            if is_optional:
                missing_used.append(col)
                cleaned[col] = np.nan
                continue
            errors.append({
                "field": col,
                "message": f"Please enter a value for {short_lbl}.",
            })
            continue

        # 3. Numeric conversion check
        try:
            val_float = float(raw_val)
            if np.isnan(val_float) or np.isinf(val_float):
                errors.append({
                    "field": col,
                    "message": f"Please enter a valid number for {short_lbl}.",
                })
                continue
        except (ValueError, TypeError):
            errors.append({
                "field": col,
                "message": f"Please enter a valid number for {short_lbl}.",
            })
            continue

        # 4. Categorical validation
        if col in CATEGORICAL_FEATURES:
            if not float(val_float).is_integer():
                errors.append({
                    "field": col,
                    "message": f"Please select a valid option for {short_lbl}.",
                })
            else:
                int_code = int(val_float)
                if int_code not in valid_categories[col]:
                    errors.append({
                        "field": col,
                        "message": f"Please select a valid option for {short_lbl}.",
                    })
                else:
                    cleaned[col] = float(int_code)

        # 5. Continuous numeric validation
        elif col in CONTINUOUS_NUMERIC_FEATURES:
            min_plausible, max_plausible = PLAUSIBLE_LIMITS[col]
            min_p_int = int(min_plausible) if min_plausible.is_integer() else min_plausible
            max_p_int = int(max_plausible) if max_plausible.is_integer() else max_plausible

            if val_float < min_plausible or val_float > max_plausible:
                if col == "age":
                    msg = f"Enter an age between {min_p_int} and {max_p_int}."
                elif col == "trestbps":
                    msg = f"Enter a resting blood pressure between {min_p_int} and {max_p_int}."
                elif col == "chol":
                    msg = f"Enter a cholesterol level between {min_p_int} and {max_p_int}."
                elif col == "thalach":
                    msg = f"Enter a maximum heart rate between {min_p_int} and {max_p_int}."
                elif col == "oldpeak":
                    msg = f"Enter an ST depression between {min_p_int} and {max_p_int}."
                else:
                    msg = f"Enter a {short_lbl} between {min_p_int} and {max_p_int}."

                errors.append({
                    "field": col,
                    "message": msg,
                })
            else:
                min_train, max_train = train_ranges[col]
                if val_float < min_train or val_float > max_train:
                    val_disp = int(round(val_float)) if col != "oldpeak" else f"{val_float:.1f}"
                    min_t_disp = int(round(min_train)) if col != "oldpeak" else f"{min_train:.1f}"
                    max_t_disp = int(round(max_train)) if col != "oldpeak" else f"{max_train:.1f}"

                    if col == "age":
                        msg = f"An age of {val_disp} is outside the range the model learned from ({min_t_disp} to {max_t_disp}), so the score may be less reliable."
                    elif col == "trestbps":
                        msg = f"A resting blood pressure of {val_disp} is outside the range the model learned from ({min_t_disp} to {max_t_disp}), so the score may be less reliable."
                    elif col == "chol":
                        msg = f"A cholesterol level of {val_disp} is outside the range the model learned from ({min_t_disp} to {max_t_disp}), so the score may be less reliable."
                    elif col == "thalach":
                        msg = f"A maximum heart rate of {val_disp} is outside the range the model learned from ({min_t_disp} to {max_t_disp}), so the score may be less reliable."
                    elif col == "oldpeak":
                        msg = f"An ST depression of {val_disp} is outside the range the model learned from ({min_t_disp} to {max_t_disp}), so the score may be less reliable."
                    else:
                        msg = f"A value of {val_disp} for {short_lbl} is outside the range the model learned from ({min_t_disp} to {max_t_disp}), so the score may be less reliable."

                    warnings.append({
                        "field": col,
                        "message": msg,
                    })
                cleaned[col] = float(val_float)

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        cleaned_inputs=cleaned,
        missing_used=missing_used,
    )


# -----------------------------------------------------------------------------
# Risk Tier Classification
# -----------------------------------------------------------------------------


def risk_tier(score: float) -> RiskTier:
    """Classify model risk score against evaluation thresholds."""
    low_cutoff, high_cutoff = get_cutoffs()
    if score < low_cutoff:
        return RiskTier("lower", TIERS["lower"]["label"], TIERS["lower"]["text"])
    elif score < high_cutoff:
        return RiskTier("moderate", TIERS["moderate"]["label"], TIERS["moderate"]["text"])
    else:
        return RiskTier("higher", TIERS["higher"]["label"], TIERS["higher"]["text"])


# -----------------------------------------------------------------------------
# Prediction Score
# -----------------------------------------------------------------------------


def predict_score(inputs: Any) -> float:
    """Compute heart disease risk model score (class 1 probability).

    Raises ValueError on invalid inputs.
    """
    val_res = validate_inputs(inputs)
    if not val_res.is_valid:
        error_msgs = [e["message"] if isinstance(e, dict) else str(e) for e in val_res.errors]
        raise ValueError(f"Invalid inputs: {'; '.join(error_msgs)}")

    df_train = load_train_features()
    expected_cols = df_train.columns.tolist()

    row_data = {col: val_res.cleaned_inputs[col] for col in expected_cols}
    row_df = pd.DataFrame([row_data], columns=expected_cols, dtype=np.float64)

    pipeline = load_model_pipeline()
    prob = pipeline.predict_proba(row_df)[0, 1]
    return float(prob)


# -----------------------------------------------------------------------------
# SHAP Explainability: 22 Columns and 13 Grouped Features
# -----------------------------------------------------------------------------


def explain_columns(inputs: Any) -> pd.DataFrame:
    """Compute TreeSHAP attributions for all 22 preprocessed encoded feature columns."""
    val_res = validate_inputs(inputs)
    if not val_res.is_valid:
        error_msgs = [e["message"] if isinstance(e, dict) else str(e) for e in val_res.errors]
        raise ValueError(f"Invalid inputs: {'; '.join(error_msgs)}")

    df_train = load_train_features()
    expected_cols = df_train.columns.tolist()

    row_data = {col: val_res.cleaned_inputs[col] for col in expected_cols}
    row_df = pd.DataFrame([row_data], columns=expected_cols, dtype=np.float64)

    pipeline = load_model_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    explainer = load_shap_explainer()

    row_trans = preprocessor.transform(row_df)
    feature_names = list(row_trans.columns)

    shap_raw = explainer.shap_values(row_trans)
    if isinstance(shap_raw, list):
        shap_values_class1 = shap_raw[1][0]
    elif isinstance(shap_raw, np.ndarray):
        if shap_raw.ndim == 3 and shap_raw.shape[2] == 2:
            shap_values_class1 = shap_raw[0, :, 1]
        elif shap_raw.ndim == 2:
            shap_values_class1 = shap_raw[0]
        else:
            raise ValueError(f"Unexpected SHAP array shape: {shap_raw.shape}")
    else:
        raise TypeError(f"Unexpected SHAP output type: {type(shap_raw)}")

    base_val = get_base_value()

    records = []
    for j, col_name in enumerate(feature_names):
        val = float(row_trans.iloc[0, j])
        amt = float(shap_values_class1[j])
        records.append({
            "feature": col_name,
            "key": col_name,
            "transformed_value": val,
            "shap_amount": amt,
            "amount": amt,
            "abs_shap": abs(amt),
            "direction": "Raised score (+)" if amt >= 0 else "Lowered score (−)",
        })

    df = pd.DataFrame(records)
    df.attrs["base_value"] = base_val
    return df


def explain_grouped(inputs: Any) -> pd.DataFrame:
    """Compute SHAP attributions aggregated into the 13 original clinical features.

    Uses exact column-name matching (num__{f}, bin__{f}, or nom__{f}_*) to strictly prevent
    collisions (such as between thal and thalach).
    """
    val_res = validate_inputs(inputs)
    if not val_res.is_valid:
        error_msgs = [e["message"] if isinstance(e, dict) else str(e) for e in val_res.errors]
        raise ValueError(f"Invalid inputs: {'; '.join(error_msgs)}")

    df_train = load_train_features()
    expected_cols = df_train.columns.tolist()

    row_data = {col: val_res.cleaned_inputs[col] for col in expected_cols}
    row_df = pd.DataFrame([row_data], columns=expected_cols, dtype=np.float64)

    pipeline = load_model_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    explainer = load_shap_explainer()

    row_trans = preprocessor.transform(row_df)
    feature_names = list(row_trans.columns)

    shap_raw = explainer.shap_values(row_trans)
    if isinstance(shap_raw, list):
        shap_values_class1 = shap_raw[1][0]
    elif isinstance(shap_raw, np.ndarray):
        if shap_raw.ndim == 3 and shap_raw.shape[2] == 2:
            shap_values_class1 = shap_raw[0, :, 1]
        elif shap_raw.ndim == 2:
            shap_values_class1 = shap_raw[0]
        else:
            raise ValueError(f"Unexpected SHAP array shape: {shap_raw.shape}")
    else:
        raise TypeError(f"Unexpected SHAP output type: {type(shap_raw)}")

    base_val = get_base_value()

    grouped_records = []
    for f in ORIGINAL_FEATURES:
        matching_cols = [
            c for c in feature_names
            if c in (f"num__{f}", f"bin__{f}") or c.startswith(f"nom__{f}_")
        ]
        col_indices = [feature_names.index(c) for c in matching_cols]
        feature_shap = float(np.sum(shap_values_class1[col_indices]))

        val_entered = val_res.cleaned_inputs.get(f, np.nan)
        meta = FIELD_METADATA.get(f, {})
        label = meta.get("label", f)
        short_label = meta.get("short_label", label)

        if np.isnan(val_entered):
            value_display = "Not provided"
        elif f in CODE_LABELS:
            value_display = CODE_LABELS[f].get(int(val_entered), str(int(val_entered)))
        elif f == "oldpeak":
            value_display = f"{val_entered:.1f} mm"
        elif f in ["trestbps", "chol", "thalach", "age"]:
            unit = meta.get("unit", "")
            value_display = f"{int(round(val_entered))} {unit}".strip()
        else:
            value_display = str(val_entered)

        grouped_records.append({
            "feature": f,
            "key": f,
            "label": label,
            "short_label": short_label,
            "value_entered": val_entered,
            "value_display": value_display,
            "shap_amount": feature_shap,
            "amount": feature_shap,
            "abs_shap": abs(feature_shap),
            "abs_amount": abs(feature_shap),
            "direction": "raised" if feature_shap >= 0 else "lowered",
            "sign_str": f"+{feature_shap:.4f}" if feature_shap >= 0 else f"{feature_shap:.4f}",
        })

    df_grouped = pd.DataFrame(grouped_records)
    df_grouped = df_grouped.sort_values(by="abs_shap", ascending=False).reset_index(drop=True)
    df_grouped.attrs["base_value"] = base_val
    return df_grouped


# -----------------------------------------------------------------------------
# What-If Sensitivity Curve
# -----------------------------------------------------------------------------


def what_if_curve(inputs: Any, feature: str, n_points: int = 25) -> pd.DataFrame:
    """Generate sensitivity curve by varying one continuous numeric feature across training range."""
    val_res = validate_inputs(inputs)
    if not val_res.is_valid:
        error_msgs = [e["message"] if isinstance(e, dict) else str(e) for e in val_res.errors]
        raise ValueError(f"Invalid inputs: {'; '.join(error_msgs)}")

    if feature not in CONTINUOUS_NUMERIC_FEATURES:
        raise ValueError(
            f"Feature '{feature}' is not an allowed continuous numeric feature for what-if analysis. "
            f"Allowed features: {CONTINUOUS_NUMERIC_FEATURES}."
        )

    df_train = load_train_features()
    valid_s = df_train[feature].dropna()
    min_val = float(valid_s.min())
    max_val = float(valid_s.max())

    grid_values = np.linspace(min_val, max_val, n_points)
    curr_val = float(val_res.cleaned_inputs[feature])
    if min_val <= curr_val <= max_val:
        closest_idx = int(np.argmin(np.abs(grid_values - curr_val)))
        grid_values[closest_idx] = curr_val
        grid_values = np.sort(grid_values)

    expected_cols = df_train.columns.tolist()
    pipeline = load_model_pipeline()

    rows = []
    for val in grid_values:
        row_dict = {c: val_res.cleaned_inputs[c] for c in expected_cols}
        row_dict[feature] = float(val)
        rows.append(row_dict)

    batch_df = pd.DataFrame(rows, columns=expected_cols, dtype=np.float64)
    probs = pipeline.predict_proba(batch_df)[:, 1]

    return pd.DataFrame({
        "value": grid_values,
        "score": probs,
    })


# -----------------------------------------------------------------------------
# Ensemble Tree Agreement
# -----------------------------------------------------------------------------


def tree_agreement(inputs: Any) -> tuple[np.ndarray, float, float]:
    """Compute individual tree predictions and consensus statistics across Random Forest."""
    val_res = validate_inputs(inputs)
    if not val_res.is_valid:
        error_msgs = [e["message"] if isinstance(e, dict) else str(e) for e in val_res.errors]
        raise ValueError(f"Invalid inputs: {'; '.join(error_msgs)}")

    df_train = load_train_features()
    expected_cols = df_train.columns.tolist()

    row_data = {col: val_res.cleaned_inputs[col] for col in expected_cols}
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

