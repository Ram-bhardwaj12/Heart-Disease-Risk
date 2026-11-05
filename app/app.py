"""Heart Disease Risk Estimator - Interactive Streamlit Application.

Provides real-time model inference, risk categorization, and patient-specific
SHAP waterfall explanations.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from utils import (
    DISPLAY_LABELS,
    explain_patient,
    get_feature_specs,
    get_recall_threshold,
    load_shap_summary,
    load_test_features,
    predict_score,
    risk_tier,
)
from components import (
    render_disclaimer_banner,
    render_how_to_read_caption,
    render_risk_badge,
)

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Risk Estimator",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Render Global Disclaimer Banner
render_disclaimer_banner()

# Title and Academic Context
st.title("🫀 Early Heart Disease Risk Prediction")
st.markdown(
    "Interactive clinical demonstration exploring patient risk classification with "
    "**Explainable Machine Learning (Random Forest + TreeSHAP)**."
)

# Load metadata and specifications
specs = get_feature_specs()
shap_summary = load_shap_summary()
test_features = load_test_features()
t_threshold = get_recall_threshold()

# Exemplar Patient Callbacks
exemplar_meta = shap_summary["selected_exemplar_rows"]


def load_exemplar(row_idx: int) -> None:
    """Callback to populate sidebar inputs from a test row."""
    patient_row = test_features.iloc[row_idx]
    for col, spec in specs.items():
        val = patient_row[col]
        if spec["widget"] == "selectbox":
            st.session_state[f"input_{col}"] = int(val)
        elif spec.get("is_float", False):
            st.session_state[f"input_{col}"] = float(val)
        else:
            st.session_state[f"input_{col}"] = float(val)


# Sidebar Header & Exemplar Buttons
st.sidebar.header("📋 Clinical Inputs")
st.sidebar.caption("Load demonstration patient profiles:")

btn_col1, btn_col2, btn_col3 = st.sidebar.columns(3)

with btn_col1:
    idx_tp = exemplar_meta["high_risk_true_positive"]["test_row_index"]
    st.button("Example: high score", on_click=load_exemplar, args=(idx_tp,), use_container_width=True)

with btn_col2:
    idx_tn = exemplar_meta["low_risk_true_negative"]["test_row_index"]
    st.button("Example: low score", on_click=load_exemplar, args=(idx_tn,), use_container_width=True)

with btn_col3:
    idx_fn = exemplar_meta["missed_case_false_negative"]["test_row_index"]
    st.button("Example: missed at 0.5", on_click=load_exemplar, args=(idx_fn,), use_container_width=True)

st.sidebar.caption("These are patients from the held-out test set, shown only to demonstrate the app.")
st.sidebar.markdown("---")

# Render Sidebar Widgets Grouped by Category
inputs = {}
groups = ["Patient", "Exercise and symptoms", "Tests"]

for grp in groups:
    st.sidebar.subheader(grp)
    for col, spec in specs.items():
        if spec["group"] != grp:
            continue

        key = f"input_{col}"
        label = spec["label"]
        help_text = spec["help"]

        if spec["widget"] == "selectbox":
            default_idx = spec["options"].index(spec["default"]) if spec["default"] in spec["options"] else 0
            if key not in st.session_state:
                st.session_state[key] = spec["options"][default_idx]
            val = st.sidebar.selectbox(
                label=label,
                options=spec["options"],
                index=spec["options"].index(st.session_state[key]) if st.session_state[key] in spec["options"] else 0,
                key=key,
                help=help_text,
            )
            inputs[col] = val

        elif spec.get("is_float", False):
            if key not in st.session_state:
                st.session_state[key] = spec["default"]
            val = st.sidebar.number_input(
                label=label,
                min_value=spec["min"],
                max_value=spec["max"],
                value=float(st.session_state[key]),
                step=spec["step"],
                key=key,
                help=help_text,
            )
            inputs[col] = val

        else:
            if key not in st.session_state:
                st.session_state[key] = spec["default"]
            val = st.sidebar.slider(
                label=label,
                min_value=int(spec["min"]),
                max_value=int(spec["max"]),
                value=int(st.session_state[key]),
                step=int(spec["step"]),
                key=key,
                help=help_text,
            )
            inputs[col] = val

# Compute Prediction & Risk Tier
score = predict_score(inputs)
score_pct = score * 100.0
tier_name, badge_text, badge_color = risk_tier(score)

# Main Area Display
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Model Risk Estimate")
    st.metric(
        label="Predicted Risk Score",
        value=f"{score_pct:.1f}%",
        help="Model estimated likelihood of coronary artery disease presence.",
    )
    render_risk_badge(tier_name, badge_text, badge_color)

    st.info(
        f"**Note:** This is the model's score, not a measured probability. Calibration was not checked. "
        f"Risk cut-offs are configured as: Low below {t_threshold:.2f} (the screening threshold chosen on training data), "
        f"Moderate {t_threshold:.2f} to 0.50, High 0.50 and above.",
        icon="ℹ️",
    )

with col_right:
    st.subheader("Patient Parameter Summary")
    summary_df = pd.DataFrame([
        {"Feature": DISPLAY_LABELS.get(k, k), "Value Entered": v}
        for k, v in inputs.items()
    ])
    st.dataframe(summary_df, use_container_width=True, height=210)

st.markdown("---")

# Local Explainability (SHAP Waterfall)
st.subheader("🔍 Patient-Specific Feature Attribution (TreeSHAP Waterfall)")

explanation = explain_patient(inputs)

fig, ax = plt.subplots(figsize=(9, 6.5))
shap.plots.waterfall(explanation, show=False)
plt.title(
    f"Individual SHAP Waterfall Attribution (Predicted Risk: {score_pct:.1f}%)",
    fontsize=12,
    pad=12,
)
st.pyplot(fig)
plt.close(fig)

render_how_to_read_caption()

# Plain-Language Breakdown of Largest Contributions
st.markdown("### Key Factors Influencing This Prediction")

feature_shap_pairs = list(zip(explanation.feature_names, explanation.values, explanation.data))
# Sort by SHAP contribution
sorted_pairs = sorted(feature_shap_pairs, key=lambda x: x[1])

# Largest negative contributions (lowered risk)
negative_contributors = [p for p in sorted_pairs if p[1] < 0][:3]
# Largest positive contributions (raised risk)
positive_contributors = [p for p in reversed(sorted_pairs) if p[1] > 0][:3]

col_pos, col_neg = st.columns(2)

with col_pos:
    st.markdown("#### 🔺 Top Factors Raising the Risk Score")
    if positive_contributors:
        for fname, s_val, d_val in positive_contributors:
            clean_name = fname.replace("num__", "").replace("bin__", "").replace("nom__", "")
            disp_label = DISPLAY_LABELS.get(clean_name, clean_name)
            st.markdown(
                f"* **{disp_label}** (entered value: `{d_val}`): "
                f"**raised the score** by `+{s_val:.4f}`"
            )
    else:
        st.write("No features pushed the risk score upward.")

with col_neg:
    st.markdown("#### 🔻 Top Factors Lowering the Risk Score")
    if negative_contributors:
        for fname, s_val, d_val in negative_contributors:
            clean_name = fname.replace("num__", "").replace("bin__", "").replace("nom__", "")
            disp_label = DISPLAY_LABELS.get(clean_name, clean_name)
            st.markdown(
                f"* **{disp_label}** (entered value: `{d_val}`): "
                f"**lowered the score** by `{s_val:.4f}`"
            )
    else:
        st.write("No features pulled the risk score downward.")

st.caption("SHAP shows how the model uses each input, not what causes disease.")
