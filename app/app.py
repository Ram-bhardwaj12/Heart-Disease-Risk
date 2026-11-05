"""Main Streamlit Application: Early Heart Disease Risk Prediction.

Provides real-time clinical risk classification and local TreeSHAP explanations.
Zero imports from src/; all loaders and utilities are resolved via app/utils.py.
"""

import io
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# Setup local imports without importing from src/
import sys
app_dir = Path(__file__).resolve().parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components import (
    render_disclaimer_banner,
    render_how_to_read_caption,
    render_risk_badge,
)
from utils import (
    DISPLAY_LABELS,
    explain_patient,
    get_feature_specs,
    get_recall_threshold,
    get_typical_values,
    load_shap_summary,
    load_test_features,
    predict_score,
    risk_tier,
    validate_inputs,
    what_if_curve,
)

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
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

# Load metadata, specifications, and default typical values
specs = get_feature_specs()
shap_summary = load_shap_summary()
test_features = load_test_features()
t_threshold = get_recall_threshold()
typical_vals = get_typical_values()

# Initialize session state for all 13 inputs if not already present
for col, default_v in typical_vals.items():
    key = f"input_{col}"
    if key not in st.session_state:
        st.session_state[key] = default_v


# Exemplar & Reset Callbacks
exemplar_meta = shap_summary["selected_exemplar_rows"]


def load_exemplar(row_idx: int) -> None:
    """Callback to populate all 13 sidebar inputs from a test row."""
    patient_row = test_features.iloc[row_idx]
    for col, spec in specs.items():
        val = patient_row[col]
        key = f"input_{col}"
        if spec["widget"] == "selectbox":
            st.session_state[key] = int(val)
        elif spec.get("is_float", False):
            st.session_state[key] = float(val)
        else:
            st.session_state[key] = int(round(val))


def reset_to_typical() -> None:
    """Callback to reset all 13 sidebar inputs to training medians/modes."""
    for col, val in typical_vals.items():
        st.session_state[f"input_{col}"] = val


# Sidebar Header & Exemplar Buttons
st.sidebar.header("📋 Clinical Inputs")
st.sidebar.caption("Load demonstration patient profiles or reset:")

btn_col1, btn_col2 = st.sidebar.columns(2)
with btn_col1:
    idx_tp = exemplar_meta["high_risk_true_positive"]["test_row_index"]
    st.button("High score (#41)", on_click=load_exemplar, args=(idx_tp,), use_container_width=True)

with btn_col2:
    idx_tn = exemplar_meta["low_risk_true_negative"]["test_row_index"]
    st.button("Low score (#6)", on_click=load_exemplar, args=(idx_tn,), use_container_width=True)

btn_col3, btn_col4 = st.sidebar.columns(2)
with btn_col3:
    idx_fn = exemplar_meta["missed_case_false_negative"]["test_row_index"]
    st.button("Missed at 0.5 (#18)", on_click=load_exemplar, args=(idx_fn,), use_container_width=True)

with btn_col4:
    st.button("Reset to typical", on_click=reset_to_typical, use_container_width=True)

st.sidebar.caption("Patients #41, #6, #18 are from the held-out test set, shown to demonstrate the app.")
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
            val = st.sidebar.selectbox(
                label=label,
                options=spec["options"],
                key=key,
                help=help_text,
            )
            inputs[col] = val

        elif spec.get("is_float", False):
            val = st.sidebar.number_input(
                label=label,
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                step=float(spec["step"]),
                key=key,
                help=help_text,
            )
            inputs[col] = val

        else:
            val = st.sidebar.slider(
                label=label,
                min_value=int(spec["min"]),
                max_value=int(spec["max"]),
                step=int(spec["step"]),
                key=key,
                help=help_text,
            )
            inputs[col] = val

# Validate inputs and compute prediction
val_res = validate_inputs(inputs)

if not val_res.is_valid:
    st.error(f"Cannot compute risk score due to invalid inputs: {'; '.join(val_res.errors)}")
    st.stop()

# Out-of-range notice
if val_res.warnings:
    st.warning(
        "⚠️ **Caution:** Some inputs are outside the range of the training data. "
        "The model score may be unreliable for this patient.\n\n"
        + "\n".join(f"* {w}" for w in val_res.warnings)
    )

score = predict_score(inputs)
score_pct = score * 100.0
tier_name, badge_text, badge_color = risk_tier(score)

# Main Area Tab Navigation
tab_res, tab_exp, tab_wi, tab_sum = st.tabs(["Result", "Explanation", "What-if", "Inputs summary"])

# TAB 1: RESULT
with tab_res:
    st.subheader("Model Risk Estimate")
    col_score1, col_score2 = st.columns([1, 1])

    with col_score1:
        st.metric(
            label="Predicted Risk Score",
            value=f"{score_pct:.1f}%",
            help="Model estimated likelihood of coronary artery disease presence.",
        )
        render_risk_badge(tier_name, badge_text, badge_color)

    with col_score2:
        st.markdown("**Score Distribution & Cut-Offs:**")
        st.progress(min(max(float(score), 0.0), 1.0))
        st.caption(
            f"Decision Cut-Offs: 🟢 **Low Risk** (< {t_threshold:.2f}) | "
            f"🟡 **Moderate Risk** ({t_threshold:.2f} – 0.50) | "
            f"🔴 **High Risk** (≥ 0.50)"
        )

    st.info(
        f"**Note:** This is the model's score, not a measured probability. Calibration was not checked. "
        f"Risk cut-offs are configured as: Low below {t_threshold:.2f} (the screening threshold chosen on training data), "
        f"Moderate {t_threshold:.2f} to 0.50, High 0.50 and above.",
        icon="ℹ️",
    )

# TAB 2: EXPLANATION
with tab_exp:
    st.subheader("🔍 Patient-Specific Feature Attribution (TreeSHAP Waterfall)")

    with st.spinner("Computing TreeSHAP patient attribution..."):
        explanation = explain_patient(inputs)

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(
        f"Individual SHAP Waterfall Attribution (Model Score: {score_pct:.1f}%)",
        fontsize=12,
        pad=12,
    )
    st.pyplot(fig)
    plt.close(fig)

    render_how_to_read_caption()

    # Plain-Language Breakdown of Largest Contributions
    st.markdown("### Key Factors Influencing This Prediction")

    feature_shap_pairs = list(zip(explanation.feature_names, explanation.values, explanation.data))
    sorted_pairs = sorted(feature_shap_pairs, key=lambda x: x[1])

    negative_contributors = [p for p in sorted_pairs if p[1] < 0][:3]
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

# TAB 3: WHAT-IF
with tab_wi:
    st.subheader("📈 Interactive What-If Sensitivity Analysis")
    st.markdown(
        "Explore how varying a single clinical test or measurement affects the predicted risk score "
        "while keeping all other 12 patient inputs constant."
    )

    numeric_what_if_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    chosen_feature = st.selectbox(
        "Choose a continuous clinical feature to vary:",
        options=numeric_what_if_features,
        format_func=lambda x: DISPLAY_LABELS.get(x, x),
        key="what_if_feature_select",
    )

    df_curve = what_if_curve(inputs, chosen_feature, n_points=25)
    curr_feature_val = float(inputs[chosen_feature])

    fig_wi, ax_wi = plt.subplots(figsize=(9, 4.5))
    ax_wi.plot(df_curve["value"], df_curve["score"] * 100.0, color="#1e88e5", lw=2.5, label="Model Score Response")
    ax_wi.scatter(
        [curr_feature_val],
        [score_pct],
        color="#c62828",
        s=100,
        zorder=5,
        label=f"Current Patient Value ({curr_feature_val})",
    )
    ax_wi.axhline(50.0, color="#d32f2f", linestyle="--", alpha=0.6, label="High Risk Threshold (50%)")
    ax_wi.axhline(t_threshold * 100.0, color="#f57f17", linestyle=":", alpha=0.6, label=f"Screening Threshold ({t_threshold*100:.0f}%)")
    ax_wi.set_xlabel(DISPLAY_LABELS.get(chosen_feature, chosen_feature), fontsize=10)
    ax_wi.set_ylabel("Predicted Risk Score (%)", fontsize=10)
    ax_wi.set_ylim(-2, 102)
    ax_wi.grid(True, linestyle=":", alpha=0.6)
    ax_wi.legend(loc="best", fontsize=9)
    st.pyplot(fig_wi)
    plt.close(fig_wi)

    st.caption("Other inputs are held at the values currently in the sidebar. This shows how the model responds, not what causes disease.")

# TAB 4: INPUTS SUMMARY
with tab_sum:
    st.subheader("📋 Patient Clinical Parameters & Summary Export")
    summary_df = pd.DataFrame([
        {"Feature": DISPLAY_LABELS.get(k, k), "Parameter Code": k, "Value Entered": v}
        for k, v in inputs.items()
    ])
    st.dataframe(summary_df[["Feature", "Value Entered"]], use_container_width=True, height=350)

    # In-memory CSV export
    csv_buffer = io.StringIO()
    csv_buffer.write("# Educational project, not a medical diagnosis.\n")
    csv_buffer.write(f"# Predicted Model Score: {score:.4f} ({score_pct:.1f}%)\n")
    csv_buffer.write(f"# Risk Tier: {tier_name}\n")
    export_rows = [
        {"feature": k, "label": DISPLAY_LABELS.get(k, k), "entered_value": v}
        for k, v in inputs.items()
    ]
    export_rows.append({"feature": "model_score", "label": "Model Score", "entered_value": round(score, 4)})
    export_rows.append({"feature": "risk_tier", "label": "Risk Tier", "entered_value": tier_name})
    export_df = pd.DataFrame(export_rows)
    export_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Inputs & Risk Summary (CSV)",
        data=csv_buffer.getvalue(),
        file_name="heart_disease_risk_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
