"""Risk Estimator View: Interactive Patient Evaluation Dashboard."""

import io
from pathlib import Path
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Setup local imports
import sys
app_dir = Path(__file__).resolve().parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components import (
    COPY,
    inject_css,
    render_disclaimer_strip,
    render_footer,
    render_hero,
    render_risk_meter,
    render_tier_pill,
)
from utils import (
    DISPLAY_LABELS,
    build_summary_sentence,
    explain_grouped,
    explain_patient,
    generate_html_report,
    get_feature_specs,
    get_recall_threshold,
    get_typical_values,
    load_shap_summary,
    load_test_features,
    predict_score,
    risk_tier,
    tree_agreement,
    validate_inputs,
    what_if_curve,
)

# Inject design system
inject_css()

# Header & Disclaimer Strip
render_disclaimer_strip()
render_hero(COPY["APP_TITLE"], COPY["ESTIMATOR_SUBTITLE"])

# Metadata & Specifications
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


# --- SIDEBAR: PRODUCT TITLE & QUICK START ---
st.sidebar.markdown(f"### {COPY['SIDEBAR_TITLE']}\n*{COPY['SIDEBAR_SUBTITLE']}*")
st.sidebar.markdown(f"**{COPY['SIDEBAR_QUICK_START']}**")
st.sidebar.caption(COPY["SIDEBAR_QUICK_START_CAPTION"])

idx_tp = exemplar_meta["high_risk_true_positive"]["test_row_index"]
st.sidebar.button(
    COPY["BTN_HIGH_SCORE"],
    on_click=load_exemplar,
    args=(idx_tp,),
    help=COPY["HELP_HIGH_SCORE"],
    use_container_width=True,
    icon=":material/personal_injury:",
)

idx_tn = exemplar_meta["low_risk_true_negative"]["test_row_index"]
st.sidebar.button(
    COPY["BTN_LOW_SCORE"],
    on_click=load_exemplar,
    args=(idx_tn,),
    help=COPY["HELP_LOW_SCORE"],
    use_container_width=True,
    icon=":material/sentiment_satisfied:",
)

idx_fn = exemplar_meta["missed_case_false_negative"]["test_row_index"]
st.sidebar.button(
    COPY["BTN_BORDERLINE"],
    on_click=load_exemplar,
    args=(idx_fn,),
    help=COPY["HELP_BORDERLINE"],
    use_container_width=True,
    icon=":material/rule:",
)

st.sidebar.button(
    COPY["BTN_RESET"],
    on_click=reset_to_typical,
    help=COPY["HELP_RESET"],
    use_container_width=True,
    icon=":material/refresh:",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Clinical Parameters**")

# Grouped Collapsible Sections
inputs = {}
groups = ["Patient", "Exercise and symptoms", "Tests"]

for grp in groups:
    with st.sidebar.expander(grp, expanded=True):
        for col, spec in specs.items():
            if spec["group"] != grp:
                continue

            key = f"input_{col}"
            label = spec["label"]
            help_text = spec["help"]

            if spec["widget"] == "selectbox":
                val = st.selectbox(
                    label=label,
                    options=spec["options"],
                    key=key,
                    help=help_text,
                )
                inputs[col] = val

            elif spec.get("is_float", False):
                val = st.number_input(
                    label=label,
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    step=float(spec["step"]),
                    key=key,
                    help=help_text,
                )
                inputs[col] = val

            else:
                val = st.slider(
                    label=label,
                    min_value=int(spec["min"]),
                    max_value=int(spec["max"]),
                    step=int(spec["step"]),
                    key=key,
                    help=help_text,
                )
                inputs[col] = val

# Input Validation
val_res = validate_inputs(inputs)
if not val_res.is_valid:
    st.error(f"Cannot compute risk score: {'; '.join(val_res.errors)}")
    st.stop()

# Out-of-range caution notice
if val_res.warnings:
    st.warning(
        f"⚠️ **{COPY['OUT_OF_RANGE_TITLE']}**\n\n"
        + "\n".join(f"* {w}" for w in val_res.warnings)
    )

# Compute Main Prediction
score = predict_score(inputs)
score_pct = score * 100.0
tier_name, badge_text, badge_color = risk_tier(score)
grouped_df = explain_grouped(inputs)
summary_sentence = build_summary_sentence(score, tier_name, grouped_df)

# Navigation Tabs
tab_res, tab_exp, tab_wi, tab_rep = st.tabs([
    f":material/speed: {COPY['TAB_RESULT']}",
    f":material/bar_chart: {COPY['TAB_EXPLANATION']}",
    f":material/tune: {COPY['TAB_WHAT_IF']}",
    f":material/description: {COPY['TAB_REPORT']}",
])

# ==============================================================================
# TAB 1: RESULT
# ==============================================================================
with tab_res:
    col_kpi, col_info = st.columns([1, 1])

    with col_kpi:
        st.metric(
            label=COPY["CARD_SCORE_TITLE"],
            value=f"{score_pct:.1f}%",
        )
        render_tier_pill(tier_name)

    with col_info:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-size:0.9rem; color:#5B6B7F; text-transform:uppercase; font-weight:600; margin-bottom:8px;">Assessment Summary</div>
                <p style="font-size:0.95rem; line-height:1.5; color:#0F1B2D; margin-bottom:8px;">{summary_sentence}</p>
                <div style="font-size:0.8rem; color:#5B6B7F;">{COPY['CALIBRATION_NOTE']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Risk Meter Component
    render_risk_meter(score, t_threshold, 0.50)

    with st.popover(COPY["POPOVER_TIERS_LABEL"], icon=":material/help:"):
        st.markdown(COPY["POPOVER_TIERS_CONTENT"].format(t_threshold=t_threshold))

    st.markdown("---")

    # Model Agreement Section
    st.subheader(COPY["MODEL_AGREEMENT_TITLE"])
    tree_scores, share_ge_05, mean_tree_score = tree_agreement(inputs)

    col_ag1, col_ag2 = st.columns([1, 2])
    with col_ag1:
        st.metric(
            label="Trees Scoring ≥ 0.50",
            value=f"{share_ge_05*100.0:.1f}%",
            help="Proportion of the 400 trees in the Random Forest whose individual probability estimate is >= 50%.",
        )
        st.metric(
            label="Mean Tree Score",
            value=f"{mean_tree_score*100.0:.1f}%",
            help="Ensemble arithmetic average across all 400 trees.",
        )

    with col_ag2:
        df_trees = pd.DataFrame({"Tree Score": tree_scores * 100.0})
        chart_trees = (
            alt.Chart(df_trees)
            .mark_bar(color="#0E7C86", opacity=0.85, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Tree Score:Q", bin=alt.Bin(maxbins=25), title="Individual Tree Score (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("count()", title="Tree Count"),
                tooltip=[alt.Tooltip("count()", title="Trees in bin")],
            )
            .properties(height=170)
        )
        st.altair_chart(chart_trees, use_container_width=True)

    st.caption(COPY["MODEL_AGREEMENT_CAPTION"])


# ==============================================================================
# TAB 2: EXPLANATION
# ==============================================================================
with tab_exp:
    st.subheader(COPY["EXPLANATION_TITLE"])
    st.caption(COPY["SHAP_UNITS_NOTE"])

    show_ungrouped = st.toggle(COPY["EXPLANATION_TOGGLE"], value=False)

    if not show_ungrouped:
        # Grouped 13-feature Altair Chart
        chart_data = grouped_df.copy()
        chart_data["Display Label"] = chart_data.apply(lambda r: f"{r['label']} ({r['sign_str']})", axis=1)

        color_scale = alt.Scale(
            domain=["Raised score (+)", "Lowered score (−)"],
            range=["#C8423B", "#1E8E5A"],
        )

        bar_chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("Display Label:N", sort=list(chart_data["Display Label"]), title=None),
                x=alt.X("shap_amount:Q", title="SHAP Contribution to Model Score"),
                color=alt.Color("direction:N", scale=color_scale, legend=alt.Legend(title="Direction")),
                tooltip=[
                    alt.Tooltip("label:N", title="Clinical Feature"),
                    alt.Tooltip("value_entered:Q", title="Value Entered"),
                    alt.Tooltip("shap_amount:Q", title="SHAP Amount", format="+.4f"),
                    alt.Tooltip("direction:N", title="Effect"),
                ],
            )
            .properties(height=420)
        )
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        # Ungrouped 22-column view
        explanation = explain_patient(inputs)
        df_ungrouped = pd.DataFrame({
            "Encoded Column": explanation.feature_names,
            "SHAP Value": explanation.values,
            "Direction": ["Raised score (+)" if v >= 0 else "Lowered score (−)" for v in explanation.values],
            "Abs SHAP": np.abs(explanation.values),
        }).sort_values(by="Abs SHAP", ascending=False)

        bar_ungrouped = (
            alt.Chart(df_ungrouped)
            .mark_bar()
            .encode(
                y=alt.Y("Encoded Column:N", sort=list(df_ungrouped["Encoded Column"]), title=None),
                x=alt.X("SHAP Value:Q", title="SHAP Contribution (22 Encodings)"),
                color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raised score (+)", "Lowered score (−)"], range=["#C8423B", "#1E8E5A"])),
                tooltip=["Encoded Column:N", alt.Tooltip("SHAP Value:Q", format="+.4f")],
            )
            .properties(height=520)
        )
        st.altair_chart(bar_ungrouped, use_container_width=True)

    with st.expander(COPY["EXPANDER_WATERFALL_TITLE"]):
        with st.spinner("Rendering waterfall attribution..."):
            explanation_wf = explain_patient(inputs)
            fig_wf, ax_wf = plt.subplots(figsize=(9, 5.5))
            shap.plots.waterfall(explanation_wf, show=False)
            plt.title(f"Individual SHAP Waterfall Attribution (Model Score: {score_pct:.1f}%)", fontsize=11, pad=10)
            st.pyplot(fig_wf)
            plt.close(fig_wf)

    st.caption(COPY["SHAP_NON_CAUSAL_NOTE"])


# ==============================================================================
# TAB 3: WHAT-IF SENSITIVITY
# ==============================================================================
with tab_wi:
    st.subheader(COPY["WHAT_IF_TITLE"])
    numeric_what_if_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

    chosen_feature = st.selectbox(
        COPY["WHAT_IF_FEATURE_SELECT"],
        options=numeric_what_if_features,
        format_func=lambda x: DISPLAY_LABELS.get(x, x),
        key="what_if_feature_dropdown",
    )

    df_curve = what_if_curve(inputs, chosen_feature, n_points=25)
    df_curve["score_pct"] = df_curve["score"] * 100.0
    curr_val = float(inputs[chosen_feature])

    # Interactive slider to test alternate value
    f_spec = specs[chosen_feature]
    col_sl, col_delta = st.columns([2, 1])

    with col_sl:
        alt_val = st.slider(
            COPY["WHAT_IF_SLIDER_LABEL"].format(feature=DISPLAY_LABELS.get(chosen_feature, chosen_feature)),
            min_value=float(f_spec["min"]),
            max_value=float(f_spec["max"]),
            value=curr_val,
            step=float(f_spec["step"]),
            key=f"what_if_slider_{chosen_feature}",
        )

    # Compute counterfactual score on copy of inputs (never mutating sidebar)
    trial_inputs = dict(inputs)
    trial_inputs[chosen_feature] = alt_val
    alt_score = predict_score(trial_inputs)
    alt_score_pct = alt_score * 100.0
    score_delta_pct = alt_score_pct - score_pct

    with col_delta:
        st.metric(
            label="Simulated Risk Score",
            value=f"{alt_score_pct:.1f}%",
            delta=f"{score_delta_pct:+.1f}% vs Current",
            delta_color="inverse",
        )

    # Altair Sensitivity Chart with shaded risk bands
    band_df = pd.DataFrame([
        {"band": "Low Risk", "y1": 0.0, "y2": t_threshold * 100.0, "color": "#E6F5EC"},
        {"band": "Moderate Risk", "y1": t_threshold * 100.0, "y2": 50.0, "color": "#FEF3D6"},
        {"band": "High Risk", "y1": 50.0, "y2": 100.0, "color": "#FDE8E7"},
    ])

    bands = (
        alt.Chart(band_df)
        .mark_rect(opacity=0.6)
        .encode(
            y=alt.Y("y1:Q"),
            y2=alt.Y2("y2:Q"),
            color=alt.Color("color:N", scale=None),
        )
    )

    line = (
        alt.Chart(df_curve)
        .mark_line(color="#0E7C86", strokeWidth=3)
        .encode(
            x=alt.X("value:Q", title=DISPLAY_LABELS.get(chosen_feature, chosen_feature)),
            y=alt.Y("score_pct:Q", title="Predicted Risk Score (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("value:Q", title=DISPLAY_LABELS.get(chosen_feature, chosen_feature)),
                alt.Tooltip("score_pct:Q", title="Predicted Score (%)", format=".1f"),
            ],
        )
    )

    curr_point_df = pd.DataFrame([{"value": curr_val, "score_pct": score_pct, "label": "Current Value"}])
    point = (
        alt.Chart(curr_point_df)
        .mark_circle(size=120, color="#C8423B")
        .encode(
            x="value:Q",
            y="score_pct:Q",
            tooltip=[alt.Tooltip("value:Q", title="Current Value"), alt.Tooltip("score_pct:Q", title="Current Score (%)", format=".1f")],
        )
    )

    chart_what_if = (bands + line + point).properties(height=320)
    st.altair_chart(chart_what_if, use_container_width=True)

    st.caption(COPY["WHAT_IF_CAPTION"])


# ==============================================================================
# TAB 4: REPORT
# ==============================================================================
with tab_rep:
    st.subheader(COPY["REPORT_TITLE"])

    col_rep_tbl, col_rep_dl = st.columns([3, 2])

    with col_rep_tbl:
        tbl_data = pd.DataFrame([
            {"Clinical Parameter": DISPLAY_LABELS.get(k, k), "Entered Value": v}
            for k, v in inputs.items()
        ])
        st.dataframe(tbl_data, use_container_width=True, hide_index=True, height=440)

    with col_rep_dl:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-size:0.85rem; color:#5B6B7F; text-transform:uppercase; font-weight:600; margin-bottom:6px;">Summary Snapshot</div>
                <div style="font-size:1.8rem; font-weight:700; color:#0F1B2D; margin-bottom:8px;">{score_pct:.1f}% ({tier_name} Risk)</div>
                <p style="font-size:0.9rem; color:#5B6B7F; line-height:1.4; margin-bottom:16px;">{summary_sentence}</p>
                <div style="font-size:0.8rem; color:#8C9BAE; margin-bottom:14px;">Screening cut-off: {t_threshold:.2f} | High risk: 0.50</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # HTML Report Generation
        html_content = generate_html_report(
            inputs=inputs,
            score=score,
            tier_name=tier_name,
            summary_sentence=summary_sentence,
            grouped_df=grouped_df,
            share_ge_05=share_ge_05,
        )
        st.download_button(
            label=f":material/html: {COPY['BTN_DOWNLOAD_HTML']}",
            data=html_content,
            file_name="heart_disease_risk_report.html",
            mime="text/html",
            use_container_width=True,
        )

        # CSV Export Generation
        csv_buffer = io.StringIO()
        csv_buffer.write(f"# {COPY['DISCLAIMER']}\n")
        csv_buffer.write(f"# Predicted Model Score: {score:.4f} ({score_pct:.1f}%)\n")
        csv_buffer.write(f"# Risk Tier: {tier_name}\n")
        csv_rows = [
            {"feature": k, "label": DISPLAY_LABELS.get(k, k), "entered_value": v}
            for k, v in inputs.items()
        ]
        csv_rows.append({"feature": "model_score", "label": "Model Score", "entered_value": round(score, 4)})
        csv_rows.append({"feature": "risk_tier", "label": "Risk Tier", "entered_value": tier_name})
        pd.DataFrame(csv_rows).to_csv(csv_buffer, index=False)

        st.download_button(
            label=f":material/csv: {COPY['BTN_DOWNLOAD_CSV']}",
            data=csv_buffer.getvalue(),
            file_name="heart_disease_risk_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Page Footer
render_footer()
