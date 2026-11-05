"""Model Performance View: Validation and Evaluation Benchmarks."""

from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

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
)
from utils import (
    REPO_ROOT,
    load_evaluation_summary,
    load_training_history,
)

inject_css()
render_disclaimer_strip()
render_hero(COPY["NAV_PERFORMANCE"], COPY["PERFORMANCE_SUBTITLE"])

eval_summary = load_evaluation_summary()
train_history = load_training_history()

test_rf = eval_summary["primary_results_threshold_0_5"]["random_forest"]
ci_rf = test_rf["confidence_intervals_95"]
test_lr = eval_summary["primary_results_threshold_0_5"]["logistic_regression"]
ci_lr = test_lr["confidence_intervals_95"]

# Row of Headline Metric Cards (Random Forest primary candidate on held-out test cohort)
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric(
        label="Test Recall (Sensitivity)",
        value=f"{test_rf['recall']:.2%}",
        help=f"95% CI: [{ci_rf['recall']['ci_lower']:.2%}, {ci_rf['recall']['ci_upper']:.2%}]",
    )
    st.caption(f"95% CI: [{ci_rf['recall']['ci_lower']:.2f}, {ci_rf['recall']['ci_upper']:.2f}]")

with m_col2:
    st.metric(
        label="Test ROC-AUC",
        value=f"{test_rf['roc_auc']:.4f}",
        help=f"95% CI: [{ci_rf['roc_auc']['ci_lower']:.4f}, {ci_rf['roc_auc']['ci_upper']:.4f}]",
    )
    st.caption(f"95% CI: [{ci_rf['roc_auc']['ci_lower']:.2f}, {ci_rf['roc_auc']['ci_upper']:.2f}]")

with m_col3:
    st.metric(
        label="Test Precision",
        value=f"{test_rf['precision']:.2%}",
        help=f"95% CI: [{ci_rf['precision']['ci_lower']:.2%}, {ci_rf['precision']['ci_upper']:.2%}]",
    )
    st.caption(f"95% CI: [{ci_rf['precision']['ci_lower']:.2f}, {ci_rf['precision']['ci_upper']:.2f}]")

with m_col4:
    st.metric(
        label="Test Brier Score",
        value=f"{test_rf['brier_score']:.4f}",
        help="Mean squared error of probabilistic predictions (lower is better calibrated).",
    )
    st.caption("Lower is better")

st.markdown("---")

# Performance Tabs
tab_cv, tab_test, tab_op = st.tabs([
    ":material/sync: Cross-Validation",
    ":material/verified: Held-Out Test",
    ":material/target: Operating Point (t = 0.30)",
])

# TAB 1: CROSS-VALIDATION
with tab_cv:
    st.subheader("Training Cross-Validation Performance (5-Fold Stratified)")
    st.caption("Hyperparameter optimization performed strictly within training folds (241 patients).")

    cv_lr = train_history["models"]["baseline_logistic_regression"]["cv_metrics_at_best_index"]
    cv_rf = train_history["models"]["candidate_random_forest"]["cv_metrics_at_best_index"]

    cv_table = pd.DataFrame([
        {
            "Model": "Logistic Regression (Baseline)",
            "ROC-AUC": f"{cv_lr['roc_auc']['mean']:.4f} ± {cv_lr['roc_auc']['std']:.4f}",
            "Recall": f"{cv_lr['recall']['mean']:.4f} ± {cv_lr['recall']['std']:.4f}",
            "Precision": f"{cv_lr['precision']['mean']:.4f} ± {cv_lr['precision']['std']:.4f}",
            "F1-Score": f"{cv_lr['f1']['mean']:.4f} ± {cv_lr['f1']['std']:.4f}",
            "Accuracy": f"{cv_lr['accuracy']['mean']:.4f} ± {cv_lr['accuracy']['std']:.4f}",
        },
        {
            "Model": "Random Forest (Tuned Candidate)",
            "ROC-AUC": f"{cv_rf['roc_auc']['mean']:.4f} ± {cv_rf['roc_auc']['std']:.4f}",
            "Recall": f"{cv_rf['recall']['mean']:.4f} ± {cv_rf['recall']['std']:.4f}",
            "Precision": f"{cv_rf['precision']['mean']:.4f} ± {cv_rf['precision']['std']:.4f}",
            "F1-Score": f"{cv_rf['f1']['mean']:.4f} ± {cv_rf['f1']['std']:.4f}",
            "Accuracy": f"{cv_rf['accuracy']['mean']:.4f} ± {cv_rf['accuracy']['std']:.4f}",
        },
    ])
    st.dataframe(cv_table, use_container_width=True, hide_index=True)


# TAB 2: HELD-OUT TEST
with tab_test:
    st.subheader("Held-Out Test Set Benchmark (n = 61, Threshold = 0.50)")
    st.caption("Evaluated once on strictly unseen patients; brackets show 95% bootstrap confidence intervals (1,000 resamples).")

    test_table = pd.DataFrame([
        {
            "Model": "Logistic Regression (Baseline)",
            "Accuracy [95% CI]": f"{test_lr['accuracy']:.4f} [{ci_lr['accuracy']['ci_lower']:.4f}, {ci_lr['accuracy']['ci_upper']:.4f}]",
            "Precision [95% CI]": f"{test_lr['precision']:.4f} [{ci_lr['precision']['ci_lower']:.4f}, {ci_lr['precision']['ci_upper']:.4f}]",
            "Recall [95% CI]": f"{test_lr['recall']:.4f} [{ci_lr['recall']['ci_lower']:.4f}, {ci_lr['recall']['ci_upper']:.4f}]",
            "F1-Score [95% CI]": f"{test_lr['f1']:.4f} [{ci_lr['f1']['ci_lower']:.4f}, {ci_lr['f1']['ci_upper']:.4f}]",
            "ROC-AUC [95% CI]": f"{test_lr['roc_auc']:.4f} [{ci_lr['roc_auc']['ci_lower']:.4f}, {ci_lr['roc_auc']['ci_upper']:.4f}]",
            "Brier Score": f"{test_lr['brier_score']:.4f}",
        },
        {
            "Model": "Random Forest (Tuned Candidate)",
            "Accuracy [95% CI]": f"{test_rf['accuracy']:.4f} [{ci_rf['accuracy']['ci_lower']:.4f}, {ci_rf['accuracy']['ci_upper']:.4f}]",
            "Precision [95% CI]": f"{test_rf['precision']:.4f} [{ci_rf['precision']['ci_lower']:.4f}, {ci_rf['precision']['ci_upper']:.4f}]",
            "Recall [95% CI]": f"{test_rf['recall']:.4f} [{ci_rf['recall']['ci_lower']:.4f}, {ci_rf['recall']['ci_upper']:.4f}]",
            "F1-Score [95% CI]": f"{test_rf['f1']:.4f} [{ci_rf['f1']['ci_lower']:.4f}, {ci_rf['f1']['ci_upper']:.4f}]",
            "ROC-AUC [95% CI]": f"{test_rf['roc_auc']:.4f} [{ci_rf['roc_auc']['ci_lower']:.4f}, {ci_rf['roc_auc']['ci_upper']:.4f}]",
            "Brier Score": f"{test_rf['brier_score']:.4f}",
        },
    ])
    st.dataframe(test_table, use_container_width=True, hide_index=True)

    # Paired Bootstrap Comparison
    diff_auc = eval_summary["bootstrap_comparison"]["diff_rf_minus_lr"]["roc_auc"]
    diff_rec = eval_summary["bootstrap_comparison"]["diff_rf_minus_lr"]["recall"]

    st.markdown(
        f"**Paired Bootstrap Comparison (Random Forest minus Baseline):**\n"
        f"* ROC-AUC Difference: `{diff_auc['mean']:+.4f}` (95% CI: `[{diff_auc['ci_lower']:+.4f}, {diff_auc['ci_upper']:+.4f}]`)\n"
        f"* Recall Difference: `{diff_rec['mean']:+.4f}` (95% CI: `[{diff_rec['ci_lower']:+.4f}, {diff_rec['ci_upper']:+.4f}]`)\n\n"
        f"Because both 95% confidence intervals include zero, the test cohort does not demonstrate a statistically significant performance gap between the two models. "
        f"Random Forest is adopted as the primary model by design, with Logistic Regression serving as the benchmark baseline."
    )


# TAB 3: OPERATING POINT
with tab_op:
    sec_rf = eval_summary["recall_oriented_operating_point"]["chosen_threshold"]["random_forest"]
    sec_lr = eval_summary["recall_oriented_operating_point"]["chosen_threshold"]["logistic_regression"]
    t_val = sec_rf["threshold"]

    st.subheader(f"Recall-Oriented Clinical Operating Point (Threshold = {t_val:.2f})")
    st.caption("Selected exclusively on out-of-fold training data as the highest threshold achieving ≥90% recall.")

    col_op_lr, col_op_rf = st.columns(2)
    cm_lr_s = sec_lr["secondary_test_result"]["confusion_matrix"]
    cm_rf_s = sec_rf["secondary_test_result"]["confusion_matrix"]

    with col_op_lr:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-weight:700; color:#0F1B2D; margin-bottom:6px;">Logistic Regression Baseline</div>
                <div><strong>Recall:</strong> {sec_lr['secondary_test_result']['recall']:.2%} (25/28 cases detected)</div>
                <div><strong>Precision:</strong> {sec_lr['secondary_test_result']['precision']:.2%} | <strong>Accuracy:</strong> {sec_lr['secondary_test_result']['accuracy']:.2%}</div>
                <div style="margin-top:8px; font-size:0.9rem; color:#5B6B7F;">
                    Confusion Matrix: TN = {cm_lr_s['tn']}, FP = {cm_lr_s['fp']}, FN = {cm_lr_s['fn']}, TP = {cm_lr_s['tp']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_op_rf:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-weight:700; color:#0F1B2D; margin-bottom:6px;">Random Forest Candidate</div>
                <div><strong>Recall:</strong> {sec_rf['secondary_test_result']['recall']:.2%} (27/28 cases detected)</div>
                <div><strong>Precision:</strong> {sec_rf['secondary_test_result']['precision']:.2%} | <strong>Accuracy:</strong> {sec_rf['secondary_test_result']['accuracy']:.2%}</div>
                <div style="margin-top:8px; font-size:0.9rem; color:#5B6B7F;">
                    Confusion Matrix: TN = {cm_rf_s['tn']}, FP = {cm_rf_s['fp']}, FN = {cm_rf_s['fn']}, TP = {cm_rf_s['tp']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        f"At the {t_val:.2f} operating-point threshold, Logistic Regression yielded {cm_lr_s['fn']} false negatives (25/28 cases detected) "
        f"and Random Forest yielded {cm_rf_s['fn']} false negative (27/28 cases detected) on the held-out test cohort."
    )

st.markdown("---")

# Visual Diagnostic Figures inside Cards
st.subheader("Diagnostic Evaluation Visualizations")
col_fig1, col_fig2 = st.columns(2)

cm_fig_path = REPO_ROOT / "reports" / "figures" / "confusion_matrix.png"
roc_fig_path = REPO_ROOT / "reports" / "figures" / "roc_curve.png"

with col_fig1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if cm_fig_path.exists():
        img_cm = Image.open(cm_fig_path)
        st.image(img_cm, caption="Confusion Matrix Heatmaps at Default Threshold (0.50)", use_container_width=True)
    else:
        st.warning(f"Diagnostic figure not found: {cm_fig_path.name}")
    st.markdown("</div>", unsafe_allow_html=True)

with col_fig2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if roc_fig_path.exists():
        img_roc = Image.open(roc_fig_path)
        st.image(img_roc, caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
    else:
        st.warning(f"Diagnostic figure not found: {roc_fig_path.name}")
    st.markdown("</div>", unsafe_allow_html=True)

render_footer()
