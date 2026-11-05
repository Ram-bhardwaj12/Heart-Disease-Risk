"""Model Performance Evaluation Page.

Displays cross-validation benchmarks, test set metrics with 95% bootstrap CIs,
the recall-oriented operating point, and diagnostic ROC / confusion matrix plots.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

# Setup local imports without importing from src/
import sys
app_dir = Path(__file__).resolve().parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components import render_disclaimer_banner
from utils import (
    REPO_ROOT,
    load_evaluation_summary,
    load_training_history,
)

st.set_page_config(page_title="Model Performance - Heart Disease Risk", page_icon="📊", layout="wide")

render_disclaimer_banner()

st.title("📊 Model Performance & Evaluation Benchmarks")
st.markdown("Comprehensive performance metrics on training cross-validation and the held-out clinical test cohort.")

eval_summary = load_evaluation_summary()
train_history = load_training_history()

# Table 1: Cross-validation on Training Set (5-fold, mean ± std)
st.subheader("1. Cross-Validation on Training Partition (5-Fold Stratified)")
st.caption("Hyperparameter optimization was performed exclusively on the 241-patient training split.")

cv_lr = train_history["models"]["baseline_logistic_regression"]["cv_metrics_at_best_index"]
cv_rf = train_history["models"]["candidate_random_forest"]["cv_metrics_at_best_index"]

cv_table_data = [
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
]
st.table(pd.DataFrame(cv_table_data))

# Table 2: Held-out Test Set (n = 61, threshold 0.5)
st.subheader("2. Unbiased Evaluation on Held-Out Test Set (n = 61, Threshold = 0.50)")
st.caption("Evaluated once on 61 strictly held-out patients; brackets represent 95% bootstrap confidence intervals (1,000 resamples).")

test_lr = eval_summary["primary_results_threshold_0_5"]["logistic_regression"]
ci_lr = test_lr["confidence_intervals_95"]
test_rf = eval_summary["primary_results_threshold_0_5"]["random_forest"]
ci_rf = test_rf["confidence_intervals_95"]

test_table_data = [
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
]
st.table(pd.DataFrame(test_table_data))

st.info(
    "⚠️ **Sample Size Caution:** The test partition contains 61 unique patient records (28 disease, 33 healthy). "
    "Because of this small sample size, test set point estimates carry sampling uncertainty, as indicated by the wide bootstrap intervals.",
    icon="ℹ️",
)

# Recall-Oriented Operating Point (Secondary Result)
st.subheader("3. Recall-Oriented Clinical Operating Point (Secondary Result)")
sec_rf = eval_summary["recall_oriented_operating_point"]["chosen_threshold"]["random_forest"]
sec_lr = eval_summary["recall_oriented_operating_point"]["chosen_threshold"]["logistic_regression"]
thresh_val = sec_rf["threshold"]

st.markdown(
    f"To prioritize sensitivity in clinical screening, a decision threshold of **{thresh_val:.2f}** was selected "
    f"based exclusively on out-of-fold training set cross-validation (the highest threshold achieving $\ge 90\%$ recall). "
    f"When applied to the held-out test cohort as a **secondary result**:"
)

col_op1, col_op2 = st.columns(2)
with col_op1:
    cm_lr_s = sec_lr["secondary_test_result"]["confusion_matrix"]
    st.markdown(
        f"**Logistic Regression (Threshold = {thresh_val:.2f}):**\n"
        f"* Confusion Matrix Counts: **TN = {cm_lr_s['tn']}**, **FP = {cm_lr_s['fp']}**, **FN = {cm_lr_s['fn']}**, **TP = {cm_lr_s['tp']}**\n"
        f"* Recall: **{sec_lr['secondary_test_result']['recall']:.4f}** | Precision: **{sec_lr['secondary_test_result']['precision']:.4f}** | Accuracy: **{sec_lr['secondary_test_result']['accuracy']:.4f}**"
    )

with col_op2:
    cm_rf_s = sec_rf["secondary_test_result"]["confusion_matrix"]
    st.markdown(
        f"**Random Forest (Threshold = {thresh_val:.2f}):**\n"
        f"* Confusion Matrix Counts: **TN = {cm_rf_s['tn']}**, **FP = {cm_rf_s['fp']}**, **FN = {cm_rf_s['fn']}**, **TP = {cm_rf_s['tp']}**\n"
        f"* Recall: **{sec_rf['secondary_test_result']['recall']:.4f}** | Precision: **{sec_rf['secondary_test_result']['precision']:.4f}** | Accuracy: **{sec_rf['secondary_test_result']['accuracy']:.4f}**"
    )

st.markdown(
    f"At the {thresh_val:.2f} operating-point threshold, Logistic Regression yielded {cm_lr_s['fn']} false negatives (25/28 cases detected) "
    f"and Random Forest yielded {cm_rf_s['fn']} false negative (27/28 cases detected) on the held-out test cohort."
)

# Statistical Comparison Note
st.subheader("4. Statistical Comparison (Paired Bootstrap)")
diff_auc = eval_summary["bootstrap_comparison"]["diff_rf_minus_lr"]["roc_auc"]
diff_rec = eval_summary["bootstrap_comparison"]["diff_rf_minus_lr"]["recall"]

diff_auc_mean = diff_auc["mean"]
diff_auc_lo = diff_auc["ci_lower"]
diff_auc_hi = diff_auc["ci_upper"]
diff_rec_mean = diff_rec["mean"]
diff_rec_lo = diff_rec["ci_lower"]
diff_rec_hi = diff_rec["ci_upper"]

st.markdown(
    f"On the held-out test cohort, the paired bootstrap difference (Random Forest minus Logistic Regression) shows:\n"
    f"* **ROC-AUC Difference:** Mean = `{diff_auc_mean:+.4f}`, 95% CI = `[{diff_auc_lo:+.4f}, {diff_auc_hi:+.4f}]`\n"
    f"* **Recall Difference:** Mean = `{diff_rec_mean:+.4f}`, 95% CI = `[{diff_rec_lo:+.4f}, {diff_rec_hi:+.4f}]`\n\n"
    f"Because both 95% confidence intervals include zero, the test data does not show a clear difference between the two models. "
    f"Random Forest is used as the project's main model by design, and Logistic Regression serves as the baseline."
)

st.markdown("---")

# Visual Diagnostic Figures
st.subheader("5. Diagnostic Visualizations")
col_fig1, col_fig2 = st.columns(2)

cm_fig_path = REPO_ROOT / "reports" / "figures" / "confusion_matrix.png"
roc_fig_path = REPO_ROOT / "reports" / "figures" / "roc_curve.png"

with col_fig1:
    if cm_fig_path.exists():
        img_cm = Image.open(cm_fig_path)
        st.image(img_cm, caption="Confusion Matrix Heatmaps at Default Threshold (0.50)", use_container_width=True)

with col_fig2:
    if roc_fig_path.exists():
        img_roc = Image.open(roc_fig_path)
        st.image(img_roc, caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)

