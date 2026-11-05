"""Phase 5: Model Evaluation on Held-Out Test Set.

Evaluates the saved baseline Logistic Regression and candidate Random Forest models
on the held-out test set (data/processed/X_test.csv, data/processed/y_test.csv).

Key Deliverables:
1. Primary Results at default threshold 0.5 with full confusion matrix counts.
2. 95% Bootstrap Confidence Intervals (1,000 resamples) and paired differences.
3. Recall-Oriented Operating Point selected via training out-of-fold predictions.
4. Evaluation Visualizations: confusion_matrix.png and roc_curve.png.
5. Summary Artifacts: metrics_comparison.csv and evaluation_summary.json.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import (
    DISEASE_COL,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)


def load_test_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load test features and binary disease labels."""
    x_test_path = PROCESSED_DIR / "X_test.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"

    if not x_test_path.exists():
        raise FileNotFoundError(f"Test features not found at: {x_test_path}")
    if not y_test_path.exists():
        raise FileNotFoundError(f"Test labels not found at: {y_test_path}")

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)[DISEASE_COL].to_numpy()
    return X_test, y_test


def load_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load training features and labels for out-of-fold operating point selection."""
    x_train_path = PROCESSED_DIR / "X_train.csv"
    y_train_path = PROCESSED_DIR / "y_train.csv"

    if not x_train_path.exists():
        raise FileNotFoundError(f"Training features not found at: {x_train_path}")
    if not y_train_path.exists():
        raise FileNotFoundError(f"Training labels not found at: {y_train_path}")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)[DISEASE_COL].to_numpy()
    return X_train, y_train


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate comprehensive classification metrics and confusion matrix at a specified threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def compute_bootstrap_cis(
    y_test: np.ndarray,
    prob_lr: np.ndarray,
    prob_rf: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compute 95% bootstrap confidence intervals and paired differences."""
    n_test = len(y_test)
    rng = np.random.RandomState(random_state)

    metrics_lr = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    metrics_rf = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    diff_auc = []
    diff_recall = []
    skipped_resamples = 0

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_test, size=n_test)
        y_b = y_test[idx]

        # Skip resamples that do not contain both positive and negative cases
        if len(np.unique(y_b)) < 2:
            skipped_resamples += 1
            continue

        p_lr_b = prob_lr[idx]
        p_rf_b = prob_rf[idx]
        pred_lr_b = (p_lr_b >= 0.5).astype(int)
        pred_rf_b = (p_rf_b >= 0.5).astype(int)

        # Logistic Regression bootstrap metrics
        acc_lr = accuracy_score(y_b, pred_lr_b)
        prec_lr = precision_score(y_b, pred_lr_b, zero_division=0)
        rec_lr = recall_score(y_b, pred_lr_b, zero_division=0)
        f1_lr = f1_score(y_b, pred_lr_b, zero_division=0)
        auc_lr = roc_auc_score(y_b, p_lr_b)

        # Random Forest bootstrap metrics
        acc_rf = accuracy_score(y_b, pred_rf_b)
        prec_rf = precision_score(y_b, pred_rf_b, zero_division=0)
        rec_rf = recall_score(y_b, pred_rf_b, zero_division=0)
        f1_rf = f1_score(y_b, pred_rf_b, zero_division=0)
        auc_rf = roc_auc_score(y_b, p_rf_b)

        metrics_lr["accuracy"].append(acc_lr)
        metrics_lr["precision"].append(prec_lr)
        metrics_lr["recall"].append(rec_lr)
        metrics_lr["f1"].append(f1_lr)
        metrics_lr["roc_auc"].append(auc_lr)

        metrics_rf["accuracy"].append(acc_rf)
        metrics_rf["precision"].append(prec_rf)
        metrics_rf["recall"].append(rec_rf)
        metrics_rf["f1"].append(f1_rf)
        metrics_rf["roc_auc"].append(auc_rf)

        diff_auc.append(auc_rf - auc_lr)
        diff_recall.append(rec_rf - rec_lr)

    cis_lr = {}
    cis_rf = {}
    for k in metrics_lr:
        cis_lr[k] = {
            "ci_lower": float(np.percentile(metrics_lr[k], 2.5)),
            "ci_upper": float(np.percentile(metrics_lr[k], 97.5)),
        }
        cis_rf[k] = {
            "ci_lower": float(np.percentile(metrics_rf[k], 2.5)),
            "ci_upper": float(np.percentile(metrics_rf[k], 97.5)),
        }

    return {
        "skipped_resamples": skipped_resamples,
        "n_valid_resamples": len(diff_auc),
        "ci_lr": cis_lr,
        "ci_rf": cis_rf,
        "diff_rf_minus_lr": {
            "roc_auc": {
                "mean": float(np.mean(diff_auc)),
                "ci_lower": float(np.percentile(diff_auc, 2.5)),
                "ci_upper": float(np.percentile(diff_auc, 97.5)),
            },
            "recall": {
                "mean": float(np.mean(diff_recall)),
                "ci_lower": float(np.percentile(diff_recall, 2.5)),
                "ci_upper": float(np.percentile(diff_recall, 97.5)),
            },
        },
    }


def select_recall_operating_point(
    lr_pipe: object,
    rf_pipe: object,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict:
    """Find operating threshold using out-of-fold training predictions only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob_lr = cross_val_predict(clone(lr_pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    oof_prob_rf = cross_val_predict(clone(rf_pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    threshold_grid = [round(float(t), 2) for t in np.arange(0.20, 0.65, 0.05)]
    oof_table_lr = []
    oof_table_rf = []

    for t in threshold_grid:
        pred_l = (oof_prob_lr >= t).astype(int)
        pred_r = (oof_prob_rf >= t).astype(int)

        prec_l = float(precision_score(y_train, pred_l, zero_division=0))
        rec_l = float(recall_score(y_train, pred_l, zero_division=0))
        oof_table_lr.append({"threshold": t, "precision": prec_l, "recall": rec_l})

        prec_r = float(precision_score(y_train, pred_r, zero_division=0))
        rec_r = float(recall_score(y_train, pred_r, zero_division=0))
        oof_table_rf.append({"threshold": t, "precision": prec_r, "recall": rec_r})

    def pick_threshold(table: list[dict]) -> tuple[float, str]:
        # Condition 1: Highest threshold whose OOF recall >= 0.90
        candidates_rec90 = [item for item in table if item["recall"] >= 0.90]
        if candidates_rec90:
            best = max(candidates_rec90, key=lambda x: x["threshold"])
            return best["threshold"], f"Highest threshold with out-of-fold recall >= 0.90 (threshold={best['threshold']}, OOF recall={best['recall']:.4f}, precision={best['precision']:.4f})"

        # Condition 2: Fallback to highest OOF recall maintaining OOF precision >= 0.70
        candidates_prec70 = [item for item in table if item["precision"] >= 0.70]
        if candidates_prec70:
            best = max(candidates_prec70, key=lambda x: x["recall"])
            return best["threshold"], f"Fallback highest recall with precision >= 0.70 (threshold={best['threshold']}, OOF recall={best['recall']:.4f}, precision={best['precision']:.4f})"

        return 0.50, "Default threshold 0.50 (no candidate satisfied criteria)"

    thresh_lr, rationale_lr = pick_threshold(oof_table_lr)
    thresh_rf, rationale_rf = pick_threshold(oof_table_rf)

    return {
        "threshold_grid": threshold_grid,
        "oof_table_lr": oof_table_lr,
        "oof_table_rf": oof_table_rf,
        "selected_lr": {"threshold": thresh_lr, "rationale": rationale_lr},
        "selected_rf": {"threshold": thresh_rf, "rationale": rationale_rf},
    }


def plot_confusion_matrices(cm_lr: dict, cm_rf: dict, output_path: Path) -> None:
    """Plot side-by-side confusion matrix heatmaps at threshold 0.5."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    labels = ["Disease absent", "Disease present"]
    matrix_lr = np.array([[cm_lr["tn"], cm_lr["fp"]], [cm_lr["fn"], cm_lr["tp"]]])
    matrix_rf = np.array([[cm_rf["tn"], cm_rf["fp"]], [cm_rf["fn"], cm_rf["tp"]]])

    sns.heatmap(
        matrix_lr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[0].set_title("Logistic Regression (Baseline)\nThreshold = 0.50", fontsize=12, pad=10)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)

    sns.heatmap(
        matrix_rf,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[1].set_title("Random Forest (Tuned Candidate)\nThreshold = 0.50", fontsize=12, pad=10)
    axes[1].set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    y_test: np.ndarray,
    prob_lr: np.ndarray,
    prob_rf: np.ndarray,
    auc_lr: float,
    auc_rf: float,
    output_path: Path,
) -> None:
    """Plot combined ROC curves with diagonal chance line."""
    fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.plot(
        fpr_rf,
        tpr_rf,
        color="#1f77b4",
        lw=2.5,
        label=f"Random Forest (AUC = {auc_rf:.4f})",
    )
    ax.plot(
        fpr_lr,
        tpr_lr,
        color="#ff7f0e",
        lw=2.0,
        linestyle="-.",
        label=f"Logistic Regression (AUC = {auc_lr:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Chance (AUC = 0.5000)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve\nHeld-Out Test Set (n = 61)", fontsize=12, pad=10)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_evaluation() -> None:
    """Execute complete Phase 5 evaluation, bootstrap analysis, operating points, and visualization."""
    print("=" * 80)
    print("PHASE 5: MODEL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 80)

    # 1. Load test data and trained model pipelines
    print("Step 1: Loading held-out test partition and trained model artifacts...")
    X_test, y_test = load_test_data()
    n_test = len(y_test)
    n_disease = int(np.sum(y_test == 1))
    n_healthy = int(np.sum(y_test == 0))
    print(f"Loaded X_test ({n_test} samples) and y_test ({n_disease} disease=1, {n_healthy} disease=0).\n")

    lr_model_path = MODELS_DIR / "baseline_logistic_regression.joblib"
    rf_model_path = MODELS_DIR / "best_random_forest.joblib"
    lr_pipe = joblib.load(lr_model_path)
    rf_pipe = joblib.load(rf_model_path)
    print(f"Loaded baseline model: {lr_model_path}")
    print(f"Loaded candidate model: {rf_model_path}\n")

    # Generate test set predicted probabilities
    prob_lr = lr_pipe.predict_proba(X_test)[:, 1]
    prob_rf = rf_pipe.predict_proba(X_test)[:, 1]

    # 2. Primary Results at threshold 0.50
    print("=" * 80)
    print("PRIMARY EVALUATION RESULTS (THRESHOLD = 0.50)")
    print("=" * 80)
    primary_lr = compute_metrics_at_threshold(y_test, prob_lr, threshold=0.50)
    primary_rf = compute_metrics_at_threshold(y_test, prob_rf, threshold=0.50)

    cm_lr = primary_lr["confusion_matrix"]
    cm_rf = primary_rf["confusion_matrix"]

    print("1. Baseline Logistic Regression:")
    print(f"   Accuracy    : {primary_lr['accuracy']:.4f}")
    print(f"   Precision   : {primary_lr['precision']:.4f}")
    print(f"   Recall      : {primary_lr['recall']:.4f}")
    print(f"   F1-Score    : {primary_lr['f1']:.4f}")
    print(f"   ROC-AUC     : {primary_lr['roc_auc']:.4f}")
    print(f"   Brier Score : {primary_lr['brier_score']:.4f}")
    print(f"   Confusion Matrix Counts: TN={cm_lr['tn']}, FP={cm_lr['fp']}, FN={cm_lr['fn']}, TP={cm_lr['tp']}\n")

    print("2. Candidate Random Forest:")
    print(f"   Accuracy    : {primary_rf['accuracy']:.4f}")
    print(f"   Precision   : {primary_rf['precision']:.4f}")
    print(f"   Recall      : {primary_rf['recall']:.4f}")
    print(f"   F1-Score    : {primary_rf['f1']:.4f}")
    print(f"   ROC-AUC     : {primary_rf['roc_auc']:.4f}")
    print(f"   Brier Score : {primary_rf['brier_score']:.4f}")
    print(f"   Confusion Matrix Counts: TN={cm_rf['tn']}, FP={cm_rf['fp']}, FN={cm_rf['fn']}, TP={cm_rf['tp']}\n")

    # 3. Bootstrap Confidence Intervals
    print("=" * 80)
    print("BOOTSTRAP UNCERTAINTY ESTIMATION (1000 RESAMPLES, 95% CI)")
    print("=" * 80)
    bootstrap_results = compute_bootstrap_cis(
        y_test=y_test,
        prob_lr=prob_lr,
        prob_rf=prob_rf,
        n_bootstraps=1000,
        random_state=RANDOM_STATE,
    )
    print(f"Bootstrap resamples generated: 1000 (Skipped unstratified resamples: {bootstrap_results['skipped_resamples']})\n")

    ci_lr = bootstrap_results["ci_lr"]
    ci_rf = bootstrap_results["ci_rf"]

    print("Logistic Regression (95% Bootstrap CIs):")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"   - {m:<12}: {primary_lr[m]:.4f} [{ci_lr[m]['ci_lower']:.4f}, {ci_lr[m]['ci_upper']:.4f}]")
    print()

    print("Random Forest (95% Bootstrap CIs):")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"   - {m:<12}: {primary_rf[m]:.4f} [{ci_rf[m]['ci_lower']:.4f}, {ci_rf[m]['ci_upper']:.4f}]")
    print()

    diff_auc = bootstrap_results["diff_rf_minus_lr"]["roc_auc"]
    diff_rec = bootstrap_results["diff_rf_minus_lr"]["recall"]
    print("Paired Model Difference (Random Forest minus Logistic Regression):")
    print(f"   - ROC-AUC Difference: Mean = {diff_auc['mean']:+.4f}, 95% CI = [{diff_auc['ci_lower']:+.4f}, {diff_auc['ci_upper']:+.4f}]")
    print(f"   - Recall Difference : Mean = {diff_rec['mean']:+.4f}, 95% CI = [{diff_rec['ci_lower']:+.4f}, {diff_rec['ci_upper']:+.4f}]\n")

    # 4. Recall-Oriented Operating Point Selection via Training Set OOF Predictions
    print("=" * 80)
    print("RECALL-ORIENTED OPERATING POINT (TRAINING OUT-OF-FOLD SEARCH)")
    print("=" * 80)
    X_train, y_train = load_training_data()
    oof_selection = select_recall_operating_point(lr_pipe, rf_pipe, X_train, y_train)

    print("Out-of-Fold Performance Table across Thresholds (Training Set Only):")
    print(f"{'Threshold':<12} {'LR Precision':<15} {'LR Recall':<15} {'RF Precision':<15} {'RF Recall':<15}")
    print("-" * 72)
    for row_l, row_r in zip(oof_selection["oof_table_lr"], oof_selection["oof_table_rf"]):
        t = row_l["threshold"]
        print(f"{t:<12.2f} {row_l['precision']:<15.4f} {row_l['recall']:<15.4f} {row_r['precision']:<15.4f} {row_r['recall']:<15.4f}")
    print("-" * 72)

    selected_lr_t = oof_selection["selected_lr"]["threshold"]
    selected_rf_t = oof_selection["selected_rf"]["threshold"]
    print(f"\nLogistic Regression Selected Threshold: {selected_lr_t:.2f} ({oof_selection['selected_lr']['rationale']})")
    print(f"Random Forest Selected Threshold      : {selected_rf_t:.2f} ({oof_selection['selected_rf']['rationale']})\n")

    # Apply single chosen threshold once to test set (Secondary Result)
    print("Secondary Evaluation on Test Set at Selected Operating Point:")
    sec_lr = compute_metrics_at_threshold(y_test, prob_lr, threshold=selected_lr_t)
    sec_rf = compute_metrics_at_threshold(y_test, prob_rf, threshold=selected_rf_t)
    cm_sec_lr = sec_lr["confusion_matrix"]
    cm_sec_rf = sec_rf["confusion_matrix"]

    print(f"1. Baseline Logistic Regression (Threshold = {selected_lr_t:.2f}):")
    print(f"   Accuracy : {sec_lr['accuracy']:.4f}")
    print(f"   Precision: {sec_lr['precision']:.4f}")
    print(f"   Recall   : {sec_lr['recall']:.4f}")
    print(f"   F1-Score : {sec_lr['f1']:.4f}")
    print(f"   Counts   : TN={cm_sec_lr['tn']}, FP={cm_sec_lr['fp']}, FN={cm_sec_lr['fn']}, TP={cm_sec_lr['tp']}\n")

    print(f"2. Candidate Random Forest (Threshold = {selected_rf_t:.2f}):")
    print(f"   Accuracy : {sec_rf['accuracy']:.4f}")
    print(f"   Precision: {sec_rf['precision']:.4f}")
    print(f"   Recall   : {sec_rf['recall']:.4f}")
    print(f"   F1-Score : {sec_rf['f1']:.4f}")
    print(f"   Counts   : TN={cm_sec_rf['tn']}, FP={cm_sec_rf['fp']}, FN={cm_sec_rf['fn']}, TP={cm_sec_rf['tp']}\n")

    # 5. Generate and Save Figures
    print("=" * 80)
    print("GENERATING EVALUATION FIGURES")
    print("=" * 80)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm_plot_path = FIGURES_DIR / "confusion_matrix.png"
    roc_plot_path = FIGURES_DIR / "roc_curve.png"

    plot_confusion_matrices(cm_lr, cm_rf, cm_plot_path)
    plot_roc_curves(y_test, prob_lr, prob_rf, primary_lr["roc_auc"], primary_rf["roc_auc"], roc_plot_path)
    print(f"  Saved confusion matrix plot: {cm_plot_path}")
    print(f"  Saved ROC curve plot       : {roc_plot_path}\n")

    # 6. Save Artifacts: CSV and JSON
    print("Step 6: Saving Reports (CSV & JSON)...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "metrics_comparison.csv"
    json_path = REPORTS_DIR / "evaluation_summary.json"

    # Save metrics comparison CSV
    csv_rows = [
        {
            "Model": "Logistic Regression (Baseline)",
            "Threshold": 0.50,
            "Accuracy": primary_lr["accuracy"],
            "Accuracy_95CI_Lower": ci_lr["accuracy"]["ci_lower"],
            "Accuracy_95CI_Upper": ci_lr["accuracy"]["ci_upper"],
            "Precision": primary_lr["precision"],
            "Precision_95CI_Lower": ci_lr["precision"]["ci_lower"],
            "Precision_95CI_Upper": ci_lr["precision"]["ci_upper"],
            "Recall": primary_lr["recall"],
            "Recall_95CI_Lower": ci_lr["recall"]["ci_lower"],
            "Recall_95CI_Upper": ci_lr["recall"]["ci_upper"],
            "F1": primary_lr["f1"],
            "F1_95CI_Lower": ci_lr["f1"]["ci_lower"],
            "F1_95CI_Upper": ci_lr["f1"]["ci_upper"],
            "ROC_AUC": primary_lr["roc_auc"],
            "ROC_AUC_95CI_Lower": ci_lr["roc_auc"]["ci_lower"],
            "ROC_AUC_95CI_Upper": ci_lr["roc_auc"]["ci_upper"],
            "Brier_Score": primary_lr["brier_score"],
            "TN": cm_lr["tn"],
            "FP": cm_lr["fp"],
            "FN": cm_lr["fn"],
            "TP": cm_lr["tp"],
        },
        {
            "Model": "Random Forest (Tuned Candidate)",
            "Threshold": 0.50,
            "Accuracy": primary_rf["accuracy"],
            "Accuracy_95CI_Lower": ci_rf["accuracy"]["ci_lower"],
            "Accuracy_95CI_Upper": ci_rf["accuracy"]["ci_upper"],
            "Precision": primary_rf["precision"],
            "Precision_95CI_Lower": ci_rf["precision"]["ci_lower"],
            "Precision_95CI_Upper": ci_rf["precision"]["ci_upper"],
            "Recall": primary_rf["recall"],
            "Recall_95CI_Lower": ci_rf["recall"]["ci_lower"],
            "Recall_95CI_Upper": ci_rf["recall"]["ci_upper"],
            "F1": primary_rf["f1"],
            "F1_95CI_Lower": ci_rf["f1"]["ci_lower"],
            "F1_95CI_Upper": ci_rf["f1"]["ci_upper"],
            "ROC_AUC": primary_rf["roc_auc"],
            "ROC_AUC_95CI_Lower": ci_rf["roc_auc"]["ci_lower"],
            "ROC_AUC_95CI_Upper": ci_rf["roc_auc"]["ci_upper"],
            "Brier_Score": primary_rf["brier_score"],
            "TN": cm_rf["tn"],
            "FP": cm_rf["fp"],
            "FN": cm_rf["fn"],
            "TP": cm_rf["tp"],
        },
    ]
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    # Save comprehensive evaluation summary JSON
    summary_data = {
        "dataset_rows": {
            "test_total": n_test,
            "disease_positive": n_disease,
            "disease_negative": n_healthy,
        },
        "primary_results_threshold_0_5": {
            "logistic_regression": {
                **primary_lr,
                "confidence_intervals_95": ci_lr,
            },
            "random_forest": {
                **primary_rf,
                "confidence_intervals_95": ci_rf,
            },
        },
        "bootstrap_comparison": {
            "n_bootstraps": 1000,
            "skipped_resamples": bootstrap_results["skipped_resamples"],
            "n_valid_resamples": bootstrap_results["n_valid_resamples"],
            "diff_rf_minus_lr": bootstrap_results["diff_rf_minus_lr"],
        },
        "recall_oriented_operating_point": {
            "selection_strategy": "Out-of-fold cross-validation on training partition only",
            "threshold_grid": oof_selection["threshold_grid"],
            "oof_table_logistic_regression": oof_selection["oof_table_lr"],
            "oof_table_random_forest": oof_selection["oof_table_rf"],
            "chosen_threshold": {
                "logistic_regression": {
                    "threshold": selected_lr_t,
                    "rationale": oof_selection["selected_lr"]["rationale"],
                    "secondary_test_result": sec_lr,
                },
                "random_forest": {
                    "threshold": selected_rf_t,
                    "rationale": oof_selection["selected_rf"]["rationale"],
                    "secondary_test_result": sec_rf,
                },
            },
        },
        "environment": {
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"  Saved metrics comparison CSV : {csv_path}")
    print(f"  Saved evaluation summary JSON: {json_path}\n")

    print("Phase 5 evaluation pipeline execution complete.")


if __name__ == "__main__":
    run_evaluation()

