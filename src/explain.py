"""Phase 6: SHAP Explainability Analysis.

Computes TreeSHAP attributions for the candidate Random Forest classifier on the
held-out test set (data/processed/X_test.csv, data/processed/y_test.csv).

Key Deliverables:
1. TreeExplainer calculation for disease = 1 class.
2. Mathematical additivity validation (base_value + sum(shap) == predict_proba).
3. Clinical direction validation via Spearman rank correlation (thalach, oldpeak, exang).
4. Feature importance rankings: 22 encoded features and 13 grouped clinical variables.
5. Visualizations: beeswarm, 22-column bar plot, 13-feature grouped bar plot, and
   three local waterfall explanations (high-risk TP, low-risk TN, missed-case FN).
6. Artifacts: models/shap_explainer.joblib, reports/shap_values_test.csv,
   and reports/shap_summary.json.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap
import sklearn

from src.config import (
    DISEASE_COL,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
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


def build_explanation_for_row(
    row_idx: int,
    shap_row: np.ndarray,
    base_val: float,
    X_test: pd.DataFrame,
    X_test_trans: pd.DataFrame,
    feature_names: list[str],
    imputer_stats: dict[str, float],
) -> shap.Explanation:
    """Build a shap.Explanation with original units for numeric/count features."""
    display_data = np.zeros(len(feature_names))

    for j, col_name in enumerate(feature_names):
        if col_name.startswith("num__"):
            orig_col = col_name.replace("num__", "")
            raw_val = X_test.iloc[row_idx][orig_col]
            if pd.isna(raw_val):
                display_data[j] = imputer_stats.get(orig_col, 0.0)
            else:
                display_data[j] = float(raw_val)
        else:
            display_data[j] = float(X_test_trans.iloc[row_idx, j])

    return shap.Explanation(
        values=shap_row,
        base_values=base_val,
        data=display_data,
        feature_names=feature_names,
    )


def run_explainability() -> None:
    """Execute complete Phase 6 SHAP attribution, verification, and figure generation."""
    print("=" * 80)
    print("PHASE 6: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 80)

    # 1. Load pipeline and test data
    print("Step 1: Loading model pipeline and held-out test partition...")
    rf_model_path = MODELS_DIR / "best_random_forest.joblib"
    if not rf_model_path.exists():
        raise FileNotFoundError(f"Model pipeline not found at: {rf_model_path}")

    rf_pipe = joblib.load(rf_model_path)
    preprocessor = rf_pipe.named_steps["preprocessor"]
    classifier = rf_pipe.named_steps["classifier"]

    X_test, y_test = load_test_data()
    n_test = len(y_test)
    print(f"Loaded X_test ({n_test} samples) and y_test ({int(np.sum(y_test == 1))} disease=1, {int(np.sum(y_test == 0))} disease=0).")

    X_test_trans = preprocessor.transform(X_test)
    feature_names = list(X_test_trans.columns)
    print(f"Transformed test partition into {X_test_trans.shape[1]} encoded features: {X_test_trans.shape}\n")

    # Extract numerical imputer statistics for original-units representation
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    num_feature_list = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    imputer_stats = dict(zip(num_feature_list, num_imputer.statistics_))

    # 2. TreeExplainer calculation
    print("Step 2: Initializing TreeExplainer and calculating SHAP values...")
    explainer = shap.TreeExplainer(classifier)
    raw_shap_values = explainer.shap_values(X_test_trans)

    # Handle shape variants across shap versions
    print(f"TreeExplainer output type: {type(raw_shap_values)}")
    if isinstance(raw_shap_values, list):
        print(f"Found list of arrays with lengths: {[arr.shape for arr in raw_shap_values]}")
        shap_values_class1 = raw_shap_values[1]
        base_value_class1 = float(explainer.expected_value[1])
        print("Selected Class 1 (disease = 1, heart disease present) from array list index 1.")
    elif isinstance(raw_shap_values, np.ndarray):
        print(f"Found 3-D array with shape: {raw_shap_values.shape}")
        if raw_shap_values.ndim == 3 and raw_shap_values.shape[2] == 2:
            shap_values_class1 = raw_shap_values[:, :, 1]
            base_value_class1 = float(explainer.expected_value[1])
            print("Selected Class 1 (disease = 1, heart disease present) from slice [:, :, 1].")
        elif raw_shap_values.ndim == 2:
            shap_values_class1 = raw_shap_values
            base_value_class1 = float(explainer.expected_value)
            print("Selected 2-D array directly as Class 1 SHAP values.")
        else:
            raise ValueError(f"Unexpected SHAP values array shape: {raw_shap_values.shape}")
    else:
        raise TypeError(f"Unexpected SHAP values type: {type(raw_shap_values)}")

    print(f"Selected SHAP values shape: {shap_values_class1.shape}")
    print(f"Selected base value (expected value for class 1): {base_value_class1:.6f}\n")

    # 3. Additivity Check
    print("=" * 80)
    print("MATHEMATICAL ADDITIVITY CHECK")
    print("=" * 80)
    pred_probs = rf_pipe.predict_proba(X_test)[:, 1]
    reconstructed_probs = base_value_class1 + np.sum(shap_values_class1, axis=1)
    abs_errors = np.abs(reconstructed_probs - pred_probs)
    max_additivity_error = float(np.max(abs_errors))

    additivity_pass = max_additivity_error <= 1e-4
    print(f"Formula: base_value + sum(shap_values) == predict_proba[:, 1]")
    print(f"Maximum absolute additivity error across all {n_test} test samples: {max_additivity_error:.10e}")
    print(f"Additivity Check: [{'PASS' if additivity_pass else 'FAIL'}] (Threshold <= 1e-4)\n")

    if not additivity_pass:
        raise RuntimeError(f"FATAL: SHAP additivity check failed! Max error: {max_additivity_error}")

    # 4. Clinical Direction Check (Spearman Rank Correlation)
    print("=" * 80)
    print("CLINICAL DIRECTION CHECK (SPEARMAN RANK CORRELATION)")
    print("=" * 80)
    direction_checks = {}

    # Check num__thalach (expected negative)
    thalach_idx = feature_names.index("num__thalach")
    corr_thalach, p_thalach = spearmanr(X_test_trans["num__thalach"], shap_values_class1[:, thalach_idx])
    thalach_pass = corr_thalach < 0.0
    direction_checks["num__thalach"] = {
        "correlation": float(corr_thalach),
        "p_value": float(p_thalach),
        "passed": bool(thalach_pass),
        "expected": "negative",
    }
    print(f"  [{'PASS' if thalach_pass else 'FAIL'}] num__thalach vs SHAP: Spearman r = {corr_thalach:+.4f} (p = {p_thalach:.4e}) [Expected: negative]")

    # Check num__oldpeak (expected positive)
    oldpeak_idx = feature_names.index("num__oldpeak")
    corr_oldpeak, p_oldpeak = spearmanr(X_test_trans["num__oldpeak"], shap_values_class1[:, oldpeak_idx])
    oldpeak_pass = corr_oldpeak > 0.0
    direction_checks["num__oldpeak"] = {
        "correlation": float(corr_oldpeak),
        "p_value": float(p_oldpeak),
        "passed": bool(oldpeak_pass),
        "expected": "positive",
    }
    print(f"  [{'PASS' if oldpeak_pass else 'FAIL'}] num__oldpeak vs SHAP: Spearman r = {corr_oldpeak:+.4f} (p = {p_oldpeak:.4e}) [Expected: positive]")

    # Check bin__exang (expected positive)
    exang_idx = feature_names.index("bin__exang")
    corr_exang, p_exang = spearmanr(X_test_trans["bin__exang"], shap_values_class1[:, exang_idx])
    exang_pass = corr_exang > 0.0
    direction_checks["bin__exang"] = {
        "correlation": float(corr_exang),
        "p_value": float(p_exang),
        "passed": bool(exang_pass),
        "expected": "positive",
    }
    print(f"  [{'PASS' if exang_pass else 'FAIL'}] bin__exang vs SHAP  : Spearman r = {corr_exang:+.4f} (p = {p_exang:.4e}) [Expected: positive]")

    all_direction_pass = thalach_pass and oldpeak_pass and exang_pass
    if not all_direction_pass:
        print("\nWARNING: One or more SHAP clinical direction checks failed! Saving artifacts regardless.\n")
    else:
        print("\nAll clinical direction checks passed successfully.\n")

    # 5. Rankings: 22 Encodings and 13 Grouped Original Features
    print("=" * 80)
    print("FEATURE IMPORTANCE RANKINGS (MEAN ABSOLUTE SHAP)")
    print("=" * 80)

    # 22-column ranking
    mean_abs_22 = np.mean(np.abs(shap_values_class1), axis=0)
    ranking_22 = sorted(zip(feature_names, mean_abs_22), key=lambda x: x[1], reverse=True)

    print("Ranking A: All 22 Encoded Features (Mean |SHAP|):")
    print(f"{'Rank':<6} {'Feature Name':<25} {'Mean |SHAP|':>12}")
    print("-" * 45)
    for rank, (fname, val) in enumerate(ranking_22, start=1):
        print(f"{rank:<6} {fname:<25} {val:>12.4f}")
    print("-" * 45 + "\n")

    # 13-feature grouped ranking
    # Sum absolute SHAP across one-hot columns per patient, then average over patients
    orig_13_features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    grouped_patient_abs = {}
    for f in orig_13_features:
        matching_cols = [col for col in feature_names if col.startswith(f"num__{f}") or col.startswith(f"bin__{f}") or col.startswith(f"nom__{f}_")]
        col_indices = [feature_names.index(c) for c in matching_cols]
        grouped_patient_abs[f] = np.sum(np.abs(shap_values_class1[:, col_indices]), axis=1)

    grouped_ranking = sorted(
        [(f, float(np.mean(vals))) for f, vals in grouped_patient_abs.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    print("Ranking B: 13 Grouped Original Clinical Features (Mean |SHAP|):")
    print(f"{'Rank':<6} {'Feature Name':<25} {'Mean |SHAP|':>12}")
    print("-" * 45)
    for rank, (fname, val) in enumerate(grouped_ranking, start=1):
        print(f"{rank:<6} {fname:<25} {val:>12.4f}")
    print("-" * 45 + "\n")

    # Top 5 Comparison: Random Forest Impurity vs. SHAP
    rf_importances = classifier.feature_importances_
    rf_imp_sorted = sorted(zip(feature_names, rf_importances), key=lambda x: x[1], reverse=True)[:5]
    top5_shap_22 = ranking_22[:5]

    print("Top 5 Comparison: Random Forest Gini Impurity vs. SHAP (22 Encodings):")
    print(f"{'Rank':<6} {'RF Impurity Feature':<25} {'Impurity':>10}  |  {'SHAP Feature':<25} {'Mean |SHAP|':>12}")
    print("-" * 84)
    for rank in range(5):
        rf_f, rf_v = rf_imp_sorted[rank]
        sh_f, sh_v = top5_shap_22[rank]
        print(f"{rank+1:<6} {rf_f:<25} {rf_v:>10.4f}  |  {sh_f:<25} {sh_v:>12.4f}")
    print("-" * 84 + "\n")

    # 6. Patient Exemplar Selection for Waterfalls
    print("=" * 80)
    print("PATIENT EXEMPLAR SELECTION FOR WATERFALL VISUALIZATIONS")
    print("=" * 80)
    pred_labels = (pred_probs >= 0.50).astype(int)

    # 1. True Positive with highest predicted probability
    tp_indices = np.where((y_test == 1) & (pred_labels == 1))[0]
    best_tp_idx = int(tp_indices[np.argmax(pred_probs[tp_indices])])
    tp_prob = float(pred_probs[best_tp_idx])
    print(f"1. High-Risk True Positive : Test Row Index {best_tp_idx} (True Label = {int(y_test[best_tp_idx])}, Predicted Prob = {tp_prob:.4f})")

    # 2. True Negative with lowest predicted probability
    tn_indices = np.where((y_test == 0) & (pred_labels == 0))[0]
    best_tn_idx = int(tn_indices[np.argmin(pred_probs[tn_indices])])
    tn_prob = float(pred_probs[best_tn_idx])
    print(f"2. Low-Risk True Negative  : Test Row Index {best_tn_idx} (True Label = {int(y_test[best_tn_idx])}, Predicted Prob = {tn_prob:.4f})")

    # 3. False Negative with highest predicted probability
    fn_indices = np.where((y_test == 1) & (pred_labels == 0))[0]
    if len(fn_indices) > 0:
        best_fn_idx = int(fn_indices[np.argmax(pred_probs[fn_indices])])
        fn_prob = float(pred_probs[best_fn_idx])
        print(f"3. Missed-Case False Neg.  : Test Row Index {best_fn_idx} (True Label = {int(y_test[best_fn_idx])}, Predicted Prob = {fn_prob:.4f})")
    else:
        best_fn_idx = None
        fn_prob = None
        print("3. Missed-Case False Neg.  : None found at threshold 0.50 (skipped).")
    print()

    # 7. Generate Figures
    print("=" * 80)
    print("GENERATING SHAP EXPLAINABILITY FIGURES")
    print("=" * 80)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Beeswarm plot of 22 features
    fig_summary = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values_class1, X_test_trans, show=False)
    plt.title("SHAP Summary Beeswarm Plot (22 Encodings)\nHeld-Out Test Set (n = 61)", fontsize=12, pad=12)
    summary_plot_path = FIGURES_DIR / "shap_summary_plot.png"
    fig_summary.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig_summary)
    print(f"  Saved summary beeswarm plot: {summary_plot_path}")

    # Figure 2: Mean |SHAP| bar plot of 22 features
    fig_bar = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values_class1, X_test_trans, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (22 Encodings)\nHeld-Out Test Set (n = 61)", fontsize=12, pad=12)
    bar_plot_path = FIGURES_DIR / "shap_bar_plot.png"
    fig_bar.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"  Saved 22-column bar plot   : {bar_plot_path}")

    # Figure 3: Grouped bar plot of 13 original features
    fig_grouped, ax_g = plt.subplots(figsize=(8, 6))
    g_features = [item[0] for item in reversed(grouped_ranking)]
    g_values = [item[1] for item in reversed(grouped_ranking)]
    ax_g.barh(g_features, g_values, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax_g.set_xlabel("Mean Absolute SHAP Value (Impact on Model Output)", fontsize=11)
    ax_g.set_ylabel("Original Clinical Feature", fontsize=11)
    ax_g.set_title("SHAP Grouped Clinical Feature Importance (13 Variables)\nHeld-Out Test Set (n = 61)", fontsize=12, pad=12)
    ax_g.grid(axis="x", linestyle=":", alpha=0.5)
    for i, v in enumerate(g_values):
        ax_g.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
    grouped_bar_path = FIGURES_DIR / "shap_bar_grouped.png"
    fig_grouped.savefig(grouped_bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig_grouped)
    print(f"  Saved 13-feature bar plot  : {grouped_bar_path}")

    # Figure 4: Waterfall High-Risk True Positive
    exp_tp = build_explanation_for_row(
        row_idx=best_tp_idx,
        shap_row=shap_values_class1[best_tp_idx],
        base_val=base_value_class1,
        X_test=X_test,
        X_test_trans=X_test_trans,
        feature_names=feature_names,
        imputer_stats=imputer_stats,
    )
    plt.figure(figsize=(8.5, 6.5))
    shap.plots.waterfall(exp_tp, show=False)
    plt.title(f"Local SHAP Explanation: High-Risk True Positive\nTest Row Index {best_tp_idx} (True Label = {int(y_test[best_tp_idx])}, Predicted Prob = {tp_prob:.4f})", fontsize=11, pad=12)
    waterfall_tp_path = FIGURES_DIR / "shap_waterfall_high_risk.png"
    plt.savefig(waterfall_tp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved high-risk waterfall  : {waterfall_tp_path}")

    # Figure 5: Waterfall Low-Risk True Negative
    exp_tn = build_explanation_for_row(
        row_idx=best_tn_idx,
        shap_row=shap_values_class1[best_tn_idx],
        base_val=base_value_class1,
        X_test=X_test,
        X_test_trans=X_test_trans,
        feature_names=feature_names,
        imputer_stats=imputer_stats,
    )
    plt.figure(figsize=(8.5, 6.5))
    shap.plots.waterfall(exp_tn, show=False)
    plt.title(f"Local SHAP Explanation: Low-Risk True Negative\nTest Row Index {best_tn_idx} (True Label = {int(y_test[best_tn_idx])}, Predicted Prob = {tn_prob:.4f})", fontsize=11, pad=12)
    waterfall_tn_path = FIGURES_DIR / "shap_waterfall_low_risk.png"
    plt.savefig(waterfall_tn_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved low-risk waterfall   : {waterfall_tn_path}")

    # Figure 6: Waterfall Missed-Case False Negative
    if best_fn_idx is not None:
        exp_fn = build_explanation_for_row(
            row_idx=best_fn_idx,
            shap_row=shap_values_class1[best_fn_idx],
            base_val=base_value_class1,
            X_test=X_test,
            X_test_trans=X_test_trans,
            feature_names=feature_names,
            imputer_stats=imputer_stats,
        )
        plt.figure(figsize=(8.5, 6.5))
        shap.plots.waterfall(exp_fn, show=False)
        plt.title(f"Local SHAP Explanation: Missed-Case False Negative\nTest Row Index {best_fn_idx} (True Label = {int(y_test[best_fn_idx])}, Predicted Prob = {fn_prob:.4f})", fontsize=11, pad=12)
        waterfall_fn_path = FIGURES_DIR / "shap_waterfall_missed_case.png"
        plt.savefig(waterfall_fn_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved missed-case waterfall: {waterfall_fn_path}\n")

    # 8. Save Artifacts
    print("Step 8: Saving Explainer, CSV and JSON Artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    explainer_path = MODELS_DIR / "shap_explainer.joblib"
    csv_shap_path = REPORTS_DIR / "shap_values_test.csv"
    json_summary_path = REPORTS_DIR / "shap_summary.json"

    # Save explainer
    joblib.dump(explainer, explainer_path)

    # Save SHAP values CSV
    df_shap_export = pd.DataFrame(shap_values_class1, columns=feature_names)
    df_shap_export.index.name = "test_row_index"
    df_shap_export.to_csv(csv_shap_path, index=True)

    # Save comprehensive summary JSON
    summary_data = {
        "environment": {
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "shap_version": shap.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "explainer": {
            "type": "TreeExplainer",
            "class_selected": 1,
            "class_description": "disease = 1 (presence of heart disease)",
            "base_value": base_value_class1,
        },
        "additivity_check": {
            "formula": "base_value + sum(shap_values) == predict_proba[:, 1]",
            "max_absolute_error": max_additivity_error,
            "threshold": 1e-4,
            "passed": bool(additivity_pass),
        },
        "direction_checks_spearman": {
            **direction_checks,
            "all_passed": bool(all_direction_pass),
        },
        "rankings": {
            "encoded_22_features": [
                {"rank": r, "feature": fname, "mean_abs_shap": round(float(val), 6)}
                for r, (fname, val) in enumerate(ranking_22, start=1)
            ],
            "grouped_13_original_features": [
                {"rank": r, "feature": fname, "mean_abs_shap": round(float(val), 6)}
                for r, (fname, val) in enumerate(grouped_ranking, start=1)
            ],
            "top5_comparison_impurity_vs_shap": [
                {
                    "rank": rank + 1,
                    "rf_impurity_feature": rf_imp_sorted[rank][0],
                    "rf_impurity_value": round(float(rf_imp_sorted[rank][1]), 6),
                    "shap_22_feature": top5_shap_22[rank][0],
                    "shap_22_mean_abs": round(float(top5_shap_22[rank][1]), 6),
                }
                for rank in range(5)
            ],
        },
        "selected_exemplar_rows": {
            "high_risk_true_positive": {
                "test_row_index": best_tp_idx,
                "true_label": int(y_test[best_tp_idx]),
                "predicted_probability": round(tp_prob, 6),
            },
            "low_risk_true_negative": {
                "test_row_index": best_tn_idx,
                "true_label": int(y_test[best_tn_idx]),
                "predicted_probability": round(tn_prob, 6),
            },
            "missed_case_false_negative": {
                "test_row_index": best_fn_idx,
                "true_label": int(y_test[best_fn_idx]) if best_fn_idx is not None else None,
                "predicted_probability": round(fn_prob, 6) if fn_prob is not None else None,
            },
        },
    }

    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"  Saved TreeExplainer artifact: {explainer_path}")
    print(f"  Saved test SHAP values CSV  : {csv_shap_path}")
    print(f"  Saved SHAP summary JSON     : {json_summary_path}\n")

    print("Phase 6 SHAP explainability pipeline complete.")


if __name__ == "__main__":
    run_explainability()
