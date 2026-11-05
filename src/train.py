"""Phase 4: Model Training and Cross-Validated Hyperparameter Tuning.

Trains a baseline Logistic Regression model and a tuned Random Forest Classifier
exclusively on the training partition (X_train.csv, y_train.csv) using 5-fold
Stratified Cross-Validation.

Artifacts produced:
- models/baseline_logistic_regression.joblib (full fitted pipeline)
- models/best_random_forest.joblib (full fitted pipeline)
- reports/cv_results_logistic_regression.csv (full CV results table)
- reports/cv_results_random_forest.csv (full CV results table)
- models/training_history.json (summary metadata, metrics, importances, sanity checks)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    DISEASE_COL,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load training features and labels. Test set is strictly avoided."""
    x_train_path = PROCESSED_DIR / "X_train.csv"
    y_train_path = PROCESSED_DIR / "y_train.csv"

    if not x_train_path.exists():
        raise FileNotFoundError(f"Training features not found at: {x_train_path}")
    if not y_train_path.exists():
        raise FileNotFoundError(f"Training labels not found at: {y_train_path}")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)[DISEASE_COL]
    return X_train, y_train


def load_unfitted_preprocessor() -> object:
    """Load saved preprocessor and return an unfitted clone with identical settings."""
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at: {preprocessor_path}")
    fitted_preprocessor = joblib.load(preprocessor_path)
    return clone(fitted_preprocessor)


def extract_cv_metrics(grid: GridSearchCV, metric_names: list[str]) -> dict:
    """Extract mean and std for each metric at the best CV index."""
    best_idx = grid.best_index_
    metrics = {}
    for m in metric_names:
        mean_val = float(grid.cv_results_[f"mean_test_{m}"][best_idx])
        std_val = float(grid.cv_results_[f"std_test_{m}"][best_idx])
        metrics[m] = {"mean": mean_val, "std": std_val}
    return metrics


def run_training() -> None:
    """Execute complete Phase 4 training, tuning, sanity checks, and serialization."""
    print("=" * 80)
    print("PHASE 4: BASELINE AND CANDIDATE MODEL TRAINING")
    print("=" * 80)

    # 1. Load training data
    print("Step 1: Loading training partition (test set is strictly held out)...")
    X_train, y_train = load_training_data()
    n_samples, n_features = X_train.shape
    print(f"Loaded X_train ({n_samples} samples, {n_features} raw features) and y_train ({len(y_train)} labels).")
    print(f"Training class distribution (disease=1 / disease=0): "
          f"{int((y_train == 1).sum())} ({float((y_train == 1).mean() * 100):.2f}%) / "
          f"{int((y_train == 0).sum())} ({float((y_train == 0).mean() * 100):.2f}%)\n")

    # 2. Build model pipelines with unfitted preprocessor clones
    print("Step 2: Building full pipelines with unfitted preprocessor clones...")
    preprocessor_lr = load_unfitted_preprocessor()
    preprocessor_rf = load_unfitted_preprocessor()

    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_lr),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    candidate_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_rf),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    print("Configured pipelines: LogisticRegression (baseline) and RandomForestClassifier (candidate).\n")

    # 3. Setup GridSearchCV configuration
    print("Step 3: Setting up 5-Fold Stratified Cross-Validation and Hyperparameter Grids...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "accuracy": "accuracy",
    }
    metric_names = list(scoring.keys())

    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
    }

    rf_param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_leaf": [1, 3, 5],
        "classifier__max_features": ["sqrt", 0.5],
    }

    print(f"Logistic Regression Grid candidates: {len(lr_param_grid['classifier__C'])}")
    total_rf_candidates = (
        len(rf_param_grid["classifier__n_estimators"])
        * len(rf_param_grid["classifier__max_depth"])
        * len(rf_param_grid["classifier__min_samples_leaf"])
        * len(rf_param_grid["classifier__max_features"])
    )
    print(f"Random Forest Grid candidates: {total_rf_candidates} (total fits: {total_rf_candidates * 5})\n")

    # Train Baseline Model
    print("-" * 80)
    print("Training Model 1: Baseline Logistic Regression (GridSearchCV)...")
    lr_grid = GridSearchCV(
        estimator=baseline_pipeline,
        param_grid=lr_param_grid,
        scoring=scoring,
        refit="roc_auc",
        cv=cv,
    )
    lr_grid.fit(X_train, y_train)
    print("Baseline Logistic Regression training complete.")

    # Train Candidate Model
    print("-" * 80)
    print("Training Model 2: Candidate Random Forest Classifier (GridSearchCV)...")
    rf_grid = GridSearchCV(
        estimator=candidate_pipeline,
        param_grid=rf_param_grid,
        scoring=scoring,
        refit="roc_auc",
        cv=cv,
    )
    rf_grid.fit(X_train, y_train)
    print("Candidate Random Forest Classifier training complete.\n")

    # 4. Print Best Parameters and CV Scores
    print("=" * 80)
    print("CROSS-VALIDATION RESULTS SUMMARY (TRAINING SET ONLY)")
    print("=" * 80)
    print("Note: These cross-validation scores are evaluated strictly on the training partition")
    print("(5-fold stratified CV) and represent a slightly optimistic estimate because the optimal")
    print("hyperparameter configuration was selected on them. Unbiased generalization performance")
    print("will be evaluated on the unseen test set in Phase 5.\n")

    # Logistic Regression Summary
    lr_metrics = extract_cv_metrics(lr_grid, metric_names)
    print("1. Baseline Logistic Regression:")
    print(f"   Best Parameters: {lr_grid.best_params_}")
    print("   Cross-Validation Metrics (Mean +/- Std across 5 folds):")
    for m in metric_names:
        print(f"     - {m:<12}: {lr_metrics[m]['mean']:.4f} +/- {lr_metrics[m]['std']:.4f}")
    print()

    # Random Forest Summary
    rf_metrics = extract_cv_metrics(rf_grid, metric_names)
    print("2. Candidate Random Forest Classifier:")
    print(f"   Best Parameters: {rf_grid.best_params_}")
    print("   Cross-Validation Metrics (Mean +/- Std across 5 folds):")
    for m in metric_names:
        print(f"     - {m:<12}: {rf_metrics[m]['mean']:.4f} +/- {rf_metrics[m]['std']:.4f}")
    print()

    # 5. Label Sanity Check on Logistic Regression
    print("=" * 80)
    print("LABEL DIRECTION SANITY CHECK (LOGISTIC REGRESSION COEFFICIENTS)")
    print("=" * 80)
    best_lr_pipe = lr_grid.best_estimator_
    lr_prep = best_lr_pipe.named_steps["preprocessor"]
    lr_clf = best_lr_pipe.named_steps["classifier"]
    feature_names = list(lr_prep.get_feature_names_out())
    coefs = lr_clf.coef_[0]

    coef_dict = dict(zip(feature_names, coefs))
    sorted_coefs = sorted(coef_dict.items(), key=lambda item: item[1])

    print(f"All {len(sorted_coefs)} Logistic Regression coefficients (sorted by value):")
    print(f"{'Feature Name':<25} {'Coefficient':>12}")
    print("-" * 38)
    for fname, val in sorted_coefs:
        print(f"{fname:<25} {val:>+12.4f}")
    print("-" * 38)

    thalach_coef = coef_dict.get("num__thalach", 0.0)
    oldpeak_coef = coef_dict.get("num__oldpeak", 0.0)
    exang_coef = coef_dict.get("bin__exang", 0.0)

    thalach_pass = thalach_coef < 0.0
    oldpeak_pass = oldpeak_coef > 0.0
    exang_pass = exang_coef > 0.0

    print("\nSanity Check Verification (disease = 1 corresponds to disease present):")
    print(f"  [{'PASS' if thalach_pass else 'FAIL'}] num__thalach coefficient is negative: {thalach_coef:+.4f} (< 0)")
    print(f"  [{'PASS' if oldpeak_pass else 'FAIL'}] num__oldpeak coefficient is positive: {oldpeak_coef:+.4f} (> 0)")
    print(f"  [{'PASS' if exang_pass else 'FAIL'}] bin__exang coefficient is positive  : {exang_coef:+.4f} (> 0)")

    sanity_passed = thalach_pass and oldpeak_pass and exang_pass
    if not sanity_passed:
        print("\nWARNING: One or more label direction sanity checks failed! Saving artifacts regardless.\n")
    else:
        print("\nAll clinical label direction sanity checks passed successfully.\n")

    # 6. Top 10 Random Forest Feature Importances
    print("=" * 80)
    print("TOP 10 RANDOM FOREST FEATURE IMPORTANCES")
    print("=" * 80)
    best_rf_pipe = rf_grid.best_estimator_
    rf_prep = best_rf_pipe.named_steps["preprocessor"]
    rf_clf = best_rf_pipe.named_steps["classifier"]
    rf_feat_names = list(rf_prep.get_feature_names_out())
    rf_importances = rf_clf.feature_importances_

    rf_imp_sorted = sorted(zip(rf_feat_names, rf_importances), key=lambda x: x[1], reverse=True)
    top10_importances = rf_imp_sorted[:10]

    print(f"{'Rank':<6} {'Feature Name':<25} {'Importance':>12}")
    print("-" * 45)
    for rank, (fname, imp) in enumerate(top10_importances, start=1):
        print(f"{rank:<6} {fname:<25} {imp:>12.4f}")
    print("-" * 45 + "\n")

    # 7. Save Artifacts
    print("Step 7: Saving Model and Report Artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lr_model_path = MODELS_DIR / "baseline_logistic_regression.joblib"
    rf_model_path = MODELS_DIR / "best_random_forest.joblib"
    lr_cv_csv_path = REPORTS_DIR / "cv_results_logistic_regression.csv"
    rf_cv_csv_path = REPORTS_DIR / "cv_results_random_forest.csv"
    history_path = MODELS_DIR / "training_history.json"

    # Save full fitted pipelines
    joblib.dump(lr_grid.best_estimator_, lr_model_path)
    joblib.dump(rf_grid.best_estimator_, rf_model_path)

    # Save full cv_results_ tables
    pd.DataFrame(lr_grid.cv_results_).to_csv(lr_cv_csv_path, index=False)
    pd.DataFrame(rf_grid.cv_results_).to_csv(rf_cv_csv_path, index=False)

    # Save models/training_history.json
    training_history = {
        "training_samples": n_samples,
        "random_state": RANDOM_STATE,
        "cv_folds": 5,
        "refit_metric": "roc_auc",
        "models": {
            "baseline_logistic_regression": {
                "best_params": lr_grid.best_params_,
                "cv_metrics_at_best_index": lr_metrics,
                "label_sanity_checks": {
                    "num__thalach_negative": {"value": float(thalach_coef), "passed": bool(thalach_pass)},
                    "num__oldpeak_positive": {"value": float(oldpeak_coef), "passed": bool(oldpeak_pass)},
                    "bin__exang_positive": {"value": float(exang_coef), "passed": bool(exang_pass)},
                    "all_passed": bool(sanity_passed),
                },
            },
            "candidate_random_forest": {
                "best_params": rf_grid.best_params_,
                "cv_metrics_at_best_index": rf_metrics,
                "top_10_feature_importances": [
                    {"rank": idx, "feature": fname, "importance": round(float(imp), 6)}
                    for idx, (fname, imp) in enumerate(top10_importances, start=1)
                ],
            },
        },
        "environment": {
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    print(f"  Saved baseline model pipeline : {lr_model_path}")
    print(f"  Saved tuned RF model pipeline : {rf_model_path}")
    print(f"  Saved Logistic Regression CV  : {lr_cv_csv_path}")
    print(f"  Saved Random Forest CV        : {rf_cv_csv_path}")
    print(f"  Saved Training History JSON   : {history_path}\n")

    print("Phase 4 model training and cross-validation complete.")


if __name__ == "__main__":
    run_training()
