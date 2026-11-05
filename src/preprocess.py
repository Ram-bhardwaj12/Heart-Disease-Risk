"""Phase 3 Preprocessing, Label Correction, and Stratified Splitting.

This script executes Phase 3 of the Early Heart Disease Risk Prediction pipeline:
1. Ingests the verified deduplicated dataset (data/processed/heart_deduplicated.csv).
2. Derives the clinically validated label: disease = 1 - target.
3. Flags and replaces invalid/anomalous codes (ca=4, thal=0) with NaN.
4. Partitions features and labels using a stratified train/test split.
5. Constructs a scikit-learn ColumnTransformer fitted exclusively on training data.
6. Serializes raw split CSVs and the fitted preprocessor pipeline.
7. Executes comprehensive verification checks and writes a JSON summary report.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    COUNT_FEATURES,
    DISEASE_COL,
    INVALID_CODES,
    MODELS_DIR,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SIZE,
)


def load_deduplicated_data(filepath: Path) -> pd.DataFrame:
    """Load deduplicated dataset without modifying original file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Deduplicated data not found at: {filepath}")
    return pd.read_csv(filepath)


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer configured for pandas DataFrame output."""
    num_cols = NUMERIC_FEATURES + COUNT_FEATURES
    bin_cols = BINARY_FEATURES
    nom_cols = NOMINAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("passthrough", "passthrough"),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("bin", binary_pipeline, bin_cols),
            ("nom", nominal_pipeline, nom_cols),
        ],
        remainder="drop",
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def run_verification(
    df_dedup: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    preprocessor_path: Path,
) -> list[str]:
    """Run all Phase 3 verification checks and return feature names."""
    print("=" * 80)
    print("PHASE 3 VERIFICATION SUITE")
    print("=" * 80)
    failures = []

    # Check 1: no row appears in both train and test
    overlap_indices = set(X_train.index).intersection(set(X_test.index))
    if len(overlap_indices) == 0:
        print(f"[PASS] Check 1: No row appears in both train and test (0 overlapping indices).")
    else:
        msg = f"[FAIL] Check 1: Found {len(overlap_indices)} overlapping indices between train and test."
        print(msg)
        failures.append(msg)

    # Check 2: train + test rows equal the deduplicated row count
    total_split_rows = len(X_train) + len(X_test)
    if total_split_rows == len(df_dedup):
        print(
            f"[PASS] Check 2: Train rows ({len(X_train)}) + test rows ({len(X_test)}) "
            f"equal deduplicated row count ({len(df_dedup)})."
        )
    else:
        msg = f"[FAIL] Check 2: Train rows ({len(X_train)}) + test rows ({len(X_test)}) = {total_split_rows} != {len(df_dedup)}."
        print(msg)
        failures.append(msg)

    # Check 3: disease % differs by less than 2 percentage points between train and test
    train_disease_pct = float((y_train == 1).mean() * 100.0)
    test_disease_pct = float((y_test == 1).mean() * 100.0)
    disease_pct_diff = abs(train_disease_pct - test_disease_pct)
    if disease_pct_diff < 2.0:
        print(
            f"[PASS] Check 3: Disease % difference between train and test is {disease_pct_diff:.4f} pp (< 2.00 pp) "
            f"| Train: {train_disease_pct:.2f}% (n={(y_train == 1).sum()}/{len(y_train)}), "
            f"Test: {test_disease_pct:.2f}% (n={(y_test == 1).sum()}/{len(y_test)})."
        )
    else:
        msg = f"[FAIL] Check 3: Disease % difference is {disease_pct_diff:.4f} pp, exceeding 2.00 pp threshold."
        print(msg)
        failures.append(msg)

    # Check 4: transformed train and test contain no NaN
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    train_nans = int(X_train_trans.isna().sum().sum())
    test_nans = int(X_test_trans.isna().sum().sum())
    if train_nans == 0 and test_nans == 0:
        print(
            f"[PASS] Check 4: Transformed train and test matrices contain zero NaN values "
            f"(Train NaNs: {train_nans}, Test NaNs: {test_nans})."
        )
    else:
        msg = f"[FAIL] Check 4: Transformed matrices contain NaNs (Train NaNs: {train_nans}, Test NaNs: {test_nans})."
        print(msg)
        failures.append(msg)

    # Check 5: preprocessor was fit on training data only (state how this is ensured)
    state_msg = (
        f"[PASS] Check 5: Preprocessor fit guarantee: preprocessor.fit() was invoked strictly and exclusively "
        f"on X_train ({len(X_train)} samples). X_test ({len(X_test)} samples) was strictly held out and only "
        f"passed to transform(), ensuring no statistical parameters (medians, means, scales, OHE categories) leaked from test data."
    )
    print(state_msg)

    # Check 6: saved preprocessor loads with joblib and transforms X_test with the same columns
    try:
        loaded_preprocessor = joblib.load(preprocessor_path)
        X_test_trans_loaded = loaded_preprocessor.transform(X_test)
        cols_match = list(X_test_trans_loaded.columns) == list(X_train_trans.columns)
        shapes_match = X_test_trans_loaded.shape == X_test_trans.shape
        values_identical = np.allclose(
            X_test_trans_loaded.to_numpy(), X_test_trans.to_numpy(), equal_nan=True
        )

        if cols_match and shapes_match and values_identical:
            print(
                f"[PASS] Check 6: Serialized preprocessor reloaded successfully via joblib. "
                f"It transforms X_test into identical shape {X_test_trans_loaded.shape} with matching {len(X_test_trans_loaded.columns)} column names."
            )
        else:
            msg = (
                f"[FAIL] Check 6: Reloaded preprocessor mismatch "
                f"(cols_match={cols_match}, shapes_match={shapes_match}, values_identical={values_identical})."
            )
            print(msg)
            failures.append(msg)
    except Exception as e:
        msg = f"[FAIL] Check 6: Exception while reloading or transforming with serialized preprocessor: {e}"
        print(msg)
        failures.append(msg)

    print("=" * 80)
    if failures:
        print(f"FAILED: {len(failures)} verification check(s) did not pass.")
        for err in failures:
            print(f"  * {err}")
        raise RuntimeError("Preprocessing verification checks failed.")
    print("ALL 6 VERIFICATION CHECKS PASSED.")
    print("=" * 80 + "\n")

    return list(X_train_trans.columns)


def run_preprocessing() -> None:
    """Execute complete Phase 3 preprocessing, partition, serialization, and reporting."""
    print("=" * 80)
    print("PHASE 3: PREPROCESSING, LABEL CORRECTION, AND STRATIFIED SPLIT")
    print("=" * 80)

    dedup_path = PROCESSED_DIR / "heart_deduplicated.csv"
    print(f"Step A: Loading deduplicated dataset from: {dedup_path}")
    df_raw = load_deduplicated_data(dedup_path)
    df = df_raw.copy()
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.\n")

    # Step B: Create disease = 1 - target; keep target_original
    print("Step B: Label Derivation")
    print(f"Creating '{DISEASE_COL}' = 1 - '{TARGET_COL}'.")
    df[DISEASE_COL] = 1 - df[TARGET_COL]
    df["target_original"] = df[TARGET_COL]

    overall_total = len(df)
    overall_disease_counts = df[DISEASE_COL].value_counts().to_dict()
    overall_disease_pct = {
        k: float((v / overall_total) * 100.0) for k, v in overall_disease_counts.items()
    }
    print(f"Derived label '{DISEASE_COL}' distribution across {overall_total} unique patients:")
    print(
        f"  disease = 1 (Present): {overall_disease_counts.get(1, 0)} ({overall_disease_pct.get(1, 0.0):.2f}%)"
    )
    print(
        f"  disease = 0 (Absent) : {overall_disease_counts.get(0, 0)} ({overall_disease_pct.get(0, 0.0):.2f}%)\n"
    )

    # Step C: Replace ca == 4 and thal == 0 with NaN
    print("Step C: Invalid / Anomalous Code Handling")
    invalid_counts_replaced = {}
    for col, invalid_val in INVALID_CODES.items():
        count = int((df[col] == invalid_val).sum())
        invalid_counts_replaced[col] = count
        df.loc[df[col] == invalid_val, col] = np.nan
        print(f"  Column '{col}': replaced {count} occurrence(s) of value {invalid_val} with NaN.")
    print()

    # Model input feature isolation
    model_feature_cols = (
        NUMERIC_FEATURES + COUNT_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES
    )
    # Ensure preserved feature order matches raw dataset
    ordered_feature_cols = [c for c in df_raw.columns if c in model_feature_cols]

    X = df[ordered_feature_cols].copy()
    y = df[DISEASE_COL].copy()

    # Step D: Stratified train_test_split
    print("Step D: Stratified Partitioning")
    print(
        f"Splitting data with TEST_SIZE={TEST_SIZE} (80% train, 20% test), "
        f"RANDOM_STATE={RANDOM_STATE}, stratified on '{DISEASE_COL}'."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_total = len(X_train)
    test_total = len(X_test)
    train_disease_counts = y_train.value_counts().to_dict()
    test_disease_counts = y_test.value_counts().to_dict()
    train_disease_pct = {
        k: float((v / train_total) * 100.0) for k, v in train_disease_counts.items()
    }
    test_disease_pct = {
        k: float((v / test_total) * 100.0) for k, v in test_disease_counts.items()
    }

    print(f"Partition results:")
    print(f"  X_train: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(
        f"    disease=1: {train_disease_counts.get(1, 0)} ({train_disease_pct.get(1, 0.0):.2f}%), "
        f"disease=0: {train_disease_counts.get(0, 0)} ({train_disease_pct.get(0, 0.0):.2f}%)"
    )
    print(f"  X_test : {X_test.shape[0]} rows, {X_test.shape[1]} features")
    print(
        f"    disease=1: {test_disease_counts.get(1, 0)} ({test_disease_pct.get(1, 0.0):.2f}%), "
        f"disease=0: {test_disease_counts.get(0, 0)} ({test_disease_pct.get(0, 0.0):.2f}%)\n"
    )

    # Step E: Build ColumnTransformer and fit ONLY on training features
    print("Step E: Preprocessor Pipeline Construction & Fitting")
    print(f"  Numeric & Count Features ({len(NUMERIC_FEATURES + COUNT_FEATURES)}): {NUMERIC_FEATURES + COUNT_FEATURES}")
    print("    -> SimpleImputer(strategy='median') + StandardScaler()")
    print(f"  Binary Features ({len(BINARY_FEATURES)}): {BINARY_FEATURES}")
    print("    -> SimpleImputer(strategy='most_frequent') + Passthrough")
    print(f"  Nominal Features ({len(NOMINAL_FEATURES)}): {NOMINAL_FEATURES}")
    print("    -> SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore', sparse_output=False)")

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    print("Preprocessor successfully fitted strictly on X_train.\n")

    # Step F: Save raw features (with NaNs where invalid) and serialized preprocessor
    print("Step F: Saving Artifacts")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train_path = PROCESSED_DIR / "X_train.csv"
    x_test_path = PROCESSED_DIR / "X_test.csv"
    y_train_path = PROCESSED_DIR / "y_train.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_frame(name=DISEASE_COL).to_csv(y_train_path, index=False)
    y_test.to_frame(name=DISEASE_COL).to_csv(y_test_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"  Saved raw train features: {x_train_path}")
    print(f"  Saved raw test features : {x_test_path}")
    print(f"  Saved train labels      : {y_train_path}")
    print(f"  Saved test labels       : {y_test_path}")
    print(f"  Saved preprocessor      : {preprocessor_path}\n")

    # Step Verification
    output_feature_names = run_verification(
        df_dedup=df_raw,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        preprocessor_path=preprocessor_path,
    )

    print(f"Output transformed feature names ({len(output_feature_names)} features):")
    for idx, fname in enumerate(output_feature_names, start=1):
        print(f"  {idx:>2}. {fname}")
    print()

    # Step G: Write reports/preprocessing_summary.json
    print("Step G: Writing Preprocessing Summary Report")
    summary_data = {
        "dataset_rows": {
            "total_deduplicated": overall_total,
            "train_rows": train_total,
            "test_rows": test_total,
            "test_split_ratio": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "disease_class_balance": {
            "overall": {
                "counts": {str(k): int(v) for k, v in overall_disease_counts.items()},
                "percentages": {str(k): round(v, 2) for k, v in overall_disease_pct.items()},
            },
            "train": {
                "counts": {str(k): int(v) for k, v in train_disease_counts.items()},
                "percentages": {str(k): round(v, 2) for k, v in train_disease_pct.items()},
            },
            "test": {
                "counts": {str(k): int(v) for k, v in test_disease_counts.items()},
                "percentages": {str(k): round(v, 2) for k, v in test_disease_pct.items()},
            },
            "train_test_disease_pct_diff": round(abs(train_disease_pct[1] - test_disease_pct[1]), 4),
        },
        "invalid_codes_replaced": invalid_counts_replaced,
        "output_feature_names": output_feature_names,
        "environment": {
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
        },
    }

    summary_path = REPORTS_DIR / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary report saved to: {summary_path}\n")
    print("Phase 3 pipeline execution complete.")


if __name__ == "__main__":
    run_preprocessing()
