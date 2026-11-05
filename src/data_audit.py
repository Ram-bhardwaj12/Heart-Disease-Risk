"""Data Audit and Deduplication Script for Heart Disease Dataset.

Phase 1: Loads raw data, audits schema, checks duplicates and label conflicts,
validates value distributions, evaluates target direction correlations,
deduplicates rows, and exports summary report.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COL,
)


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load the raw dataset without modifying it."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    return pd.read_csv(filepath)


def audit_structure(df: pd.DataFrame) -> dict:
    """Audit dataset dimensions, column names, dtypes, and missing values."""
    print("=" * 80)
    print("A. DATASET STRUCTURE & SCHEMA AUDIT")
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print(f"{'Column':<15} {'Dtype':<12} {'Missing Values':<16} {'Missing %':<12}")
    print("-" * 55)
    missing_info = {}
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        pct_missing = (n_missing / len(df)) * 100
        print(f"{col:<15} {str(df[col].dtype):<12} {n_missing:<16} {pct_missing:<12.2f}%")
        missing_info[col] = {
            "dtype": str(df[col].dtype),
            "missing_count": n_missing,
            "missing_pct": round(pct_missing, 4),
        }
    print()
    return missing_info


def audit_duplicates(df: pd.DataFrame) -> tuple[int, int, int, int, dict, dict]:
    """Audit exact duplicates and conflicting label groups across features."""
    print("=" * 80)
    print("B. DUPLICATE OBSERVATIONS & LABEL CONFLICT AUDIT")
    print("=" * 80)

    raw_rows = len(df)
    exact_duplicates = int(df.duplicated().sum())
    unique_rows = raw_rows - exact_duplicates

    print(f"Total raw rows:               {raw_rows}")
    print(f"Exact duplicate rows:         {exact_duplicates} ({exact_duplicates / raw_rows * 100:.2f}%)")
    print(f"Unique rows after dedup:      {unique_rows} ({unique_rows / raw_rows * 100:.2f}%)\n")

    # Conflicting label check: same 13 features but differing target value
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    grouped_targets = df.groupby(feature_cols)[TARGET_COL].nunique()
    conflicting_groups = grouped_targets[grouped_targets > 1]
    n_conflicting_groups = len(conflicting_groups)

    if n_conflicting_groups > 0:
        conflicting_keys = conflicting_groups.index
        conflicting_mask = df.set_index(feature_cols).index.isin(conflicting_keys)
        n_conflicting_rows = int(conflicting_mask.sum())
    else:
        n_conflicting_rows = 0

    print(f"Conflicting label groups (identical 13 features with differing target): {n_conflicting_groups}")
    print(f"Total rows involved in conflicting label groups:                        {n_conflicting_rows}\n")

    # Class balance before deduplication
    before_counts = df[TARGET_COL].value_counts().to_dict()
    before_balance = {
        int(k): {
            "count": int(v),
            "percentage": round(float(v / raw_rows * 100), 2),
        }
        for k, v in before_counts.items()
    }

    # Class balance after deduplication
    df_dedup = df.drop_duplicates(keep="first")
    after_counts = df_dedup[TARGET_COL].value_counts().to_dict()
    after_balance = {
        int(k): {
            "count": int(v),
            "percentage": round(float(v / len(df_dedup) * 100), 2),
        }
        for k, v in after_counts.items()
    }

    print("Class Balance BEFORE Deduplication:")
    for cls, stats in sorted(before_balance.items()):
        print(f"  Target {cls}: {stats['count']:>4} ({stats['percentage']:>5.2f}%)")

    print("\nClass Balance AFTER Deduplication:")
    for cls, stats in sorted(after_balance.items()):
        print(f"  Target {cls}: {stats['count']:>4} ({stats['percentage']:>5.2f}%)")
    print()

    return (
        raw_rows,
        exact_duplicates,
        unique_rows,
        n_conflicting_groups,
        before_balance,
        after_balance,
    )


def audit_values(df_dedup: pd.DataFrame) -> tuple[dict, dict, list[str]]:
    """Audit categorical and numeric value distributions and flag anomalies."""
    print("=" * 80)
    print("C. FEATURE VALUE AUDIT & ANOMALY DETECTION (DEDUPLICATED DATA)")
    print("=" * 80)

    categorical_results = {}
    anomalies = []

    print("--- Categorical Feature Value Counts ---")
    for feat in CATEGORICAL_FEATURES:
        vc = df_dedup[feat].value_counts().sort_index().to_dict()
        vc_clean = {int(k) if isinstance(k, (int, np.integer)) else k: int(v) for k, v in vc.items()}
        categorical_results[feat] = vc_clean
        print(f"\nFeature: '{feat}'")
        for val, count in vc_clean.items():
            pct = count / len(df_dedup) * 100
            print(f"  Value {val}: {count:>3} ({pct:>5.2f}%)")

    # Domain / README boundary checks
    # ca check: standard clinical range is 0 to 3 major vessels
    if "ca" in df_dedup.columns:
        ca_anom = (df_dedup["ca"] == 4).sum()
        if ca_anom > 0:
            anom_msg = f"Feature 'ca' contains {ca_anom} row(s) with value 4 (outside standard 0-3 range; historically used in Cleveland dataset as missing/uncolored code)."
            anomalies.append(anom_msg)

    # thal check: clinical codes are typically 1 (normal), 2 (fixed), 3 (reversible)
    if "thal" in df_dedup.columns:
        thal_zero = (df_dedup["thal"] == 0).sum()
        if thal_zero > 0:
            anom_msg = f"Feature 'thal' contains {thal_zero} row(s) with value 0 (outside standard 1-3 range; used as null/unrecorded entry in original source)."
            anomalies.append(anom_msg)

    # slope check: values are 0, 1, 2
    if "slope" in df_dedup.columns:
        invalid_slope = (~df_dedup["slope"].isin([0, 1, 2])).sum()
        if invalid_slope > 0:
            anomalies.append(f"Feature 'slope' contains {invalid_slope} row(s) outside permissible [0, 1, 2].")

    # cp check: values are 0, 1, 2, 3
    if "cp" in df_dedup.columns:
        invalid_cp = (~df_dedup["cp"].isin([0, 1, 2, 3])).sum()
        if invalid_cp > 0:
            anomalies.append(f"Feature 'cp' contains {invalid_cp} row(s) outside permissible [0, 1, 2, 3].")

    print("\n" + "-" * 55)
    print("--- Numeric Feature Statistics ---")
    print(f"{'Feature':<12} {'Min':<8} {'Max':<8} {'Mean':<10} {'Median':<8} {'Zero Count':<12}")
    print("-" * 55)

    numeric_results = {}
    for feat in NUMERIC_FEATURES:
        col = df_dedup[feat]
        col_min = float(col.min())
        col_max = float(col.max())
        col_mean = float(col.mean())
        col_median = float(col.median())
        zero_count = int((col == 0).sum())

        numeric_results[feat] = {
            "min": round(col_min, 4),
            "max": round(col_max, 4),
            "mean": round(col_mean, 4),
            "median": round(col_median, 4),
            "zero_count": zero_count,
        }
        print(f"{feat:<12} {col_min:<8.1f} {col_max:<8.1f} {col_mean:<10.2f} {col_median:<8.1f} {zero_count:<12}")

    print("\n--- Flagged Discrepancies / Boundary Anomalies ---")
    if anomalies:
        for idx, anom in enumerate(anomalies, 1):
            print(f"  [{idx}] {anom}")
    else:
        print("  No boundary anomalies found.")
    print()

    return categorical_results, numeric_results, anomalies


def audit_target_direction(df_dedup: pd.DataFrame) -> dict:
    """Compute target direction relationships on deduplicated data without drawing conclusions."""
    print("=" * 80)
    print("D. TARGET DIRECTION CHECK (DEDUPLICATED DATA)")
    print("=" * 80)

    # 1. Continuous feature means grouped by target
    means_by_target = df_dedup.groupby(TARGET_COL)[["thalach", "oldpeak", "age"]].mean()
    print("Continuous Features Mean by Target:")
    print(means_by_target.to_string())
    print()

    # 2. Proportion of exang == 1 grouped by target
    exang_prop = df_dedup.groupby(TARGET_COL)["exang"].apply(lambda s: (s == 1).mean())
    print("Proportion of exang = 1 by Target:")
    print(exang_prop.to_string())
    print()

    # 3. Target rate grouped by cp, thal, slope, ca
    rate_by_cp = df_dedup.groupby("cp")[TARGET_COL].mean()
    print("Target Rate (Mean of Target) by Chest Pain Type ('cp'):")
    print(rate_by_cp.to_string())
    print()

    rate_by_thal = df_dedup.groupby("thal")[TARGET_COL].mean()
    print("Target Rate (Mean of Target) by Thalassemia ('thal'):")
    print(rate_by_thal.to_string())
    print()

    rate_by_slope = df_dedup.groupby("slope")[TARGET_COL].mean()
    print("Target Rate (Mean of Target) by ST Segment Slope ('slope'):")
    print(rate_by_slope.to_string())
    print()

    rate_by_ca = df_dedup.groupby("ca")[TARGET_COL].mean()
    print("Target Rate (Mean of Target) by Major Vessels ('ca'):")
    print(rate_by_ca.to_string())
    print()

    target_direction_results = {
        "means_by_target": {
            str(k): {col: round(float(val), 4) for col, val in row.items()}
            for k, row in means_by_target.iterrows()
        },
        "exang_prop_by_target": {str(k): round(float(v), 4) for k, v in exang_prop.items()},
        "target_rate_by_cp": {str(k): round(float(v), 4) for k, v in rate_by_cp.items()},
        "target_rate_by_thal": {str(k): round(float(v), 4) for k, v in rate_by_thal.items()},
        "target_rate_by_slope": {str(k): round(float(v), 4) for k, v in rate_by_slope.items()},
        "target_rate_by_ca": {str(k): round(float(v), 4) for k, v in rate_by_ca.items()},
    }

    return target_direction_results


def run_data_audit() -> None:
    """Execute complete Phase 1 pipeline."""
    # 1. Load data
    df_raw = load_raw_data(RAW_DATA_PATH)

    # 2. Structural audit
    missing_info = audit_structure(df_raw)

    # 3. Duplicate audit
    (
        raw_rows,
        duplicate_rows,
        unique_rows,
        conflicting_groups,
        class_balance_before,
        class_balance_after,
    ) = audit_duplicates(df_raw)

    # 4. Perform deduplication
    df_dedup = df_raw.drop_duplicates(keep="first").reset_index(drop=True)

    # 5. Value audit on deduplicated data
    categorical_counts, numeric_summary, anomalies = audit_values(df_dedup)

    # 6. Target direction audit
    target_direction = audit_target_direction(df_dedup)

    # 7. Save deduplicated dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dedup_path = PROCESSED_DIR / "heart_deduplicated.csv"
    df_dedup.to_csv(dedup_path, index=False)
    print("=" * 80)
    print("E. EXPORT DEDUPLICATED DATASET")
    print("=" * 80)
    print(f"Saved deduplicated data to: {dedup_path}")
    print(f"Final shape: {df_dedup.shape[0]} rows, {df_dedup.shape[1]} columns\n")

    # 8. Save audit summary JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "data_audit_summary.json"
    summary_data = {
        "raw_rows": raw_rows,
        "duplicate_rows": duplicate_rows,
        "unique_rows": unique_rows,
        "conflicting_label_groups": conflicting_groups,
        "class_balance_before": class_balance_before,
        "class_balance_after": class_balance_after,
        "anomalies_found": anomalies,
        "categorical_value_counts": categorical_counts,
        "numeric_summary": numeric_summary,
        "target_direction_check": target_direction,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("=" * 80)
    print("F. EXPORT AUDIT SUMMARY REPORT")
    print("=" * 80)
    print(f"Saved summary JSON report to: {report_path}\n")

    # 9. Final Readable Summary
    print("=" * 80)
    print("G. PHASE 1 DATA AUDIT SUMMARY")
    print("=" * 80)
    print(f"* Raw Record Count:            {raw_rows}")
    print(f"* Duplicate Rows Purged:       {duplicate_rows} ({duplicate_rows / raw_rows * 100:.2f}%)")
    print(f"* Verified Unique Patients:    {unique_rows}")
    print(f"* Conflicting Label Groups:    {conflicting_groups}")
    print(f"* Deduplicated Class Balance:  Target 0 = {class_balance_after[0]['count']} ({class_balance_after[0]['percentage']}%), Target 1 = {class_balance_after[1]['count']} ({class_balance_after[1]['percentage']}%)")
    print(f"* Anomalies Detected:          {len(anomalies)}")
    print(f"* Deduplicated File Created:   {dedup_path.name}")
    print(f"* Summary JSON Created:        {report_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    run_data_audit()
