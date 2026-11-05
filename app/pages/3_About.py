"""Project Methodology, Data Pipeline, and Limitations Page."""

from pathlib import Path
import streamlit as st

import sys
app_dir = Path(__file__).resolve().parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components import render_disclaimer_banner
from utils import (
    load_data_audit_summary,
    load_preprocessing_summary,
)

st.set_page_config(page_title="About & Methodology - Heart Disease Risk", page_icon="📖", layout="wide")

render_disclaimer_banner()

st.title("📖 Project Methodology, Data Integrity & Limitations")
st.markdown(
    "Academic machine learning capstone project for **B.Tech CSE (Machine Learning & Data Science)** "
    "at **Amity University Madhya Pradesh**."
)

audit_data = load_data_audit_summary()
prep_data = load_preprocessing_summary()

raw_rows = audit_data.get("structure_audit", {}).get("total_rows", 1025)
unique_rows = audit_data.get("duplicate_audit", {}).get("unique_rows", 302)
dup_rows = audit_data.get("duplicate_audit", {}).get("exact_duplicate_rows", 723)
dup_pct = audit_data.get("duplicate_audit", {}).get("exact_duplicate_pct", 70.54)

# Section 1: Dataset & Rigorous Deduplication
st.subheader("1. Clinical Dataset & Deduplication Audit")
st.markdown(
    f"* **Source Dataset:** Kaggle Heart Disease Dataset derived from the Cleveland Clinic foundation database.\n"
    f"* **Raw Record Count:** `{raw_rows}` observations across 13 clinical features.\n"
    f"* **Duplicate Observations Detected:** `{dup_rows}` repeated entries (`{dup_pct:.2f}%`).\n"
    f"* **Verified Unique Patient Records:** `{unique_rows}` distinct clinical individuals.\n"
    f"* **Leakage Prevention:** To avoid synthetic performance inflation and severe train-test data leakage, "
    f"**all duplicates were purged strictly before partitioning the dataset**."
)

# Section 2: Label Direction Verification
st.subheader("2. Label Direction & Verification (`disease = 1 - target`)")
st.markdown(
    "A rigorous clinical audit in Phase 1 revealed that patients labeled `target = 1` exhibited higher maximum heart rates (`thalach`), "
    "lower exercise ST depression (`oldpeak`), significantly lower exercise angina (`exang`), and zero fluoroscopy vessels (`ca = 0`), "
    "which corresponds to the physiological profile of healthy individuals. "
    "To adhere to standard diagnostic orientation, the project models a derived label: **`disease = 1 - target`**, "
    "where `disease = 1` denotes heart disease presence."
)

# Section 3: Invalid Code Handling & Preprocessing Pipeline
st.subheader("3. Data Preprocessing & Missing Code Handling")
invalid_ca = prep_data.get("invalid_codes_replaced", {}).get("ca", 4)
invalid_thal = prep_data.get("invalid_codes_replaced", {}).get("thal", 2)

st.markdown(
    f"* **Invalid Clinical Codes:** `ca = 4` ({invalid_ca} occurrences) and `thal = 0` ({invalid_thal} occurrences) "
    f"were converted to missing values (`NaN`) and imputed with training-set statistics.\n"
    f"* **Pipeline Steps:**\n"
    f"  * **Numeric & Count features** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`): Median imputation followed by `StandardScaler`.\n"
    f"  * **Binary features** (`sex`, `fbs`, `exang`): Most frequent imputation and passthrough.\n"
    f"  * **Nominal categorical features** (`cp`, `restecg`, `slope`, `thal`): Most frequent imputation followed by `OneHotEncoder(handle_unknown='ignore')`."
)

# Section 4: Machine Learning Models
st.subheader("4. Predictive Modeling Framework")
st.markdown(
    "* **Baseline Model:** L2-regularized Logistic Regression ($C = 0.1$) optimized across 5-fold cross-validation.\n"
    "* **Candidate Primary Model:** Random Forest Classifier (`n_estimators = 400`, `max_depth = 3`, `min_samples_leaf = 5`, `max_features = 'sqrt'`).\n"
    "* **Explainability Engine:** TreeSHAP (game-theoretic feature attribution) providing mathematical additivity on model probabilities."
)

# Section 5: Clinical Limitations
st.subheader("5. Clinical & Statistical Limitations")
st.markdown(
    "1. **Small Sample Cohort:** After removing duplicate records, the dataset comprises only 302 unique patients (61 in test split), "
    "limiting statistical power and introducing sampling noise into point metrics.\n"
    "2. **Single-Source Regional Bias:** Data collection originates from historical clinical cohorts that do not adequately reflect modern heterogeneous demographic, socioeconomic, or regional populations.\n"
    "3. **Absence of Prospective Clinical Validation:** The application is developed strictly as an academic benchmarking tool and has not been validated in a clinical trial.\n"
    "4. **Unverified Feature Encodings:** Value codes for `cp`, `slope`, and `thal` are reported directly as raw codes without secondary clinical verification against original hospital archives.\n"
    "5. **Unchecked Calibration:** The model's predicted score reflects relative classification confidence, not calibrated true empirical probabilities."
)

# Section 6: Academic Disclaimer
st.subheader("6. Academic Disclaimer")
st.markdown(
    "*This software and associated documentation are developed exclusively for academic and educational evaluation "
    "in partial fulfillment of the B.Tech CSE (Machine Learning and Data Science) curriculum at Amity University Madhya Pradesh. "
    "It is not an FDA-approved medical diagnostic device and must not be used to direct or determine patient diagnosis, "
    "medical management, or clinical interventions.*"
)
