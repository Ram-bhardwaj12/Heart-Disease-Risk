"""About View: Project Methodology, Data Pipeline, and Limitations."""

from pathlib import Path
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
    load_data_audit_summary,
    load_preprocessing_summary,
)

inject_css()
render_disclaimer_strip()
render_hero(COPY["NAV_ABOUT"], COPY["ABOUT_SUBTITLE"])

audit_data = load_data_audit_summary()
prep_data = load_preprocessing_summary()

raw_rows = audit_data.get("structure_audit", {}).get("total_rows", 1025)
unique_rows = audit_data.get("duplicate_audit", {}).get("unique_rows", 302)
dup_rows = audit_data.get("duplicate_audit", {}).get("exact_duplicate_rows", 723)
dup_pct = audit_data.get("duplicate_audit", {}).get("exact_duplicate_pct", 70.54)
invalid_ca = prep_data.get("invalid_codes_replaced", {}).get("ca", 4)
invalid_thal = prep_data.get("invalid_codes_replaced", {}).get("thal", 2)

# --- 4-STEP "HOW IT WORKS" FLOW ---
st.subheader("How It Works")
step_col1, step_col2, step_col3, step_col4 = st.columns(4)

with step_col1:
    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:1.5rem; font-weight:700; color:#0E7C86;">01</div>
            <div style="font-weight:700; margin:6px 0;">Data Audit</div>
            <div style="font-size:0.85rem; color:#5B6B7F; line-height:1.4;">
                Audited {raw_rows} records from the Cleveland Foundation dataset; identified and removed {dup_rows} exact duplicates ({dup_pct:.1f}%).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_col2:
    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:1.5rem; font-weight:700; color:#0E7C86;">02</div>
            <div style="font-weight:700; margin:6px 0;">Preprocessing</div>
            <div style="font-size:0.85rem; color:#5B6B7F; line-height:1.4;">
                Imputed missing/invalid codes (ca=4: {invalid_ca}, thal=0: {invalid_thal}). Scaled numeric predictors and one-hot encoded nominals inside CV folds.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_col3:
    st.markdown(
        """
        <div class="card">
            <div style="font-size:1.5rem; font-weight:700; color:#0E7C86;">03</div>
            <div style="font-weight:700; margin:6px 0;">Model Training</div>
            <div style="font-size:0.85rem; color:#5B6B7F; line-height:1.4;">
                Tuned 400-tree Random Forest on 241 training patients. Identified clinical screening operating point (t = 0.30) achieving ≥90% recall.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_col4:
    st.markdown(
        """
        <div class="card">
            <div style="font-size:1.5rem; font-weight:700; color:#0E7C86;">04</div>
            <div style="font-weight:700; margin:6px 0;">TreeSHAP XAI</div>
            <div style="font-size:0.85rem; color:#5B6B7F; line-height:1.4;">
                Computed additive feature attributions for every prediction, decomposing individual patient risk scores into grouped clinical contributions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- DETAIL EXPANDERS ---
with st.expander(":material/database: Dataset & Deduplication Details"):
    st.markdown(
        f"- **Raw Record Count:** {raw_rows} rows.\n"
        f"- **Exact Duplicate Rows:** {dup_rows} ({dup_pct:.2f}%).\n"
        f"- **Verified Unique Patients:** {unique_rows} clinical records.\n"
        f"- **Leakage Prevention:** All duplicate rows were removed prior to partitioning to prevent identical patient leakage between train and test splits."
    )

with st.expander(":material/swap_horiz: Label Direction Rationale (disease = 1 - target)"):
    st.markdown(
        "Phase 1 clinical verification revealed that patients labeled `target = 1` had higher maximum heart rates (`thalach`), "
        "lower ST depression (`oldpeak`), and lower exercise-induced angina (`exang`)—the classic physiological profile of healthy patients. "
        "The project models a derived binary label: **`disease = 1 - target`**, where `disease = 1` denotes coronary artery disease presence."
    )

with st.expander(":material/warning: Clinical & Statistical Limitations"):
    st.markdown(
        "1. **Sample Size:** The dataset contains 302 unique patients (61 held-out test records), leading to wide confidence intervals.\n"
        "2. **Regional Cohort Bias:** Historical Cleveland Clinic foundation data does not capture modern heterogeneous patient populations.\n"
        "3. **No Prospective Validation:** The application is developed as an academic working model and has not been evaluated in clinical practice.\n"
        "4. **Uncalibrated Model Scores:** Model scores reflect relative classification ranking, not empirically calibrated absolute probabilities."
    )

with st.expander(":material/policy: Academic Project Disclaimer"):
    st.markdown(f"*{COPY['DISCLAIMER']}*")
    st.caption("Developed in partial fulfillment of the B.Tech CSE (Machine Learning & Data Science) curriculum at Amity University Madhya Pradesh.")

render_footer()
