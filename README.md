# Early Heart Disease Risk Prediction using Explainable Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## Project Overview

* **Project Title:** Early Heart Disease Risk Prediction using Explainable Machine Learning
* **Core Concepts:** Supervised Classification, Random Forest Classifier, SHAP (SHapley Additive exPlanations)
* **Academic Context:** Data Science Working-Model Project, B.Tech CSE (Machine Learning and Data Science), Amity University Madhya Pradesh (AUMP)
* **Submission Deadline:** 16 November 2026

Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide. Early detection can save lives, but conventional "black-box" machine learning models often fail to gain clinician trust. This project builds a complete, reproducible clinical risk prediction pipeline using Random Forest classification paired with SHAP explainability. Every prediction is paired with intuitive local feature attribution—showing exactly which clinical indicators increased or decreased an individual patient's cardiovascular risk.

---

## Project Team

| Role | Member Name & Enrollment | Assigned Responsibilities | Owns |
| :--- | :--- | :--- | :--- |
| **Developer** | Ram Bhardwaj, A60205224028 | End-to-end data pipeline, model training, SHAP explainability analysis, Streamlit application, and cloud deployment. | `src/`, `models/`, `data/`, later notebooks, deployment |
| **Teammate** | `[Name, Enrollment No.]` | Works independently; contributions to be updated. | EDA notebook, `reports/`, Streamlit UI pages |

---

## Problem Statement and Objectives

### Problem Statement
Heart disease screening is critical for timely clinical intervention and preventive therapy. While standard machine learning algorithms can achieve high statistical performance, healthcare professionals cannot rely on opaque "black-box" predictions when patient care is at stake. To be clinically viable, an automated screening system must not only deliver high diagnostic sensitivity (recall) to minimize missed diagnoses (false negatives) but also provide granular, interpretable explanations for every individual risk prediction.

### Measurable Objectives
1. **Clean and Deduplicate Data:** Audit the dataset for structural anomalies, handle missing or corrupted values, and remove identical duplicate records prior to splitting to eliminate data leakage.
2. **Train Baseline and Candidate Models:** Establish a baseline with Logistic Regression and develop a hyperparameter-tuned Random Forest classifier.
3. **Optimize with Recall-Focused Metrics:** Evaluate models prioritizing Recall (Sensitivity), ROC-AUC, and F1-score to strictly minimize false negative predictions.
4. **Explain Predictions with SHAP:** Compute TreeSHAP values to yield both global feature importance (summary plots, feature importance rankings) and local explanations (patient-level waterfall plots).
5. **Deploy an Interactive Streamlit App:** Build and deploy a public-facing Streamlit web app providing real-time risk stratification and patient-specific SHAP explanation plots.

---

## Dataset Description

The project utilizes the widely studied **Kaggle Heart Disease Dataset** (derived from the classic Cleveland, Hungarian, Switzerland, and Long Beach V databases).

* **Total Records:** 1,025 rows
* **Feature Count:** 13 clinical predictors + 1 binary target column
* **Target Definition:** `target` (0 = No Heart Disease Present, 1 = Heart Disease Present)

### Feature Dictionary

| Feature Name | Clinical Description | Data Type | Permissible / Observed Range |
| :--- | :--- | :--- | :--- |
| `age` | Age of patient in years | Integer | 29 – 77 |
| `sex` | Biological sex (`0` = Female, `1` = Male) | Categorical (Binary) | 0, 1 |
| `cp` | Chest pain type (`0` = Asymptomatic (no chest pain), `1` = Atypical angina, `2` = Non-anginal pain, `3` = Typical angina); labels inferred from value counts and disease rates (see reports/category_label_evidence.md); the dataset provider does not document the encoding | Categorical (Nominal) | 0, 1, 2, 3 |
| `trestbps` | Resting blood pressure in mm Hg on hospital admission | Continuous (Integer) | 94 – 200 mm Hg |
| `chol` | Serum cholesterol level in mg/dl | Continuous (Integer) | 126 – 564 mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl (`1` = true, `0` = false) | Categorical (Binary) | 0, 1 |
| `restecg` | Resting electrocardiographic results (`0` = Left ventricular hypertrophy, `1` = Normal, `2` = ST-T wave abnormality); labels inferred from value counts and disease rates (see reports/category_label_evidence.md); the dataset provider does not document the encoding | Categorical (Nominal) | 0, 1, 2 |
| `thalach` | Maximum heart rate achieved during stress test | Continuous (Integer) | 71 – 202 bpm |
| `exang` | Exercise-induced angina (`1` = yes, `0` = no) | Categorical (Binary) | 0, 1 |
| `oldpeak` | ST depression induced by exercise relative to rest | Continuous (Float) | 0.0 – 6.2 |
| `slope` | Slope of the peak exercise ST segment (`0` = Downsloping, `1` = Flat, `2` = Upsloping); labels inferred from value counts and disease rates (see reports/category_label_evidence.md); the dataset provider does not document the encoding | Categorical (Nominal) | 0, 1, 2 |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy | Discrete (Integer) | ca = 4 (4 rows) and thal = 0 (2 rows) occur in the data and are treated as missing |
| `thal` | Thallium stress test result (`1` = Fixed defect, `2` = Normal, `3` = Reversible defect); labels inferred from value counts and disease rates (see reports/category_label_evidence.md); the dataset provider does not document the encoding | Categorical (Nominal) | ca = 4 (4 rows) and thal = 0 (2 rows) occur in the data and are treated as missing |
| `target` | Original label column from the Kaggle file. Audit evidence in this file indicates target = 1 corresponds to ABSENCE of heart disease. The project uses a derived column 'disease' = 1 - target (1 = disease present). | Binary Target | 0, 1 |

### Known Data Issues
* **Duplicate Observations:** The raw 1,025-row dataset contains repeated observations representing approximately 300 unique clinical patients.
* **Leakage Prevention:** To avoid synthetic performance inflation and severe data leakage between training and testing sets, **all duplicate records are purged strictly BEFORE the train/test split**.
* **Audit & Verification:** The exact count of unique instances will be verified systematically in code during Phase 1 and recorded directly into the project's Results section.
* **Encoding verification:** Value codes for sex, cp, restecg, slope and thal: labels inferred from value counts and disease rates (see reports/category_label_evidence.md); the dataset provider does not document the encoding.
* Label direction: the audit showed target = 1 groups have higher thalach, lower oldpeak, lower exercise angina and ca = 0 more often, which is the clinical pattern of healthy patients. The project therefore uses disease = 1 - target. Sanity check after training passed: for disease = 1, logistic regression coefficients are negative for thalach and positive for oldpeak and exang.
* Invalid codes: ca = 4 and thal = 0 are converted to missing values and imputed with the most frequent training value.
* Small test set: with 302 unique patients the test split has about 61 rows, so test metrics are noisy and are reported together with cross-validation results.
* Rare category: restecg = 2 occurs in 4 patients in total, all of which fall in the training split.

---

## Tech Stack & Tooling

* **Python 3.11:** Core execution language offering robust performance, broad library compatibility, and typing stability.
* **pandas:** High-performance tabular data structures for data ingestion, deduplication, and transformation.
* **numpy:** Vectorized numerical operations and multi-dimensional array support.
* **scikit-learn:** Comprehensive machine learning suite for data preprocessing, baseline model training, hyperparameter grid search, and evaluation metrics.
* **SHAP:** Game-theoretic feature attribution library utilizing TreeExplainer to produce interpretable global and local model explanations.
* **matplotlib:** Foundational plotting library for formatting and rendering diagnostic visual evaluations.
* **seaborn:** Statistical data visualization library for correlation heatmaps and feature distributions.
* **joblib:** Serialization engine for persistent storage and rapid reloading of trained model artifacts and preprocessing scalers.
* **Streamlit:** Python-native framework for rapid development of interactive, clinician-friendly web applications.
* **GitHub:** Version control repository hosting project code, documentation, and revision tracking.
* **Streamlit Community Cloud:** Managed cloud hosting environment enabling continuous public web app accessibility directly from GitHub.
* **Antigravity:** AI coding assistant utilized prompt-by-prompt across planning, code generation, debugging, and documentation stages.

---

## Project Workflow

The project executes through eight sequential, modular phases:

```
[Phase 1: Ingestion & Deduplication]  [Phase 2: EDA]  [Phase 3: Preprocessing & Split]
                                                                        ⬇
[Phase 6: SHAP Explainability] ⬅ [Phase 5: Evaluation] ⬅ [Phase 4: Model Training]
             ⬇
[Phase 7: Streamlit App]  [Phase 8: Cloud Deployment]
```

1. **Phase 1: Data Loading, Validation, and Deduplication**
   * **Goal:** Inspect schema, detect duplicate rows, validate ranges, and deduplicate dataset.
   * **Input:** `data/raw/heart.csv`
   * **Output Files:** `data/processed/heart_deduplicated.csv`, `reports/data_audit_summary.json`

2. **Phase 2: Exploratory Data Analysis (EDA)**
   * **Goal:** Examine class balance, correlations, feature distributions, and clinical trends.
   * **Input:** `data/processed/heart_deduplicated.csv`
   * **Output Files:** `notebooks/01_eda.ipynb`, `reports/figures/correlation_matrix.png`, `reports/figures/feature_distributions.png`

3. **Phase 3: Data Preprocessing and Partitioning**
   * **Goal:** Handle anomalies/outliers, encode categorical features, scale numerical features, and perform a stratified train/test split (`random_state=42`).
   * **Input:** `data/processed/heart_deduplicated.csv`
   * **Output Files:** `data/processed/X_train.csv`, `data/processed/X_test.csv`, `data/processed/y_train.csv`, `data/processed/y_test.csv`, `models/preprocessor.joblib`

4. **Phase 4: Baseline & Candidate Model Training**
   * **Goal:** Train a Logistic Regression baseline and conduct cross-validated hyperparameter tuning (`GridSearchCV`) for a Random Forest classifier.
   * **Input:** `data/processed/X_train.csv`, `data/processed/y_train.csv`
   * **Output Files:** `models/baseline_logistic_regression.joblib`, `models/best_random_forest.joblib`, `models/training_history.json`

5. **Phase 5: Model Evaluation & Benchmarking**
   * **Goal:** Compute classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Brier score) and confusion matrices on unseen test data, prioritizing high sensitivity.
   * **Input:** `models/best_random_forest.joblib`, `data/processed/X_test.csv`, `data/processed/y_test.csv`
   * **Output Files:** `reports/metrics_comparison.csv`, `reports/figures/confusion_matrix.png`, `reports/figures/roc_curve.png`

6. **Phase 6: SHAP Explainability Analysis**
   * **Goal:** Extract TreeSHAP values for the test partition; produce global summary/bar plots and local single-patient waterfall explanation charts.
   * **Input:** `models/best_random_forest.joblib`, `data/processed/X_test.csv`
   * **Output Files:** `models/shap_explainer.joblib`, `reports/figures/shap_summary_plot.png`, `reports/figures/shap_bar_plot.png`

7. **Phase 7: Streamlit Application Development**
   * **Goal:** Build a modular, interactive user interface supporting single-patient inference, risk tier classification, and dynamic local SHAP waterfall diagrams.
   * **Input:** `models/best_random_forest.joblib`, `models/preprocessor.joblib`, `models/shap_explainer.joblib`
   * **Output Files:** `app/app.py`, `app/utils.py`, `app/components.py`

8. **Phase 8: Production Deployment**
   * **Goal:** Connect GitHub repository with Streamlit Community Cloud and verify end-to-end cloud accessibility.
   * **Input:** GitHub Repository (`main` branch), `requirements.txt`
   * **Output Files:** Production URL, `reports/deployment_checklist.md`

---

## CardioLens Web Application

The primary presentation layer is **CardioLens**, a responsive single-page web application powered by a framework-independent Python engine (`engine/`) and a high-performance FastAPI backend (`api/`). The front end (`web/`) is built entirely with vanilla HTML, CSS, and ES-module JavaScript—with zero third-party framework dependencies, zero CDN requests, and inline SVG graphics.

Key features include:
* **Top Navigation:** Responsive header with route links (`#/check`, `#/how`, `#/details`, `#/about`), mobile drawer toggle, and persistent medical disclaimer banner.
* **Three-Step Assessment Wizard:** Interactive input form organized into three accessible segments ("About you", "Report, part 1", "Report, part 2") with real-time client-side validation and empirical range indicators.
* **Result View:** Dynamic risk visualization featuring an animated circular score ring, calibrated risk tier badges (Lower Risk < 0.30, Moderate Risk 0.30 - 0.50, Higher Risk >= 0.50), a visual risk meter bar, and ranked top contributors from local TreeSHAP explanations.
* **Interactive What-If Sensitivity:** Real-time counterfactual exploration allowing users to adjust continuous parameters via an interactive slider and observe immediate score updates.
* **Printable Clinical Summary:** Browser print media stylesheet (`web/css/print.css`) that formats patient inputs, risk tier, and key contributing factors into a clean, single-page printable report while hiding navigation bars and buttons.
* **Model Details Page:** Transparent technical documentation across four horizontal sub-tabs: *Data* (deduplication audit and class balance), *Performance* (held-out metrics, 95% bootstrap intervals, CV benchmarks, operating-point confusion matrix), *Explanations* (top 5 grouped factors and global SHAP summary figures), and *Limitations*.

### Legacy Fallback Application
The original Streamlit application located in `app/` is maintained as a legacy fallback. It provides multi-page exploration and local inference using Streamlit's native component library.

---

## Repository Directory Structure

```text
heart-disease-prediction/
├── api/                     # FastAPI REST API endpoints and schema serialization
├── engine/                  # Framework-independent pure Python core engine and text
├── web/                     # CardioLens front end (plain HTML, CSS, vanilla ES-module JS)
│   ├── css/                 # Design tokens (tokens.css), base styles, components, and print styles
│   ├── js/                  # Routing, state management, UI components, and page views
│   ├── assets/              # SVG logos and favicon
│   └── fonts/               # Self-hosted Inter font and SIL Open Font License
├── app/                     # Multi-page Streamlit web application (legacy fallback)
├── src/                     # Production machine learning pipeline (cleaning, training, SHAP)
├── models/                  # Serialized scikit-learn models and training history
├── data/
│   ├── raw/                 # Original, untouched Kaggle heart.csv
│   └── processed/           # Deduplicated, cleaned, and split datasets (X_train, X_test, etc.)
├── notebooks/               # Interactive EDA and modeling experiments
├── tests/                   # Automated pytest suite (unit, API, and Playwright e2e verification)
├── reports/                 # Evaluation summaries, audit logs, and figure plots
│   └── figures/             # Exported PNG plots (confusion matrix, ROC curve, SHAP charts)
├── requirements.txt         # Minimal backend production dependencies
├── requirements-dev.txt     # Development, testing, and research dependencies
└── README.md                # Comprehensive project documentation
```

---

## Project Schedule (Gantt Chart)

```mermaid
gantt
    dateFormat  YYYY-MM-DD
    axisFormat  %d %b
    title       Early Heart Disease Prediction Project Schedule

    section Planning
    A - README and project setup                  :crit, active, 2026-11-03, 3d

    section Data
    B - Data cleaning and deduplication          :crit, 2026-11-05, 1d
    C - EDA                                      :crit, 2026-11-06, 1d
    D - Preprocessing and train/test split       :crit, 2026-11-06, 1d

    section Modeling
    E - Baseline and Random Forest training with tuning :crit, 2026-11-07, 2d
    F - Model evaluation                         :crit, 2026-11-08, 2d
    G - SHAP explainability analysis             :crit, 2026-11-09, 2d

    section Application
    H - Streamlit app development                :crit, 2026-11-11, 2d
    I - Deployment on Streamlit Community Cloud  :crit, 2026-11-13, 1d

    section Finalization
    J - Testing and bug fixes                    :crit, 2026-11-13, 2d
    K - Report and PPT                           :2026-11-11, 5d
    L - Buffer day                               :crit, milestone, 2026-11-16, 0d
```

---

## PERT Analysis

Project Evaluation and Review Technique (PERT) manages task variability and quantifies project duration confidence.

### Part 1: PERT Activity Estimates Table

$$\mu = T_E = \frac{O + 4M + P}{6}, \quad \sigma^2 = \left(\frac{P - O}{6}\right)^2$$

| Task ID | Task Description | Predecessors | Optimistic $O$ (days) | Most Likely $M$ (days) | Pessimistic $P$ (days) | Expected Time $T_E$ (days) | Variance $\sigma^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | README and project setup | None | 1.0 | 2.0 | 3.0 | **2.00** | $\frac{4}{36} \approx 0.1111$ |
| **B** | Data cleaning and deduplication | A | 0.5 | 1.0 | 1.5 | **1.00** | $\frac{1}{36} \approx 0.0278$ |
| **C** | Exploratory Data Analysis (EDA) | B | 0.5 | 1.0 | 1.5 | **1.00** | $\frac{1}{36} \approx 0.0278$ |
| **D** | Preprocessing and train/test split | B | 0.5 | 1.0 | 1.5 | **1.00** | $\frac{1}{36} \approx 0.0278$ |
| **E** | Baseline & Random Forest training with tuning | C, D | 1.0 | 1.5 | 2.0 | **1.50** | $\frac{1}{36} \approx 0.0278$ |
| **F** | Model evaluation | E | 0.5 | 1.0 | 1.5 | **1.00** | $\frac{1}{36} \approx 0.0278$ |
| **G** | SHAP explainability analysis | F | 1.0 | 1.5 | 2.0 | **1.50** | $\frac{1}{36} \approx 0.0278$ |
| **H** | Streamlit app development | G | 1.0 | 2.0 | 3.0 | **2.00** | $\frac{4}{36} \approx 0.1111$ |
| **I** | Deployment on Streamlit Community Cloud | H | 0.5 | 1.0 | 1.5 | **1.00** | $\frac{1}{36} \approx 0.0278$ |
| **J** | Testing and bug fixes | I | 1.0 | 1.5 | 2.0 | **1.50** | $\frac{1}{36} \approx 0.0278$ |
| **K** | Report and presentation preparation | G | 1.0 | 2.0 | 3.0 | **2.00** | $\frac{4}{36} \approx 0.1111$ |
| **L** | Final review & buffer day | J, K | 1.0 | 1.0 | 1.0 | **1.00** | $\frac{0}{36} = 0.0000$ |

---

### Part 2: PERT Network Diagram

```mermaid
flowchart LR
    classDef crit fill:#ffebee,stroke:#d32f2f,stroke-width:2.5px,color:#b71c1c;
    classDef noncrit fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1.5px,color:#424242;

    A["<b>A: README & Setup</b><br>TE = 2.0d"]:::crit
    B["<b>B: Data Cleaning</b><br>TE = 1.0d"]:::crit
    C["<b>C: EDA</b><br>TE = 1.0d"]:::crit
    D["<b>D: Preprocessing & Split</b><br>TE = 1.0d"]:::crit
    E["<b>E: Model Training & Tuning</b><br>TE = 1.5d"]:::crit
    F["<b>F: Model Evaluation</b><br>TE = 1.0d"]:::crit
    G["<b>G: SHAP Analysis</b><br>TE = 1.5d"]:::crit
    H["<b>H: Streamlit App</b><br>TE = 2.0d"]:::crit
    I["<b>I: Deployment</b><br>TE = 1.0d"]:::crit
    J["<b>J: Testing & Bug Fixes</b><br>TE = 1.5d"]:::crit
    K["<b>K: Report & PPT</b><br>TE = 2.0d"]:::noncrit
    L["<b>L: Buffer Day</b><br>TE = 1.0d"]:::crit

    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    G -.-> K
    J --> L
    K -.-> L

    linkStyle 0,1,2,3,4,5,6,7,8,9,11 stroke:#d32f2f,stroke-width:2.5px;
    linkStyle 10,12 stroke:#9e9e9e,stroke-width:1.5px,stroke-dasharray: 4 4;
```

> **Diagram Legend:**  
> *  **Solid red nodes and arrows:** Critical path (slack = 0)  
> *  **Grey node with dashed arrows:** non-critical task (slack > 0)

---

### Part 3: Critical Path Analysis

Forward and backward passes evaluate schedule float across the workflow:

| Task ID | Task Description | Expected $T_E$ | Earliest Start (ES) | Earliest Finish (EF) | Latest Start (LS) | Latest Finish (LF) | Total Slack ($LS - ES$) | Critical Path? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | README and project setup | 2.00 | 0.00 | 2.00 | 0.00 | 2.00 | **0.00** | **Yes** |
| **B** | Data cleaning and deduplication | 1.00 | 2.00 | 3.00 | 2.00 | 3.00 | **0.00** | **Yes** |
| **C** | Exploratory Data Analysis | 1.00 | 3.00 | 4.00 | 3.00 | 4.00 | **0.00** | **Yes** (tied) |
| **D** | Preprocessing and train/test split | 1.00 | 3.00 | 4.00 | 3.00 | 4.00 | **0.00** | **Yes** (tied) |
| **E** | Model training and tuning | 1.50 | 4.00 | 5.50 | 4.00 | 5.50 | **0.00** | **Yes** |
| **F** | Model evaluation | 1.00 | 5.50 | 6.50 | 5.50 | 6.50 | **0.00** | **Yes** |
| **G** | SHAP explainability analysis | 1.50 | 6.50 | 8.00 | 6.50 | 8.00 | **0.00** | **Yes** |
| **H** | Streamlit app development | 2.00 | 8.00 | 10.00 | 8.00 | 10.00 | **0.00** | **Yes** |
| **I** | Deployment on Streamlit Cloud | 1.00 | 10.00 | 11.00 | 10.00 | 11.00 | **0.00** | **Yes** |
| **J** | Testing and bug fixes | 1.50 | 11.00 | 12.50 | 11.00 | 12.50 | **0.00** | **Yes** |
| **K** | Report and PPT preparation | 2.00 | 8.00 | 10.00 | 10.50 | 12.50 | **2.50** | No |
| **L** | Buffer day | 1.00 | 12.50 | 13.50 | 12.50 | 13.50 | **0.00** | **Yes** |

#### Summary & Statistical Parameters
* **Critical Path Sequence:** $\text{A} \rightarrow \text{B} \rightarrow \{\text{C}, \text{D}\} \rightarrow \text{E} \rightarrow \text{F} \rightarrow \text{G} \rightarrow \text{H} \rightarrow \text{I} \rightarrow \text{J} \rightarrow \text{L}$
* **Parallel Criticality Tie:** Tasks **C** and **D** run in parallel and share identical durations ($T_E = 1.00\text{ day}$) and dependencies; both share zero float and tie for criticality.
* **Non-Critical Path Slack:** Task **K** (Report and PPT preparation) operates in parallel with the application pipeline (**H**, **I**, and **J**). Because the application chain requires $2.0 + 1.0 + 1.5 = 4.5\text{ days}$ versus $2.0\text{ days}$ for Task **K**, Task **K** possesses **2.50 days of total slack** ($\text{LS} = 10.50$, $\text{ES} = 8.00$).
* **Total Expected Project Duration ($T_{\text{proj}}$):**
  $$T_{\text{proj}} = 2.0 + 1.0 + 1.0 + 1.5 + 1.0 + 1.5 + 2.0 + 1.0 + 1.5 + 1.0 = \mathbf{13.50\text{ days}}$$
* **Critical Path Total Variance ($\sigma^2_{\text{CP}}$):**
  $$\sigma^2_{\text{CP}} = \sigma^2_A + \sigma^2_B + \sigma^2_{C/D} + \sigma^2_E + \sigma^2_F + \sigma^2_G + \sigma^2_H + \sigma^2_I + \sigma^2_J + \sigma^2_L$$
  $$\sigma^2_{\text{CP}} = \frac{4 + 1 + 1 + 1 + 1 + 1 + 4 + 1 + 1 + 0}{36} = \frac{15}{36} \approx \mathbf{0.417}$$
* **Project Standard Deviation ($\sigma_{\text{CP}}$):**
  $$\sigma_{\text{CP}} = \sqrt{0.4167} \approx \mathbf{0.65\text{ days}}$$

---

## Contributing

We welcome team collaboration! Please consult [CONTRIBUTING.md](CONTRIBUTING.md) for branch guidelines, notebook hygiene, and ownership policies.
All feature work must be performed on dedicated branches (`feature/<yourname>-<task>`) and merged into `main` exclusively through a reviewed Pull Request.
Direct commits to `main` are strictly disallowed to prevent merge conflicts and preserve deployment stability.

---

## Setup and Installation Guide

### Prerequisites
* Python 3.11 or higher
* Git command line utility

### 1. Clone the Repository
```bash
git clone https://github.com/[Your-Username]/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create and Activate Virtual Environment
* **Windows (Command Prompt / PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
* **To run the Streamlit application:**
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
* **To work on Jupyter notebooks and research:**
  ```bash
  pip install -r requirements-dev.txt
  ```

### 4. Launch the Web Application

**Primary Application (CardioLens):**
```bash
uvicorn api.main:app --port 8600
```
Open your web browser and navigate to `http://localhost:8600`.

**Legacy Fallback Application (Streamlit):**
```bash
streamlit run app/app.py
```
Open your web browser and navigate to `http://localhost:8501`.

### 5. Run the Automated Tests

**Unit and API Tests:**
```bash
python -m pytest -q
```

**End-to-End Browser Tests (Playwright Chromium):**
```bash
# Requires Playwright Chromium installation:
playwright install chromium

# Run all browser tests across all viewport widths:
python -m pytest -m e2e -q
```

---

## Deployment

CardioLens is configured for production hosting on Render (native Python 3 runtime on the free tier with automated build and health checking). Complete step-by-step instructions, blueprint specifications, runtime cold-start characteristics, and verification checklists are documented in [DEPLOYMENT.md](DEPLOYMENT.md).

For cloud deployment using the Render Blueprint:
1. Ensure all changes are committed and pushed to GitHub on branch `main`.
2. Connect the repository in Render using the Blueprint configured in `render.yaml` or as a new Web Service using `requirements-api.txt`.
3. Set the build command to `pip install -r requirements-api.txt` and start command to `uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1`.
4. Set environment variable `PYTHON_VERSION=3.11.9` and health check path to `/api/health`.

The repository also retains the legacy Streamlit interface in `app/app.py` as an optional fallback for hosting on Streamlit Community Cloud.

Live app URL: [to be added after deployment]

---

## Project Execution Checklist

- [x] **Phase 1:** Data loading, schema validation, and deduplication audit
- [ ] **Phase 2:** Exploratory data analysis & statistical correlation reporting
- [x] **Phase 3:** Feature encoding, scaling, and stratified train/test split
- [x] **Phase 4:** Baseline Logistic Regression & tuned Random Forest training
- [x] **Phase 5:** Model evaluation with recall, ROC-AUC, and confusion matrix benchmarking
- [x] **Phase 6:** Global and local SHAP explainability visualizations
- [x] **Phase 7:** Streamlit interactive dashboard development and validation
- [ ] **Phase 8:** Streamlit Community Cloud production deployment & documentation

---

## Experimental Results & Evaluation

*(All fields will be updated upon completion of Phase 1, Phase 4, and Phase 5 experiments).*

### Data Audit Summary
* Raw record count: `1025`
* Duplicate rows detected: `723` (70.54%)
* Verified unique patient records: `302` (29.46%)
* Disease present (disease = 1, i.e. target = 0): 45.70% (n=138); disease absent: 54.30% (n=164)

### Model Performance Comparison

#### Table 1: Cross-validation on training set (5-fold, mean ± std)

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.9146 ± 0.0291 | 0.7909 ± 0.0680 | 0.9160 ± 0.0829 | 0.8446 ± 0.0466 | 0.8669 ± 0.0413 |
| **Random Forest (Tuned Candidate)** | 0.9212 ± 0.0253 | 0.7909 ± 0.0979 | 0.8724 ± 0.0712 | 0.8226 ± 0.0491 | 0.8462 ± 0.0393 |

#### Table 2: Held-out test set (n = 61, threshold 0.5)

| Model | Accuracy [95% CI] | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] | ROC-AUC [95% CI] | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.8361 [0.7377, 0.9180] | 0.8750 [0.7273, 1.0000] | 0.7500 [0.5861, 0.9000] | 0.8077 [0.6789, 0.9123] | 0.9091 [0.8319, 0.9729] | 0.1217 |
| **Random Forest (Tuned Candidate)** | 0.8033 [0.7049, 0.9016] | 0.8077 [0.6538, 0.9501] | 0.7500 [0.5861, 0.9000] | 0.7778 [0.6500, 0.8852] | 0.9026 [0.8200, 0.9662] | 0.1372 |

**Operating point:** A recall-oriented decision threshold of 0.30 was selected based exclusively on training set out-of-fold cross-validation predictions (the highest threshold maintaining out-of-fold recall ≥ 0.90). As a secondary test result, applying threshold 0.30 to the test set yielded:
* **Logistic Regression:** Recall = 0.8929 (25/28), Precision = 0.7353, Accuracy = 0.8033, F1 = 0.8065 (TN=24, FP=9, FN=3, TP=25).
* **Random Forest:** Recall = 0.9643 (27/28), Precision = 0.6750, Accuracy = 0.7705, F1 = 0.7941 (TN=20, FP=13, FN=1, TP=27).

**Model comparison note:** The bootstrap paired difference between Random Forest and Logistic Regression (RF minus LR) on the held-out test set shows a mean difference in ROC-AUC of -0.0068 (95% CI: [-0.0379, +0.0249]) and a mean difference in Recall of +0.0000 (95% CI: [+0.0000, +0.0000]). Because both 95% confidence intervals include zero, the data does not show a clear difference between the two models. Random Forest is used as the project's main model by design, and Logistic Regression is the baseline.

### Explainability Findings

* **Top 5 Grouped Features (Mean |SHAP|):**
  1. `thal`: 0.107299
  2. `cp`: 0.079776
  3. `ca`: 0.078887
  4. `exang`: 0.041587
  5. `oldpeak`: 0.038433
* **Clinical Direction Checks:** Spearman rank correlation confirmed that higher maximum heart rate reduces predicted disease risk (`num__thalach`: r = -0.9628, p = 3.1189e-35), while greater ST depression (`num__oldpeak`: r = +0.8518, p = 3.2750e-18) and exercise-induced angina (`bin__exang`: r = +0.8231, p = 3.9879e-16) both increase predicted disease risk.
* SHAP values describe how the model uses each feature; they do not show causes.
* SHAP values were computed on 61 test patients, so rankings beyond the top few are noisy.
* Value codes for cp, slope and thal are shown as raw codes without clinical interpretation.
* At threshold 0.5, both the baseline Logistic Regression and tuned Random Forest models missed the exact same test patients (test row indices 18, 19, 27, 31, 37, 42, and 51).
* The mathematical additivity check passed with a maximum absolute error of 1.332268e-15 (well within the 1e-4 threshold).

### Evaluation Artifacts
![Test Set Confusion Matrix](reports/figures/confusion_matrix.png)
![ROC Curve Comparison](reports/figures/roc_curve.png)
![SHAP Global Summary Beeswarm Plot](reports/figures/shap_summary_plot.png)
![SHAP 22-Column Feature Importance Bar Plot](reports/figures/shap_bar_plot.png)
![SHAP 13-Variable Grouped Clinical Importance Bar Plot](reports/figures/shap_bar_grouped.png)

#### Local Patient SHAP Waterfall Explanations
![High-Risk True Positive Waterfall](reports/figures/shap_waterfall_high_risk.png)
*High-Risk True Positive: Test Row Index 41, True Label = 1, Model score = 0.9312*

![Low-Risk True Negative Waterfall](reports/figures/shap_waterfall_low_risk.png)
*Low-Risk True Negative: Test Row Index 6, True Label = 0, Model score = 0.1011*

![Missed-Case False Negative Waterfall](reports/figures/shap_waterfall_missed_case.png)
*Missed-Case False Negative: Test Row Index 18, True Label = 1, Model score = 0.4630*

---

## Limitations

1. **Small Unique Sample Size:** After purging duplicate patient records, the dataset comprises only approximately 300 unique individuals, limiting statistical power and high-dimensional generalization.
2. **Single-Source Regional Bias:** Data collection originates from historical clinical cohorts that do not adequately reflect modern heterogeneous demographic, socioeconomic, or regional populations.
3. **Absence of Clinical Validation:** Predictions are produced solely as an academic machine learning benchmark and have not been validated in a prospective clinical trial.
4. Value codes for cp, slope and thal: labels inferred from value counts and disease rates (see reports/category_label_evidence.md); the dataset provider does not document the encoding.

---

## Future Scope

* **External Cohort Validation:** Ingest and benchmark models against modern open medical cohorts (e.g., UK Biobank, PhysioNet, MIMIC-IV).
* **Deep Tabular Architectures:** Benchmark Random Forest against gradient boosting ensembles (XGBoost, LightGBM, CatBoost) and TabNet architectures.
* **Calibrated Uncertainty Estimation:** Implement conformal prediction intervals to provide reliable prediction confidence bounds alongside point probabilities.
* **FHIR/EHR Integration:** Design compliant HL7/FHIR microservice endpoints to interface risk prediction with hospital information management systems.

---

## References

1. **Kaggle Dataset Source:** [Heart Disease Dataset (Cleveland & Aggregated)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
2. **Clinical Origin:** Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. American Journal of Cardiology, 64(5), 304–310. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
3. **Explainable AI (SHAP):** Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 4765–4774.
4. **Machine Learning Framework:** Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. **PERT Methodology:** Malcolm, D. G., Roseboom, J. H., Clark, C. E., & Fazar, W. (1959). *Application of a Technique for Research and Development Program Evaluation*. Operations Research, 7(5), 646–669.

---

### Disclaimer
*This software and associated documentation are developed exclusively for academic and educational evaluation in partial fulfillment of the B.Tech CSE (Machine Learning and Data Science) curriculum at Amity University Madhya Pradesh. It is not an FDA-approved medical diagnostic device and must not be used to direct or determine patient diagnosis, medical management, or clinical interventions.*

