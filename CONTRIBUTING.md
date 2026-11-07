# Contributing Guidelines

Welcome to the **Early Heart Disease Risk Prediction** project! This guide ensures seamless, conflict-free collaboration between team members. Please read and follow these practical rules before starting any task.

---

## 🌿 1. Branch Rules & Workflow
* **Never commit directly to `main`:** The `main` branch is our production-ready source of truth.
* **Feature Branches:** Create a descriptive branch for every new task using the convention:
  ```bash
  feature/<yourname>-<task>
  ```
  *Example:* `feature/teammate-eda`, `feature/developer-shap-plots`
* **Pull Request (PR) Reviews:** Merge code into `main` **only** via a Pull Request reviewed and approved by the Developer.

---

## 🔄 2. Daily Collaboration Routine
1. **Pull the latest changes:** Before writing any code, synchronize your local `main` branch:
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Branch out or rebase:** Create or update your task branch:
   ```bash
   git checkout -b feature/<yourname>-<task>
   ```
3. **Commit often with clear messages:** Make small, self-contained commits with informative messages (e.g., `Add correlation heatmap to reports/figures/`).
4. **Push and open a PR:** Push your branch to GitHub and submit a Pull Request using the repository's PR template.

---

## 3. Folder & File Ownership Table

To prevent merge conflicts and accidental overwrites, follow this division of ownership:

| Folder / File | Owner | Notes |
| :--- | :--- | :--- |
| `engine/`, `api/` | **Developer** | Core framework-independent engine, business logic, and FastAPI REST endpoints. |
| `tests/` | **Developer** | Automated unit, API, and Playwright end-to-end verification suites. |
| `README.md`, `CONTRIBUTING.md`, requirements files | **Developer** | Repository documentation and dependency definitions (`requirements.txt`, `requirements-dev.txt`). |
| `web/` (`css/`, `js/`, `assets/`) | **Teammate** | Front-end presentation layer, styles, views, components, and client-side assets. |
| `src/`, `models/`, `data/` | **Developer** | Core data engineering, feature transformations, trained model artifacts (read-only in Phase 2). |
| `notebooks/01_eda.ipynb` | **Teammate** | Exploratory Data Analysis and statistical visualizations. |
| `reports/` (Report & PPT) | **Teammate** | Academic project report, slide deck, and presentation assets. |

> **Ownership Rule:** UI changes must be made only in `web/`. Run both test commands before every Pull Request and paste the results into the description:
> ```bash
> python -m pytest -q
> python -m pytest -m e2e -q
> ```

---

## 📓 4. Jupyter Notebook Best Practices
* **One Person Per Notebook:** Never work concurrently on the same `.ipynb` file.
* **Clear All Outputs Before Committing:** Jupyter outputs contain binary images and metadata that create severe merge conflicts. Always click **Kernel $\rightarrow$ Restart & Clear All Outputs**, then save before running `git add`.

---

## 📦 5. Managing Dependencies
* If your work requires a new Python library, verify it works in Python 3.11.
* Add the package to `requirements.txt` inside your feature branch and document the addition in your Pull Request description.
* Avoid OS-specific packages so the app remains deployable on Streamlit Community Cloud.

---

## 🤖 6. Model Artifacts Policy
* **Only the Developer retrains or overwrites files in `models/`:** Serialized pipelines (`.joblib`) are directly tied to the deployed Streamlit application on Streamlit Community Cloud. Modifying model weights without coordination will break production.

---

## 🚫 7. What Never to Commit
* `venv/` (Virtual environments)
* `.env` or credential files
* Large temporary datasets or cache directories (`__pycache__/`, `.ipynb_checkpoints/`)

---

## 🛠️ 8. How to Resolve a Merge Conflict in 4 Simple Steps
If Git flags a conflict when updating your branch:
1. **Pull the latest `main` into your feature branch:**
   ```bash
   git checkout feature/<yourname>-<task>
   git merge origin/main
   ```
2. **Open the conflicting file(s):** Look for Git's conflict markers:
   ```text
   <<<<<<< HEAD
   (Your local changes)
   =======
   (Changes from main)
   >>>>>>> origin/main
   ```
3. **Resolve the differences:** Delete the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) and keep the correct combined code.
4. **Stage and commit the resolution:**
   ```bash
   git add <resolved-file>
   git commit -m "Resolve merge conflict with main"
   ```

---

## 💻 9. Running the Project Locally
To run and verify the project on your machine:
```bash
# 1. Activate the virtual environment
.\venv\Scripts\activate      # Windows
source venv/bin/activate       # Linux/macOS

# 2. Ensure all development and notebook dependencies are installed
pip install -r requirements-dev.txt

# 3. Launch the Streamlit application
streamlit run app/app.py
```

---

## 🎨 10. Working on the UI
* **Application Architecture:** The presentation layer is modularized under `app/`:
  - `app/styles.css`: CSS variables (`:root` design tokens for colors, typography, cards, badges, and meters).
  - `app/copy.py`: Centralized user-facing strings, microcopy, and academic medical disclaimers (`COPY` dictionary).
  - `app/components.py`: Reusable UI widgets and layout blocks (CSS injection, disclaimer strips, footers, tier badges, and risk meters).
  - `app/views/`: Dedicated page views rendered via `st.navigation` (`estimator.py`, `performance.py`, `explanations.py`, `about.py`).
  - `app/pages/`: Backward-compatible page entry points forwarding to `app/views/`.
  - `.streamlit/config.toml`: Global theme settings (`primaryColor`, background colors, toolbar modes).
* **Where to Add or Edit Copy:** All text, button labels, tooltips, and explanations must be edited **only in `app/copy.py`**. Never hardcode user-facing copy inside views or components.
* **Where to Change Colors or Tokens:** Theme colors and visual tokens must be modified in `app/styles.css` (CSS custom properties) and synchronized with `.streamlit/config.toml`.
* **Inference Logic Integrity:** Do not edit `app/utils.py` without coordinating with the Developer, as it houses production model inference, input validation, tree agreement, and SHAP pipelines.
* **Testing Requirement:** `python -m pytest -q` must pass with zero failures and zero project warnings after any change to UI views, copy, or components. Always run the test suite and paste the real output in your PR.


