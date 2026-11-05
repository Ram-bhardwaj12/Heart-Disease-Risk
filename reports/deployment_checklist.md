# Streamlit Community Cloud Deployment Checklist

## App Configuration
* **Repository:** `[Your GitHub Repository URL]`
* **Branch:** `main`
* **Main file path:** `app/app.py`
* **Python version:** `3.11` (selected under Advanced settings)
* **Dependencies:** `requirements.txt` (pinned dependencies)
* **Live App URL:** 
* **Deployment Date:** 

---

## Deployment Steps
1. Fork or push the repository to your GitHub account on branch `main`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub credentials.
3. Click **New app** (or **Deploy an app**).
4. Configure the app settings:
   * **Repository:** Select your forked or pushed repository.
   * **Branch:** `main`
   * **Main file path:** `app/app.py`
5. Click **Advanced settings**, set the **Python version** dropdown to `3.11`, and click **Save**.
6. Click **Deploy!**. Streamlit Cloud will launch the container, install pinned dependencies from `requirements.txt`, and start the app.

---

## Post-Deployment Test List
- [ ] Open every page from the sidebar navigation:
  - [ ] Main Page (`app.py` - Early Heart Disease Risk Prediction)
  - [ ] Page 1 (`1_Model_Performance.py` - Model Performance & Evaluation Benchmarks)
  - [ ] Page 2 (`2_Global_Explanations.py` - Global Model Explanations)
  - [ ] Page 3 (`3_About.py` - Project Methodology & Limitations)
- [ ] On the main page, click the three example patient buttons:
  - [ ] High-Risk True Positive (Patient #41)
  - [ ] Low-Risk True Negative (Patient #6)
  - [ ] Borderline Missed Case (Patient #18)
- [ ] Change several input sliders and selectboxes to verify real-time recalculation of the model score and SHAP waterfall plot.
- [ ] Open the live application URL on a mobile device / smartphone browser to confirm responsive layout rendering.
- [ ] Record the live app URL and deployment date above.
