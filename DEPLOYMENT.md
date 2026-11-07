# CardioLens Deployment Guide (Render)

This guide documents the production deployment of CardioLens to Render's free tier.

- Live URL: 
- Deployment Date: 

---

## 1. Prerequisites

1. All commits are merged to the `main` branch.
2. The repository is pushed to a remote GitHub repository accessible by Render.

---

## 2. Render Deployment Steps

1. Create or log in to your account at [Render](https://render.com/).
2. From the dashboard, select **New +** and choose **Web Service** (or use Blueprint with `render.yaml`).
3. Connect your GitHub repository containing the CardioLens source code.
4. Configure service parameters:
   - **Name:** `cardiolens`
   - **Branch:** `main`
   - **Region:** Select closest region to your users (e.g., Oregon, Frankfurt).
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements-api.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Instance Type:** `Free` (512 MB RAM, 0.1 CPU)
5. Environment Variables:
   - Add variable `PYTHON_VERSION` with value `3.11.9`.
6. Advanced Settings:
   - **Health Check Path:** `/api/health`
7. Click **Create Web Service**. Render will initiate the build and launch the container.

---

## 3. Expected Runtime Behaviour on Free Tier

- **Idle Spin-Down:** The free instance automatically spins down after 15 minutes of inactivity.
- **Cold-Start Wakeup:** When spun down, the first incoming request can take approximately 50 to 60 seconds to boot the Python runtime and bind the port.
- **Background Loading:** Once the port is bound, `/api/health` responds immediately with status ok. The model and explainability engine load in a background thread and warm up.
- **Front-End Resilience:** The front-end client pings `/api/health` on load and step transitions to wake the instance early. Requests to `/api/predict` and `/api/whatif` use an extended 75-second initial timeout, display a waking-up notice after 8 seconds, and automatically retry once after 3 seconds upon receiving 502/503/504 responses or network timeouts.

---

## 4. Post-Deployment Verification Checklist

Perform these verification steps after deployment:

1. **Routing and Layout (Desktop & Mobile 390px):**
   - Check risk assessment page (`#/check`).
   - How it works page (`#/how`).
   - Model details page (`#/details`), including all 4 subtabs (Data, Performance, Explanations, Limitations).
   - About page (`#/about`).
2. **Interactive Form:**
   - Verify step 1 default age is 55 years.
   - Complete steps 1 through 3 manually.
3. **Exemplar Profiles:**
   - Test Example 1 and verify result card rendering.
   - Test Example 2 and Example 3.
4. **Result Card & What-If Analysis:**
   - Check risk tier pill, score ring percentage, and factor contribution bars.
   - Expand "Try a change" and adjust the sensitivity slider across points.
5. **Print Preview:**
   - Open browser print dialog and confirm top navigation and print buttons are hidden while result details format cleanly.

---

## 5. Rollback and Fallback Plan

- **Rollback:** In the Render dashboard, navigate to **Deploys** and select a previous successful deploy, then click **Rollback to this deploy**.
- **Streamlit Fallback:** The repository maintains the legacy Streamlit application in `app/` configured via `requirements.txt`. If needed, it can be hosted on Streamlit Community Cloud with main file `app/app.py`.

