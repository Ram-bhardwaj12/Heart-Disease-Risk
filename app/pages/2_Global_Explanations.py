"""Global Model Explanations Page.

Displays cohort-level TreeSHAP attributions, feature importance rankings,
and beeswarm summary charts.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

import sys
app_dir = Path(__file__).resolve().parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components import render_disclaimer_banner
from utils import REPO_ROOT, load_shap_summary

st.set_page_config(page_title="Global Explanations - Heart Disease Risk", page_icon="🌐", layout="wide")

render_disclaimer_banner()

st.title("🌐 Global Model Explanations (TreeSHAP)")
st.markdown("Cohort-level feature attribution explaining overall Random Forest decision logic.")

shap_summary = load_shap_summary()

# Top 5 Grouped Features Table
st.subheader("1. Top Clinical Feature Importance (13 Grouped Variables)")
grouped_ranking = shap_summary["rankings"]["grouped_13_original_features"]
top5_grouped = grouped_ranking[:5]

top5_df = pd.DataFrame([
    {
        "Rank": item["rank"],
        "Clinical Feature": item["feature"],
        "Mean Absolute SHAP Value": f"{item['mean_abs_shap']:.6f}",
    }
    for item in top5_grouped
])
st.table(top5_df)

st.caption("SHAP values describe how the model uses each feature; they do not show causes.")
st.caption("Rankings beyond the top few are noisy (61 test patients).")
st.caption("Value codes for cp, slope and thal are shown as raw codes without clinical interpretation.")

st.markdown("---")

# Visual Diagnostic Figures
st.subheader("2. Global Feature Importance Visualizations")

fig_bar_grouped_path = REPO_ROOT / "reports" / "figures" / "shap_bar_grouped.png"
fig_bar_path = REPO_ROOT / "reports" / "figures" / "shap_bar_plot.png"
fig_summary_path = REPO_ROOT / "reports" / "figures" / "shap_summary_plot.png"

col_g1, col_g2 = st.columns(2)

with col_g1:
    if fig_bar_grouped_path.exists():
        st.image(Image.open(fig_bar_grouped_path), caption="Grouped Clinical Variable Importance (13 Variables)", use_container_width=True)
    else:
        st.warning(f"Visualization not found: {fig_bar_grouped_path.name}")

with col_g2:
    if fig_bar_path.exists():
        st.image(Image.open(fig_bar_path), caption="Individual Encoding Importance (22 Encodings)", use_container_width=True)
    else:
        st.warning(f"Visualization not found: {fig_bar_path.name}")

st.markdown("---")
st.subheader("3. Feature Impact & Directionality (Beeswarm)")

if fig_summary_path.exists():
    st.image(
        Image.open(fig_summary_path),
        caption="SHAP Summary Beeswarm: Color represents feature value (red = high, blue = low); position indicates positive or negative risk attribution.",
        use_container_width=True,
    )
else:
    st.warning(f"Visualization not found: {fig_summary_path.name}")

