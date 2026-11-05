"""Global Explanations View: Cohort-level TreeSHAP feature attributions."""

from pathlib import Path
from PIL import Image
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
    REPO_ROOT,
    load_shap_summary,
)

inject_css()
render_disclaimer_strip()
render_hero(COPY["NAV_GLOBAL"], COPY["GLOBAL_SUBTITLE"])

shap_summary = load_shap_summary()

# Top 5 Grouped Features Cards
st.subheader("Top Clinical Features by Global Impact (Grouped)")
st.caption("Mean absolute SHAP value across the 61-patient held-out test cohort.")

grouped_ranking = shap_summary["rankings"]["grouped_13_original_features"]
top5_grouped = grouped_ranking[:5]

card_cols = st.columns(5)
for i, item in enumerate(top5_grouped):
    with card_cols[i]:
        st.markdown(
            f"""
            <div class="card" style="text-align:center; padding:16px 8px;">
                <div style="font-size:0.75rem; color:#5B6B7F; text-transform:uppercase; font-weight:700;">Rank #{item['rank']}</div>
                <div style="font-size:1.25rem; font-weight:700; color:#0F1B2D; margin:6px 0;">{item['feature']}</div>
                <div style="font-size:0.9rem; color:#0E7C86; font-weight:600;">{item['mean_abs_shap']:.4f}</div>
                <div style="font-size:0.75rem; color:#8C9BAE;">mean |SHAP|</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("SHAP values describe how the model uses each feature; they do not show biological causes.")
st.caption("Rankings beyond the top few are noisy due to sample size (61 test patients).")
st.caption("Value codes for cp, slope, and thal are shown as raw codes without clinical interpretation.")

st.markdown("---")

# Visual Diagnostic Figures in Tabs
st.subheader("Global Explainability Visualizations")

tab_g1, tab_g2, tab_g3 = st.tabs([
    ":material/stacked_bar_chart: Grouped Clinical Features",
    ":material/bar_chart: 22 Encodings",
    ":material/scatter_plot: Summary Beeswarm",
])

fig_bar_grouped_path = REPO_ROOT / "reports" / "figures" / "shap_bar_grouped.png"
fig_bar_path = REPO_ROOT / "reports" / "figures" / "shap_bar_plot.png"
fig_summary_path = REPO_ROOT / "reports" / "figures" / "shap_summary_plot.png"

with tab_g1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if fig_bar_grouped_path.exists():
        st.image(Image.open(fig_bar_grouped_path), caption="Grouped Clinical Variable Importance (13 Variables)", use_container_width=True)
    else:
        st.warning(f"Visualization not found: {fig_bar_grouped_path.name}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_g2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if fig_bar_path.exists():
        st.image(Image.open(fig_bar_path), caption="Individual Encoding Importance (22 Encodings)", use_container_width=True)
    else:
        st.warning(f"Visualization not found: {fig_bar_path.name}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_g3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if fig_summary_path.exists():
        st.image(
            Image.open(fig_summary_path),
            caption="SHAP Summary Beeswarm: Color represents feature value (red = high, blue = low); position indicates positive or negative risk attribution.",
            use_container_width=True,
        )
    else:
        st.warning(f"Visualization not found: {fig_summary_path.name}")
    st.markdown("</div>", unsafe_allow_html=True)

render_footer()
