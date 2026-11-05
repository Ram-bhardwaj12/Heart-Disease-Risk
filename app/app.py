"""Entry point for the Heart Disease Risk Prediction Streamlit Application.

Uses st.navigation with st.Page to define clean sidebar routing.
Deployment path remains app/app.py.
"""

from pathlib import Path
import streamlit as st

app_dir = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(app_dir / "views" / "estimator.py", title="Risk Estimator", icon=":material/cardiology:", default=True),
    st.Page(app_dir / "views" / "performance.py", title="Model Performance", icon=":material/analytics:"),
    st.Page(app_dir / "views" / "explanations.py", title="Global Explanations", icon=":material/public:"),
    st.Page(app_dir / "views" / "about.py", title="About", icon=":material/info:"),
]

pg = st.navigation(pages)
pg.run()
