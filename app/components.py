"""Reusable UI components and design system helpers for the Streamlit application."""

import html
import importlib.util
from pathlib import Path
import streamlit as st

_copy_path = Path(__file__).resolve().parent / "copy.py"
_spec = importlib.util.spec_from_file_location("app_ui_copy", str(_copy_path))
_copy_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_copy_mod)
COPY = _copy_mod.COPY

REPO_ROOT = Path(__file__).resolve().parent.parent
STYLES_PATH = Path(__file__).resolve().parent / "styles.css"


def inject_css() -> None:
    """Load and inject design system styles.css into the active page."""
    if STYLES_PATH.exists():
        with open(STYLES_PATH, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_disclaimer_strip() -> None:
    """Render compact one-line academic medical disclaimer at the top of the page."""
    html_strip = f"""
    <div class="disclaimer-strip">
        <span style="font-weight:700; color:#B86E00;">⚠️ Academic Project:</span>
        <span>{html.escape(COPY['DISCLAIMER'])}</span>
    </div>
    """
    st.markdown(html_strip, unsafe_allow_html=True)


def render_footer() -> None:
    """Render standardized footer with dataset details, model specification, and disclaimer."""
    html_footer = f"""
    <div class="footer">
        <div><strong>Dataset:</strong> Kaggle Heart Disease (Cleveland Clinic Foundation) | <strong>Unique Patients:</strong> 302 | <strong>Model:</strong> Tuned Random Forest (400 trees)</div>
        <div style="margin-top:4px;">{html.escape(COPY['DISCLAIMER'])}</div>
    </div>
    """
    st.markdown(html_footer, unsafe_allow_html=True)


def render_tier_pill(tier_name: str) -> None:
    """Render accessible tier pill with emoji and text combined."""
    tier_lower = tier_name.lower()
    if "low" in tier_lower:
        pill_class = "pill-low"
        symbol = "🟢"
        label = "Low Risk"
    elif "moderate" in tier_lower:
        pill_class = "pill-moderate"
        symbol = "🟡"
        label = "Moderate Risk"
    else:
        pill_class = "pill-high"
        symbol = "🔴"
        label = "High Risk"

    pill_html = f"""
    <div class="pill {pill_class}">
        <span>{symbol}</span>
        <span>{label}</span>
    </div>
    """
    st.markdown(pill_html, unsafe_allow_html=True)


def render_risk_meter(score: float, t_threshold: float, high_threshold: float = 0.50) -> None:
    """Render three-zone risk meter with pointer at current score."""
    pct = min(max(float(score) * 100.0, 0.0), 100.0)
    low_pct = max(float(t_threshold) * 100.0, 1.0)
    mod_pct = max((high_threshold - t_threshold) * 100.0, 1.0)
    high_pct = max((1.0 - high_threshold) * 100.0, 1.0)

    meter_html = f"""
    <div class="meter">
        <div class="meter-pointer-area">
            <div class="meter-pointer-label" style="left:{pct:.1f}%;">{pct:.1f}%</div>
            <div class="meter-pointer" style="left:{pct:.1f}%;"></div>
        </div>
        <div class="meter-track">
            <div class="meter-zone meter-zone-low" style="width:{low_pct:.2f}%;"></div>
            <div class="meter-zone meter-zone-moderate" style="width:{mod_pct:.2f}%;"></div>
            <div class="meter-zone meter-zone-high" style="width:{high_pct:.2f}%;"></div>
        </div>
        <div class="meter-labels">
            <span>Low Risk (0 – {t_threshold:.2f})</span>
            <span>Moderate ({t_threshold:.2f} – {high_threshold:.2f})</span>
            <span>High (≥ {high_threshold:.2f})</span>
        </div>
    </div>
    """
    st.markdown(meter_html, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    """Render clean, compact hero header without long text paragraphs."""
    hero_html = f"""
    <div class="hero">
        <h1>{html.escape(title)}</h1>
        <p>{html.escape(subtitle)}</p>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
