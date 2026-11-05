"""Reusable UI components and visual elements for the Streamlit application."""

import streamlit as st


def render_disclaimer_banner() -> None:
    """Render prominent academic medical disclaimer banner."""
    st.warning(
        "⚠️ **Disclaimer:** Educational project, not a medical diagnosis. "
        "Do not use this tool to make health decisions.",
        icon=None,
    )


def render_risk_badge(tier_name: str, badge_text: str, color_hex: str) -> None:
    """Render high-visibility clinical risk category badge.

    Always includes text and emoji (🟢 Low, 🟡 Moderate, 🔴 High)
    so information is accessible and never depends on color alone.
    """
    badge_html = f"""
    <div style="
        display: inline-block;
        padding: 8px 18px;
        border-radius: 20px;
        background-color: {color_hex}15;
        border: 2px solid {color_hex};
        color: {color_hex};
        font-weight: 700;
        font-size: 1.15rem;
        margin: 6px 0;
    ">
        {badge_text}
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)


def render_how_to_read_caption() -> None:
    """Render explanatory caption for the local explanation waterfall."""
    st.caption(
        "ℹ️ **How to read this plot:** The base value (E[f(x)]) represents the average disease "
        "risk across the training cohort. Red bars indicate patient attributes pushing predicted risk "
        "upward, while blue bars represent attributes pulling predicted risk downward. "
        "SHAP values reflect statistical contributions to this model's score, not biological causes."
    )
