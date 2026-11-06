"""Centralized user-facing strings and copy for the Heart Disease Risk application.

All text, tooltips, labels, and captions are defined here in one dictionary so
the presentation layer can be audited and edited without modifying logic.
"""

COPY = {
    # Mandatory Medical Disclaimer
    "DISCLAIMER": "Educational project, not a medical diagnosis. Do not use this tool to make health decisions.",

    # Navigation Labels
    "NAV_ESTIMATOR": "Risk Estimator",
    "NAV_PERFORMANCE": "Model Performance",
    "NAV_GLOBAL": "Global Explanations",
    "NAV_ABOUT": "About",

    # Application Header & Hero
    "APP_TITLE": "Early Heart Disease Risk Prediction",
    "ESTIMATOR_SUBTITLE": "Interactive clinical risk estimation and local feature attribution powered by Random Forest and TreeSHAP.",
    "PERFORMANCE_SUBTITLE": "Cross-validation benchmarks, held-out test cohort evaluation, and secondary operating-point metrics.",
    "GLOBAL_SUBTITLE": "Ensemble feature importance rankings, directional impact beeswarm plots, and Gini vs. SHAP comparisons.",
    "ABOUT_SUBTITLE": "Methodology, dataset deduplication audit, pipeline leakage safeguards, and academic limitations.",

    # Sidebar
    "SIDEBAR_TITLE": "Heart Disease Risk",
    "SIDEBAR_SUBTITLE": "Explainable ML Screening",
    "SIDEBAR_QUICK_START": "Quick start",
    "SIDEBAR_QUICK_START_CAPTION": "Load held-out demonstration profiles or reset:",

    # Quick Start Buttons & Tooltips
    "BTN_HIGH_SCORE": "High-score patient",
    "HELP_HIGH_SCORE": "Loads held-out test patient #41 (true label = 1, model score = 0.9312).",
    "BTN_LOW_SCORE": "Low-score patient",
    "HELP_LOW_SCORE": "Loads held-out test patient #6 (true label = 0, model score = 0.1011).",
    "BTN_BORDERLINE": "Borderline patient (missed at 0.5)",
    "HELP_BORDERLINE": "Loads held-out test patient #18 (true label = 1, model score = 0.4630; detected at 0.30 screening threshold).",
    "BTN_RESET": "Reset to typical values",
    "HELP_RESET": "Resets all 13 clinical inputs to empirical training cohort medians and modes.",

    # Tabs
    "TAB_RESULT": "Result",
    "TAB_EXPLANATION": "Explanation",
    "TAB_WHAT_IF": "What-if",
    "TAB_REPORT": "Report",

    # Result Tab
    "CARD_SCORE_TITLE": "Predicted Model Score",
    "CALIBRATION_NOTE": "Model score, not a measured probability. Calibration was not checked.",
    "POPOVER_TIERS_LABEL": "How are the tiers defined?",
    "POPOVER_TIERS_CONTENT": (
        "Risk cut-offs are configured based exclusively on training cross-validation:\n"
        "- 🟢 **Low Risk:** Score < {t_threshold:.2f} (the recall-oriented screening threshold achieving ≥90% training recall).\n"
        "- 🟡 **Moderate Risk:** Score {t_threshold:.2f} to 0.50.\n"
        "- 🔴 **High Risk:** Score ≥ 0.50.\n\n"
        "Calibration was not checked on this cohort. These bands stratify screening priority, not true empirical probabilities."
    ),
    "OUT_OF_RANGE_TITLE": "Some inputs are outside the range of the training data. The model score may be unreliable for this patient.",
    "MODEL_AGREEMENT_TITLE": "Ensemble Tree Agreement",
    "MODEL_AGREEMENT_CAPTION": "Each of the forest's trees gives its own score. A wide spread means the trees disagree. This is not a confidence interval.",

    # Explanation Tab
    "EXPLANATION_TITLE": "Patient-Specific Feature Attribution",
    "EXPLANATION_TOGGLE": "Show the 22 encoded columns",
    "SHAP_UNITS_NOTE": "Contributions are in units of the model score (0 to 1 scale): each bar shows how much that input moved the score up or down.",
    "SHAP_NON_CAUSAL_NOTE": "SHAP shows how the model uses each input, not what causes disease.",
    "EXPANDER_WATERFALL_TITLE": "Classic waterfall view",

    # What-If Tab
    "WHAT_IF_TITLE": "Sensitivity Curve (One Input Varied)",
    "WHAT_IF_FEATURE_SELECT": "Choose a continuous test or measurement to vary:",
    "WHAT_IF_SLIDER_LABEL": "Try a different value for {feature}:",
    "WHAT_IF_CAPTION": "Other inputs are held at the values currently in the sidebar. This shows how the model responds, not what causes disease.",

    # Report Tab
    "REPORT_TITLE": "Patient Assessment Summary",
    "BTN_DOWNLOAD_HTML": "Download One-Page Report (HTML)",
    "BTN_DOWNLOAD_CSV": "Download Summary (CSV)",
}

