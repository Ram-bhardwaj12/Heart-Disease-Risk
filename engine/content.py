"""CardioLens centralized product copy, field definitions, and wording."""

from typing import Any

PRODUCT_NAME = "CardioLens"
TAGLINE = "Understand what is behind your heart risk score."
DISCLAIMER = "Educational project, not a medical diagnosis. Do not use this tool to make health decisions."

TIERS = {
    "lower": {
        "key": "lower",
        "label": "Lower",
        "text": "Score is below the screening threshold.",
    },
    "moderate": {
        "key": "moderate",
        "label": "Moderate",
        "text": "Score is between the screening threshold and 0.50.",
    },
    "higher": {
        "key": "higher",
        "label": "Higher",
        "text": "Score is at or above 0.50.",
    },
}

UNVERIFIED_CODES = ["cp", "restecg", "slope", "thal"]

CODE_LABELS = {
    "cp": {0: "Option 0", 1: "Option 1", 2: "Option 2", 3: "Option 3"},
    "restecg": {0: "Option 0", 1: "Option 1", 2: "Option 2"},
    "slope": {0: "Option 0", 1: "Option 1", 2: "Option 2"},
    "thal": {1: "Option 1", 2: "Option 2", 3: "Option 3"},
    "sex": {0: "Option 0", 1: "Option 1"},
    "fbs": {0: "No", 1: "Yes"},
    "exang": {0: "No", 1: "Yes"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
}

FIELD_GROUPS = {
    "basic": ["age", "sex", "trestbps", "chol", "fbs"],
    "report": ["cp", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
}

FIELD_METADATA = {
    "age": {
        "label": "Age (years)",
        "help": "Patient age in years.",
        "group": "basic",
        "type": "number",
        "unit": "years",
        "step": 1.0,
        "optional": False,
    },
    "sex": {
        "label": "Biological sex (code)",
        "help": "Dataset code for biological sex (Option 0 or Option 1).",
        "group": "basic",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "trestbps": {
        "label": "Resting blood pressure (mm Hg)",
        "help": "Resting blood pressure in mm Hg measured upon hospital admission.",
        "group": "basic",
        "type": "number",
        "unit": "mm Hg",
        "step": 1.0,
        "optional": False,
    },
    "chol": {
        "label": "Serum cholesterol (mg/dl)",
        "help": "Serum cholesterol measurement in mg/dl.",
        "group": "basic",
        "type": "number",
        "unit": "mg/dl",
        "step": 1.0,
        "optional": False,
    },
    "fbs": {
        "label": "Fasting blood sugar > 120 mg/dl",
        "help": "Indicator whether fasting blood sugar exceeds 120 mg/dl.",
        "group": "basic",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "cp": {
        "label": "Chest pain type (code)",
        "help": "Dataset code for chest pain category (Option 0 to Option 3).",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "restecg": {
        "label": "Resting electrocardiogram (code)",
        "help": "Dataset code for resting electrocardiographic result (Option 0 to Option 2).",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "thalach": {
        "label": "Maximum heart rate achieved (bpm)",
        "help": "Maximum heart rate achieved during exercise testing in beats per minute.",
        "group": "report",
        "type": "number",
        "unit": "bpm",
        "step": 1.0,
        "optional": False,
    },
    "exang": {
        "label": "Exercise-induced angina",
        "help": "Indicator whether angina was induced during exercise testing.",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "oldpeak": {
        "label": "ST depression (mm)",
        "help": "ST depression induced by exercise relative to rest in millimeters.",
        "group": "report",
        "type": "number",
        "unit": "mm",
        "step": 0.1,
        "optional": False,
    },
    "slope": {
        "label": "Peak exercise ST segment slope (code)",
        "help": "Dataset code for the slope of the peak exercise ST segment (Option 0 to Option 2).",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "ca": {
        "label": "Number of major vessels (0-3)",
        "help": "Number of major blood vessels colored by fluoroscopy (0 to 3). Optional field.",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": True,
    },
    "thal": {
        "label": "Thalassemia type (code)",
        "help": "Dataset code for thalassemia type (Option 1 to Option 3). Optional field.",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": True,
    },
}

OPTIONAL_FIELDS = ["ca", "thal"]
REQUIRED_FIELDS = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "cp",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
]


def build_summary(score: float, tier: Any, contributions: Any) -> str:
    """Build neutral, factual one-sentence summary of model score and contributions.

    Never contains causal or diagnostic language (e.g., 'you have', 'diagnosis', 'at risk').
    """
    pct = round(score * 100.0, 1)
    if isinstance(tier, dict):
        tier_label = tier.get("label", str(tier.get("key", "Moderate"))).capitalize()
    else:
        tier_label = str(tier).capitalize()

    items = contributions
    if hasattr(items, "to_dict") and callable(items.to_dict):
        items = items.to_dict(orient="records")
    elif not isinstance(items, list):
        items = list(items)

    raising = []
    lowering = []

    for item in items:
        amt = item.get("amount") if "amount" in item else item.get("shap_amount", 0.0)
        lbl = item.get("label") or item.get("feature") or item.get("key", "")
        if amt > 0:
            raising.append((lbl, amt))
        elif amt < 0:
            lowering.append((lbl, amt))

    raising.sort(key=lambda x: x[1], reverse=True)
    lowering.sort(key=lambda x: abs(x[1]), reverse=True)

    parts = []
    if raising:
        top_r = [f"{lbl} (+{amt:.4f})" for lbl, amt in raising[:2]]
        parts.append(f"{' and '.join(top_r)} raised the score")
    if lowering:
        top_l = [f"{lbl} ({amt:.4f})" for lbl, amt in lowering[:2]]
        parts.append(f"{' and '.join(top_l)} lowered the score")

    desc = "; ".join(parts) if parts else "all features had zero attribution"
    return f"Model score: {pct}% ({tier_label} tier). {desc}."
