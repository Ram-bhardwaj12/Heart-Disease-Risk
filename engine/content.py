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

INFERRED_CODES = ["cp", "restecg", "slope", "thal"]
UNVERIFIED_CODES = INFERRED_CODES  # Backwards compatibility alias
INFERRED_CODES_NOTE = (
    "Labels for chest pain type, resting ECG, ST slope and thallium test were "
    "inferred by matching the dataset's value counts and disease rates to the "
    "published Cleveland data; the Kaggle file does not document its encoding."
)

CODE_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        0: "Asymptomatic (no chest pain)",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "Typical angina",
    },
    "restecg": {
        0: "Left ventricular hypertrophy",
        1: "Normal",
        2: "ST-T wave abnormality",
    },
    "slope": {
        0: "Downsloping",
        1: "Flat",
        2: "Upsloping",
    },
    "thal": {
        1: "Fixed defect",
        2: "Normal",
        3: "Reversible defect",
    },
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
        "short_label": "Age",
        "help": "Patient age in years.",
        "group": "basic",
        "type": "number",
        "unit": "years",
        "step": 1.0,
        "optional": False,
    },
    "sex": {
        "label": "Biological sex",
        "short_label": "Biological sex",
        "help": "Biological sex (Female or Male).",
        "group": "basic",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "trestbps": {
        "label": "Resting blood pressure (mm Hg)",
        "short_label": "Resting blood pressure",
        "help": "Resting blood pressure in mm Hg measured upon hospital admission.",
        "group": "basic",
        "type": "number",
        "unit": "mm Hg",
        "step": 1.0,
        "optional": False,
    },
    "chol": {
        "label": "Serum cholesterol (mg/dl)",
        "short_label": "Cholesterol",
        "help": "Serum cholesterol measurement in mg/dl.",
        "group": "basic",
        "type": "number",
        "unit": "mg/dl",
        "step": 1.0,
        "optional": False,
    },
    "fbs": {
        "label": "Fasting blood sugar > 120 mg/dl",
        "short_label": "Fasting blood sugar",
        "help": "Indicator whether fasting blood sugar exceeds 120 mg/dl.",
        "group": "basic",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "cp": {
        "label": "Chest pain type",
        "short_label": "Chest pain type",
        "help": "Chest pain category (Asymptomatic, Atypical angina, Non-anginal pain, Typical angina).",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "restecg": {
        "label": "Resting ECG",
        "short_label": "Resting ECG",
        "help": "Resting electrocardiographic result (Left ventricular hypertrophy, Normal, ST-T wave abnormality).",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "thalach": {
        "label": "Maximum heart rate achieved (bpm)",
        "short_label": "Maximum heart rate",
        "help": "Maximum heart rate achieved during exercise testing in beats per minute.",
        "group": "report",
        "type": "number",
        "unit": "bpm",
        "step": 1.0,
        "optional": False,
    },
    "exang": {
        "label": "Exercise-induced angina",
        "short_label": "Exercise-induced angina",
        "help": "Indicator whether angina was induced during exercise testing.",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "oldpeak": {
        "label": "ST depression (mm)",
        "short_label": "ST depression",
        "help": "ST depression induced by exercise relative to rest in millimeters.",
        "group": "report",
        "type": "number",
        "unit": "mm",
        "step": 0.1,
        "optional": False,
    },
    "slope": {
        "label": "ST slope",
        "short_label": "ST slope",
        "help": "ST slope (Downsloping, Flat, Upsloping).",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": False,
    },
    "ca": {
        "label": "Number of major vessels (0-3)",
        "short_label": "Number of major vessels",
        "help": "Number of major blood vessels colored by fluoroscopy (0 to 3). Optional field.",
        "group": "report",
        "type": "choice",
        "unit": "",
        "step": None,
        "optional": True,
    },
    "thal": {
        "label": "Thallium stress test result",
        "short_label": "Thallium stress test result",
        "help": "Thallium stress test result (Fixed defect, Normal, Reversible defect). Optional field.",
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


def build_summary(
    score: float,
    tier: Any,
    top_raising_or_contributions: Any,
    top_lowering: Any = None,
) -> str:
    """Build neutral, factual summary of model score and contributions.

    Template:
    'The model gives a {tier label} score ({percent}%). Answers that raised the score most: {label} ({value_display}), {label} ({value_display}). Answers that lowered it most: {label} ({value_display}).'
    Omit the last sentence if nothing lowered the score. Neutral wording only.
    """
    pct = round(score * 100.0, 1)
    if isinstance(tier, dict):
        tier_label = tier.get("label", str(tier.get("key", "Moderate"))).capitalize()
    else:
        tier_label = str(tier).capitalize()

    if top_lowering is None:
        items = top_raising_or_contributions
        if hasattr(items, "to_dict") and callable(items.to_dict):
            items = items.to_dict(orient="records")
        elif not isinstance(items, list):
            items = list(items)

        r_all = [c for c in items if (c.get("amount") if "amount" in c else c.get("shap_amount", 0.0)) > 0]
        r_all.sort(
            key=lambda x: (x.get("amount") if "amount" in x else x.get("shap_amount", 0.0)),
            reverse=True,
        )
        raising = r_all[:3]

        l_all = [c for c in items if (c.get("amount") if "amount" in c else c.get("shap_amount", 0.0)) < 0]
        l_all.sort(
            key=lambda x: abs(x.get("amount") if "amount" in x else x.get("shap_amount", 0.0)),
            reverse=True,
        )
        lowering = l_all[:3]
    else:
        raising = top_raising_or_contributions
        lowering = top_lowering

    sentence = f"The model gives a {tier_label} score ({pct}%)."
    if raising:
        r_parts = [
            f"{item.get('short_label', item['label'])} ({item['value_display']})"
            for item in raising[:2]
        ]
        sentence += f" Answers that raised the score most: {', '.join(r_parts)}."
    if lowering:
        l_parts = [
            f"{item.get('short_label', item['label'])} ({item['value_display']})"
            for item in lowering[:1]
        ]
        sentence += f" Answers that lowered it most: {', '.join(l_parts)}."

    return sentence

