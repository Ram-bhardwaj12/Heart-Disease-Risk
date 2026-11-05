"""Configuration constants and paths for the Heart Disease Prediction project."""

from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "heart.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Modeling Hyperparameters and Data Partitioning
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"

# Feature Subsets
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
