
"""
Configuration settings for SmartLoan project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Model settings
TARGET_AUC = 0.88
TARGET_PRECISION = 0.85
TARGET_RECALL = 0.80

# Data generation settings
N_SAMPLES = 10000
DEFAULT_RISK_RATE = 0.15  # 15% default rate

# Feature engineering settings
CREDIT_SCORE_BINS = [300, 580, 670, 740, 850]
CREDIT_SCORE_LABELS = ['Poor', 'Fair', 'Good', 'Excellent']

INCOME_BINS = [0, 30000, 60000, 100000, float('inf')]
INCOME_LABELS = ['Low', 'Medium', 'High', 'Very High']

# Model file paths
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_model.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"

# API settings
API_HOST = "127.0.0.1"
API_PORT = 8000
API_TIMEOUT = 30

# Streamlit settings
STREAMLIT_PORT = 8501
PAGE_TITLE = "SmartLoan - Credit Risk Analyzer"
PAGE_ICON = "💰"

# Feature columns
CATEGORICAL_FEATURES = [
    'employment_type', 'home_ownership', 'loan_purpose',
    'credit_score_tier', 'income_category'
]

NUMERICAL_FEATURES = [
    'loan_amount', 'annual_income', 'credit_score', 'employment_length',
    'debt_to_income_ratio', 'loan_to_income_ratio', 'payment_to_income_ratio'
]

# Display settings
PLOT_STYLE = 'whitegrid'
COLOR_PALETTE = 'viridis'
FIGURE_SIZE = (10, 6)