# data/__init__.py
"""
Data generation and handling modules for SmartLoan
"""

from .data_generator import LoanDataGenerator

__all__ = ['LoanDataGenerator']

# preprocessing/__init__.py
"""
Data preprocessing and feature engineering modules
"""

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.data_preprocessor import DataPreprocessor

__all__ = ['FeatureEngineer', 'DataPreprocessor']

# analysis/__init__.py
"""
Statistical analysis and exploratory data analysis modules
"""

from analysis.hypothesis_testing import HypothesisTests
from analysis.eda import LoanEDA

__all__ = ['HypothesisTests', 'LoanEDA']

# models/__init__.py
"""
Machine learning model training and evaluation modules
"""

from models.model_trainer import ModelTrainer
from models.model_evaluator import ModelEvaluator

__all__ = ['ModelTrainer', 'ModelEvaluator']

# api/__init__.py
"""
FastAPI microservice modules
"""

from api.main import app
from api.schemas import (
    LoanApplicationRequest, BatchLoanRequest, PredictionResponse,
    BatchPredictionResponse, ModelInfo, HealthResponse, ErrorResponse
)

__all__ = [
    'app', 'LoanApplicationRequest', 'BatchLoanRequest', 'PredictionResponse',
    'BatchPredictionResponse', 'ModelInfo', 'HealthResponse', 'ErrorResponse'
]

# ui/__init__.py
"""
Streamlit user interface modules
"""

from ui.streamlit_app import SmartLoanApp

__all__ = ['SmartLoanApp']

# ui/components/__init__.py
"""
UI component modules
"""

from ui.components.prediction_form import PredictionForm
from ui.components.dashboard import AnalyticsDashboard

__all__ = ['PredictionForm', 'AnalyticsDashboard']

# utils/__init__.py
"""
Utility modules and configuration
"""

from utils.config import *

__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'MODELS_DIR', 'LOGS_DIR',
    'RANDOM_STATE', 'TEST_SIZE', 'VALIDATION_SIZE',
    'TARGET_AUC', 'TARGET_PRECISION', 'TARGET_RECALL',
    'N_SAMPLES', 'DEFAULT_RISK_RATE',
    'CREDIT_SCORE_BINS', 'CREDIT_SCORE_LABELS',
    'INCOME_BINS', 'INCOME_LABELS',
    'CATEGORICAL_FEATURES', 'NUMERICAL_FEATURES'
]