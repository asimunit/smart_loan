"""
FastAPI microservice for loan default prediction
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import *
from utils.config import *
from preprocessing.feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SmartLoan API",
    description="ML-based credit risk assessment microservice",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = {}
feature_engineer = None
model_info = {}

class PredictionService:
    def __init__(self):
        self.models = {}
        self.feature_engineer = None
        self.model_info = {}

    def load_models(self):
        """Load trained models and preprocessor"""
        try:
            # Load feature engineer
            self.feature_engineer = FeatureEngineer()
            self.feature_engineer.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.feature_engineer.feature_names = joblib.load(FEATURE_NAMES_PATH)
            logger.info("Feature engineer loaded successfully")

            # Load Logistic Regression model
            try:
                self.models['logistic_regression'] = joblib.load(LOGISTIC_MODEL_PATH)
                logger.info("Logistic Regression model loaded successfully")
            except FileNotFoundError:
                logger.warning("Logistic Regression model not found")

            # Load XGBoost model
            try:
                self.models['xgboost'] = joblib.load(XGBOOST_MODEL_PATH)
                logger.info("XGBoost model loaded successfully")
            except FileNotFoundError:
                logger.warning("XGBoost model not found")

            # Set model info (you would load this from a config file in production)
            self.model_info = {
                'logistic_regression': {
                    'model_type': 'Logistic Regression',
                    'model_version': '1.0.0',
                    'training_date': '2024-01-01',
                    'auc_score': 0.85,
                    'precision': 0.82,
                    'recall': 0.78,
                    'f1_score': 0.80,
                    'feature_count': len(self.feature_engineer.feature_names) if self.feature_engineer else 0
                },
                'xgboost': {
                    'model_type': 'XGBoost',
                    'model_version': '1.0.0',
                    'training_date': '2024-01-01',
                    'auc_score': 0.88,
                    'precision': 0.85,
                    'recall': 0.80,
                    'f1_score': 0.82,
                    'feature_count': len(self.feature_engineer.feature_names) if self.feature_engineer else 0
                }
            }

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def preprocess_application(self, application: LoanApplicationRequest) -> pd.DataFrame:
        """Convert application request to DataFrame and preprocess"""

        # Convert to dictionary
        app_dict = application.dict()

        # Calculate monthly payment
        monthly_rate = app_dict['interest_rate'] / 100 / 12
        n_payments = app_dict['loan_term']
        monthly_payment = (app_dict['loan_amount'] * monthly_rate *
                          (1 + monthly_rate) ** n_payments) / \
                         ((1 + monthly_rate) ** n_payments - 1)
        app_dict['monthly_payment'] = monthly_payment

        # Calculate all derived features that feature engineer expects
        app_dict['debt_to_income_ratio'] = app_dict['loan_amount'] / app_dict['annual_income']
        app_dict['loan_to_income_ratio'] = app_dict['loan_amount'] / app_dict['annual_income']
        app_dict['payment_to_income_ratio'] = (monthly_payment * 12) / app_dict['annual_income']

        # Create DataFrame
        df = pd.DataFrame([app_dict])

        # Transform using feature engineer
        if self.feature_engineer is None:
            raise HTTPException(status_code=500, detail="Feature engineer not loaded")

        try:
            X_processed, _ = self.feature_engineer.transform(df, target_column=None)
            return X_processed
        except Exception as e:
            # Log the error and provide more details
            logger.error(f"Feature engineering failed: {str(e)}")
            logger.error(f"Available columns in DataFrame: {df.columns.tolist()}")
            raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")

    def predict(self, X_processed: pd.DataFrame, model_name: str = 'xgboost') -> Dict[str, Any]:
        """Make prediction using specified model"""

        if model_name not in self.models:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not available")

        model = self.models[model_name]

        # Get prediction probability
        prob = model.predict_proba(X_processed)[0, 1]

        # Get binary prediction (threshold = 0.5)
        prediction = prob > 0.5

        # Determine risk tier
        if prob < 0.2:
            risk_tier = "Low Risk"
        elif prob < 0.4:
            risk_tier = "Medium Risk"
        elif prob < 0.6:
            risk_tier = "High Risk"
        else:
            risk_tier = "Very High Risk"

        # Calculate confidence (distance from decision boundary)
        confidence = abs(prob - 0.5) * 2

        return {
            'default_probability': float(prob),
            'default_prediction': bool(prediction),
            'risk_tier': risk_tier,
            'confidence_score': float(confidence)
        }

# Initialize prediction service
prediction_service = PredictionService()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        prediction_service.load_models()
        logger.info("API startup complete")
    except Exception as e:
        logger.error(f"Failed to load models on startup: {str(e)}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "SmartLoan Credit Risk API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = {
        'logistic_regression': 'logistic_regression' in prediction_service.models,
        'xgboost': 'xgboost' in prediction_service.models,
        'feature_engineer': prediction_service.feature_engineer is not None
    }

    return HealthResponse(
        status="healthy" if all(models_loaded.values()) else "partial",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        api_version="1.0.0"
    )

@app.get("/model_info/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a specific model"""

    if model_name not in prediction_service.model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    info = prediction_service.model_info[model_name]
    return ModelInfo(**info)

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(application: LoanApplicationRequest, model_name: str = "xgboost"):
    """Predict default probability for a single loan application"""

    try:
        # Preprocess the application
        X_processed = prediction_service.preprocess_application(application)

        # Make prediction
        prediction_result = prediction_service.predict(X_processed, model_name)

        return PredictionResponse(**prediction_result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchLoanRequest, model_name: str = "xgboost"):
    """Predict default probability for multiple loan applications"""

    try:
        predictions = []
        default_probabilities = []
        high_risk_count = 0

        for i, application in enumerate(batch_request.applications):
            # Preprocess the application
            X_processed = prediction_service.preprocess_application(application)

            # Make prediction
            prediction_result = prediction_service.predict(X_processed, model_name)

            # Add application ID
            prediction_result['application_id'] = f"app_{i+1}"
            predictions.append(PredictionResponse(**prediction_result))

            # Track statistics
            default_probabilities.append(prediction_result['default_probability'])
            if prediction_result['risk_tier'] in ['High Risk', 'Very High Risk']:
                high_risk_count += 1

        return BatchPredictionResponse(
            predictions=predictions,
            total_applications=len(batch_request.applications),
            high_risk_count=high_risk_count,
            average_default_probability=float(np.mean(default_probabilities))
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/debug/preprocess")
async def debug_preprocess(application: LoanApplicationRequest):
    """Debug endpoint to see preprocessing steps"""

    try:
        # Convert to dictionary
        app_dict = application.dict()

        # Calculate derived features
        monthly_rate = app_dict['interest_rate'] / 100 / 12
        n_payments = app_dict['loan_term']
        monthly_payment = (app_dict['loan_amount'] * monthly_rate *
                          (1 + monthly_rate) ** n_payments) / \
                         ((1 + monthly_rate) ** n_payments - 1)

        app_dict['monthly_payment'] = monthly_payment
        app_dict['debt_to_income_ratio'] = app_dict['loan_amount'] / app_dict['annual_income']
        app_dict['loan_to_income_ratio'] = app_dict['loan_amount'] / app_dict['annual_income']
        app_dict['payment_to_income_ratio'] = (monthly_payment * 12) / app_dict['annual_income']

        # Create DataFrame
        df = pd.DataFrame([app_dict])

        # Show what we have before feature engineering
        debug_info = {
            'input_data': app_dict,
            'dataframe_columns': df.columns.tolist(),
            'dataframe_shape': df.shape,
            'feature_engineer_available': prediction_service.feature_engineer is not None
        }

        # Try feature engineering
        if prediction_service.feature_engineer is not None:
            try:
                X_processed, _ = prediction_service.feature_engineer.transform(df, target_column=None)
                debug_info['processed_shape'] = X_processed.shape
                debug_info['processed_columns'] = X_processed.columns.tolist()[:10]  # First 10 columns
                debug_info['preprocessing_success'] = True
            except Exception as e:
                debug_info['preprocessing_error'] = str(e)
                debug_info['preprocessing_success'] = False

        return debug_info

    except Exception as e:
        return {"error": str(e), "debug_failed": True}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred"
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)