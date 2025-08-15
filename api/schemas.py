"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class EmploymentType(str, Enum):
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    SELF_EMPLOYED = "Self-employed"
    UNEMPLOYED = "Unemployed"


class HomeOwnership(str, Enum):
    OWN = "Own"
    RENT = "Rent"
    MORTGAGE = "Mortgage"


class LoanPurpose(str, Enum):
    DEBT_CONSOLIDATION = "debt_consolidation"
    HOME_IMPROVEMENT = "home_improvement"
    MAJOR_PURCHASE = "major_purchase"
    MEDICAL = "medical"
    VACATION = "vacation"
    CAR = "car"
    BUSINESS = "business"
    OTHER = "other"


class LoanApplicationRequest(BaseModel):
    """Single loan application request"""

    # Personal information
    age: int = Field(..., ge=18, le=100, description="Applicant's age")
    annual_income: float = Field(..., ge=0, le=500000,
                                 description="Annual income in USD")
    employment_length: float = Field(..., ge=0, le=50,
                                     description="Employment length in years")
    employment_type: EmploymentType = Field(...,
                                            description="Type of employment")
    home_ownership: HomeOwnership = Field(...,
                                          description="Home ownership status")

    # Loan details
    loan_amount: float = Field(..., ge=1000, le=100000,
                               description="Requested loan amount")
    loan_purpose: LoanPurpose = Field(..., description="Purpose of the loan")
    interest_rate: float = Field(..., ge=0, le=50,
                                 description="Interest rate percentage")
    loan_term: int = Field(..., description="Loan term in months")

    # Credit information
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")

    @validator('loan_term')
    def validate_loan_term(cls, v):
        if v not in [36, 60]:
            raise ValueError('Loan term must be 36 or 60 months')
        return v


class BatchLoanRequest(BaseModel):
    """Batch loan applications request"""
    applications: List[LoanApplicationRequest] = Field(..., min_items=1,
                                                       max_items=100)


class PredictionResponse(BaseModel):
    """Single prediction response"""
    application_id: Optional[str] = Field(None,
                                          description="Application identifier")
    default_probability: float = Field(...,
                                       description="Probability of default (0-1)")
    default_prediction: bool = Field(...,
                                     description="Binary default prediction")
    risk_tier: str = Field(..., description="Risk tier classification")
    confidence_score: float = Field(..., description="Model confidence (0-1)")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_applications: int
    high_risk_count: int
    average_default_probability: float


class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str
    model_version: str
    training_date: str
    auc_score: float
    precision: float
    recall: float
    f1_score: float
    feature_count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    api_version: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None