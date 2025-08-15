"""
Feature engineering for loan default prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *

class FeatureEngineer:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def create_features(self, df):
        """Create additional features from existing data"""
        df_copy = df.copy()

        # Ensure required base features exist
        required_features = ['age', 'annual_income', 'employment_length', 'employment_type',
                           'credit_score', 'loan_amount', 'monthly_payment']

        missing_features = [f for f in required_features if f not in df_copy.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Calculate derived features if they don't exist
        if 'debt_to_income_ratio' not in df_copy.columns:
            df_copy['debt_to_income_ratio'] = df_copy['loan_amount'] / df_copy['annual_income']

        if 'loan_to_income_ratio' not in df_copy.columns:
            df_copy['loan_to_income_ratio'] = df_copy['loan_amount'] / df_copy['annual_income']

        if 'payment_to_income_ratio' not in df_copy.columns:
            df_copy['payment_to_income_ratio'] = (df_copy['monthly_payment'] * 12) / df_copy['annual_income']

        # Create categorical features if they don't exist
        if 'credit_score_tier' not in df_copy.columns:
            df_copy['credit_score_tier'] = pd.cut(
                df_copy['credit_score'],
                bins=[300, 580, 670, 740, 850],
                labels=['Poor', 'Fair', 'Good', 'Excellent'],
                include_lowest=True
            )

        if 'income_category' not in df_copy.columns:
            df_copy['income_category'] = pd.cut(
                df_copy['annual_income'],
                bins=[0, 30000, 60000, 100000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )

        # Age groups
        df_copy['age_group'] = pd.cut(
            df_copy['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['Young', 'Young_Adult', 'Middle_Age', 'Senior', 'Elder']
        )

        # Employment stability score
        df_copy['employment_stability'] = np.where(
            (df_copy['employment_type'] == 'Full-time') &
            (df_copy['employment_length'] >= 2), 1, 0
        )

        # High risk loan indicator
        df_copy['high_risk_loan'] = np.where(
            (df_copy['debt_to_income_ratio'] > 0.4) |
            (df_copy['payment_to_income_ratio'] > 0.3), 1, 0
        )

        # Credit utilization proxy (based on credit score and loan amount)
        df_copy['credit_utilization_proxy'] = (
            (850 - df_copy['credit_score']) / 550 *
            df_copy['loan_amount'] / df_copy['annual_income']
        )

        # Loan affordability score
        df_copy['affordability_score'] = (
            df_copy['annual_income'] / df_copy['monthly_payment'] / 12
        )

        # Risk tier based on multiple factors with more granular scoring
        # Credit score component (40% weight)
        credit_component = np.where(df_copy['credit_score'] >= 740, 4,
                          np.where(df_copy['credit_score'] >= 670, 3,
                          np.where(df_copy['credit_score'] >= 600, 2, 1)))

        # DTI component (30% weight)
        dti_component = np.where(df_copy['debt_to_income_ratio'] <= 0.2, 4,
                        np.where(df_copy['debt_to_income_ratio'] <= 0.3, 3,
                        np.where(df_copy['debt_to_income_ratio'] <= 0.4, 2, 1)))

        # Income stability component (20% weight)
        income_component = np.where(df_copy['annual_income'] >= 75000, 4,
                          np.where(df_copy['annual_income'] >= 50000, 3,
                          np.where(df_copy['annual_income'] >= 30000, 2, 1)))

        # Employment component (10% weight)
        emp_component = np.where(df_copy['employment_type'] == 'Full-time', 4,
                        np.where(df_copy['employment_type'] == 'Self-employed', 3,
                        np.where(df_copy['employment_type'] == 'Part-time', 2, 1)))

        # Weighted risk score
        risk_score = (credit_component * 0.4 + dti_component * 0.3 +
                     income_component * 0.2 + emp_component * 0.1)

        conditions = [
            risk_score >= 3.5,
            risk_score >= 2.5,
            risk_score >= 1.5,
        ]
        choices = ['Low_Risk', 'Medium_Risk', 'High_Risk']
        df_copy['risk_tier'] = np.select(conditions, choices, default='Very_High_Risk')

        # Additional advanced features
        # Payment burden score
        df_copy['payment_burden_score'] = (
            df_copy['payment_to_income_ratio'] * 0.6 +
            df_copy['debt_to_income_ratio'] * 0.4
        )

        # Credit utilization estimate
        df_copy['estimated_credit_utilization'] = np.clip(
            (850 - df_copy['credit_score']) / 550, 0, 1
        )

        # Financial stress indicator
        df_copy['financial_stress'] = np.where(
            (df_copy['debt_to_income_ratio'] > 0.4) &
            (df_copy['payment_to_income_ratio'] > 0.3) &
            (df_copy['credit_score'] < 650), 1, 0
        )

        return df_copy

    def prepare_preprocessor(self, df):
        """Prepare preprocessing pipeline"""

        # Define categorical and numerical features
        categorical_features = [
            'employment_type', 'home_ownership', 'loan_purpose',
            'credit_score_tier', 'income_category', 'age_group', 'risk_tier'
        ]

        numerical_features = [
            'age', 'annual_income', 'employment_length', 'credit_score',
            'loan_amount', 'interest_rate', 'loan_term', 'monthly_payment',
            'debt_to_income_ratio', 'loan_to_income_ratio', 'payment_to_income_ratio',
            'employment_stability', 'high_risk_loan', 'credit_utilization_proxy',
            'affordability_score', 'payment_burden_score', 'estimated_credit_utilization',
            'financial_stress'
        ]

        # Numerical pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        return self.preprocessor

    def fit_transform(self, df, target_column='default'):
        """Fit preprocessor and transform data"""

        # Create additional features
        df_engineered = self.create_features(df)

        # Separate features and target
        if target_column in df_engineered.columns:
            X = df_engineered.drop(columns=[target_column])
            y = df_engineered[target_column]
        else:
            X = df_engineered
            y = None

        # Prepare and fit preprocessor
        self.prepare_preprocessor(X)
        X_transformed = self.preprocessor.fit_transform(X)

        # Get feature names after transformation
        self.feature_names = self._get_feature_names()

        # Convert to DataFrame for easier handling
        X_transformed = pd.DataFrame(
            X_transformed,
            columns=self.feature_names,
            index=X.index
        )

        return X_transformed, y

    def transform(self, df, target_column='default'):
        """Transform new data using fitted preprocessor"""

        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        # Create additional features
        df_engineered = self.create_features(df)

        # Separate features and target
        if target_column in df_engineered.columns:
            X = df_engineered.drop(columns=[target_column])
            y = df_engineered[target_column]
        else:
            X = df_engineered
            y = None

        # Transform data
        X_transformed = self.preprocessor.transform(X)

        # Convert to DataFrame
        X_transformed = pd.DataFrame(
            X_transformed,
            columns=self.feature_names,
            index=X.index
        )

        return X_transformed, y

    def _get_feature_names(self):
        """Get feature names after preprocessing"""
        feature_names = []

        # Get transformers
        transformers = self.preprocessor.transformers_

        for name, transformer, features in transformers:
            if name == 'num':
                # Numerical features keep their names
                feature_names.extend(features)
            elif name == 'cat':
                # Get one-hot encoded feature names
                if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                    feature_names.extend(cat_features)
                else:
                    # Fallback for older sklearn versions
                    categories = transformer.named_steps['onehot'].categories_
                    for i, feature in enumerate(features):
                        for category in categories[i][1:]:  # Skip first category (dropped)
                            feature_names.append(f"{feature}_{category}")

        return feature_names

    def get_feature_importance_summary(self, feature_importance, top_n=20):
        """Get summary of feature importance"""

        if self.feature_names is None:
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

if __name__ == "__main__":
    # Test the feature engineering
    from data.data_generator import LoanDataGenerator

    # Generate sample data
    generator = LoanDataGenerator(n_samples=1000)
    df = generator.generate_data()

    # Apply feature engineering
    engineer = FeatureEngineer()
    X_transformed, y = engineer.fit_transform(df)

    print("Original features:", df.shape[1])
    print("Transformed features:", X_transformed.shape[1])
    print("\nFeature names:")
    print(engineer.feature_names[:10])  # Show first 10 features