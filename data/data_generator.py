"""
Generate synthetic loan data for credit risk modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *

class LoanDataGenerator:
    def __init__(self, n_samples=N_SAMPLES, random_state=RANDOM_STATE):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_data(self):
        """Generate synthetic loan dataset with realistic patterns"""

        # Basic borrower demographics
        data = {}

        # Age (21-80)
        data['age'] = np.random.normal(40, 12, self.n_samples).astype(int)
        data['age'] = np.clip(data['age'], 21, 80)

        # Annual income (correlated with age)
        age_factor = (data['age'] - 21) / 59  # Normalize age
        base_income = 25000 + age_factor * 75000
        data['annual_income'] = np.random.normal(base_income, 15000)
        data['annual_income'] = np.clip(data['annual_income'], 15000, 200000)

        # Employment length (correlated with age)
        max_employment = np.maximum(0, data['age'] - 21)
        data['employment_length'] = np.random.uniform(0, max_employment)
        data['employment_length'] = np.round(data['employment_length'], 1)

        # Employment type
        employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
        employment_weights = [0.7, 0.15, 0.12, 0.03]
        data['employment_type'] = np.random.choice(
            employment_types, self.n_samples, p=employment_weights
        )

        # Home ownership
        home_ownership = ['Own', 'Rent', 'Mortgage']
        # Higher income more likely to own
        income_terciles = np.percentile(data['annual_income'], [33, 67])
        home_probs = []
        for income in data['annual_income']:
            if income <= income_terciles[0]:
                home_probs.append([0.2, 0.7, 0.1])  # Low income
            elif income <= income_terciles[1]:
                home_probs.append([0.4, 0.4, 0.2])  # Medium income
            else:
                home_probs.append([0.6, 0.2, 0.2])  # High income

        data['home_ownership'] = [
            np.random.choice(home_ownership, p=prob)
            for prob in home_probs
        ]

        # Credit score (300-850, correlated with income and employment)
        base_credit = 550
        income_boost = (data['annual_income'] - 15000) / 185000 * 200
        employment_boost = np.where(data['employment_type'] == 'Full-time', 50, 0)
        unemployment_penalty = np.where(data['employment_type'] == 'Unemployed', -100, 0)

        data['credit_score'] = (base_credit + income_boost +
                               employment_boost + unemployment_penalty +
                               np.random.normal(0, 50, self.n_samples))
        data['credit_score'] = np.clip(data['credit_score'], 300, 850).astype(int)

        # Loan amount (correlated with income)
        loan_to_income_ratio = np.random.uniform(0.1, 2.0, self.n_samples)
        data['loan_amount'] = data['annual_income'] * loan_to_income_ratio
        data['loan_amount'] = np.round(data['loan_amount'], 2)

        # Loan purpose
        loan_purposes = ['debt_consolidation', 'home_improvement', 'major_purchase',
                        'medical', 'vacation', 'car', 'business', 'other']
        purpose_weights = [0.25, 0.15, 0.12, 0.1, 0.08, 0.1, 0.1, 0.1]
        data['loan_purpose'] = np.random.choice(
            loan_purposes, self.n_samples, p=purpose_weights
        )

        # Interest rate (based on credit score and loan amount)
        credit_score_norm = (data['credit_score'] - 300) / 550
        base_rate = 15 - credit_score_norm * 10  # 15% to 5%
        risk_adjustment = np.random.normal(0, 2, self.n_samples)
        data['interest_rate'] = np.clip(base_rate + risk_adjustment, 3, 25)
        data['interest_rate'] = np.round(data['interest_rate'], 2)

        # Loan term (months)
        loan_terms = [36, 60]
        # Larger loans tend to have longer terms
        term_probs = np.where(data['loan_amount'] > 15000, 0.7, 0.3)
        data['loan_term'] = [
            np.random.choice(loan_terms, p=[1-p, p])
            for p in term_probs
        ]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Feature engineering
        df = self._engineer_features(df)

        # Generate target variable (default)
        df = self._generate_target(df)

        return df

    def _engineer_features(self, df):
        """Engineer additional features"""

        # Debt-to-income ratio
        df['debt_to_income_ratio'] = df['loan_amount'] / df['annual_income']

        # Loan-to-income ratio
        df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']

        # Monthly payment calculation
        monthly_rate = df['interest_rate'] / 100 / 12
        n_payments = df['loan_term']
        df['monthly_payment'] = (df['loan_amount'] * monthly_rate *
                                (1 + monthly_rate) ** n_payments) / \
                               ((1 + monthly_rate) ** n_payments - 1)

        # Payment-to-income ratio
        df['payment_to_income_ratio'] = (df['monthly_payment'] * 12) / df['annual_income']

        # Credit score tiers
        df['credit_score_tier'] = pd.cut(
            df['credit_score'],
            bins=CREDIT_SCORE_BINS,
            labels=CREDIT_SCORE_LABELS,
            include_lowest=True
        )

        # Income categories
        df['income_category'] = pd.cut(
            df['annual_income'],
            bins=INCOME_BINS,
            labels=INCOME_LABELS,
            include_lowest=True
        )

        return df

    def _generate_target(self, df):
        """Generate realistic default probabilities and target variable"""

        # Base default probability
        base_prob = 0.15

        # Enhanced risk factors with stronger signals
        credit_risk = np.where(df['credit_score'] < 600, 0.25,
                      np.where(df['credit_score'] < 670, 0.1, -0.1))

        income_risk = np.where(df['annual_income'] < 30000, 0.15,
                      np.where(df['annual_income'] < 50000, 0.05, -0.08))

        employment_risk = np.where(df['employment_type'] == 'Unemployed', 0.4,
                         np.where(df['employment_type'] == 'Part-time', 0.1,
                         np.where(df['employment_type'] == 'Self-employed', 0.05, -0.05)))

        dti_risk = np.where(df['debt_to_income_ratio'] > 0.5, 0.2,
                   np.where(df['debt_to_income_ratio'] > 0.4, 0.15,
                   np.where(df['debt_to_income_ratio'] > 0.3, 0.05, -0.05)))

        payment_risk = np.where(df['payment_to_income_ratio'] > 0.4, 0.25,
                       np.where(df['payment_to_income_ratio'] > 0.3, 0.1, -0.05))

        # Age risk (younger and older borrowers might be riskier)
        age_risk = np.where((df['age'] < 25) | (df['age'] > 65), 0.1, -0.02)

        # Employment length risk
        emp_length_risk = np.where(df['employment_length'] < 1, 0.1,
                         np.where(df['employment_length'] < 2, 0.05, -0.02))

        # Loan purpose risk
        purpose_risk = np.where(df['loan_purpose'].isin(['medical', 'vacation']), 0.08,
                       np.where(df['loan_purpose'] == 'debt_consolidation', 0.05, -0.02))

        # Calculate probability with interaction effects
        default_prob = (base_prob + credit_risk + income_risk + employment_risk +
                       dti_risk + payment_risk + age_risk + emp_length_risk + purpose_risk)

        # Add interaction effects for even stronger signals
        # High DTI + Low credit score = very high risk
        interaction_1 = np.where((df['debt_to_income_ratio'] > 0.4) & (df['credit_score'] < 600), 0.2, 0)

        # Unemployed + High loan amount = very high risk
        interaction_2 = np.where((df['employment_type'] == 'Unemployed') &
                                (df['loan_amount'] > df['annual_income']), 0.25, 0)

        # Low income + High payment ratio = high risk
        interaction_3 = np.where((df['annual_income'] < 30000) &
                                (df['payment_to_income_ratio'] > 0.3), 0.15, 0)

        default_prob += interaction_1 + interaction_2 + interaction_3

        # Ensure probabilities are within valid range
        default_prob = np.clip(default_prob, 0.01, 0.85)

        # Generate binary target with some randomness
        random_factor = np.random.normal(0, 0.05, len(df))
        adjusted_prob = np.clip(default_prob + random_factor, 0.01, 0.99)

        df['default'] = np.random.binomial(1, adjusted_prob)

        return df

    def save_data(self, filepath=None):
        """Generate and save data to CSV"""
        if filepath is None:
            filepath = DATA_DIR / "loan_data.csv"

        df = self.generate_data()
        df.to_csv(filepath, index=False)

        print(f"Generated {len(df)} loan records")
        print(f"Default rate: {df['default'].mean():.2%}")
        print(f"Data saved to: {filepath}")

        return df

if __name__ == "__main__":
    generator = LoanDataGenerator()
    df = generator.save_data()
    print("\nData preview:")
    print(df.head())