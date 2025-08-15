"""
Streamlit components for loan prediction form
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.config import *

class PredictionForm:
    def __init__(self):
        self.form_data = {}

    def render_personal_info_section(self):
        """Render personal information section"""
        st.markdown("### 👤 Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider(
                "Age",
                min_value=18,
                max_value=80,
                value=35,
                help="Applicant's age in years"
            )

            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=15000,
                max_value=500000,
                value=50000,
                step=5000,
                format="%d",
                help="Total annual income before taxes"
            )

            employment_length = st.slider(
                "Employment Length (years)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help="Years at current job"
            )

        with col2:
            employment_type = st.selectbox(
                "Employment Type",
                options=["Full-time", "Part-time", "Self-employed", "Unemployed"],
                index=0,
                help="Current employment status"
            )

            home_ownership = st.selectbox(
                "Home Ownership",
                options=["Own", "Rent", "Mortgage"],
                index=1,
                help="Current housing situation"
            )

            # Display income insights
            income_percentile = self._calculate_income_percentile(annual_income)
            st.info(f"💡 Your income is in the {income_percentile} percentile")

        return {
            'age': age,
            'annual_income': annual_income,
            'employment_length': employment_length,
            'employment_type': employment_type,
            'home_ownership': home_ownership
        }

    def render_loan_info_section(self):
        """Render loan information section"""
        st.markdown("### 💰 Loan Information")

        col1, col2 = st.columns(2)

        with col1:
            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=1000,
                max_value=100000,
                value=15000,
                step=1000,
                format="%d",
                help="Requested loan amount"
            )

            loan_purpose = st.selectbox(
                "Loan Purpose",
                options=[
                    "debt_consolidation",
                    "home_improvement",
                    "major_purchase",
                    "medical",
                    "vacation",
                    "car",
                    "business",
                    "other"
                ],
                format_func=lambda x: x.replace("_", " ").title(),
                help="Primary purpose for the loan"
            )

            loan_term = st.selectbox(
                "Loan Term (months)",
                options=[36, 60],
                index=0,
                help="Repayment period in months"
            )

        with col2:
            credit_score = st.slider(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650,
                help="FICO credit score"
            )

            interest_rate = st.slider(
                "Interest Rate (%)",
                min_value=3.0,
                max_value=25.0,
                value=12.0,
                step=0.1,
                help="Annual percentage rate (APR)"
            )

            # Display credit score tier
            credit_tier = self._get_credit_score_tier(credit_score)
            tier_color = self._get_tier_color(credit_tier)
            st.markdown(f"**Credit Tier:** <span style='color: {tier_color}'>{credit_tier}</span>",
                       unsafe_allow_html=True)

        return {
            'loan_amount': loan_amount,
            'loan_purpose': loan_purpose,
            'loan_term': loan_term,
            'credit_score': credit_score,
            'interest_rate': interest_rate
        }

    def render_calculated_metrics(self, personal_info, loan_info):
        """Render calculated financial metrics"""
        st.markdown("### 📊 Calculated Metrics")

        # Calculate monthly payment
        monthly_rate = loan_info['interest_rate'] / 100 / 12
        n_payments = loan_info['loan_term']
        monthly_payment = (loan_info['loan_amount'] * monthly_rate *
                          (1 + monthly_rate) ** n_payments) / \
                         ((1 + monthly_rate) ** n_payments - 1)

        # Calculate ratios
        debt_to_income = loan_info['loan_amount'] / personal_info['annual_income']
        payment_to_income = (monthly_payment * 12) / personal_info['annual_income']
        loan_to_income = loan_info['loan_amount'] / personal_info['annual_income']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Monthly Payment",
                f"${monthly_payment:,.2f}",
                help="Estimated monthly loan payment"
            )

        with col2:
            dti_status = "🟢 Good" if debt_to_income <= 0.3 else "🟡 Moderate" if debt_to_income <= 0.4 else "🔴 High"
            st.metric(
                "Debt-to-Income Ratio",
                f"{debt_to_income:.1%}",
                delta=dti_status,
                help="Loan amount as percentage of annual income"
            )

        with col3:
            pti_status = "🟢 Good" if payment_to_income <= 0.2 else "🟡 Moderate" if payment_to_income <= 0.3 else "🔴 High"
            st.metric(
                "Payment-to-Income Ratio",
                f"{payment_to_income:.1%}",
                delta=pti_status,
                help="Annual payments as percentage of income"
            )

        with col4:
            lti_status = "🟢 Good" if loan_to_income <= 1.0 else "🟡 Moderate" if loan_to_income <= 1.5 else "🔴 High"
            st.metric(
                "Loan-to-Income Ratio",
                f"{loan_to_income:.1%}",
                delta=lti_status,
                help="Total loan as percentage of annual income"
            )

        # Risk assessment visualization
        self._render_risk_indicators(debt_to_income, payment_to_income, personal_info['credit_score'])

        return {
            'monthly_payment': monthly_payment,
            'debt_to_income_ratio': debt_to_income,
            'payment_to_income_ratio': payment_to_income,
            'loan_to_income_ratio': loan_to_income
        }

    def render_model_selection(self):
        """Render model selection section"""
        st.markdown("### 🤖 Model Selection")

        col1, col2 = st.columns([2, 1])

        with col1:
            model_choice = st.radio(
                "Choose Prediction Model",
                options=["XGBoost (Recommended)", "Logistic Regression"],
                index=0,
                help="Select the machine learning model for prediction"
            )

            # Model information
            if "XGBoost" in model_choice:
                st.info("🎯 **XGBoost**: Advanced gradient boosting model with high accuracy (Target AUC: 0.88)")
            else:
                st.info("📈 **Logistic Regression**: Interpretable linear model with good baseline performance")

        with col2:
            st.markdown("**Model Features:**")
            st.markdown("- Statistical significance testing")
            st.markdown("- Feature engineering")
            st.markdown("- Hyperparameter optimization")
            st.markdown("- Cross-validation")

        model_name = "xgboost" if "XGBoost" in model_choice else "logistic_regression"
        return model_name

    def render_complete_form(self):
        """Render the complete loan application form"""
        st.markdown("## 📋 Loan Application Form")
        st.markdown("Fill out the information below to get a credit risk assessment.")

        # Personal Information
        personal_info = self.render_personal_info_section()
        st.divider()

        # Loan Information
        loan_info = self.render_loan_info_section()
        st.divider()

        # Calculated Metrics
        calculated_metrics = self.render_calculated_metrics(personal_info, loan_info)
        st.divider()

        # Model Selection
        model_name = self.render_model_selection()

        # Combine all data
        application_data = {
            **personal_info,
            **loan_info
        }

        # Calculate monthly payment and derived ratios
        monthly_rate = application_data['interest_rate'] / 100 / 12
        n_payments = application_data['loan_term']
        monthly_payment = (application_data['loan_amount'] * monthly_rate *
                          (1 + monthly_rate) ** n_payments) / \
                         ((1 + monthly_rate) ** n_payments - 1)

        # Add calculated fields
        application_data['monthly_payment'] = monthly_payment
        application_data['debt_to_income_ratio'] = calculated_metrics['debt_to_income_ratio']
        application_data['loan_to_income_ratio'] = calculated_metrics['loan_to_income_ratio']
        application_data['payment_to_income_ratio'] = calculated_metrics['payment_to_income_ratio']

        return application_data, model_name

    def _calculate_income_percentile(self, income):
        """Calculate income percentile based on typical distribution"""
        # Simplified percentile calculation based on US income distribution
        if income < 25000:
            return "20th"
        elif income < 40000:
            return "40th"
        elif income < 60000:
            return "60th"
        elif income < 85000:
            return "80th"
        else:
            return "90th+"

    def _get_credit_score_tier(self, score):
        """Get credit score tier"""
        if score >= 740:
            return "Excellent"
        elif score >= 670:
            return "Good"
        elif score >= 580:
            return "Fair"
        else:
            return "Poor"

    def _get_tier_color(self, tier):
        """Get color for credit tier"""
        colors = {
            "Excellent": "#4CAF50",
            "Good": "#8BC34A",
            "Fair": "#FF9800",
            "Poor": "#F44336"
        }
        return colors.get(tier, "#666666")

    def _render_risk_indicators(self, dti, pti, credit_score):
        """Render risk indicator visualization"""

        # Calculate overall risk score
        risk_factors = []

        if dti > 0.4:
            risk_factors.append("High DTI Ratio")
        if pti > 0.3:
            risk_factors.append("High Payment Ratio")
        if credit_score < 600:
            risk_factors.append("Low Credit Score")

        if not risk_factors:
            st.success("✅ **Low Risk Profile**: All key metrics are within acceptable ranges")
        elif len(risk_factors) == 1:
            st.warning(f"⚠️ **Moderate Risk**: {risk_factors[0]}")
        else:
            st.error(f"🔴 **High Risk**: Multiple risk factors detected: {', '.join(risk_factors)}")

        # Risk factor details
        with st.expander("📊 View Risk Factor Details"):
            risk_data = {
                'Metric': ['Debt-to-Income', 'Payment-to-Income', 'Credit Score'],
                'Your Value': [f"{dti:.1%}", f"{pti:.1%}", f"{credit_score}"],
                'Good Threshold': ['≤ 30%', '≤ 20%', '≥ 670'],
                'Status': [
                    '✅ Good' if dti <= 0.3 else '⚠️ Risk',
                    '✅ Good' if pti <= 0.2 else '⚠️ Risk',
                    '✅ Good' if credit_score >= 670 else '⚠️ Risk'
                ]
            }

            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    # Test the prediction form
    form = PredictionForm()

    st.title("Test Prediction Form")
    application_data, model_name = form.render_complete_form()

    if st.button("Show Form Data"):
        st.json(application_data)
        st.write(f"Selected model: {model_name}")