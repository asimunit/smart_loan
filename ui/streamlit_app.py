"""
Main Streamlit application for SmartLoan credit risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class SmartLoanApp:
    def __init__(self):
        self.api_base_url = f"http://{API_HOST}:{API_PORT}"

    def check_api_health(self):
        """Check if the API is available"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def make_prediction(self, application_data, model_name="xgboost"):
        """Make API call for prediction"""
        try:
            response = requests.post(
                f"{self.api_base_url}/predict",
                params={"model_name": model_name},
                json=application_data,
                timeout=API_TIMEOUT
            )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {str(e)}")
            return None

    def create_risk_gauge(self, probability, risk_tier):
        """Create risk probability gauge chart"""

        # Determine color based on risk
        if probability < 0.3:
            color = "green"
        elif probability < 0.6:
            color = "orange"
        else:
            color = "red"

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Risk %"},
            delta = {'reference': 15},  # Average default rate
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 40], 'color': "yellow"},
                    {'range': [40, 60], 'color': "orange"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))

        fig.update_layout(height=300)
        return fig

    def create_comparison_chart(self, user_data, avg_data):
        """Create comparison chart between user application and averages"""

        categories = ['Credit Score', 'Annual Income', 'DTI Ratio', 'Loan Amount']

        # Normalize values for comparison (0-100 scale)
        user_values = [
            (user_data['credit_score'] - 300) / 550 * 100,
            min(user_data['annual_income'] / 100000 * 100, 100),
            min((user_data['loan_amount'] / user_data['annual_income']) / 2 * 100, 100),
            min(user_data['loan_amount'] / 50000 * 100, 100)
        ]

        avg_values = [60, 50, 35, 40]  # Average benchmarks

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=user_values,
            theta=categories,
            fill='toself',
            name='Your Application',
            line_color='blue'
        ))

        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=categories,
            fill='toself',
            name='Market Average',
            line_color='red',
            opacity=0.6
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Application Profile Comparison",
            height=400
        )

        return fig

    def render_prediction_form(self):
        """Render the loan application form"""

        st.subheader("📋 Loan Application Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Personal Information")
            age = st.slider("Age", 18, 80, 35)
            annual_income = st.number_input("Annual Income ($)", 15000, 500000, 50000, step=5000)
            employment_length = st.slider("Employment Length (years)", 0.0, 50.0, 5.0, 0.5)
            employment_type = st.selectbox("Employment Type",
                                         ["Full-time", "Part-time", "Self-employed", "Unemployed"])
            home_ownership = st.selectbox("Home Ownership", ["Own", "Rent", "Mortgage"])

        with col2:
            st.markdown("#### Loan Information")
            loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, step=1000)
            loan_purpose = st.selectbox("Loan Purpose", [
                "debt_consolidation", "home_improvement", "major_purchase",
                "medical", "vacation", "car", "business", "other"
            ])
            credit_score = st.slider("Credit Score", 300, 850, 650)
            interest_rate = st.slider("Interest Rate (%)", 3.0, 25.0, 12.0, 0.1)
            loan_term = st.selectbox("Loan Term (months)", [36, 60])

        # Model selection
        st.markdown("#### Model Selection")
        model_choice = st.radio("Choose Prediction Model",
                               ["XGBoost (Recommended)", "Logistic Regression"],
                               horizontal=True)
        model_name = "xgboost" if "XGBoost" in model_choice else "logistic_regression"

        # Create application data
        application_data = {
            "age": age,
            "annual_income": annual_income,
            "employment_length": employment_length,
            "employment_type": employment_type,
            "home_ownership": home_ownership,
            "loan_amount": loan_amount,
            "loan_purpose": loan_purpose,
            "interest_rate": interest_rate,
            "loan_term": loan_term,
            "credit_score": credit_score
        }

        return application_data, model_name

    def render_prediction_results(self, prediction_result, application_data):
        """Render prediction results"""

        st.subheader("🎯 Prediction Results")

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        # Risk probability
        risk_prob = prediction_result['default_probability']
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{risk_prob:.1%}</h3>
                <p>Default Probability</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk tier
        risk_tier = prediction_result['risk_tier']
        risk_class = "risk-low" if "Low" in risk_tier else ("risk-medium" if "Medium" in risk_tier else "risk-high")
        with col2:
            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <h3>{risk_tier}</h3>
                <p>Risk Classification</p>
            </div>
            """, unsafe_allow_html=True)

        # Prediction
        prediction = "APPROVE" if not prediction_result['default_prediction'] else "DECLINE"
        pred_class = "risk-low" if prediction == "APPROVE" else "risk-high"
        with col3:
            st.markdown(f"""
            <div class="metric-card {pred_class}">
                <h3>{prediction}</h3>
                <p>Recommendation</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence
        confidence = prediction_result['confidence_score']
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{confidence:.1%}</h3>
                <p>Model Confidence</p>
            </div>
            """, unsafe_allow_html=True)

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Risk gauge
            gauge_fig = self.create_risk_gauge(risk_prob, risk_tier)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col2:
            # Comparison radar chart
            comparison_fig = self.create_comparison_chart(application_data, {})
            st.plotly_chart(comparison_fig, use_container_width=True)

    def render_analytics_dashboard(self):
        """Render analytics dashboard"""

        st.subheader("📊 Credit Risk Analytics Dashboard")

        # Load sample data for demonstration
        try:
            from data.data_generator import LoanDataGenerator
            generator = LoanDataGenerator(n_samples=1000)
            df = generator.generate_data()

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Applications", len(df))
            with col2:
                st.metric("Default Rate", f"{df['default'].mean():.1%}")
            with col3:
                st.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")
            with col4:
                st.metric("Avg Loan Amount", f"${df['loan_amount'].mean():,.0f}")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                # Default rate by employment type
                default_by_emp = df.groupby('employment_type')['default'].agg(['mean', 'count']).reset_index()
                fig = px.bar(default_by_emp, x='employment_type', y='mean',
                           title="Default Rate by Employment Type")
                fig.update_yaxis(title="Default Rate")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Credit score distribution
                fig = px.histogram(df, x='credit_score', color='default',
                                 title="Credit Score Distribution by Default Status",
                                 nbins=30)
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                # Income vs DTI scatter
                fig = px.scatter(df, x='annual_income', y='debt_to_income_ratio',
                               color='default', title="Income vs Debt-to-Income Ratio")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Loan purpose distribution
                purpose_counts = df['loan_purpose'].value_counts()
                fig = px.pie(values=purpose_counts.values, names=purpose_counts.index,
                           title="Loan Purpose Distribution")
                st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.warning("Analytics data not available. Please ensure the data module is accessible.")

    def run(self):
        """Main application runner"""

        # Header
        st.markdown('<h1 class="main-header">💰 SmartLoan Credit Risk Analyzer</h1>',
                   unsafe_allow_html=True)

        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page",
                                   ["Loan Prediction", "Analytics Dashboard", "Model Information"])

        # API health check
        if not self.check_api_health():
            st.error("⚠️ API service is not available. Please ensure the FastAPI server is running.")
            st.info(f"Expected API URL: {self.api_base_url}")
            return
        else:
            st.sidebar.success("✅ API Connected")

        if page == "Loan Prediction":
            # Prediction form
            application_data, model_name = self.render_prediction_form()

            # Predict button
            if st.button("🔮 Predict Default Risk", type="primary"):
                with st.spinner("Analyzing loan application..."):
                    prediction_result = self.make_prediction(application_data, model_name)

                    if prediction_result:
                        self.render_prediction_results(prediction_result, application_data)

        elif page == "Analytics Dashboard":
            self.render_analytics_dashboard()

        elif page == "Model Information":
            st.subheader("🤖 Model Information")

            # Get model info from API
            try:
                for model_name in ["xgboost", "logistic_regression"]:
                    response = requests.get(f"{self.api_base_url}/model_info/{model_name}")
                    if response.status_code == 200:
                        model_info = response.json()

                        st.markdown(f"### {model_info['model_type']}")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("AUC Score", f"{model_info['auc_score']:.3f}")
                        with col2:
                            st.metric("Precision", f"{model_info['precision']:.3f}")
                        with col3:
                            st.metric("Recall", f"{model_info['recall']:.3f}")
                        with col4:
                            st.metric("F1 Score", f"{model_info['f1_score']:.3f}")

                        st.markdown(f"**Features:** {model_info['feature_count']}")
                        st.markdown(f"**Training Date:** {model_info['training_date']}")
                        st.markdown("---")

            except Exception as e:
                st.error(f"Error loading model information: {str(e)}")

if __name__ == "__main__":
    app = SmartLoanApp()
    app.run()