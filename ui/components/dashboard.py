"""
Streamlit dashboard components for analytics and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.config import *


class AnalyticsDashboard:
    def __init__(self, df=None):
        self.df = df
        self.setup_sample_data()

    def setup_sample_data(self):
        """Setup sample data if none provided"""
        if self.df is None:
            try:
                from data.data_generator import LoanDataGenerator
                generator = LoanDataGenerator(n_samples=1000)
                self.df = generator.generate_data()
            except ImportError:
                st.error(
                    "Unable to load sample data. Please ensure data module is available.")
                self.df = pd.DataFrame()

    def render_kpi_metrics(self):
        """Render key performance indicators"""
        if self.df.empty:
            st.warning("No data available for KPIs")
            return

        st.markdown("### 📊 Key Performance Indicators")

        col1, col2, col3, col4, col5 = st.columns(5)

        # Total applications
        total_apps = len(self.df)
        with col1:
            st.metric(
                "Total Applications",
                f"{total_apps:,}",
                help="Total number of loan applications"
            )

        # Default rate
        default_rate = self.df['default'].mean()
        with col2:
            st.metric(
                "Default Rate",
                f"{default_rate:.1%}",
                delta=f"{(default_rate - 0.15):.1%}",
                help="Percentage of loans that defaulted"
            )

        # Average credit score
        avg_credit = self.df['credit_score'].mean()
        with col3:
            st.metric(
                "Avg Credit Score",
                f"{avg_credit:.0f}",
                delta=f"{(avg_credit - 650):.0f}",
                help="Average FICO credit score"
            )

        # Average loan amount
        avg_loan = self.df['loan_amount'].mean()
        with col4:
            st.metric(
                "Avg Loan Amount",
                f"${avg_loan:,.0f}",
                help="Average requested loan amount"
            )

        # Risk distribution
        if 'risk_tier' in self.df.columns:
            high_risk_pct = (self.df['risk_tier'].isin(
                ['High_Risk', 'Very_High_Risk'])).mean()
        else:
            high_risk_pct = (self.df['default'] == 1).mean()

        with col5:
            st.metric(
                "High Risk %",
                f"{high_risk_pct:.1%}",
                delta=f"{(high_risk_pct - 0.25):.1%}",
                help="Percentage of high-risk applications"
            )

    def render_default_analysis(self):
        """Render default rate analysis charts"""
        if self.df.empty:
            return

        st.markdown("### 🎯 Default Rate Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Default rate by employment type
            if 'employment_type' in self.df.columns:
                emp_default = self.df.groupby('employment_type')[
                    'default'].agg(['mean', 'count']).reset_index()
                emp_default['default_rate'] = emp_default['mean']

                fig = px.bar(
                    emp_default,
                    x='employment_type',
                    y='default_rate',
                    title="Default Rate by Employment Type",
                    labels={'default_rate': 'Default Rate',
                            'employment_type': 'Employment Type'},
                    color='default_rate',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Default rate by home ownership
            if 'home_ownership' in self.df.columns:
                home_default = self.df.groupby('home_ownership')[
                    'default'].agg(['mean', 'count']).reset_index()
                home_default['default_rate'] = home_default['mean']

                fig = px.pie(
                    home_default,
                    values='count',
                    names='home_ownership',
                    title="Application Volume by Home Ownership",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)

    def render_credit_analysis(self):
        """Render credit score analysis"""
        if self.df.empty or 'credit_score' not in self.df.columns:
            return

        st.markdown("### 💳 Credit Score Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Credit score distribution by default status
            fig = px.histogram(
                self.df,
                x='credit_score',
                color='default',
                title="Credit Score Distribution by Default Status",
                nbins=30,
                labels={'default': 'Default Status'},
                color_discrete_map={0: '#2E86AB', 1: '#F24236'}
            )
            fig.update_layout(barmode='overlay')
            fig.update_traces(opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Credit score vs default rate
            credit_bins = pd.cut(self.df['credit_score'], bins=10)
            credit_default = self.df.groupby(credit_bins)['default'].agg(
                ['mean', 'count']).reset_index()
            credit_default['credit_score_range'] = credit_default[
                'credit_score'].astype(str)

            fig = px.line(
                credit_default,
                x='credit_score_range',
                y='mean',
                title="Default Rate by Credit Score Range",
                labels={'mean': 'Default Rate',
                        'credit_score_range': 'Credit Score Range'},
                markers=True
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    def render_financial_analysis(self):
        """Render financial metrics analysis"""
        if self.df.empty:
            return

        st.markdown("### 💰 Financial Metrics Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Income vs Loan Amount scatter
            if 'annual_income' in self.df.columns and 'loan_amount' in self.df.columns:
                fig = px.scatter(
                    self.df.sample(min(500, len(self.df))),
                    # Sample for performance
                    x='annual_income',
                    y='loan_amount',
                    color='default',
                    title="Annual Income vs Loan Amount",
                    labels={'annual_income': 'Annual Income ($)',
                            'loan_amount': 'Loan Amount ($)'},
                    color_discrete_map={0: '#2E86AB', 1: '#F24236'},
                    opacity=0.6
                )
                fig.update_layout(legend_title_text='Default Status')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # DTI ratio distribution
            if 'debt_to_income_ratio' in self.df.columns:
                fig = px.box(
                    self.df,
                    x='default',
                    y='debt_to_income_ratio',
                    title="Debt-to-Income Ratio by Default Status",
                    labels={'default': 'Default Status',
                            'debt_to_income_ratio': 'DTI Ratio'},
                    color='default',
                    color_discrete_map={0: '#2E86AB', 1: '#F24236'}
                )
                st.plotly_chart(fig, use_container_width=True)

    def render_loan_purpose_analysis(self):
        """Render loan purpose analysis"""
        if self.df.empty or 'loan_purpose' not in self.df.columns:
            return

        st.markdown("### 🎯 Loan Purpose Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Loan purpose distribution
            purpose_counts = self.df['loan_purpose'].value_counts()

            fig = px.bar(
                x=purpose_counts.index,
                y=purpose_counts.values,
                title="Applications by Loan Purpose",
                labels={'x': 'Loan Purpose', 'y': 'Number of Applications'},
                color=purpose_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_xaxis(tickangle=45)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Default rate by loan purpose
            purpose_default = self.df.groupby('loan_purpose')['default'].agg(
                ['mean', 'count']).reset_index()
            purpose_default = purpose_default[
                purpose_default['count'] >= 10]  # Filter small groups

            fig = px.bar(
                purpose_default,
                x='loan_purpose',
                y='mean',
                title="Default Rate by Loan Purpose",
                labels={'mean': 'Default Rate',
                        'loan_purpose': 'Loan Purpose'},
                color='mean',
                color_continuous_scale='Reds'
            )
            fig.update_xaxis(tickangle=45)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    def render_risk_correlation_heatmap(self):
        """Render correlation heatmap for risk factors"""
        if self.df.empty:
            return

        st.markdown("### 🔥 Risk Factor Correlation Analysis")

        # Select numerical columns for correlation
        risk_columns = [
            'credit_score', 'annual_income', 'loan_amount', 'age',
            'debt_to_income_ratio', 'employment_length', 'default'
        ]
        risk_columns = [col for col in risk_columns if col in self.df.columns]

        if len(risk_columns) > 2:
            corr_matrix = self.df[risk_columns].corr()

            fig = px.imshow(
                corr_matrix,
                title="Risk Factor Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto',
                labels={'color': 'Correlation'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Correlation insights
            default_corr = corr_matrix['default'].abs().sort_values(
                ascending=False)
            default_corr = default_corr[default_corr.index != 'default']

            st.markdown("**Top Risk Factors (correlation with default):**")
            for i, (factor, corr) in enumerate(default_corr.head(3).items(),
                                               1):
                direction = "positively" if corr_matrix.loc[
                                                'default', factor] > 0 else "negatively"
                st.write(
                    f"{i}. **{factor.replace('_', ' ').title()}**: {direction} correlated ({corr:.3f})")

    def render_model_performance_summary(self):
        """Render model performance summary"""
        st.markdown("### 🤖 Model Performance Summary")

        # Mock model performance data (replace with actual model metrics)
        performance_data = {
            'Model': ['XGBoost', 'Logistic Regression'],
            'AUC Score': [0.88, 0.85],
            'Precision': [0.85, 0.82],
            'Recall': [0.80, 0.78],
            'F1 Score': [0.82, 0.80]
        }

        perf_df = pd.DataFrame(performance_data)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

        with col2:
            # Performance comparison chart
            metrics = ['AUC Score', 'Precision', 'Recall', 'F1 Score']

            fig = go.Figure()

            for i, model in enumerate(performance_data['Model']):
                values = [performance_data[metric][i] for metric in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=model
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Performance Comparison"
            )

            st.plotly_chart(fig, use_container_width=True)

    def render_interactive_filters(self):
        """Render interactive filters for dashboard"""
        st.sidebar.markdown("### 🔍 Dashboard Filters")

        filters = {}

        if not self.df.empty:
            # Employment type filter
            if 'employment_type' in self.df.columns:
                emp_options = ['All'] + list(
                    self.df['employment_type'].unique())
                filters['employment_type'] = st.sidebar.selectbox(
                    "Employment Type",
                    emp_options,
                    index=0
                )

            # Credit score range
            if 'credit_score' in self.df.columns:
                min_credit, max_credit = int(
                    self.df['credit_score'].min()), int(
                    self.df['credit_score'].max())
                filters['credit_score_range'] = st.sidebar.slider(
                    "Credit Score Range",
                    min_credit, max_credit,
                    (min_credit, max_credit)
                )

            # Income range
            if 'annual_income' in self.df.columns:
                min_income, max_income = int(
                    self.df['annual_income'].min()), int(
                    self.df['annual_income'].max())
                filters['income_range'] = st.sidebar.slider(
                    "Annual Income Range ($)",
                    min_income, max_income,
                    (min_income, max_income),
                    step=5000
                )

            # Loan purpose filter
            if 'loan_purpose' in self.df.columns:
                purpose_options = ['All'] + list(
                    self.df['loan_purpose'].unique())
                filters['loan_purpose'] = st.sidebar.selectbox(
                    "Loan Purpose",
                    purpose_options,
                    index=0
                )

        return filters

    def apply_filters(self, filters):
        """Apply filters to the dataset"""
        filtered_df = self.df.copy()

        if filters.get('employment_type') and filters[
            'employment_type'] != 'All':
            filtered_df = filtered_df[
                filtered_df['employment_type'] == filters['employment_type']]

        if filters.get('credit_score_range'):
            min_credit, max_credit = filters['credit_score_range']
            filtered_df = filtered_df[
                (filtered_df['credit_score'] >= min_credit) &
                (filtered_df['credit_score'] <= max_credit)
                ]

        if filters.get('income_range'):
            min_income, max_income = filters['income_range']
            filtered_df = filtered_df[
                (filtered_df['annual_income'] >= min_income) &
                (filtered_df['annual_income'] <= max_income)
                ]

        if filters.get('loan_purpose') and filters['loan_purpose'] != 'All':
            filtered_df = filtered_df[
                filtered_df['loan_purpose'] == filters['loan_purpose']]

        return filtered_df

    def render_complete_dashboard(self):
        """Render the complete analytics dashboard"""
        st.markdown("## 📊 Credit Risk Analytics Dashboard")
        st.markdown(
            "Comprehensive analysis of loan applications and risk factors")

        # Interactive filters
        filters = self.render_interactive_filters()

        # Apply filters
        if filters:
            original_df = self.df
            self.df = self.apply_filters(filters)

            if len(self.df) != len(original_df):
                st.info(
                    f"Showing {len(self.df):,} of {len(original_df):,} applications based on filters")

        # Dashboard sections
        self.render_kpi_metrics()
        st.divider()

        self.render_default_analysis()
        st.divider()

        self.render_credit_analysis()
        st.divider()

        self.render_financial_analysis()
        st.divider()

        self.render_loan_purpose_analysis()
        st.divider()

        self.render_risk_correlation_heatmap()
        st.divider()

        self.render_model_performance_summary()


if __name__ == "__main__":
    # Test the dashboard
    dashboard = AnalyticsDashboard()

    st.title("Test Analytics Dashboard")
    dashboard.render_complete_dashboard()