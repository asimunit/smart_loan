"""
Exploratory Data Analysis for loan default prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *

class LoanEDA:
    def __init__(self, df):
        self.df = df.copy()
        self.setup_plotting()

    def setup_plotting(self):
        """Setup plotting configurations"""
        plt.style.use(PLOT_STYLE)
        sns.set_palette(COLOR_PALETTE)

    def basic_info(self):
        """Display basic dataset information"""
        print("Dataset Information")
        print("=" * 50)
        print(f"Shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        print(f"Default rate: {self.df['default'].mean():.2%}")
        print("\nData types:")
        print(self.df.dtypes.value_counts())

        print("\nNumerical variables summary:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numerical_cols].describe())

        print("\nCategorical variables summary:")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts().head())

    def create_target_analysis(self):
        """Analyze target variable distribution"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Default Distribution', 'Default Rate by Employment Type',
                          'Default Rate by Home Ownership', 'Default Rate by Loan Purpose'),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )

        # Default distribution
        default_counts = self.df['default'].value_counts()
        fig.add_trace(go.Bar(x=['No Default', 'Default'], y=default_counts.values,
                            name='Count', showlegend=False), row=1, col=1)

        # Default rate by employment type
        if 'employment_type' in self.df.columns:
            emp_default = self.df.groupby('employment_type')['default'].agg(['mean', 'count'])
            fig.add_trace(go.Bar(x=emp_default.index, y=emp_default['mean'],
                                name='Default Rate', showlegend=False), row=1, col=2)

        # Default rate by home ownership
        if 'home_ownership' in self.df.columns:
            home_default = self.df.groupby('home_ownership')['default'].agg(['mean', 'count'])
            fig.add_trace(go.Bar(x=home_default.index, y=home_default['mean'],
                                name='Default Rate', showlegend=False), row=2, col=1)

        # Default rate by loan purpose
        if 'loan_purpose' in self.df.columns:
            purpose_default = self.df.groupby('loan_purpose')['default'].agg(['mean', 'count'])
            fig.add_trace(go.Bar(x=purpose_default.index, y=purpose_default['mean'],
                                name='Default Rate', showlegend=False), row=2, col=2)

        fig.update_layout(height=800, title_text="Target Variable Analysis", showlegend=False)
        return fig

    def create_numerical_analysis(self):
        """Analyze numerical variables"""
        numerical_cols = ['annual_income', 'credit_score', 'loan_amount', 'debt_to_income_ratio']
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]

        if len(numerical_cols) == 0:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{col.replace("_", " ").title()} Distribution' for col in numerical_cols[:4]]
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for i, col in enumerate(numerical_cols[:4]):
            row, col_pos = positions[i]

            # Distribution by default status
            no_default = self.df[self.df['default'] == 0][col].dropna()
            default = self.df[self.df['default'] == 1][col].dropna()

            fig.add_trace(go.Histogram(x=no_default, name='No Default', opacity=0.7,
                                     legendgroup='group1', showlegend=(i==0)),
                         row=row, col=col_pos)
            fig.add_trace(go.Histogram(x=default, name='Default', opacity=0.7,
                                     legendgroup='group2', showlegend=(i==0)),
                         row=row, col=col_pos)

        fig.update_layout(height=800, title_text="Numerical Variables Distribution by Default Status")
        return fig

    def create_correlation_analysis(self):
        """Create correlation analysis"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numerical_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title='Correlation Matrix of Numerical Variables',
            xaxis_tickangle=-45,
            height=600,
            width=800
        )

        return fig

    def create_risk_factor_analysis(self):
        """Analyze key risk factors"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Credit Score vs Default Rate', 'Income vs Default Rate',
                          'DTI Ratio vs Default Rate', 'Loan Amount vs Default Rate')
        )

        # Credit Score vs Default Rate
        if 'credit_score' in self.df.columns:
            credit_bins = pd.cut(self.df['credit_score'], bins=10)
            credit_default = self.df.groupby(credit_bins)['default'].mean()
            fig.add_trace(go.Scatter(x=[str(x) for x in credit_default.index],
                                   y=credit_default.values,
                                   mode='lines+markers', name='Credit Score',
                                   showlegend=False), row=1, col=1)

        # Income vs Default Rate
        if 'annual_income' in self.df.columns:
            income_bins = pd.qcut(self.df['annual_income'], q=10, duplicates='drop')
            income_default = self.df.groupby(income_bins)['default'].mean()
            fig.add_trace(go.Scatter(x=list(range(len(income_default))),
                                   y=income_default.values,
                                   mode='lines+markers', name='Income',
                                   showlegend=False), row=1, col=2)

        # DTI Ratio vs Default Rate
        if 'debt_to_income_ratio' in self.df.columns:
            dti_bins = pd.cut(self.df['debt_to_income_ratio'], bins=10)
            dti_default = self.df.groupby(dti_bins)['default'].mean()
            fig.add_trace(go.Scatter(x=[str(x) for x in dti_default.index],
                                   y=dti_default.values,
                                   mode='lines+markers', name='DTI Ratio',
                                   showlegend=False), row=2, col=1)

        # Loan Amount vs Default Rate
        if 'loan_amount' in self.df.columns:
            loan_bins = pd.qcut(self.df['loan_amount'], q=10, duplicates='drop')
            loan_default = self.df.groupby(loan_bins)['default'].mean()
            fig.add_trace(go.Scatter(x=list(range(len(loan_default))),
                                   y=loan_default.values,
                                   mode='lines+markers', name='Loan Amount',
                                   showlegend=False), row=2, col=2)

        fig.update_layout(height=800, title_text="Risk Factors vs Default Rate")
        return fig

    def create_business_insights(self):
        """Generate business insights"""
        insights = []

        # Default rate insight
        default_rate = self.df['default'].mean()
        insights.append(f"Overall default rate: {default_rate:.2%}")

        # Employment type insights
        if 'employment_type' in self.df.columns:
            emp_default = self.df.groupby('employment_type')['default'].mean().sort_values(ascending=False)
            highest_risk_emp = emp_default.index[0]
            lowest_risk_emp = emp_default.index[-1]
            insights.append(f"Highest risk employment: {highest_risk_emp} ({emp_default.iloc[0]:.2%} default rate)")
            insights.append(f"Lowest risk employment: {lowest_risk_emp} ({emp_default.iloc[-1]:.2%} default rate)")

        # Credit score insights
        if 'credit_score' in self.df.columns:
            high_credit = self.df[self.df['credit_score'] >= 700]['default'].mean()
            low_credit = self.df[self.df['credit_score'] < 600]['default'].mean()
            insights.append(f"High credit score (≥700) default rate: {high_credit:.2%}")
            insights.append(f"Low credit score (<600) default rate: {low_credit:.2%}")

        # Income insights
        if 'annual_income' in self.df.columns:
            income_median = self.df['annual_income'].median()
            high_income = self.df[self.df['annual_income'] >= income_median]['default'].mean()
            low_income = self.df[self.df['annual_income'] < income_median]['default'].mean()
            insights.append(f"Above median income default rate: {high_income:.2%}")
            insights.append(f"Below median income default rate: {low_income:.2%}")

        # DTI insights
        if 'debt_to_income_ratio' in self.df.columns:
            high_dti = self.df[self.df['debt_to_income_ratio'] > 0.4]['default'].mean()
            low_dti = self.df[self.df['debt_to_income_ratio'] <= 0.4]['default'].mean()
            insights.append(f"High DTI (>40%) default rate: {high_dti:.2%}")
            insights.append(f"Low DTI (≤40%) default rate: {low_dti:.2%}")

        return insights

    def create_comprehensive_dashboard(self):
        """Create a comprehensive EDA dashboard"""
        # Create individual plots
        target_fig = self.create_target_analysis()
        numerical_fig = self.create_numerical_analysis()
        correlation_fig = self.create_correlation_analysis()
        risk_fig = self.create_risk_factor_analysis()

        # Generate insights
        insights = self.create_business_insights()

        return {
            'target_analysis': target_fig,
            'numerical_analysis': numerical_fig,
            'correlation_analysis': correlation_fig,
            'risk_analysis': risk_fig,
            'insights': insights
        }

    def save_eda_report(self, output_dir=None):
        """Save EDA report as HTML"""
        if output_dir is None:
            output_dir = LOGS_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        dashboard = self.create_comprehensive_dashboard()

        # Save individual plots
        for name, fig in dashboard.items():
            if name != 'insights' and fig is not None:
                fig.write_html(output_dir / f"eda_{name}.html")

        # Create summary report
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SmartLoan EDA Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .insights {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .insight {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>SmartLoan Exploratory Data Analysis Report</h1>
            
            <h2>Dataset Overview</h2>
            <p>Dataset shape: {self.df.shape}</p>
            <p>Default rate: {self.df['default'].mean():.2%}</p>
            
            <h2>Key Business Insights</h2>
            <div class="insights">
                {''.join([f'<div class="insight">• {insight}</div>' for insight in dashboard['insights']])}
            </div>
            
            <h2>Detailed Analysis</h2>
            <p>View the following detailed analysis files:</p>
            <ul>
                <li><a href="eda_target_analysis.html">Target Variable Analysis</a></li>
                <li><a href="eda_numerical_analysis.html">Numerical Variables Analysis</a></li>
                <li><a href="eda_correlation_analysis.html">Correlation Analysis</a></li>
                <li><a href="eda_risk_analysis.html">Risk Factors Analysis</a></li>
            </ul>
        </body>
        </html>
        """

        with open(output_dir / "eda_report.html", "w") as f:
            f.write(report_html)

        print(f"EDA report saved to {output_dir}")
        return dashboard

if __name__ == "__main__":
    # Test EDA with sample data
    from data.data_generator import LoanDataGenerator

    # Generate sample data
    generator = LoanDataGenerator(n_samples=2000)
    df = generator.generate_data()

    # Run EDA
    eda = LoanEDA(df)
    eda.basic_info()

    # Create and save dashboard
    dashboard = eda.save_eda_report()
    print("\nEDA completed successfully!")