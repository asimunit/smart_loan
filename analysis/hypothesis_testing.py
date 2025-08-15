"""
Statistical hypothesis testing for loan default prediction
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *


class HypothesisTests:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.results = {}

    def chi_square_test(self, df, categorical_var, target_var='default'):
        """
        Perform chi-square test for independence between categorical variable and default
        """
        # Create contingency table
        contingency_table = pd.crosstab(df[categorical_var], df[target_var])

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        result = {
            'test': 'Chi-square test',
            'variable': categorical_var,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'significant': p_value < self.alpha,
            'contingency_table': contingency_table,
            'expected_frequencies': expected
        }

        self.results[f'chi2_{categorical_var}'] = result
        return result

    def t_test_independent(self, df, numerical_var, target_var='default'):
        """
        Perform independent t-test between defaulters and non-defaulters
        """
        group_0 = df[df[target_var] == 0][numerical_var].dropna()
        group_1 = df[df[target_var] == 1][numerical_var].dropna()

        # Perform t-test
        t_stat, p_value = ttest_ind(group_0, group_1, equal_var=False)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group_0) - 1) * group_0.var() +
                              (len(group_1) - 1) * group_1.var()) /
                             (len(group_0) + len(group_1) - 2))
        cohens_d = (group_1.mean() - group_0.mean()) / pooled_std

        result = {
            'test': 'Independent t-test',
            'variable': numerical_var,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < self.alpha,
            'group_0_mean': group_0.mean(),
            'group_1_mean': group_1.mean(),
            'group_0_std': group_0.std(),
            'group_1_std': group_1.std(),
            'group_0_n': len(group_0),
            'group_1_n': len(group_1)
        }

        self.results[f'ttest_{numerical_var}'] = result
        return result

    def anova_test(self, df, numerical_var, categorical_var):
        """
        Perform one-way ANOVA to test if means differ across groups
        """
        groups = []
        group_names = []

        for group_name in df[categorical_var].unique():
            if pd.notna(group_name):
                group_data = df[df[categorical_var] == group_name][
                    numerical_var].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
                    group_names.append(group_name)

        if len(groups) < 2:
            return None

        # Perform ANOVA
        f_stat, p_value = f_oneway(*groups)

        # Calculate effect size (eta-squared)
        ss_between = sum(
            len(group) * (group.mean() - df[numerical_var].mean()) ** 2
            for group in groups)
        ss_total = ((df[numerical_var] - df[numerical_var].mean()) ** 2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Group statistics
        group_stats = {}
        for i, group_name in enumerate(group_names):
            group_stats[group_name] = {
                'mean': groups[i].mean(),
                'std': groups[i].std(),
                'n': len(groups[i])
            }

        result = {
            'test': 'One-way ANOVA',
            'numerical_var': numerical_var,
            'categorical_var': categorical_var,
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < self.alpha,
            'group_stats': group_stats
        }

        self.results[f'anova_{numerical_var}_{categorical_var}'] = result
        return result

    def correlation_analysis(self, df, target_var='default'):
        """
        Perform correlation analysis between numerical variables and target
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_var]

        correlations = {}
        for col in numerical_cols:
            if df[col].notna().sum() > 10:  # Ensure sufficient data
                corr_coef = df[col].corr(df[target_var])

                # Test significance of correlation
                n = len(df[[col, target_var]].dropna())
                t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef ** 2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                correlations[col] = {
                    'correlation': corr_coef,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'n': n
                }

        self.results['correlations'] = correlations
        return correlations

    def run_comprehensive_tests(self, df, target_var='default'):
        """
        Run comprehensive hypothesis tests on the dataset
        """
        print("Running comprehensive hypothesis tests...")
        print("=" * 50)

        # 1. Chi-square tests for categorical variables
        categorical_vars = ['employment_type', 'home_ownership',
                            'loan_purpose',
                            'credit_score_tier', 'income_category']

        print("\n1. Chi-square tests for categorical variables:")
        print("-" * 40)
        for var in categorical_vars:
            if var in df.columns:
                result = self.chi_square_test(df, var, target_var)
                print(f"{var}:")
                print(
                    f"  Chi2 = {result['chi2_statistic']:.4f}, p = {result['p_value']:.4f}")
                print(f"  Cramér's V = {result['cramers_v']:.4f}")
                print(f"  Significant: {result['significant']}")
                print()

        # 2. T-tests for numerical variables
        numerical_vars = ['annual_income', 'credit_score', 'loan_amount',
                          'debt_to_income_ratio', 'employment_length']

        print("\n2. T-tests for numerical variables:")
        print("-" * 40)
        for var in numerical_vars:
            if var in df.columns:
                result = self.t_test_independent(df, var, target_var)
                print(f"{var}:")
                print(
                    f"  t = {result['t_statistic']:.4f}, p = {result['p_value']:.4f}")
                print(f"  Cohen's d = {result['cohens_d']:.4f}")
                print(f"  Non-default mean: {result['group_0_mean']:.2f}")
                print(f"  Default mean: {result['group_1_mean']:.2f}")
                print(f"  Significant: {result['significant']}")
                print()

        # 3. ANOVA tests
        print("\n3. ANOVA tests:")
        print("-" * 40)
        anova_pairs = [
            ('annual_income', 'employment_type'),
            ('credit_score', 'credit_score_tier'),
            ('debt_to_income_ratio', 'income_category')
        ]

        for num_var, cat_var in anova_pairs:
            if num_var in df.columns and cat_var in df.columns:
                result = self.anova_test(df, num_var, cat_var)
                if result:
                    print(f"{num_var} by {cat_var}:")
                    print(
                        f"  F = {result['f_statistic']:.4f}, p = {result['p_value']:.4f}")
                    print(f"  Eta-squared = {result['eta_squared']:.4f}")
                    print(f"  Significant: {result['significant']}")
                    print()

        # 4. Correlation analysis
        print("\n4. Correlation analysis with default:")
        print("-" * 40)
        correlations = self.correlation_analysis(df, target_var)

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(),
                             key=lambda x: abs(x[1]['correlation']),
                             reverse=True)

        for var, stats in sorted_corr[:10]:  # Top 10 correlations
            print(f"{var}:")
            print(
                f"  r = {stats['correlation']:.4f}, p = {stats['p_value']:.4f}")
            print(f"  Significant: {stats['significant']}")
            print()

        return self.results

    def create_statistical_summary_plot(self, df, target_var='default'):
        """
        Create visualization of statistical test results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Correlation heatmap
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_cols].corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    ax=axes[0, 0], cbar_kws={'shrink': 0.8})
        axes[0, 0].set_title('Correlation Matrix')

        # 2. Default rate by employment type
        if 'employment_type' in df.columns:
            default_by_employment = df.groupby('employment_type')[
                target_var].agg(['mean', 'count'])
            default_by_employment['mean'].plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Default Rate by Employment Type')
            axes[0, 1].set_ylabel('Default Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Credit score distribution by default status
        if 'credit_score' in df.columns:
            df[df[target_var] == 0]['credit_score'].hist(alpha=0.7,
                                                         label='No Default',
                                                         bins=30,
                                                         ax=axes[1, 0])
            df[df[target_var] == 1]['credit_score'].hist(alpha=0.7,
                                                         label='Default',
                                                         bins=30,
                                                         ax=axes[1, 0])
            axes[1, 0].set_title('Credit Score Distribution by Default Status')
            axes[1, 0].set_xlabel('Credit Score')
            axes[1, 0].legend()

        # 4. Income vs DTI ratio with default coloring
        if 'annual_income' in df.columns and 'debt_to_income_ratio' in df.columns:
            scatter = axes[1, 1].scatter(df['annual_income'],
                                         df['debt_to_income_ratio'],
                                         c=df[target_var], alpha=0.6,
                                         cmap='RdYlBu_r')
            axes[1, 1].set_title('Income vs Debt-to-Income Ratio')
            axes[1, 1].set_xlabel('Annual Income')
            axes[1, 1].set_ylabel('Debt-to-Income Ratio')
            plt.colorbar(scatter, ax=axes[1, 1], label='Default')

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Test the hypothesis testing
    from data.data_generator import LoanDataGenerator

    # Generate sample data
    generator = LoanDataGenerator(n_samples=2000)
    df = generator.generate_data()

    # Run hypothesis tests
    tester = HypothesisTests()
    results = tester.run_comprehensive_tests(df)

    # Create plots
    fig = tester.create_statistical_summary_plot(df)
    plt.show()