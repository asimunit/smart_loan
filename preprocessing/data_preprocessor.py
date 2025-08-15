"""
Data preprocessing utilities for loan default prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *


class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_stats = {}

    def detect_outliers(self, df, method='iqr', columns=None):
        """Detect outliers using IQR or Z-score method"""

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        outliers = pd.DataFrame(index=df.index)

        for col in columns:
            if col in df.columns:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers[col] = (df[col] < lower_bound) | (
                                df[col] > upper_bound)

                elif method == 'zscore':
                    z_scores = np.abs(
                        (df[col] - df[col].mean()) / df[col].std())
                    outliers[col] = z_scores > 3

        return outliers

    def handle_outliers(self, df, method='cap', outlier_columns=None):
        """Handle outliers using capping, removal, or transformation"""

        df_processed = df.copy()

        if outlier_columns is None:
            outlier_columns = df.select_dtypes(include=[np.number]).columns

        for col in outlier_columns:
            if col in df_processed.columns:
                if method == 'cap':
                    # Cap outliers at 5th and 95th percentiles
                    lower_cap = df_processed[col].quantile(0.05)
                    upper_cap = df_processed[col].quantile(0.95)
                    df_processed[col] = df_processed[col].clip(lower=lower_cap,
                                                               upper=upper_cap)

                elif method == 'winsorize':
                    # Winsorize at 1st and 99th percentiles
                    lower_win = df_processed[col].quantile(0.01)
                    upper_win = df_processed[col].quantile(0.99)
                    df_processed[col] = df_processed[col].clip(lower=lower_win,
                                                               upper=upper_win)

                elif method == 'log_transform':
                    # Log transformation for positive values
                    if (df_processed[col] > 0).all():
                        df_processed[col] = np.log1p(df_processed[col])

        return df_processed

    def handle_missing_values(self, df, strategy='auto'):
        """Handle missing values with various strategies"""

        df_processed = df.copy()
        missing_summary = df_processed.isnull().sum()

        print("Missing Values Summary:")
        print(missing_summary[missing_summary > 0])

        if strategy == 'auto':
            # Automatic strategy based on data type and missing percentage
            for col in df_processed.columns:
                missing_pct = df_processed[col].isnull().mean()

                if missing_pct > 0:
                    if missing_pct > 0.5:
                        # Drop columns with >50% missing
                        print(f"Dropping {col} (>{missing_pct:.1%} missing)")
                        df_processed = df_processed.drop(columns=[col])

                    elif df_processed[col].dtype in ['object', 'category']:
                        # Categorical: mode or 'Unknown'
                        if df_processed[col].mode().empty:
                            fill_value = 'Unknown'
                        else:
                            fill_value = df_processed[col].mode()[0]
                        df_processed[col] = df_processed[col].fillna(
                            fill_value)
                        print(f"Filled {col} categorical with: {fill_value}")

                    else:
                        # Numerical: median for skewed, mean for normal
                        skewness = abs(df_processed[col].skew())
                        if skewness > 1:
                            fill_value = df_processed[col].median()
                            strategy_used = 'median'
                        else:
                            fill_value = df_processed[col].mean()
                            strategy_used = 'mean'
                        df_processed[col] = df_processed[col].fillna(
                            fill_value)
                        print(
                            f"Filled {col} numerical with {strategy_used}: {fill_value:.2f}")

        return df_processed

    def create_interaction_features(self, df):
        """Create interaction features between important variables"""

        df_processed = df.copy()

        # Income-based ratios
        if 'annual_income' in df_processed.columns and 'loan_amount' in df_processed.columns:
            df_processed['income_loan_ratio'] = df_processed['annual_income'] / \
                                                df_processed['loan_amount']

        if 'annual_income' in df_processed.columns and 'monthly_payment' in df_processed.columns:
            df_processed['income_payment_ratio'] = df_processed[
                                                       'annual_income'] / (
                                                               df_processed[
                                                                   'monthly_payment'] * 12)

        # Credit score interactions
        if 'credit_score' in df_processed.columns and 'annual_income' in df_processed.columns:
            df_processed['credit_income_interaction'] = df_processed[
                                                            'credit_score'] * np.log1p(
                df_processed['annual_income'])

        if 'credit_score' in df_processed.columns and 'employment_length' in df_processed.columns:
            df_processed['credit_employment_interaction'] = df_processed[
                                                                'credit_score'] * \
                                                            df_processed[
                                                                'employment_length']

        # Age interactions
        if 'age' in df_processed.columns and 'annual_income' in df_processed.columns:
            df_processed['age_income_interaction'] = df_processed['age'] * \
                                                     df_processed[
                                                         'annual_income']

        return df_processed

    def create_polynomial_features(self, df, degree=2, columns=None):
        """Create polynomial features for numerical variables"""

        df_processed = df.copy()

        if columns is None:
            # Select key numerical columns for polynomial features
            columns = ['annual_income', 'credit_score', 'debt_to_income_ratio']
            columns = [col for col in columns if col in df_processed.columns]

        for col in columns:
            if col in df_processed.columns:
                for d in range(2, degree + 1):
                    new_col = f"{col}_poly_{d}"
                    df_processed[new_col] = df_processed[col] ** d

        return df_processed

    def create_binned_features(self, df):
        """Create binned categorical features from numerical variables"""

        df_processed = df.copy()

        # Age bins
        if 'age' in df_processed.columns:
            df_processed['age_bin'] = pd.cut(
                df_processed['age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['Young', 'Young_Adult', 'Middle_Age', 'Senior',
                        'Elder']
            )

        # Income bins (percentile-based)
        if 'annual_income' in df_processed.columns:
            income_percentiles = [0, 0.25, 0.5, 0.75, 1.0]
            income_bins = df_processed['annual_income'].quantile(
                income_percentiles).values
            df_processed['income_quartile'] = pd.cut(
                df_processed['annual_income'],
                bins=income_bins,
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                include_lowest=True
            )

        # Loan amount bins
        if 'loan_amount' in df_processed.columns:
            loan_percentiles = [0, 0.33, 0.67, 1.0]
            loan_bins = df_processed['loan_amount'].quantile(
                loan_percentiles).values
            df_processed['loan_size'] = pd.cut(
                df_processed['loan_amount'],
                bins=loan_bins,
                labels=['Small', 'Medium', 'Large'],
                include_lowest=True
            )

        return df_processed

    def scale_features(self, df, method='standard', exclude_columns=None):
        """Scale numerical features using different methods"""

        df_processed = df.copy()

        if exclude_columns is None:
            exclude_columns = ['default']  # Don't scale target variable

        numerical_cols = df_processed.select_dtypes(
            include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if
                          col not in exclude_columns]

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(
                "Method must be 'standard', 'minmax', or 'robust'")

        # Fit and transform numerical columns
        df_processed[numerical_cols] = scaler.fit_transform(
            df_processed[numerical_cols])

        # Store scaler for later use
        self.scalers[method] = scaler

        return df_processed

    def create_preprocessing_pipeline(self, df, target_column='default'):
        """Create a complete preprocessing pipeline"""

        print("Starting data preprocessing pipeline...")

        # Separate features and target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df.copy()
            y = None

        print(f"Initial shape: {X.shape}")

        # Step 1: Handle missing values
        print("\n1. Handling missing values...")
        X = self.handle_missing_values(X, strategy='auto')
        print(f"Shape after missing value handling: {X.shape}")

        # Step 2: Handle outliers
        print("\n2. Handling outliers...")
        numerical_cols = ['annual_income', 'loan_amount',
                          'debt_to_income_ratio']
        numerical_cols = [col for col in numerical_cols if col in X.columns]
        X = self.handle_outliers(X, method='cap',
                                 outlier_columns=numerical_cols)

        # Step 3: Create interaction features
        print("\n3. Creating interaction features...")
        X = self.create_interaction_features(X)
        print(f"Shape after interaction features: {X.shape}")

        # Step 4: Create binned features
        print("\n4. Creating binned features...")
        X = self.create_binned_features(X)
        print(f"Shape after binned features: {X.shape}")

        # Step 5: Store feature statistics
        self.feature_stats = {
            'numerical_features': list(
                X.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(
                X.select_dtypes(include=['object', 'category']).columns),
            'total_features': X.shape[1],
            'missing_values': X.isnull().sum().to_dict()
        }

        print(f"\nPreprocessing completed!")
        print(f"Final shape: {X.shape}")
        print(
            f"Numerical features: {len(self.feature_stats['numerical_features'])}")
        print(
            f"Categorical features: {len(self.feature_stats['categorical_features'])}")

        if y is not None:
            return X, y
        else:
            return X

    def get_preprocessing_summary(self):
        """Get summary of preprocessing steps applied"""

        summary = {
            'scalers_used': list(self.scalers.keys()),
            'imputers_used': list(self.imputers.keys()),
            'feature_statistics': self.feature_stats,
            'preprocessing_steps': [
                'Missing value handling',
                'Outlier treatment',
                'Interaction feature creation',
                'Binned feature creation',
                'Feature scaling (if applied)'
            ]
        }

        return summary


if __name__ == "__main__":
    # Test preprocessing with sample data
    from data.data_generator import LoanDataGenerator

    # Generate sample data
    generator = LoanDataGenerator(n_samples=2000)
    df = generator.generate_data()

    # Add some missing values for testing
    np.random.seed(42)
    missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)),
                                       replace=False)
    df.loc[missing_indices, 'employment_length'] = np.nan

    # Run preprocessing
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.create_preprocessing_pipeline(df)

    print("\nPreprocessing Summary:")
    summary = preprocessor.get_preprocessing_summary()
    print(f"Steps applied: {summary['preprocessing_steps']}")
    print(
        f"Final feature count: {summary['feature_statistics']['total_features']}")