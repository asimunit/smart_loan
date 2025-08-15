"""
Model training for loan default prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *
from preprocessing.feature_engineering import FeatureEngineer

class ModelTrainer:
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.logistic_model = None
        self.xgboost_model = None
        self.feature_engineer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, df, target_column='default'):
        """Prepare and split data for training"""

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()

        # Apply feature engineering
        X, y = self.feature_engineer.fit_transform(df, target_column)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state, stratify=y
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Features: {self.X_train.shape[1]}")
        print(f"Default rate in training: {self.y_train.mean():.2%}")
        print(f"Default rate in test: {self.y_test.mean():.2%}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_logistic_regression(self, perform_grid_search=True):
        """Train Logistic Regression model with hyperparameter tuning"""

        print("\nTraining Logistic Regression...")

        if perform_grid_search:
            # Define hyperparameter grid with compatible solver/penalty combinations
            param_grid = [
                {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'max_iter': [1000, 2000]
                },
                {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['saga'],
                    'max_iter': [1000, 2000],
                    'l1_ratio': [0.5]  # Required for elasticnet
                }
            ]

            # Handle elasticnet penalty
            logistic = LogisticRegression(random_state=self.random_state, class_weight='balanced')

            # Grid search with cross-validation
            grid_search = GridSearchCV(
                logistic, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )

            grid_search.fit(self.X_train, self.y_train)
            self.logistic_model = grid_search.best_estimator_

            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV AUC: {grid_search.best_score_:.4f}")

        else:
            # Train with default parameters
            self.logistic_model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
            self.logistic_model.fit(self.X_train, self.y_train)

        # Cross-validation score
        cv_scores = cross_val_score(
            self.logistic_model, self.X_train, self.y_train,
            cv=5, scoring='roc_auc'
        )
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return self.logistic_model

    def train_xgboost(self, perform_grid_search=True):
        """Train XGBoost model with hyperparameter tuning"""

        print("\nTraining XGBoost...")

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        if perform_grid_search:
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }

            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric='auc'
            )

            # Grid search with cross-validation
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )

            grid_search.fit(self.X_train, self.y_train)
            self.xgboost_model = grid_search.best_estimator_

            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV AUC: {grid_search.best_score_:.4f}")

        else:
            # Train with default parameters
            self.xgboost_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric='auc'
            )

            self.xgboost_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                early_stopping_rounds=10,
                verbose=False
            )

        # Cross-validation score
        cv_scores = cross_val_score(
            self.xgboost_model, self.X_train, self.y_train,
            cv=5, scoring='roc_auc'
        )
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return self.xgboost_model

    def evaluate_models(self):
        """Evaluate both models on test set"""

        results = {}

        # Evaluate Logistic Regression
        if self.logistic_model is not None:
            lr_pred = self.logistic_model.predict(self.X_test)
            lr_pred_proba = self.logistic_model.predict_proba(self.X_test)[:, 1]
            lr_auc = roc_auc_score(self.y_test, lr_pred_proba)

            results['logistic_regression'] = {
                'predictions': lr_pred,
                'probabilities': lr_pred_proba,
                'auc': lr_auc,
                'classification_report': classification_report(self.y_test, lr_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(self.y_test, lr_pred)
            }

            print(f"\nLogistic Regression Test AUC: {lr_auc:.4f}")
            print("Classification Report:")
            print(classification_report(self.y_test, lr_pred))

        # Evaluate XGBoost
        if self.xgboost_model is not None:
            xgb_pred = self.xgboost_model.predict(self.X_test)
            xgb_pred_proba = self.xgboost_model.predict_proba(self.X_test)[:, 1]
            xgb_auc = roc_auc_score(self.y_test, xgb_pred_proba)

            results['xgboost'] = {
                'predictions': xgb_pred,
                'probabilities': xgb_pred_proba,
                'auc': xgb_auc,
                'classification_report': classification_report(self.y_test, xgb_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(self.y_test, xgb_pred)
            }

            print(f"\nXGBoost Test AUC: {xgb_auc:.4f}")
            print("Classification Report:")
            print(classification_report(self.y_test, xgb_pred))

        return results

    def plot_model_comparison(self, results):
        """Create comparison plots for both models"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ROC Curves
        if 'logistic_regression' in results:
            lr_fpr, lr_tpr, _ = roc_curve(self.y_test, results['logistic_regression']['probabilities'])
            axes[0, 0].plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {results['logistic_regression']['auc']:.3f})")

        if 'xgboost' in results:
            xgb_fpr, xgb_tpr, _ = roc_curve(self.y_test, results['xgboost']['probabilities'])
            axes[0, 0].plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {results['xgboost']['auc']:.3f})")

        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Feature Importance (XGBoost)
        if 'xgboost' in results and hasattr(self.xgboost_model, 'feature_importances_'):
            importance_df = self.feature_engineer.get_feature_importance_summary(
                self.xgboost_model.feature_importances_, top_n=15
            )

            sns.barplot(data=importance_df, y='feature', x='importance', ax=axes[0, 1])
            axes[0, 1].set_title('XGBoost Feature Importance (Top 15)')
            axes[0, 1].set_xlabel('Importance')

        # Confusion Matrices
        if 'logistic_regression' in results:
            sns.heatmap(results['logistic_regression']['confusion_matrix'],
                       annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_title('Logistic Regression Confusion Matrix')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')

        if 'xgboost' in results:
            sns.heatmap(results['xgboost']['confusion_matrix'],
                       annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('XGBoost Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')

        plt.tight_layout()
        return fig

    def save_models(self):
        """Save trained models and preprocessor"""

        # Create models directory
        MODELS_DIR.mkdir(exist_ok=True)

        # Save models
        if self.logistic_model is not None:
            joblib.dump(self.logistic_model, LOGISTIC_MODEL_PATH)
            print(f"Logistic Regression model saved to {LOGISTIC_MODEL_PATH}")

        if self.xgboost_model is not None:
            joblib.dump(self.xgboost_model, XGBOOST_MODEL_PATH)
            print(f"XGBoost model saved to {XGBOOST_MODEL_PATH}")

        # Save preprocessor and feature names
        if self.feature_engineer is not None:
            joblib.dump(self.feature_engineer.preprocessor, PREPROCESSOR_PATH)
            joblib.dump(self.feature_engineer.feature_names, FEATURE_NAMES_PATH)
            print(f"Preprocessor saved to {PREPROCESSOR_PATH}")
            print(f"Feature names saved to {FEATURE_NAMES_PATH}")

    def load_models(self):
        """Load saved models and preprocessor"""

        try:
            self.logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
            print(f"Logistic Regression model loaded from {LOGISTIC_MODEL_PATH}")
        except FileNotFoundError:
            print("Logistic Regression model not found")

        try:
            self.xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
            print(f"XGBoost model loaded from {XGBOOST_MODEL_PATH}")
        except FileNotFoundError:
            print("XGBoost model not found")

        try:
            self.feature_engineer = FeatureEngineer()
            self.feature_engineer.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.feature_engineer.feature_names = joblib.load(FEATURE_NAMES_PATH)
            print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
        except FileNotFoundError:
            print("Preprocessor not found")

if __name__ == "__main__":
    # Test the model training
    from data.data_generator import LoanDataGenerator

    # Generate data
    generator = LoanDataGenerator(n_samples=5000)
    df = generator.generate_data()

    # Train models
    trainer = ModelTrainer()
    trainer.prepare_data(df)

    # Train both models (quick training for testing)
    trainer.train_logistic_regression(perform_grid_search=False)
    trainer.train_xgboost(perform_grid_search=False)

    # Evaluate models
    results = trainer.evaluate_models()

    # Create comparison plots
    fig = trainer.plot_model_comparison(results)
    plt.show()

    # Save models
    trainer.save_models()