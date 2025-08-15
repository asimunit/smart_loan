"""
Model evaluation utilities for loan default prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, log_loss,
    brier_score_loss
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.calibration import calibration_curve
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *


class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}

    def calculate_business_metrics(self, y_true, y_pred, y_prob,
                                   loan_amounts=None):
        """Calculate business-specific metrics for loan default prediction"""

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (
                                                                                  precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Business metrics
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Financial impact (if loan amounts provided)
        financial_metrics = {}
        if loan_amounts is not None:
            # Assume 100% loss on defaults, administrative cost for false positives
            avg_loan_amount = np.mean(loan_amounts)
            admin_cost_ratio = 0.05  # 5% administrative cost for processing

            # True positives: correctly identified defaults (money saved)
            money_saved = tp * avg_loan_amount

            # False negatives: missed defaults (money lost)
            money_lost = fn * avg_loan_amount

            # False positives: rejected good loans (opportunity cost)
            opportunity_cost = fp * avg_loan_amount * 0.1  # Assume 10% profit margin

            # Administrative costs
            admin_cost = (tp + fp) * avg_loan_amount * admin_cost_ratio

            net_benefit = money_saved - money_lost - opportunity_cost - admin_cost

            financial_metrics = {
                'money_saved': money_saved,
                'money_lost': money_lost,
                'opportunity_cost': opportunity_cost,
                'admin_cost': admin_cost,
                'net_benefit': net_benefit,
                'benefit_per_loan': net_benefit / len(y_true) if len(
                    y_true) > 0 else 0
            }

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'auc_roc': roc_auc_score(y_true, y_prob),
            'auc_pr': average_precision_score(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),
            'brier_score': brier_score_loss(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            **financial_metrics
        }

        return metrics

    def evaluate_model_performance(self, model, X_test, y_test, model_name,
                                   loan_amounts=None):
        """Comprehensive model evaluation"""

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = self.calculate_business_metrics(y_test, y_pred, y_prob,
                                                  loan_amounts)

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_test,
                                                                  y_prob)

        # Calibration data
        fraction_pos, mean_pred_prob = calibration_curve(y_test, y_prob,
                                                         n_bins=10)

        evaluation_result = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': class_report,
            'roc_data': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_data': {'precision': precision, 'recall': recall,
                        'thresholds': pr_thresholds},
            'calibration_data': {'fraction_pos': fraction_pos,
                                 'mean_pred_prob': mean_pred_prob},
            'predictions': {'y_pred': y_pred, 'y_prob': y_prob}
        }

        self.evaluation_results[model_name] = evaluation_result
        return evaluation_result

    def compare_models(self, models_dict, X_test, y_test, loan_amounts=None):
        """Compare multiple models"""

        comparison_results = {}

        for model_name, model in models_dict.items():
            print(f"Evaluating {model_name}...")
            result = self.evaluate_model_performance(
                model, X_test, y_test, model_name, loan_amounts
            )
            comparison_results[model_name] = result

        # Create comparison summary
        comparison_df = pd.DataFrame({
            name: result['metrics'] for name, result in
            comparison_results.items()
        }).T

        return comparison_results, comparison_df

    def plot_roc_comparison(self, models_results=None):
        """Plot ROC curves for model comparison"""

        if models_results is None:
            models_results = self.evaluation_results

        fig = go.Figure()

        # Plot each model
        for model_name, result in models_results.items():
            roc_data = result['roc_data']
            auc_score = result['metrics']['auc_roc']

            fig.add_trace(go.Scatter(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                mode='lines',
                name=f"{model_name} (AUC = {auc_score:.3f})",
                line=dict(width=2)
            ))

        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))

        fig.update_layout(
            title='ROC Curve Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500,
            legend=dict(x=0.6, y=0.1)
        )

        return fig

    def plot_precision_recall_comparison(self, models_results=None):
        """Plot Precision-Recall curves for model comparison"""

        if models_results is None:
            models_results = self.evaluation_results

        fig = go.Figure()

        for model_name, result in models_results.items():
            pr_data = result['pr_data']
            auc_pr = result['metrics']['auc_pr']

            fig.add_trace(go.Scatter(
                x=pr_data['recall'],
                y=pr_data['precision'],
                mode='lines',
                name=f"{model_name} (AUC = {auc_pr:.3f})",
                line=dict(width=2)
            ))

        fig.update_layout(
            title='Precision-Recall Curve Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500
        )

        return fig

    def plot_calibration_curves(self, models_results=None):
        """Plot calibration curves for probability calibration assessment"""

        if models_results is None:
            models_results = self.evaluation_results

        fig = go.Figure()

        for model_name, result in models_results.items():
            cal_data = result['calibration_data']

            fig.add_trace(go.Scatter(
                x=cal_data['mean_pred_prob'],
                y=cal_data['fraction_pos'],
                mode='lines+markers',
                name=model_name,
                line=dict(width=2)
            ))

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))

        fig.update_layout(
            title='Calibration Curves',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=600,
            height=500
        )

        return fig

    def plot_confusion_matrices(self, models_results=None):
        """Plot confusion matrices for all models"""

        if models_results is None:
            models_results = self.evaluation_results

        n_models = len(models_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(models_results.keys()),
            specs=[[{"type": "xy"}] * cols for _ in range(rows)]
        )

        for i, (model_name, result) in enumerate(models_results.items()):
            row = i // cols + 1
            col = i % cols + 1

            cm = result['metrics']['confusion_matrix']

            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['Predicted No Default', 'Predicted Default'],
                    y=['Actual No Default', 'Actual Default'],
                    colorscale='Blues',
                    showscale=False,
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 14}
                ),
                row=row, col=col
            )

        fig.update_layout(
            title='Confusion Matrices Comparison',
            height=300 * rows,
            width=300 * cols
        )

        return fig

    def create_business_impact_summary(self, models_results=None):
        """Create business impact summary"""

        if models_results is None:
            models_results = self.evaluation_results

        business_summary = []

        for model_name, result in models_results.items():
            metrics = result['metrics']

            summary = {
                'Model': model_name,
                'AUC-ROC': f"{metrics['auc_roc']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}",
                'False Positive Rate': f"{metrics['false_positive_rate']:.3f}",
                'False Negative Rate': f"{metrics['false_negative_rate']:.3f}"
            }

            # Add financial metrics if available
            if 'net_benefit' in metrics:
                summary.update({
                    'Net Benefit ($)': f"{metrics['net_benefit']:,.0f}",
                    'Benefit per Loan ($)': f"{metrics['benefit_per_loan']:.2f}",
                    'Money Saved ($)': f"{metrics['money_saved']:,.0f}",
                    'Money Lost ($)': f"{metrics['money_lost']:,.0f}"
                })

            business_summary.append(summary)

        return pd.DataFrame(business_summary)

    def generate_evaluation_report(self, models_results=None, save_path=None):
        """Generate comprehensive evaluation report"""

        if models_results is None:
            models_results = self.evaluation_results

        # Create plots
        roc_fig = self.plot_roc_comparison(models_results)
        pr_fig = self.plot_precision_recall_comparison(models_results)
        cal_fig = self.plot_calibration_curves(models_results)
        cm_fig = self.plot_confusion_matrices(models_results)

        # Business summary
        business_df = self.create_business_impact_summary(models_results)

        report = {
            'business_summary': business_df,
            'roc_comparison': roc_fig,
            'precision_recall_comparison': pr_fig,
            'calibration_curves': cal_fig,
            'confusion_matrices': cm_fig,
            'detailed_results': models_results
        }

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)

            # Save plots
            roc_fig.write_html(save_path / "roc_comparison.html")
            pr_fig.write_html(save_path / "precision_recall_comparison.html")
            cal_fig.write_html(save_path / "calibration_curves.html")
            cm_fig.write_html(save_path / "confusion_matrices.html")

            # Save business summary
            business_df.to_csv(save_path / "business_summary.csv", index=False)

            print(f"Evaluation report saved to {save_path}")

        return report


if __name__ == "__main__":
    # Test model evaluation
    from data.data_generator import LoanDataGenerator
    from models.model_trainer import ModelTrainer

    # Generate data and train models
    generator = LoanDataGenerator(n_samples=2000)
    df = generator.generate_data()

    trainer = ModelTrainer()
    trainer.prepare_data(df)
    trainer.train_logistic_regression(perform_grid_search=False)
    trainer.train_xgboost(perform_grid_search=False)

    # Evaluate models
    evaluator = ModelEvaluator()

    models_dict = {
        'Logistic Regression': trainer.logistic_model,
        'XGBoost': trainer.xgboost_model
    }

    # Generate loan amounts for business metrics
    loan_amounts = df['loan_amount'].values[trainer.X_test.index]

    comparison_results, comparison_df = evaluator.compare_models(
        models_dict, trainer.X_test, trainer.y_test, loan_amounts
    )

    print("Model Comparison Summary:")
    print(comparison_df[['auc_roc', 'precision', 'recall', 'f1_score']])

    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(
        save_path=LOGS_DIR / "evaluation_report")
    print("\nEvaluation completed!")