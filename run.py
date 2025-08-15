"""
Main runner script for SmartLoan project
"""

import sys
import argparse
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import *
from data.data_generator import LoanDataGenerator
from preprocessing.feature_engineering import FeatureEngineer
from analysis.hypothesis_testing import HypothesisTests
from models.model_trainer import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'smartloan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartLoanPipeline:
    def __init__(self):
        self.logger = logger

    def generate_data(self, n_samples=N_SAMPLES, save=True):
        """Generate synthetic loan data"""
        self.logger.info(f"Generating {n_samples} loan records...")

        generator = LoanDataGenerator(n_samples=n_samples)
        df = generator.generate_data()

        if save:
            df.to_csv(DATA_DIR / "loan_data.csv", index=False)
            self.logger.info(f"Data saved to {DATA_DIR / 'loan_data.csv'}")

        self.logger.info(f"Generated {len(df)} records with {df['default'].mean():.2%} default rate")
        return df

    def run_hypothesis_tests(self, df=None):
        """Run comprehensive hypothesis tests"""
        self.logger.info("Running hypothesis tests...")

        if df is None:
            df = self.load_data()

        tester = HypothesisTests()
        results = tester.run_comprehensive_tests(df)

        # Save statistical plots
        try:
            fig = tester.create_statistical_summary_plot(df)
            fig.savefig(LOGS_DIR / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Statistical plots saved to {LOGS_DIR / 'statistical_analysis.png'}")
        except Exception as e:
            self.logger.warning(f"Could not save statistical plots: {str(e)}")

        return results

    def train_models(self, df=None, quick_training=False):
        """Train machine learning models"""
        self.logger.info("Training models...")

        if df is None:
            df = self.load_data()

        trainer = ModelTrainer()

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)

        # Train models
        self.logger.info("Training Logistic Regression...")
        lr_model = trainer.train_logistic_regression(perform_grid_search=not quick_training)

        self.logger.info("Training XGBoost...")
        xgb_model = trainer.train_xgboost(perform_grid_search=not quick_training)

        # Evaluate models
        self.logger.info("Evaluating models...")
        results = trainer.evaluate_models()

        # Print results
        for model_name, model_results in results.items():
            auc = model_results['auc']
            report = model_results['classification_report']
            precision = report['1']['precision']
            recall = report['1']['recall']
            f1 = report['1']['f1-score']

            self.logger.info(f"{model_name.replace('_', ' ').title()}:")
            self.logger.info(f"  AUC: {auc:.4f}")
            self.logger.info(f"  Precision: {precision:.4f}")
            self.logger.info(f"  Recall: {recall:.4f}")
            self.logger.info(f"  F1-Score: {f1:.4f}")

            # Check if model meets targets
            if auc >= TARGET_AUC:
                self.logger.info(f"  ✅ AUC target ({TARGET_AUC}) achieved!")
            else:
                self.logger.warning(f"  ⚠️ AUC target ({TARGET_AUC}) not achieved")

        # Save models
        trainer.save_models()

        # Save comparison plots
        try:
            fig = trainer.plot_model_comparison(results)
            fig.savefig(LOGS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison plots saved to {LOGS_DIR / 'model_comparison.png'}")
        except Exception as e:
            self.logger.warning(f"Could not save model plots: {str(e)}")

        return trainer, results

    def load_data(self):
        """Load data from CSV file"""
        data_path = DATA_DIR / "loan_data.csv"

        if not data_path.exists():
            self.logger.warning("Data file not found. Generating new data...")
            return self.generate_data()

        import pandas as pd
        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(df)} records from {data_path}")
        return df

    def run_full_pipeline(self, n_samples=N_SAMPLES, quick_training=False):
        """Run the complete ML pipeline"""
        self.logger.info("="*60)
        self.logger.info("STARTING SMARTLOAN ML PIPELINE")
        self.logger.info("="*60)

        try:
            # Step 1: Generate data
            self.logger.info("\n📊 STEP 1: Data Generation")
            df = self.generate_data(n_samples)

            # Step 2: Hypothesis testing
            self.logger.info("\n🔬 STEP 2: Statistical Analysis")
            hypothesis_results = self.run_hypothesis_tests(df)

            # Step 3: Train models
            self.logger.info("\n🤖 STEP 3: Model Training")
            trainer, model_results = self.train_models(df, quick_training)

            # Step 4: Summary
            self.logger.info("\n📋 PIPELINE SUMMARY")
            self.logger.info("-" * 40)
            self.logger.info(f"Data samples: {len(df)}")
            self.logger.info(f"Features engineered: {trainer.X_train.shape[1]}")
            self.logger.info(f"Statistical tests completed: {len(hypothesis_results)}")

            best_model = "xgboost" if model_results['xgboost']['auc'] > model_results['logistic_regression']['auc'] else "logistic_regression"
            best_auc = model_results[best_model]['auc']
            self.logger.info(f"Best model: {best_model} (AUC: {best_auc:.4f})")

            if best_auc >= TARGET_AUC:
                self.logger.info("🎉 TARGET AUC ACHIEVED!")
            else:
                self.logger.info(f"📈 Target AUC: {TARGET_AUC} (Current: {best_auc:.4f})")

            self.logger.info("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("Next steps:")
            self.logger.info("1. Start API server: uvicorn api.main:app --reload --port 8000")
            self.logger.info("2. Start Streamlit UI: streamlit run ui/streamlit_app.py")

            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return False

    def test_api_prediction(self):
        """Test the trained models with sample prediction"""
        self.logger.info("Testing model prediction...")

        try:
            from models.model_trainer import ModelTrainer
            import pandas as pd

            # Load trained models
            trainer = ModelTrainer()
            trainer.load_models()

            if trainer.logistic_model is None and trainer.xgboost_model is None:
                self.logger.error("No trained models found. Please run training first.")
                return False

            # Create sample application
            sample_application = {
                'age': 35,
                'annual_income': 60000,
                'employment_length': 5.0,
                'employment_type': 'Full-time',
                'home_ownership': 'Rent',
                'loan_amount': 15000,
                'loan_purpose': 'debt_consolidation',
                'interest_rate': 12.5,
                'loan_term': 36,
                'credit_score': 650
            }

            # Calculate derived features that the model expects
            monthly_rate = sample_application['interest_rate'] / 100 / 12
            n_payments = sample_application['loan_term']
            monthly_payment = (sample_application['loan_amount'] * monthly_rate *
                              (1 + monthly_rate) ** n_payments) / \
                             ((1 + monthly_rate) ** n_payments - 1)
            sample_application['monthly_payment'] = monthly_payment

            # Add the debt-to-income ratio that the feature engineer expects
            sample_application['debt_to_income_ratio'] = sample_application['loan_amount'] / sample_application['annual_income']

            # Create DataFrame and transform
            df_sample = pd.DataFrame([sample_application])
            X_processed, _ = trainer.feature_engineer.transform(df_sample, target_column=None)

            # Make predictions
            if trainer.xgboost_model is not None:
                xgb_prob = trainer.xgboost_model.predict_proba(X_processed)[0, 1]
                self.logger.info(f"XGBoost prediction: {xgb_prob:.3f} default probability")

            if trainer.logistic_model is not None:
                lr_prob = trainer.logistic_model.predict_proba(X_processed)[0, 1]
                self.logger.info(f"Logistic Regression prediction: {lr_prob:.3f} default probability")

            self.logger.info("✅ Model prediction test successful")
            return True

        except Exception as e:
            self.logger.error(f"Prediction test failed: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="SmartLoan ML Pipeline")
    parser.add_argument('--mode', choices=['full', 'data', 'stats', 'train', 'test'],
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--samples', type=int, default=N_SAMPLES,
                       help='Number of samples to generate')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training without grid search')

    args = parser.parse_args()

    pipeline = SmartLoanPipeline()

    if args.mode == 'full':
        success = pipeline.run_full_pipeline(args.samples, args.quick)
        if success:
            pipeline.test_api_prediction()
    elif args.mode == 'data':
        pipeline.generate_data(args.samples)
    elif args.mode == 'stats':
        pipeline.run_hypothesis_tests()
    elif args.mode == 'train':
        pipeline.train_models(quick_training=args.quick)
    elif args.mode == 'test':
        pipeline.test_api_prediction()

if __name__ == "__main__":
    main()