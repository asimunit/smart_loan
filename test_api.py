"""
Quick API test script to debug prediction issues
"""

import requests
import json


def test_api():
    """Test API endpoints with sample data"""

    base_url = "http://localhost:8000"

    # Sample application data
    sample_application = {
        "age": 35,
        "annual_income": 60000,
        "employment_length": 5.0,
        "employment_type": "Full-time",
        "home_ownership": "Rent",
        "loan_amount": 15000,
        "loan_purpose": "debt_consolidation",
        "interest_rate": 12.5,
        "loan_term": 36,
        "credit_score": 650
    }

    print("🧪 Testing SmartLoan API")
    print("=" * 50)

    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ Health Check Passed")
            print(f"   Status: {health['status']}")
            print(f"   Models Loaded: {health['models_loaded']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check Error: {str(e)}")

    # Test 2: Debug Preprocessing
    print("\n2. Testing Debug Preprocessing...")
    try:
        response = requests.post(f"{base_url}/debug/preprocess",
                                 json=sample_application)
        if response.status_code == 200:
            debug_info = response.json()
            print("✅ Debug Preprocessing Passed")
            print(f"   Input columns: {len(debug_info.get('input_data', {}))}")
            print(f"   DataFrame shape: {debug_info.get('dataframe_shape')}")
            print(
                f"   Preprocessing success: {debug_info.get('preprocessing_success')}")
            if not debug_info.get('preprocessing_success'):
                print(f"   Error: {debug_info.get('preprocessing_error')}")
        else:
            print(f"❌ Debug Preprocessing Failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Debug Preprocessing Error: {str(e)}")

    # Test 3: Actual Prediction
    print("\n3. Testing Prediction...")
    try:
        response = requests.post(f"{base_url}/predict",
                                 json=sample_application)
        if response.status_code == 200:
            prediction = response.json()
            print("✅ Prediction Passed")
            print(
                f"   Default Probability: {prediction.get('default_probability', 0):.2%}")
            print(f"   Risk Tier: {prediction.get('risk_tier')}")
            print(
                f"   Prediction: {'APPROVE' if not prediction.get('default_prediction') else 'DECLINE'}")
        else:
            print(f"❌ Prediction Failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")

    # Test 4: Model Info
    print("\n4. Testing Model Info...")
    for model_name in ["xgboost", "logistic_regression"]:
        try:
            response = requests.get(f"{base_url}/model_info/{model_name}")
            if response.status_code == 200:
                model_info = response.json()
                print(f"✅ {model_name.title()} Info Retrieved")
                print(f"   AUC Score: {model_info.get('auc_score', 0):.3f}")
            else:
                print(
                    f"❌ {model_name.title()} Info Failed: {response.status_code}")
        except Exception as e:
            print(f"❌ {model_name.title()} Info Error: {str(e)}")

    print("\n" + "=" * 50)
    print("🧪 API Testing Complete")


if __name__ == "__main__":
    test_api()