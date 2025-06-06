#!/usr/bin/env python3
"""
Test script to verify LSTM prediction integration
Run this script to test if everything is working correctly
"""

import requests
import json
import sys
import os

# API base URL
BASE_URL = "http://localhost:8000/api"

def test_server_connection():
    """Test if the Flask server is running"""
    try:
        response = requests.get(f"{BASE_URL}/stocks", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
            return True
        else:
            print(f"❌ Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure Flask app is running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {str(e)}")
        return False

def test_stocks_endpoint():
    """Test the stocks listing endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/stocks")
        if response.status_code == 200:
            data = response.json()
            stocks = data.get('stocks', [])
            print(f"✅ Stocks endpoint working. Found {len(stocks)} stocks")
            if stocks:
                print(f"   Sample stocks: {', '.join([s['symbol'] for s in stocks[:3]])}")
            return True
        else:
            print(f"❌ Stocks endpoint failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing stocks endpoint: {str(e)}")
        return False

def test_predictions_endpoint():
    """Test the predictions endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/stocks/predictions")
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            print(f"✅ Predictions endpoint working. Found {len(predictions)} predictions")
            if predictions:
                sample = predictions[0]
                print(f"   Sample prediction: {sample.get('symbol')} -> {sample.get('recommendation')}")
            return True
        else:
            print(f"❌ Predictions endpoint failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing predictions endpoint: {str(e)}")
        return False

def test_individual_prediction():
    """Test individual stock prediction"""
    try:
        # Test with RELIANCE.NS as it's likely to have a model
        test_symbol = "RELIANCE.NS"
        response = requests.get(f"{BASE_URL}/stocks/predict/{test_symbol}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Individual prediction working for {test_symbol}")
            print(f"   Current Price: ₹{data.get('current_price', 'N/A')}")
            print(f"   Predicted Price: ₹{data.get('predicted_price', 'N/A')}")
            print(f"   Expected Return: {data.get('predicted_return', 'N/A')}%")
            print(f"   Recommendation: {data.get('recommendation', 'N/A')}")
            return True
        else:
            print(f"❌ Individual prediction failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing individual prediction: {str(e)}")
        return False

def test_analysis_endpoint():
    """Test the analysis endpoint used by frontend"""
    try:
        test_data = {
            "symbol": "RELIANCE.NS",
            "days": 30,
            "model_type": "auto",
            "risk_tolerance": "medium"
        }
        response = requests.post(f"{BASE_URL}/analyze/stock", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis endpoint working")
            print(f"   Company: {data.get('company_name', 'N/A')}")
            print(f"   Model Type: {data.get('model_type', 'N/A')}")
            print(f"   Recommendation: {data.get('recommendation', {}).get('action', 'N/A')}")
            return True
        else:
            print(f"❌ Analysis endpoint failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing analysis endpoint: {str(e)}")
        return False

def check_prediction_files():
    """Check if prediction CSV files exist"""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    prediction_files = [f for f in os.listdir(src_dir) if f.startswith('stock_predictions_') and f.endswith('.csv')]
    
    if prediction_files:
        latest_file = sorted(prediction_files)[-1]
        print(f"✅ Prediction files found. Latest: {latest_file}")
        return True
    else:
        print("❌ No prediction CSV files found. Run predict_all_stocks.py first")
        return False

def main():
    print("🧪 LSTM Stock Prediction Integration Test")
    print("=" * 50)
    
    tests = [
        ("Server Connection", test_server_connection),
        ("Prediction Files", check_prediction_files),
        ("Stocks Endpoint", test_stocks_endpoint),
        ("Predictions Endpoint", test_predictions_endpoint),
        ("Individual Prediction", test_individual_prediction),
        ("Analysis Endpoint", test_analysis_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  This may affect frontend functionality")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is working correctly.")
        print("\n📝 Next steps:")
        print("   1. Start your React frontend")
        print("   2. Navigate to the LSTM Dashboard component")
        print("   3. Test individual stock analysis")
        print("   4. View bulk predictions table")
    else:
        print("⚠️  Some tests failed. Check the issues above before using the frontend.")
        print("\n🔧 Troubleshooting:")
        if not test_server_connection():
            print("   - Make sure Flask server is running: python app.py")
        if not check_prediction_files():
            print("   - Generate predictions: python predict_all_stocks.py")
        print("   - Check the logs: prediction_v2.log")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
