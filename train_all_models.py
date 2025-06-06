import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from lstm_model import LSTMModel
from data_loader import load_historical_data
from routes import prepare_features

def train_model_for_stock(symbol):
    """Train a LSTM model for a specific stock symbol"""
    logger.info(f"Starting training for {symbol}")
    
    try:
        # Load historical data
        historical_data = load_historical_data(symbol, start_date="2018-01-01")
        
        if historical_data is None or historical_data.empty:
            logger.error(f"No historical data available for {symbol}")
            return False
            
        logger.info(f"Loaded {len(historical_data)} data points for {symbol}")
        
        # Prepare features
        data = prepare_features(historical_data)
        
        if data is None or len(data) < 30:  # Ensure we have enough data
            logger.error(f"Not enough data for {symbol} after preprocessing")
            return False
        
        # Check for and handle problematic values
        # Replace inf/-inf with NaN
        data = data.replace([float('inf'), -float('inf')], float('nan'))
        
        # Check if any columns have very large values
        max_threshold = 1e12  # Adjust as needed for your data
        cols_with_large_values = data.columns[(data.abs() > max_threshold).any()].tolist()
        if cols_with_large_values:
            logger.warning(f"Columns with extremely large values: {cols_with_large_values}")
            # Option 1: Drop problematic columns if they're not essential
            data = data.drop(columns=cols_with_large_values)
            logger.info(f"Dropped {len(cols_with_large_values)} columns with extreme values")
            
            # If we dropped too many columns, we might want to abort
            if len(data.columns) < 5:  # Adjust threshold as needed
                logger.error(f"Too many problematic features for {symbol}, aborting")
                return False
                
        # Handle NaN values - you can fill with means, medians, or other strategies
        data = data.fillna(data.median())
        
        # Final check to ensure no inf or nan values remain
        if data.isnull().any().any() or np.isinf(data.values).any():
            logger.error(f"Could not clean data for {symbol}, still contains NaN or inf values")
            return False
            
        # Features - convert to list of strings
        features = list(data.columns)
        
        # Create model and train
        model = LSTMModel()
        if model.train(data.values, features, symbol):
            logger.info(f"Successfully trained model for {symbol}")
            return True
        else:
            logger.error(f"Failed to train model for {symbol}")
            return False
        
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
def main():
    """Train models for all stocks in the Indian stocks list"""
    logger.info("Starting batch training for all Indian stocks")
    
    # Get the list of Indian stocks from config
    stocks = Config.INDIAN_STOCKS
    
    successful = 0
    failed = 0
    
    results = []
    
    for i, symbol in enumerate(stocks):
        logger.info(f"Processing {i+1}/{len(stocks)}: {symbol}")
        status = "Success" if train_model_for_stock(symbol) else "Failed"
        
        if status == "Success":
            successful += 1
        else:
            failed += 1
            
        results.append({"symbol": symbol, "status": status})
            
    logger.info(f"Training complete! Successful: {successful}, Failed: {failed}")
    
    # Save results to CSV
    pd.DataFrame(results).to_csv("training_results.csv", index=False)
    logger.info("Results saved to training_results.csv")
    
if __name__ == "__main__":
    main()