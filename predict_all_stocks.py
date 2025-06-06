import os
import sys
import logging
import pandas as pd
import os
import sys
import logging
import traceback
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from lstm_model import LSTMModel
from data_loader import load_historical_data

def predict_stock(symbol):
    """Make prediction for a specific stock symbol"""
    logger.info(f"Making prediction for {symbol}")
    
    try:
        # Load the model
        model = LSTMModel()
        if not model.load_model(symbol):
            logger.error(f"No trained model found for {symbol}")
            return {"error": f"No trained model found for {symbol}", "symbol": symbol}
            
        # Load historical data
        historical_data = load_historical_data(symbol, start_date="2022-01-01")
        
        if historical_data is None:
            logger.error(f"No historical data available for {symbol}")
            return {"error": f"No historical data available for {symbol}", "symbol": symbol}
            
        if historical_data.empty:
            logger.error(f"Empty historical data for {symbol}")
            return {"error": f"Empty historical data for {symbol}", "symbol": symbol}
            
        # Make prediction
        prediction = model.predict_next_day(historical_data)
        
        # Ensure symbol is included in the result
        if prediction and "error" not in prediction:
            prediction["symbol"] = symbol
            return prediction
        else:
            # Handle error case
            error_msg = prediction.get("error", "Unknown prediction error") if prediction else "Prediction returned None"
            return {"error": error_msg, "symbol": symbol}
        
    except Exception as e:
        logger.error(f"Error predicting for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to make prediction: {str(e)}", "symbol": symbol}
        
def main():
    """Make predictions for all stocks in the Indian stocks list"""
    logger.info("Starting batch prediction for all Indian stocks")
    
    # Get the list of Indian stocks from config
    stocks = Config.INDIAN_STOCKS
    
    successful = 0
    failed = 0
    
    results = []
    
    for i, symbol in enumerate(stocks):
        logger.info(f"Processing {i+1}/{len(stocks)}: {symbol}")
        prediction = predict_stock(symbol)
        
        if prediction and "error" not in prediction:
            successful += 1
            results.append(prediction)
        else:
            failed += 1
            # Still add failed predictions to results for debugging
            if prediction:
                results.append(prediction)
            else:
                results.append({"error": "Unknown error", "symbol": symbol})
            
    logger.info(f"Prediction complete! Successful: {successful}, Failed: {failed}")
    
    if results:
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Separate successful and failed predictions
        successful_df = df[~df.get('error', pd.Series([None]*len(df))).notna()]
        failed_df = df[df.get('error', pd.Series([None]*len(df))).notna()]
        
        # Sort successful predictions by predicted return
        if not successful_df.empty and "predicted_return" in successful_df.columns:
            successful_df = successful_df.sort_values(by="predicted_return", ascending=False)
        
        # Save all results to CSV
        filename = f"stock_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"All results saved to {filename}")
        
        # Save successful predictions separately
        if not successful_df.empty:
            success_filename = f"successful_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
            successful_df.to_csv(success_filename, index=False)
            logger.info(f"Successful predictions saved to {success_filename}")
            
            # Display top 5 successful stocks
            print(f"\nTop 5 Stocks by Predicted Return (out of {len(successful_df)} successful predictions):")
            print(successful_df.head(5))
        else:
            print("\nNo successful predictions were made.")
            
        # Display failed predictions
        if not failed_df.empty:
            print(f"\nFailed predictions ({len(failed_df)}):")
            print(failed_df[['symbol', 'error']].head(10))
    else:
        print("No results to save.")
    
if __name__ == "__main__":
    main()