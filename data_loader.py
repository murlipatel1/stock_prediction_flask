from datetime import datetime
import yfinance as yf
import pandas as pd
import os

def load_historical_data(symbol, start_date=None, end_date=None):
    if start_date is None:
        start_date = "2018-01-01"
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data found for the given symbol.")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error loading data for {symbol}: {str(e)}")
        return None

def save_data_to_csv(data, symbol, directory="data"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, f"{symbol}.csv")
    data.to_csv(file_path, index=False)
    print(f"Data for {symbol} saved to {file_path}")

def load_data_from_csv(symbol, directory="data"):
    file_path = os.path.join(directory, f"{symbol}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"No CSV file found for {symbol} in {directory}.")
        return None