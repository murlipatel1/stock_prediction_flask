from pandas import DataFrame
import numpy as np

def calculate_moving_average(data: DataFrame, window: int) -> DataFrame:
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data: DataFrame, window: int = 14) -> DataFrame:
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data: DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> DataFrame:
    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data: DataFrame, window: int = 20, num_std_dev: int = 2) -> DataFrame:
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def calculate_technical_indicators(data: DataFrame) -> DataFrame:
    data['MA5'] = calculate_moving_average(data, 5)
    data['MA20'] = calculate_moving_average(data, 20)
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'] = calculate_macd(data)
    data['UpperBand'], data['LowerBand'] = calculate_bollinger_bands(data)
    return data.fillna(0)