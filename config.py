# Configuration settings for the Indian Stock Predictor application

class Config:
    # Flask configuration
    DEBUG = True
    TESTING = False

    # Model parameters
    SEQUENCE_LENGTH = 10
    EPOCHS = 50
    BATCH_SIZE = 32

    # Data source settings
    DATA_SOURCE = 'yahoo'
    
    # Stock symbols
    INDIAN_STOCKS = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "HINDUNILVR.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "TATAMOTORS.NS",
        "WIPRO.NS",
        "HCLTECH.NS",
        "SUNPHARMA.NS"
    ]