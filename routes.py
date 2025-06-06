from flask import Blueprint, request, jsonify
from flask_cors import CORS
from lstm_model import LSTMModel
from data_loader import load_historical_data
from config import Config
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
from technical_indicators import calculate_technical_indicators
import yfinance as yf
import json

# Change the name to api_bp to match what's imported in app.py
api_bp = Blueprint('api', __name__)

# Enable CORS for all routes
CORS(api_bp)

# Define the routes for the API
@api_bp.route('/stocks', methods=['GET'])
def get_stocks():
    """Get the list of available stocks"""
    try:
        stocks = Config.INDIAN_STOCKS
        
        # Convert to the format expected by frontend
        formatted_stocks = []
        for symbol in stocks:
            # Extract company name from symbol (remove .NS)
            name = symbol.replace('.NS', '')
            formatted_stocks.append({
                'symbol': symbol,
                'name': name
            })
        
        return jsonify({"stocks": formatted_stocks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New API to get detailed Yahoo Finance data for Power BI dashboard
@api_bp.route('/stocks/yfinance/data', methods=['GET'])
def get_yfinance_data():
    """Get detailed stock data from Yahoo Finance for all configured stocks"""
    try:
        # Use fixed parameters for comprehensive data collection
        period = request.args.get('period', '1y')  # Default to 1 year
        interval = request.args.get('interval', '1d')  # Default to daily data
        symbol = request.args.get('symbol')  # Still allow filtering by symbol if needed
        
        # Always use all configured stocks unless a specific symbol is requested
        if not symbol:
            symbols = Config.INDIAN_STOCKS
        else:
            symbols = [symbol]
            
        logging.info(f"Fetching yfinance data for {len(symbols)} stocks with period={period}, interval={interval}")
        
        result = {
            'stocks_data': {},
            'metadata': {
                'total_stocks': len(symbols),
                'period': period,
                'interval': interval,
                'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        success_count = 0
        error_count = 0
        
        for stock_symbol in symbols:
            try:
                logging.info(f"Processing stock: {stock_symbol}")
                
                # Get historical market data
                data = yf.download(
                    tickers=stock_symbol,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    timeout=15  # Add timeout to prevent hanging
                )
                
                if data.empty:
                    logging.warning(f"No data returned for {stock_symbol}")
                    result['stocks_data'][stock_symbol] = {"error": "No data available"}
                    error_count += 1
                    continue
                
                # Get additional info about the stock
                stock_info = yf.Ticker(stock_symbol)
                info = stock_info.info
                
                # Convert timestamps to string for JSON serialization
                df = data.reset_index()
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                
                # Calculate basic statistics
                price_stats = {
                    'current_price': float(df['Close'].iloc[-1]) if not df.empty else None,
                    'avg_price': float(df['Close'].mean()) if not df.empty else None,
                    'min_price': float(df['Close'].min()) if not df.empty else None,
                    'max_price': float(df['Close'].max()) if not df.empty else None,
                    'price_change': float((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100) if not df.empty and len(df) > 1 else None
                }
                
                # Format the result with comprehensive data
                stock_data = {
                    'symbol': stock_symbol,
                    'company_name': stock_symbol.replace('.NS', ''),
                    'historical_data': df.to_dict('records'),
                    'price_statistics': price_stats,
                    'info': {k: v for k, v in info.items() if not isinstance(v, (dict, list, pd.DataFrame)) and v is not None}
                }
                
                # Add recent financials if available
                try:
                    balance_sheet = stock_info.balance_sheet
                    if not balance_sheet.empty:
                        # Convert index to strings for JSON serialization
                        balance_sheet_copy = balance_sheet.copy()
                        if isinstance(balance_sheet_copy.index, pd.DatetimeIndex):
                            balance_sheet_copy.index = balance_sheet_copy.index.strftime('%Y-%m-%d')
                        balance_sheet_dict = balance_sheet_copy.reset_index().to_dict('records')
                        stock_data['balance_sheet'] = balance_sheet_dict
                except Exception as e:
                    logging.warning(f"Could not get balance sheet for {stock_symbol}: {str(e)}")
                    
                try:
                    income_stmt = stock_info.income_stmt
                    if isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
                        # Handle DataFrame index for JSON serialization
                        income_stmt_copy = income_stmt.copy()
                        if isinstance(income_stmt_copy.index, pd.DatetimeIndex):
                            income_stmt_copy.index = income_stmt_copy.index.strftime('%Y-%m-%d')
                        stock_data['income_statement'] = income_stmt_copy.reset_index().to_dict('records')
                except Exception as e:
                    logging.warning(f"Could not get income statement for {stock_symbol}: {str(e)}")
                
                try:
                    cashflow = stock_info.cashflow
                    if isinstance(cashflow, pd.DataFrame) and not cashflow.empty:
                        # Handle DataFrame index for JSON serialization
                        cashflow_copy = cashflow.copy()
                        if isinstance(cashflow_copy.index, pd.DatetimeIndex):
                            cashflow_copy.index = cashflow_copy.index.strftime('%Y-%m-%d')
                        stock_data['cashflow'] = cashflow_copy.reset_index().to_dict('records')
                except Exception as e:
                    logging.warning(f"Could not get cashflow data for {stock_symbol}: {str(e)}")
                
                # Add to the result
                result['stocks_data'][stock_symbol] = stock_data
                success_count += 1
                
            except Exception as e:
                logging.error(f"Error getting data for {stock_symbol}: {str(e)}")
                result['stocks_data'][stock_symbol] = {"error": str(e)}
                error_count += 1
        
        # Update metadata with success/error counts
        result['metadata']['success_count'] = success_count
        result['metadata']['error_count'] = error_count
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in Yahoo Finance API: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    
# New API for technical analysis specifically for Power BI integration
@api_bp.route('/stocks/analysis/technical', methods=['GET'])
def get_technical_analysis():
    """Get technical analysis data for one or multiple stocks - formatted for Power BI"""
    try:
        symbol = request.args.get('symbol')
        days = int(request.args.get('days', '365'))  # Default to 1 year
        
        # If no symbol provided, get data for all configured stocks
        if not symbol:
            symbols = Config.INDIAN_STOCKS
        else:
            symbols = [symbol]
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        result = []
        for stock_symbol in symbols:
            try:
                # Get historical data
                data = load_historical_data(stock_symbol, 
                                           start_date=start_date.strftime('%Y-%m-%d'), 
                                           end_date=end_date.strftime('%Y-%m-%d'))
                
                if data is None or data.empty:
                    result.append({
                        'symbol': stock_symbol,
                        'error': 'No data available'
                    })
                    continue
                
                # Calculate technical indicators
                data_with_indicators = calculate_technical_indicators(data)
                
                # # Format for PowerBI - convert to records
                # records = data_with_indicators.to_dict('records')
                
                # # Convert datetime to string for JSON serialization
                # for record in records:
                #     record['Date'] = record['Date'].strftime('%Y-%m-%d')
                #     record['Symbol'] = stock_symbol
                #     record['CompanyName'] = stock_symbol.replace('.NS', '')

                # Format for PowerBI - convert to records
                records = []
                for index, row in data_with_indicators.iterrows():
                    record = row.to_dict()
                    # Convert datetime to string for JSON serialization
                    if isinstance(record['Date'], datetime):
                        record['Date'] = record['Date'].strftime('%Y-%m-%d')
                    record['Symbol'] = stock_symbol
                    record['CompanyName'] = stock_symbol.replace('.NS', '')
                    records.append(record)
                
                result.extend(records)
                
            except Exception as e:
                logging.error(f"Error analyzing {stock_symbol}: {str(e)}")
                result.append({
                    'symbol': stock_symbol,
                    'error': str(e)
                })
        
        return jsonify({'technical_analysis': result})
    except Exception as e:
        logging.error(f"Error in technical analysis API: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Additional API for Power BI to get performance metrics
@api_bp.route('/stocks/performance', methods=['GET'])
def get_stock_performance():
    """Get performance metrics for stocks"""
    try:
        symbol = request.args.get('symbol')
        period = request.args.get('period', '1y')  # Default to 1 year
        
        # If no symbol provided, get data for all configured stocks
        if not symbol:
            symbols = Config.INDIAN_STOCKS
        else:
            symbols = [symbol]
        
        performance_data = []
        for stock_symbol in symbols:
            try:
                # Get ticker data
                stock = yf.Ticker(stock_symbol)
                hist = stock.history(period=period)
                
                if hist.empty:
                    performance_data.append({
                        'symbol': stock_symbol,
                        'error': 'No historical data available'
                    })
                    continue
                
                # Calculate performance metrics
                first_price = hist['Close'].iloc[0]
                last_price = hist['Close'].iloc[-1]
                highest_price = hist['High'].max()
                lowest_price = hist['Low'].min()
                total_return = ((last_price - first_price) / first_price) * 100
                
                # Volatility (standard deviation of returns)
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * 100
                
                # Calculate average volume
                avg_volume = hist['Volume'].mean()
                
                # Get company info
                info = stock.info
                
                performance = {
                    'symbol': stock_symbol,
                    'company_name': stock_symbol.replace('.NS', ''),
                    'current_price': last_price,
                    'period_start_price': first_price,
                    'highest_price': highest_price,
                    'lowest_price': lowest_price,
                    'total_return_percentage': total_return,
                    'volatility_percentage': volatility,
                    'average_volume': avg_volume,
                    'market_cap': info.get('marketCap', None),
                    'pe_ratio': info.get('trailingPE', None),
                    'dividend_yield': info.get('dividendYield', None) * 100 if info.get('dividendYield') is not None else None,
                    'period': period,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                performance_data.append(performance)
                
            except Exception as e:
                logging.error(f"Error getting performance for {stock_symbol}: {str(e)}")
                performance_data.append({
                    'symbol': stock_symbol,
                    'error': str(e)
                })
        
        return jsonify({'performance_data': performance_data})
    except Exception as e:
        logging.error(f"Error in performance API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/stocks/predictions', methods=['GET'])
def get_all_predictions():
    """Get all stock predictions from the latest CSV file"""
    try:
        # Look for the most recent predictions CSV file
        csv_file = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for today's predictions first
        today = datetime.now().strftime('%Y%m%d')
        today_file = f"stock_predictions_{today}.csv"
        
        if os.path.exists(os.path.join(current_dir, today_file)):
            csv_file = os.path.join(current_dir, today_file)
        else:
            # Look for any predictions file
            for file in os.listdir(current_dir):
                if file.startswith('stock_predictions_') and file.endswith('.csv'):
                    csv_file = os.path.join(current_dir, file)
                    break
        
        if not csv_file:
            return jsonify({"error": "No predictions file found"}), 404
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Convert to JSON format
        predictions = df.to_dict('records')
        
        return jsonify({
            "predictions": predictions,
            "file_date": os.path.basename(csv_file),
            "total_stocks": len(predictions)
        })
        
    except Exception as e:
        logging.error(f"Error getting predictions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/stocks/predict/<symbol>', methods=['GET'])
def get_stock_prediction(symbol):
    """Get prediction for a specific stock"""
    try:
        # Look for the most recent predictions CSV file
        csv_file = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for today's predictions first
        today = datetime.now().strftime('%Y%m%d')
        today_file = f"stock_predictions_{today}.csv"
        
        if os.path.exists(os.path.join(current_dir, today_file)):
            csv_file = os.path.join(current_dir, today_file)
        else:
            # Look for any predictions file
            for file in os.listdir(current_dir):
                if file.startswith('stock_predictions_') and file.endswith('.csv'):
                    csv_file = os.path.join(current_dir, file)
                    break
        
        if not csv_file:
            return jsonify({"error": "No predictions file found"}), 404
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Find the specific stock
        stock_data = df[df['symbol'] == symbol]
        
        if stock_data.empty:
            return jsonify({"error": f"No prediction found for {symbol}"}), 404
        
        # Convert to dict and return
        prediction = stock_data.iloc[0].to_dict()
        
        # Add additional analysis information
        prediction['analysis_date'] = prediction.get('prediction_date', datetime.now().strftime('%Y-%m-%d'))
        prediction['model_type'] = 'LSTM'
        
        # Calculate risk level based on predicted return
        predicted_return = prediction.get('predicted_return', 0)
        if abs(predicted_return) > 15:
            risk_level = "High"
        elif abs(predicted_return) > 5:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        prediction['risk_level'] = risk_level
        
        return jsonify(prediction)
        
    except Exception as e:
        logging.error(f"Error getting prediction for {symbol}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/analyze/stock', methods=['POST'])
def analyze_stock():
    """Analyze a stock with user parameters - compatible with frontend"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        days = data.get('days', 30)
        model_type = data.get('model_type', 'auto')
        risk_tolerance = data.get('risk_tolerance', 'medium')
        
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
        
        # Get prediction from CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = None
        
        # Look for today's predictions first
        today = datetime.now().strftime('%Y%m%d')
        today_file = f"stock_predictions_{today}.csv"
        
        if os.path.exists(os.path.join(current_dir, today_file)):
            csv_file = os.path.join(current_dir, today_file)
        else:
            # Look for any predictions file
            for file in os.listdir(current_dir):
                if file.startswith('stock_predictions_') and file.endswith('.csv'):
                    csv_file = os.path.join(current_dir, file)
                    break
        
        if not csv_file:
            return jsonify({"error": "No predictions available"}), 404
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Find the specific stock
        stock_data = df[df['symbol'] == symbol]
        
        if stock_data.empty:
            return jsonify({"error": f"No prediction found for {symbol}"}), 404
        
        stock_info = stock_data.iloc[0]
        
        # Format response to match frontend expectations
        analysis_result = {
            "symbol": symbol,
            "company_name": symbol.replace('.NS', ''),
            "current_price": float(stock_info.get('current_price', 0)),
            "predicted_price": float(stock_info.get('predicted_price', 0)),
            "predicted_return": float(stock_info.get('predicted_return', 0)),
            "model_type": "LSTM",
            "analysis_date": stock_info.get('prediction_date', datetime.now().strftime('%Y-%m-%d')),
            "recommendation": {
                "action": stock_info.get('recommendation', 'HOLD'),
                "confidence": stock_info.get('confidence', 'Medium'),
                "recommendation": stock_info.get('recommendation', 'HOLD'),
                "reason": f"Based on LSTM model prediction with {stock_info.get('confidence', 'Medium').lower()} confidence. Predicted return: {stock_info.get('predicted_return', 0):.2f}%"
            },
            "trend_analysis": {
                "overall_trend": "Bullish" if float(stock_info.get('predicted_return', 0)) > 0 else "Bearish"
            },
            "risk_metrics": {
                "risk": abs(float(stock_info.get('predicted_return', 0))) / 100
            },
            "prediction": {
                "next_day_price": float(stock_info.get('predicted_price', 0)),
                "return_percent": float(stock_info.get('predicted_return', 0)),
                "confidence": stock_info.get('confidence', 'Medium')
            }
        }
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logging.error(f"Error analyzing stock: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/stocks/predict', methods=['POST'])
def predict_stock():
    data = request.get_json()
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    
    model = LSTMModel(symbol)
    model.load_model(symbol)  # Add symbol parameter here
    
    historical_data = load_historical_data(symbol)
    if historical_data is None:
        return jsonify({"error": "Failed to load historical data"}), 500
    
    # This method doesn't exist yet, we need to add it
    prediction = model.predict_next_day(historical_data)
    
    return jsonify(prediction)

@api_bp.route('/stocks/train', methods=['POST'])
def train_stock_model():
    data = request.get_json()
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    
    # Get historical data
    historical_data = load_historical_data(symbol)
    if historical_data is None:
        return jsonify({"error": "Failed to load historical data"}), 500
    
    # Prepare data for training
    data_features = prepare_features(historical_data)
    features = list(data_features.columns)
    
    model = LSTMModel()
    success = model.train(data_features.values, features, symbol)
    
    if not success:
        return jsonify({"error": "Failed to train model"}), 500
    
    return jsonify({"message": f"Model trained and saved for {symbol}"})

# Add this helper function
def prepare_features(df):
    """Prepare data with technical indicators for model training"""
    data = df.copy()
    
    # Technical indicators
    # Moving Averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Price and Volume change
    data['Price_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # Drop NaN values
    data.dropna(inplace=True)
    
    # Features to use
    features = ['Close', 'Volume', 'MA5', 'MA20', 'MACD', 'RSI', 'Price_Change', 'Volume_Change']
    return data[features]