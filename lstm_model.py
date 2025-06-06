# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# import pandas as pd
# import joblib
# import os

# class LSTMModel:
#     def __init__(self, sequence_length=10, model_dir='stock_models_lstm'):
#         self.sequence_length = sequence_length
#         self.model = None
#         self.scaler_X = None
#         self.scaler_y = None
#         self.model_dir = model_dir

#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)

#     def _get_model_path(self, symbol):
#         clean_symbol = symbol.replace('.', '_').replace('-', '_')
#         return os.path.join(self.model_dir, f"{clean_symbol}_lstm_model.h5")

#     def _get_scaler_X_path(self, symbol):
#         clean_symbol = symbol.replace('.', '_').replace('-', '_')
#         return os.path.join(self.model_dir, f"{clean_symbol}_scaler_X.joblib")

#     def _get_scaler_y_path(self, symbol):
#         clean_symbol = symbol.replace('.', '_').replace('-', '_')
#         return os.path.join(self.model_dir, f"{clean_symbol}_scaler_y.joblib")

#     def create_model(self, input_shape):
#         model = Sequential()
#         model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
#         model.add(Dropout(0.3))
#         model.add(LSTM(64, return_sequences=True))
#         model.add(Dropout(0.3))
#         model.add(LSTM(64))
#         model.add(Dropout(0.3))
#         model.add(Dense(32, activation='relu'))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         return model


#     def train(self, data, features, symbol):
#         try:
#             self.scaler_X = MinMaxScaler()
#             scaled_data = self.scaler_X.fit_transform(data)

#             X, y = self._create_sequences(scaled_data, features)

#             self.scaler_y = MinMaxScaler()
#             y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

#             self.model = self.create_model((self.sequence_length, len(features)))
#             self.model.fit(X, y, epochs=50, batch_size=32, verbose=1)

#             self.save_model(symbol)
#             return True
#         except Exception as e:
#             print(f"Error training model: {str(e)}")
#             return False

#     def _create_sequences(self, data, features):
#         X, y = [], []
#         for i in range(len(data) - self.sequence_length):
#             X.append(data[i:i + self.sequence_length, :])
#             y.append(data[i + self.sequence_length, 0])  # Assuming 'Close' is the first feature
#         return np.array(X), np.array(y)

#     def save_model(self, symbol):
#         model_path = self._get_model_path(symbol)
#         self.model.save(model_path)
#         joblib.dump(self.scaler_X, self._get_scaler_X_path(symbol))
#         joblib.dump(self.scaler_y, self._get_scaler_y_path(symbol))

#     def load_model(self, symbol):
#         model_path = self._get_model_path(symbol)
#         scaler_X_path = self._get_scaler_X_path(symbol)
#         scaler_y_path = self._get_scaler_y_path(symbol)

#         if os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
#             try:
#                 self.model = load_model(model_path)
#                 self.scaler_X = joblib.load(scaler_X_path)
#                 self.scaler_y = joblib.load(scaler_y_path)
#                 return True
#             except Exception as e:
#                 print(f"Error loading model: {str(e)}")
#                 return False
#         return False

#     def predict_next_day(self, historical_data):
#         """Predict the next day's price using the trained model"""
#         try:
#             # Prepare the data with features
#             from routes import prepare_features
#             data = prepare_features(historical_data)
            
#             if data is None or len(data) < self.sequence_length:
#                 return {"error": "Not enough data for prediction"}
            
#             # Get the last sequence for prediction
#             last_sequence = data.values[-self.sequence_length:]
            
#             # Scale the sequence
#             last_sequence_scaled = self.scaler_X.transform(last_sequence)
            
#             # Reshape for LSTM input
#             prediction_input = last_sequence_scaled.reshape(1, self.sequence_length, last_sequence.shape[1])
            
#             # Make prediction
#             predicted_price_scaled = self.model.predict(prediction_input, verbose=0)[0][0]
            
#             # Inverse transform to get the actual price
#             predicted_price = self.scaler_y.inverse_transform(
#                 np.array([[predicted_price_scaled]])
#             )[0][0]
            
#             # Current price
#             current_price = data['Close'].iloc[-1]
            
#             # Calculate predicted return
#             predicted_return = (predicted_price - current_price) / current_price * 100
            
#             # Get company name
#             company_name = historical_data['Symbol'].iloc[0] if 'Symbol' in historical_data.columns else "Unknown"
            
#             # Create prediction result
#             return {
#                 "symbol": company_name,
#                 "current_price": float(current_price),
#                 "predicted_price": float(predicted_price),
#                 "predicted_return": float(predicted_return),
#                 "prediction_date": pd.Timestamp.now().strftime("%Y-%m-%d")
#             }
            
#         except Exception as e:
#             print(f"Error making prediction: {str(e)}")
#             return {"error": f"Failed to make prediction: {str(e)}"}


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import joblib
import os

class LSTMModel:
    def __init__(self, sequence_length=10, model_dir='stock_models_lstm_v2'):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.model_dir = model_dir
        self.feature_names = None

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_model_path(self, symbol):
        clean_symbol = symbol.replace('.', '_').replace('-', '_')
        return os.path.join(self.model_dir, f"{clean_symbol}_lstm_model.h5")

    def _get_scaler_X_path(self, symbol):
        clean_symbol = symbol.replace('.', '_').replace('-', '_')
        return os.path.join(self.model_dir, f"{clean_symbol}_scaler_X.joblib")

    def _get_scaler_y_path(self, symbol):
        clean_symbol = symbol.replace('.', '_').replace('-', '_')
        return os.path.join(self.model_dir, f"{clean_symbol}_scaler_y.joblib")

    def _get_features_path(self, symbol):
        clean_symbol = symbol.replace('.', '_').replace('-', '_')
        return os.path.join(self.model_dir, f"{clean_symbol}_features.joblib")

    def create_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def train(self, data, features, symbol):
        try:
            # Store feature names - handle both list and tuple
            if isinstance(features, (list, tuple)):
                self.feature_names = list(features)
            else:
                self.feature_names = features
            
            # Find the close price column index
            close_idx = None
            
            # Handle case where features might be column names from DataFrame
            if isinstance(self.feature_names[0], str):
                for i, feature in enumerate(self.feature_names):
                    if 'close' in feature.lower():
                        close_idx = i
                        break
            else:
                # If features are not strings, look for Close in the actual data
                # Assume data comes from prepare_features which should have Close column
                for i in range(data.shape[1]):
                    # Check if this column looks like price data (positive values in reasonable range)
                    col_data = data[:, i]
                    if np.all(col_data > 0) and np.mean(col_data) > 10:  # Assuming stock prices > 10
                        close_idx = i
                        break
            
            if close_idx is None:
                # If no 'Close' column found, assume it's the first column
                close_idx = 0
                print(f"Warning: No 'Close' column found for {symbol}, using column 0")

            # Separate features and target
            X_data = data.copy()
            y_data = data[:, close_idx].copy()  # Target is the close price
            
            # Validate data
            if np.any(np.isnan(X_data)) or np.any(np.isinf(X_data)):
                print(f"Warning: Found NaN or inf values in data for {symbol}")
                return False
            
            # Scale features
            self.scaler_X = MinMaxScaler(feature_range=(0, 1))
            X_scaled = self.scaler_X.fit_transform(X_data)
            
            # Scale target separately
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
            y_scaled = self.scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()

            # Create sequences
            X, y = self._create_sequences(X_scaled, y_scaled)
            
            if len(X) == 0:
                print(f"Not enough data to create sequences for {symbol}")
                return False

            # Split data into train and validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Create and train model
            self.model = self.create_model((self.sequence_length, X_data.shape[1]))
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=50,  # Reduced for faster training
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=0,  # Reduced verbosity
                shuffle=False
            )

            self.save_model(symbol)
            print(f"Model trained successfully for {symbol}")
            return True
            
        except Exception as e:
            print(f"Error training model for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def _create_sequences(self, X_data, y_data):
        X, y = [], []
        for i in range(len(X_data) - self.sequence_length):
            X.append(X_data[i:i + self.sequence_length])
            y.append(y_data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def save_model(self, symbol):
        model_path = self._get_model_path(symbol)
        self.model.save(model_path)
        joblib.dump(self.scaler_X, self._get_scaler_X_path(symbol))
        joblib.dump(self.scaler_y, self._get_scaler_y_path(symbol))
        joblib.dump(self.feature_names, self._get_features_path(symbol))

    def load_model(self, symbol):
        model_path = self._get_model_path(symbol)
        scaler_X_path = self._get_scaler_X_path(symbol)
        scaler_y_path = self._get_scaler_y_path(symbol)
        features_path = self._get_features_path(symbol)

        if all(os.path.exists(path) for path in [model_path, scaler_X_path, scaler_y_path]):
            try:
                self.model = load_model(model_path)
                self.scaler_X = joblib.load(scaler_X_path)
                self.scaler_y = joblib.load(scaler_y_path)
                
                # Load feature names if available
                if os.path.exists(features_path):
                    self.feature_names = joblib.load(features_path)
                
                return True
            except Exception as e:
                print(f"Error loading model for {symbol}: {str(e)}")
                return False
        return False

    def predict_next_day(self, historical_data):
        """Predict the next day's price using the trained model"""
        try:
            # Prepare the data with features
            from routes import prepare_features
            data = prepare_features(historical_data)
            
            # More explicit checking to avoid Series ambiguity
            if data is None:
                return {"error": "Failed to prepare features from historical data"}
            
            if not isinstance(data, pd.DataFrame):
                return {"error": "Prepared data is not a DataFrame"}
            
            if data.empty:
                return {"error": "Prepared data is empty"}
            
            if len(data) < self.sequence_length:
                return {"error": f"Not enough data for prediction. Need {self.sequence_length}, got {len(data)}"}
            
            # Ensure we have the Close column
            if 'Close' not in data.columns:
                return {"error": "No 'Close' column found in prepared data"}
            
            # Get the last sequence for prediction
            last_sequence = data.values[-self.sequence_length:]
            
            # Validate the sequence
            if np.any(np.isnan(last_sequence)) or np.any(np.isinf(last_sequence)):
                return {"error": "Last sequence contains NaN or infinite values"}
            
            # Scale the sequence using the same scaler used during training
            try:
                last_sequence_scaled = self.scaler_X.transform(last_sequence)
            except Exception as e:
                return {"error": f"Failed to scale input data: {str(e)}"}
            
            # Reshape for LSTM input
            prediction_input = last_sequence_scaled.reshape(1, self.sequence_length, last_sequence.shape[1])
            
            # Make prediction (returns scaled value)
            try:
                predicted_price_scaled = self.model.predict(prediction_input, verbose=0)[0][0]
            except Exception as e:
                return {"error": f"Model prediction failed: {str(e)}"}
            
            # Inverse transform to get the actual price
            try:
                predicted_price = self.scaler_y.inverse_transform(
                    np.array([[predicted_price_scaled]])
                )[0][0]
            except Exception as e:
                return {"error": f"Failed to inverse transform prediction: {str(e)}"}
            
            # Current price (last available close price)
            current_price = float(data['Close'].iloc[-1])
            
            # Validate prices
            if predicted_price <= 0:
                return {"error": "Predicted price is not positive"}
            
            if current_price <= 0:
                return {"error": "Current price is not positive"}
            
            # Calculate predicted return
            predicted_return = (predicted_price - current_price) / current_price * 100
            
            # Determine recommendation
            recommendation = self._get_recommendation(predicted_return)
            
            # Extract symbol from historical data
            symbol = "Unknown"
            if isinstance(historical_data, pd.DataFrame):
                if 'Symbol' in historical_data.columns and not historical_data['Symbol'].empty:
                    symbol = str(historical_data['Symbol'].iloc[0])
                elif hasattr(historical_data, 'name') and historical_data.name:
                    symbol = str(historical_data.name)
            
            # Create prediction result
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "predicted_price": float(predicted_price),
                "predicted_return": float(predicted_return),
                "recommendation": recommendation,
                "confidence": self._calculate_confidence(predicted_return),
                "prediction_date": pd.Timestamp.now().strftime("%Y-%m-%d")
            }     
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Detailed error in predict_next_day: {error_details}")
            return {"error": f"Failed to make prediction: {str(e)}"}
        

    def _get_recommendation(self, predicted_return):
        """Generate trading recommendation based on predicted return"""
        if predicted_return > 5:
            return "STRONG BUY"
        elif predicted_return > 2:
            return "BUY"
        elif predicted_return > -2:
            return "HOLD"
        elif predicted_return > -5:
            return "SELL"
        else:
            return "STRONG SELL"

    def _calculate_confidence(self, predicted_return):
        """Calculate confidence level based on prediction magnitude"""
        abs_return = abs(predicted_return)
        if abs_return > 10:
            return "High"
        elif abs_return > 5:
            return "Medium"
        else:
            return "Low"

    def predict_multiple_days(self, historical_data, days=5):
        """Predict prices for multiple days ahead"""
        try:
            # Ensure days is an integer
            days = int(days)
            
            from routes import prepare_features
            data = prepare_features(historical_data)
            
            if data is None or len(data) < self.sequence_length:
                return {"error": "Not enough data for prediction"}
            
            predictions = []
            current_data = data.values.copy()
            
            for day in range(days):
                # Get the last sequence
                last_sequence = current_data[-self.sequence_length:]
                last_sequence_scaled = self.scaler_X.transform(last_sequence)
                prediction_input = last_sequence_scaled.reshape(1, self.sequence_length, last_sequence.shape[1])
                
                # Predict next day
                predicted_scaled = self.model.predict(prediction_input, verbose=0)[0][0]
                predicted_price = self.scaler_y.inverse_transform(np.array([[predicted_scaled]]))[0][0]
                
                predictions.append({
                    "day": day + 1,
                    "predicted_price": float(predicted_price),
                    "date": (pd.Timestamp.now() + pd.Timedelta(days=day+1)).strftime("%Y-%m-%d")
                })
                
                # Update current_data with prediction for next iteration
                # Create a new row with the predicted price
                new_row = current_data[-1].copy()
                # Close price is typically at index 0, but let's be sure by using column name
                close_index = 0  # Default 
                if hasattr(data, 'columns') and 'Close' in data.columns:
                    # Find the index of 'Close' in the dataframe columns
                    close_index = list(data.columns).index('Close')
                new_row[close_index] = predicted_price
                current_data = np.vstack([current_data, new_row])
            
            # Get symbol information safely
            symbol = "Unknown"
            if isinstance(historical_data, pd.DataFrame):
                if 'Symbol' in historical_data.columns and len(historical_data['Symbol']) > 0:
                    symbol = str(historical_data['Symbol'].iloc[0])
                elif hasattr(historical_data, 'name') and historical_data.name:
                    symbol = str(historical_data.name)
            
            return {
                "symbol": symbol,
                "current_price": float(data['Close'].iloc[-1]),
                "predictions": predictions
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error making multi-day prediction: {str(e)}")
            print(f"Detailed error: {error_details}")
            return {"error": f"Failed to make multi-day prediction: {str(e)}"}