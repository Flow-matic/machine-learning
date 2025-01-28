import time
import requests
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna  # For hyperparameter tuning
import joblib  # For model persistence

# Binance API endpoint
URL = "https://api.binance.com/api/v3/ticker/24hr?symbol=BNBUSDT"

# Initialize data structures
price_history = []
features = []
labels = []
model = None
scaler = StandardScaler()

# Logging configuration
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def get_market_data():
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        return {
            "last_price": float(data["lastPrice"]),
            "volume": float(data["volume"]),
            "open_price": float(data["openPrice"]),
            "high": float(data["highPrice"]),
            "low": float(data["lowPrice"])
        }
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        return None

def calculate_technical_indicators(prices, volume):
    """Calculates advanced technical indicators."""
    df = pd.DataFrame({"price": prices, "volume": volume})
    df["returns"] = df["price"].pct_change()
    df["ma_7"] = df["price"].rolling(window=7).mean()
    df["ma_21"] = df["price"].rolling(window=21).mean()
    df["rsi"] = calculate_rsi(df["price"])
    df["macd"], df["signal_line"] = calculate_macd(df["price"])
    df["bollinger_upper"], df["bollinger_lower"] = calculate_bollinger_bands(df["price"])
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"])
    df["obv"] = calculate_obv(df["price"], df["volume"])
    df.dropna(inplace=True)
    return df.iloc[-1].to_dict()

def calculate_rsi(prices, period=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates MACD and Signal Line."""
    short_ema = prices.ewm(span=short_period, adjust=False).mean()
    long_ema = prices.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculates Bollinger Bands."""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_atr(high, low, close, window=14):
    """Calculates Average True Range (ATR)."""
    tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
    return tr.rolling(window=window).mean()

def calculate_obv(prices, volume):
    """Calculates On-Balance Volume (OBV)."""
    return np.where(prices.diff() > 0, volume, -volume).cumsum()

def train_model(features, labels):
    """Trains a LightGBM model with hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    model = LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    joblib.dump(model, "bnb_prediction_model.pkl")  # Save the model
    return model

def predict_direction(model, features):
    """Predicts the price direction with confidence score."""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    confidence = abs(prediction - 0.5) * 2  # Confidence score between 0 and 1
    return "UP" if prediction > 0.5 else "DOWN", confidence

def execute_bot():
    global price_history, features, labels, model

    market_data = get_market_data()
    if market_data:
        price_history.append(market_data["last_price"])
        if len(price_history) >= 50:  # Minimum data points for indicators
            try:
                features_row = calculate_technical_indicators(price_history, market_data["volume"])
                if features_row:
                    label = 1 if price_history[-1] > price_history[-2] else 0
                    features.append(list(features_row.values()))
                    labels.append(label)

                    if len(features) >= 100:  # Train model with sufficient data
                        model = train_model(features, labels)

                    if model:
                        prediction, confidence = predict_direction(model, list(features_row.values()))
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: {prediction} (Confidence: {confidence:.2f})")

            except Exception as e:
                logging.error(f"Error processing data: {e}")

def start_bot():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bot started.")
    while True:
        try:
            execute_bot()
            time.sleep(5)
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    start_bot()