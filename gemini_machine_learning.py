import time
import requests
from datetime import datetime, timezone
import logging
import pandas as pd
from xgboost import XGBRegressor
import numpy as np

logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Binance API endpoint
URL = "https://api.binance.com/api/v3/ticker/24hr?symbol=BNBUSDT"

# Initialize data structures
price_history = []
features = []
labels = []
model = None

# Recommended: Define hyperparameters for XGBoost
xgb_params = {
    'n_estimators': 100,  # Number of boosting rounds
    'learning_rate': 0.1, 
    'max_depth': 3, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    'random_state': 42
}

def get_market_data():
    try:
        response = requests.get(URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        last_price = float(data["lastPrice"])
        volume = float(data["volume"])
        open_price = float(data["openPrice"])
        high = float(data["highPrice"])
        low = float(data["lowPrice"])

        return last_price, volume, open_price, high, low
    except (requests.exceptions.RequestException, ValueError) as e:
        logging.error(f"Error fetching market data: {e}")
        return None, None, None, None, None

def moving_average(prices, period):
    """Calculates a simple moving average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    if len(prices) < period:
        return None
    gains = [prices[i] - prices[i - 1] for i in range(1, len(prices)) if prices[i] > prices[i - 1]]
    losses = [prices[i - 1] - prices[i] for i in range(1, len(prices)) if prices[i] < prices[i - 1]]
    avg_gain = sum(gains[-period:]) / period if gains else 0
    avg_loss = sum(losses[-period:]) / period if losses else 0
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates MACD and Signal Line."""
    if len(prices) < max(short_period, long_period, signal_period):
        return None, None
    short_ma = moving_average(prices[-short_period:], short_period)
    long_ma = moving_average(prices[-long_period:], long_period)
    macd_line = short_ma - long_ma if short_ma and long_ma else None
    signal_line = moving_average(prices[-signal_period:], signal_period) if macd_line else None
    return macd_line, signal_line

def calculate_volatility(prices, window=20):
    """Calculates volatility using standard deviation."""
    if len(prices) < window:
        return None
    return np.std(prices[-window:])

def calculate_technical_indicators(prices):
    """Calculates technical indicators for the given price history."""
    short_ma = moving_average(prices, 3)
    long_ma = moving_average(prices, 5)
    rsi = calculate_rsi(prices, 14)
    macd_line, signal_line = calculate_macd(prices)
    volatility = calculate_volatility(prices) 

    return short_ma, long_ma, rsi, macd_line, signal_line, volatility

def train_model(features, labels):
    """Trains an XGBoost model to predict price direction."""
    model = XGBRegressor(**xgb_params)  # Use defined hyperparameters
    model.fit(features, labels)
    return model

def predict_direction(model, price_history):
    """Predicts the price direction using the trained model."""
    if len(price_history) < max(14, 12, 26, 9):  # Ensure enough data for all indicators
        return None

    try:
        features = calculate_technical_indicators(price_history)
        if None not in features:
            prediction = model.predict([features])[0]
            return "UP" if prediction > 0.5 else "DOWN" 
        else:
            return None
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None

def execute_bot():
    global price_history, features, labels, model

    price, volume, open_price, high, low = get_market_data()
    if price is not None and volume is not None:
        price_history.append(price)

        if len(price_history) >= max(14, 12, 26, 9):
            try:
                # Calculate features and label
                features_row = calculate_technical_indicators(price_history)
                if None not in features_row:
                    label = 1 if price_history[-1] > price_history[-2] else 0  # 1 for UP, 0 for DOWN
                    features.append(features_row)
                    labels.append(label)

                # Train the model if enough data is available
                if len(features) >= 50:  # Increase training data threshold
                    model = train_model(features, labels)

                # Make prediction using the trained model
                if model:
                    prediction = predict_direction(model, price_history)
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: {prediction}")

            except Exception as e:
                logging.error(f"Error processing data: {e}")

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Current Price: {price}, Volume: {volume}, Open: {open_price}, High: {high}, Low: {low}")
        print(f"Price history length: {len(price_history)}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: Could not retrieve market data.")

def start_bot():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bot started.")
    while True:
        try:
            execute_bot()
            time.sleep(5)  # Adjusted for faster testing
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    start_bot()