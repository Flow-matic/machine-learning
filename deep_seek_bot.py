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
import optuna
import joblib

# Binance API endpoint
URL = "https://api.binance.com/api/v3/ticker/24hr?symbol=BNBUSDT"

# Initialize data
price_history = []
volume_history = []
features = []
labels = []
model = None
scaler = None  # Initialize scaler to None

# Logging configuration
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_market_data():
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        print("API Response:", data)  # Debugging
        return {
            "last_price": float(data["lastPrice"]),
            "volume": float(data["volume"]),
            "high": float(data["highPrice"]),
            "low": float(data["lowPrice"]),
            "close": float(data["lastPrice"])
        }
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        print(f"Error fetching market data: {e}")
        return None

def execute_bot():
    global model, scaler
    
    market_data = get_market_data()
    if market_data:
        price_history.append(market_data["last_price"])
        volume_history.append(market_data["volume"])

        print(f"Price history length: {len(price_history)}, Volume history length: {len(volume_history)}")

        if len(price_history) >= 10:  # Reduce data collection requirement
            features_row = {"price": price_history[-1], "volume": volume_history[-1]}  # Basic features
            features.append(list(features_row.values()))
            labels.append(1 if price_history[-1] > price_history[-2] else 0)

            if model is None and len(features) >= 10:
                train_model()

            if model and scaler:
                try:
                    features_scaled = scaler.transform([list(features_row.values())])
                    prediction = model.predict(features_scaled)[0]

                    direction = "UP" if prediction > 0.5 else "DOWN"
                    confidence = abs(prediction - 0.5) * 2

                    now = datetime.now()
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

                    print("=" * 50)
                    print(f"[{timestamp}] BNB Price Prediction")
                    print("=" * 50)
                    print(f"Current Price: {market_data['last_price']:.2f}")
                    print(f"Prediction: BNB will go {direction} in the next epoch.")
                    print(f"Confidence: {confidence * 100:.2f}%")
                    print("=" * 50)
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    logging.exception("Error during prediction")
            else:
                print("Model or scaler not yet trained or loaded. Skipping prediction.")

def train_model():
    global model, scaler
    if len(features) < 10:
        print("Not enough data to train yet.")
        return

    X = np.array(features)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LGBMRegressor()
    model.fit(X_train_scaled, y_train)
    joblib.dump((model, scaler), "bnb_model.pkl")
    print("Model trained and saved.")

if __name__ == "__main__":
    print(f"[{datetime.now()}] Bot started.")
    while True:
        try:
            execute_bot()
            time.sleep(240)  # Run every 4 minutes to predict before the next epoch
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Error: {e}")