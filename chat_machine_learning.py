import time
import requests
from datetime import datetime, timezone
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Binance API endpoint
URL = "https://api.binance.com/api/v3/ticker/24hr?symbol=BNBUSDT"

# Initialize data structures
price_history = []
features = []
labels = []
model = None
scaler = StandardScaler()

# Technical Indicators
def get_market_data():
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()

        last_price = float(data["lastPrice"])
        volume = float(data["volume"])
        return last_price, volume
    except (requests.exceptions.RequestException, ValueError) as e:
        logging.error(f"Error fetching market data: {e}")
        return None, None

def calculate_technical_indicators(prices):
    short_ma = moving_average(prices, 3)
    long_ma = moving_average(prices, 5)
    rsi = calculate_rsi(prices, 14)
    atr = calculate_atr(prices)
    upper_bb, lower_bb = calculate_bollinger_bands(prices, 20)
    return short_ma, long_ma, rsi, atr, upper_bb, lower_bb

def moving_average(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_rsi(prices, period=14):
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

def calculate_atr(prices, period=14):
    if len(prices) < period:
        return None
    true_ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    return sum(true_ranges[-period:]) / period

def calculate_bollinger_bands(prices, period=20):
    if len(prices) < period:
        return None, None
    sma = moving_average(prices, period)
    std_dev = np.std(prices[-period:])
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    return upper_band, lower_band

# Machine Learning Model
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Scale features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, predictions))

    return best_model

def predict_direction(model, features_row):
    try:
        scaled_features = scaler.transform([features_row])
        prediction = model.predict(scaled_features)[0]
        return "UP" if prediction == 1 else "DOWN"
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None

def execute_bot():
    global price_history, features, labels, model

    price, volume = get_market_data()
    if price is not None and volume is not None:
        price_history.append(price)

        if len(price_history) >= 20:  # Ensure enough data for all indicators
            try:
                # Calculate features and label
                features_row = calculate_technical_indicators(price_history)
                if None not in features_row:
                    label = 1 if (price / price_history[-2] - 1) > 0.001 else 0  # 0.1% threshold

                    # Update features and labels
                    features.append(features_row)
                    labels.append(label)

                    # Train the model if enough data is available
                    if len(features) >= 100:  # Minimum training set size
                        model = train_model(features, labels)

                    # Make prediction using the trained model
                    if model:
                        prediction = predict_direction(model, features_row)
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: {prediction}")

            except Exception as e:
                logging.error(f"Error processing data: {e}")

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Current Price: {price}, Volume: {volume}")

def start_bot():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for the next round...")
    while True:
        try:
            execute_bot()
            time.sleep(300)  # Run every 5 minutes
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    start_bot()