import time
import requests
from datetime import datetime, timezone
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Binance API endpoint
URL = "https://api.binance.com/api/v3/ticker/24hr?symbol=BNBUSDT"

# Initialize data structures
price_history = []
features = []
labels = []
model = None

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

def calculate_technical_indicators(prices):
    """Calculates technical indicators for the given price history."""
    short_ma = moving_average(prices, 3)
    long_ma = moving_average(prices, 5)
    rsi = calculate_rsi(prices, 14)
    macd_line, signal_line = calculate_macd(prices)

    return short_ma, long_ma, rsi, macd_line[-1], signal_line[-1]

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

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    if len(prices) < max(short_period, long_period, signal_period):
        return None, None
    short_ema = [None] * (short_period - 1)
    long_ema = [None] * (long_period - 1)
    for i in range(short_period - 1):
        short_ema.append(prices[i])
    for i in range(long_period - 1):
        long_ema.append(prices[i])
    for i in range(short_period - 1, len(prices)):
        if i == short_period - 1:
            short_ema.append(sum(prices[:short_period]) / short_period)
        else:
            short_ema.append((prices[i] - short_ema[-1]) * (2 / (short_period + 1)) + short_ema[-1])
    for i in range(long_period - 1, len(prices)):
        if i == long_period - 1:
            long_ema.append(sum(prices[:long_period]) / long_period)
        else:
            long_ema.append((prices[i] - long_ema[-1]) * (2 / (long_period + 1)) + long_ema[-1])
    macd_line = [short_ema[i] - long_ema[i] for i in range(len(short_ema))]
    signal_line = [None] * (signal_period - 1)
    for i in range(signal_period - 1, len(macd_line)):
        signal_line.append(sum(macd_line[i - signal_period + 1:i + 1]) / signal_period)
    return macd_line, signal_line

def train_model(features, labels):
    """Trains a machine learning model to predict price direction."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {accuracy}")
    return model

def predict_direction(model, price_history):
    """Predicts the price direction using the trained model."""
    if len(price_history) < max(14, 12, 26, 9):  # Ensure enough data for all indicators
        return None

    try:
        features = calculate_technical_indicators(price_history)
        prediction = model.predict([features])[0]
        if prediction == 1:
            return "UP"
        elif prediction == 0:
            return "DOWN"
        else:
            return None
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None

def get_next_round_start():
    now = datetime.now(timezone.utc)
    seconds_to_next_round = 300 - ((now.minute % 5) * 60 + now.second)
    return seconds_to_next_round

def execute_bot():
    global price_history, features, labels, model

    price, volume, open_price, high, low = get_market_data()
    if price is not None and volume is not None:
        price_history.append(price)

        if len(price_history) >= max(14, 12, 26, 9):
            try:
                # Calculate features and label
                features_row = calculate_technical_indicators(price_history)
                # Determine label based on price change
                label = 1 if price_history[-1] > price_history[-2] else 0  # 1 for UP, 0 for DOWN

                # Update features and labels for training
                features.append(features_row)
                labels.append(label)

                # Train the model if enough data is available
                if len(features) >= 50:  # Adjust this threshold as needed
                    model = train_model(features, labels)

                # Make prediction using the trained model
                prediction = predict_direction(model, price_history)

                if prediction:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: {prediction}")
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Not enough data for prediction.")

            except Exception as e:
                logging.error(f"Error processing data: {e}")

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Current Price: {price}, Volume: {volume}, Open: {open_price}, High: {high}, Low: {low}")
        print(f"Price history length: {len(price_history)}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: Could not retrieve market data.")

def start_bot():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for the next round...")
    while True:
        try:
            seconds_to_next_round = get_next_round_start()
            print(f"Seconds to next round start: {seconds_to_next_round} seconds.")
            time.sleep(seconds_to_next_round)
            execute_bot()
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}")