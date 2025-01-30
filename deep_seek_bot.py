import time
import requests
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

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

def load_model():
    global model, scaler
    if os.path.exists("bnb_model.h5") and os.path.exists("scaler.pkl"):
        model = create_model()
        model.load_weights("bnb_model.h5")
        scaler = joblib.load("scaler.pkl")
        print("Model and scaler loaded from file.")
    else:
        print("No saved model found. Training required.")

def save_data():
    np.save("price_history.npy", price_history)
    np.save("volume_history.npy", volume_history)
    np.save("features.npy", features)
    np.save("labels.npy", labels)
    print("Historical data saved.")

def load_data():
    global price_history, volume_history, features, labels
    if os.path.exists("price_history.npy"):
        price_history = np.load("price_history.npy").tolist()
    if os.path.exists("volume_history.npy"):
        volume_history = np.load("volume_history.npy").tolist()
    if os.path.exists("features.npy"):
        features = np.load("features.npy").tolist()
    if os.path.exists("labels.npy"):
        labels = np.load("labels.npy").tolist()
    print("Historical data loaded.")

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

def calculate_technical_indicators():
    if len(price_history) < 21:
        return None
    
    df = pd.DataFrame({"price": price_history, "volume": volume_history})
    df["returns"] = df["price"].pct_change()
    df["ma_7"] = df["price"].rolling(window=7).mean()
    df["ma_21"] = df["price"].rolling(window=21).mean()
    df["rsi"] = calculate_rsi(df["price"])
    df["macd"], df["signal_line"] = calculate_macd(df["price"])
    df.dropna(inplace=True)
    return df.iloc[-1].to_dict()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    short_ema = prices.ewm(span=short_period, adjust=False).mean()
    long_ema = prices.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

def create_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(1, 6)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def execute_bot():
    global model, scaler
    
    market_data = get_market_data()
    if market_data:
        price_history.append(market_data["last_price"])
        volume_history.append(market_data["volume"])

        print(f"Price history length: {len(price_history)}, Volume history length: {len(volume_history)}")

        if len(price_history) >= 21:
            features_row = calculate_technical_indicators()
            if features_row:
                label = 1 if price_history[-1] > price_history[-2] else 0
                features.append(list(features_row.values()))
                labels.append(label)
                save_data()

                if model is None and len(features) >= 50:
                    train_model()

                if model and scaler:
                    try:
                        features_scaled = scaler.transform([list(features_row.values())]).reshape(1, 1, -1)
                        prediction = model.predict(features_scaled)[0][0]

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

def train_model():
    global model, scaler
    if len(features) < 50:
        print("Not enough data to train yet.")
        return

    X = np.array(features)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train).reshape(-1, 1, 6)
    X_test_scaled = scaler.transform(X_test).reshape(-1, 1, 6)

    model = create_model()
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=8, verbose=1)
    model.save_weights("bnb_model.h5")
    joblib.dump(scaler, "scaler.pkl")
    print("Model trained and saved.")

if __name__ == "__main__":
    load_data()
    load_model()
    print(f"[{datetime.now()}] Bot started.")
    while True:
        try:
            execute_bot()
            time.sleep(240)
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Error: {e}")