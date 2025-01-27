import time
import requests
from datetime import datetime, timedelta
from web3 import Web3
import logging

# Configure logging
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize Web3 (optional, for PancakeSwap interaction)
bsc = Web3(Web3.HTTPProvider("https://bsc-dataseed.binance.org/"))

# Function to get live BNB/USD data
def get_market_data():
    try:
        url = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=BNBUSDT"
        url_volume = "https://api.binance.com/api/v3/ticker/24hr?symbol=BNBUSDT"
        response = requests.get(url)
        response_volume = requests.get(url_volume)

        data = response.json()
        volume_data = response_volume.json()

        # Extract relevant information
        bid_price = float(data["bidPrice"])
        ask_price = float(data["askPrice"])
        last_price = (bid_price + ask_price) / 2  # Approximation of last price
        volume = float(volume_data["volume"])

        return last_price, volume, bid_price, ask_price
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        return None, None, None, None

# Function to calculate a moving average
def moving_average(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

# Function to calculate RSI
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

# Prediction function
def predict_direction(price_history, volume_history, bid, ask):
    short_ma = moving_average(price_history, 3)
    long_ma = moving_average(price_history, 5)
    rsi = calculate_rsi(price_history, 14)
    bid_ask_diff = bid - ask

    if short_ma is None or long_ma is None or rsi is None:
        return "UP"

    if short_ma > long_ma and rsi < 70 and bid_ask_diff > 0:
        return "UP"
    elif short_ma < long_ma and rsi > 30 and bid_ask_diff < 0:
        return "DOWN"
    else:
        avg_volume = sum(volume_history[-5:]) / 5 if len(volume_history) >= 5 else 0
        return "UP" if avg_volume > volume_history[-1] else "DOWN"

# Function to save data
def save_data_to_csv(timestamp, price, volume, bid, ask, prediction):
    try:
        with open("price_data.csv", mode="a") as file:
            file.write(f"{timestamp},{price},{volume},{bid},{ask},{prediction}\n")
    except Exception as e:
        logging.error(f"Error saving data: {e}")

# Function to wait until 4:30 into the next 5-minute round
def wait_until_next_execution():
    now = datetime.now()
    seconds_to_next_execution = 270 - ((now.minute % 5) * 60 + now.second)
    if seconds_to_next_execution <= 0:
        seconds_to_next_execution += 300
    print(f"Waiting {seconds_to_next_execution} seconds until the next round...")
    time.sleep(seconds_to_next_execution)

# Main bot execution
def execute_bot():
    price_history = []
    volume_history = []

    price, volume, bid, ask = get_market_data()
    if price is not None and volume is not None:
        price_history.append(price)
        volume_history.append(volume)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Price: {price}, Volume: {volume}, Bid: {bid}, Ask: {ask}")
        
        if len(price_history) >= 2:
            prediction = predict_direction(price_history, volume_history, bid, ask)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: {prediction}")
            save_data_to_csv(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), price, volume, bid, ask, prediction)

# Start the bot
def start_bot():
    print("Starting prediction bot...")
    while True:
        try:
            wait_until_next_execution()
            execute_bot()
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    start_bot()

