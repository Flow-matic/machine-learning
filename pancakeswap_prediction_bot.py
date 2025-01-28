import time
import requests
from datetime import datetime, timezone
from web3 import Web3
import logging

logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

bsc = Web3(Web3.HTTPProvider("https://bsc-dataseed.binance.org/"))

# Store historical data globally
price_history = []
volume_history = []

def get_market_data():
    try:
        url = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=BNBUSDT"
        url_volume = "https://api.binance.com/api/v3/ticker/24hr?symbol=BNBUSDT"
        response = requests.get(url)
        response_volume = requests.get(url_volume)

        data = response.json()
        volume_data = response_volume.json()

        bid_price = float(data["bidPrice"])
        ask_price = float(data["askPrice"])
        last_price = (bid_price + ask_price) / 2
        volume = float(volume_data["volume"])

        return last_price, volume, bid_price, ask_price
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        return None, None, None, None

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

def predict_direction(price_history, volume_history, bid, ask):
    short_ma = moving_average(price_history, 3)
    long_ma = moving_average(price_history, 5)
    rsi = calculate_rsi(price_history, 14)

    if short_ma is None or long_ma is None or rsi is None:
        return None

    bid_ask_diff = bid - ask

    if short_ma > long_ma and rsi < 70 and bid_ask_diff > 0:
        return "UP"
    elif short_ma < long_ma and rsi > 30 and bid_ask_diff < 0:
        return "DOWN"
    else:
        avg_volume = sum(volume_history[-5:]) / 5 if len(volume_history) >= 5 else 0
        return "UP" if avg_volume > volume_history[-1] else "DOWN"

def save_data_to_csv(timestamp, price, volume, bid, ask, prediction):
    try:
        with open("price_data.csv", mode="a") as file:
            file.write(f"{timestamp},{price},{volume},{bid},{ask},{prediction}\n")
    except Exception as e:
        logging.error(f"Error saving data to CSV: {e}")

def get_next_round_start():
    now = datetime.now(timezone.utc)
    seconds_to_next_round = 300 - ((now.minute % 5) * 60 + now.second)
    return seconds_to_next_round

def execute_bot():
    global price_history, volume_history
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running the prediction bot...")

    price, volume, bid, ask = get_market_data()
    if price is not None and volume is not None:
        price_history.append(price)
        volume_history.append(volume)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Current Price: {price}, Volume: {volume}, Bid: {bid}, Ask: {ask}")

        if len(price_history) >= 5:  # Ensure enough data points for prediction
            prediction = predict_direction(price_history, volume_history, bid, ask)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: {prediction}")
            save_data_to_csv(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), price, volume, bid, ask, prediction)
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Not enough data for prediction.")
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

if __name__ == "__main__":
    start_bot()






