import time
import requests
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Function to get live BNB/USD price from Binance API
def get_price_data():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BNBUSDT"  # Binance API for BNB/USDT price
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        logging.error(f"Error fetching price data: {e}")
        return None

# Function to calculate a simple moving average
def moving_average(prices, period):
    if len(prices) < period:
        return None  # Not enough data to calculate SMA
    return sum(prices[-period:]) / period

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return None
    gains = [prices[i] - prices[i - 1] for i in range(1, len(prices)) if prices[i] > prices[i - 1]]
    losses = [prices[i - 1] - prices[i] for i in range(1, len(prices)) if prices[i] < prices[i - 1]]
    avg_gain = sum(gains[-period:]) / period if gains else 0
    avg_loss = sum(losses[-period:]) / period if losses else 0
    if avg_loss == 0:
        return 100  # Max RSI when no losses
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Function to make a prediction using multiple indicators
def predict_direction(price_history):
    # Calculate indicators
    short_ma = moving_average(price_history, 3)  # Short-term SMA (3 periods)
    long_ma = moving_average(price_history, 5)   # Long-term SMA (5 periods)
    rsi = calculate_rsi(price_history, 14)      # RSI with a 14-period window

    # Ensure enough data for predictions
    if short_ma is None or long_ma is None or rsi is None:
        return "UP"  # Default prediction

    # Enhanced prediction logic
    if short_ma > long_ma and rsi < 70:
        return "UP"
    elif short_ma < long_ma and rsi > 30:
        return "DOWN"
    else:
        return "UP" if rsi < 50 else "DOWN"

# Function to save data to a CSV file
def save_data_to_csv(timestamp, price, prediction):
    try:
        with open("price_data.csv", mode="a") as file:
            file.write(f"{timestamp},{price},{prediction}\n")
    except Exception as e:
        logging.error(f"Error saving data to CSV: {e}")

# Main function to run the bot
def start_bot():
    price_history = []  # Store historical prices
    round_time = 5 * 60  # 5 minutes in seconds
    buffer_time = 30  # Process data in the last 30 seconds of the round

    print("Starting prediction bot for BNB/USD...")
    logging.info("Bot started.")
    
    while True:
        try:
            # Step 1: Wait for 4 minutes 30 seconds into the round
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for 4 minutes 30 seconds...")
            time.sleep(round_time - buffer_time)  # Sleep for 4 minutes 30 seconds

            # Step 2: Fetch current price
            price = get_price_data()
            if price is not None:
                price_history.append(price)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Current Price: {price}")

            # Step 3: Predict UP or DOWN
            if len(price_history) >= 2:  # Ensure we have enough data
                prediction = predict_direction(price_history)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: {prediction}")
                save_data_to_csv(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), price, prediction)
                logging.info(f"Price: {price}, Prediction: {prediction}")

            # Step 4: Wait for the remaining 30 seconds before the next round starts
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for the next round...")
            time.sleep(buffer_time)

        except KeyboardInterrupt:
            print("Bot stopped by user.")
            logging.info("Bot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}")

# Run the bot
if __name__ == "__main__":
    start_bot()