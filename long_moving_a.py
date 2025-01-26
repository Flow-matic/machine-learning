import time
import requests
from datetime import datetime

# Function to get live BNB/USD price from Binance API
def get_price_data():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BNBUSDT"  # Binance API for BNB/USDT price
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return None

# Function to calculate a simple moving average
def moving_average(prices, period):
    if len(prices) < period:
        return None  # Not enough data to calculate SMA
    return sum(prices[-period:]) / period

# Function to predict UP or DOWN based on simple logic
def predict_direction(price_history):
    # Calculate short-term and long-term moving averages
    short_ma = moving_average(price_history, 3)  # Short-term SMA (3 periods)
    long_ma = moving_average(price_history, 5)   # Long-term SMA (5 periods)

    # Default prediction for the first few rounds
    if short_ma is None or long_ma is None:
        return "UP"

    # Predict based on moving averages
    if short_ma > long_ma:
        return "UP"
    elif short_ma < long_ma:
        return "DOWN"
    else:
        return "UP" if price_history[-1] > price_history[-2] else "DOWN"  # Fallback to last price trend

# Main function to run the bot
def start_bot():
    price_history = []  # Store historical prices
    round_time = 5 * 60  # 5 minutes in seconds
    buffer_time = 30  # Process data in the last 30 seconds of the round

    print("Starting prediction bot for BNB/USD...")
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

            # Step 4: Wait for the remaining 30 seconds before the next round starts
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for the next round...")
            time.sleep(buffer_time)

        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

# Run the bot
if __name__ == "__main__":
    start_bot()