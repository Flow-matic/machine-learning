import requests
import time
import csv
import matplotlib.pyplot as plt

# Replace with your API keys and secrets
CRYPTO_API_URL = "https://api.binance.com/api/v3/ticker/price"

# Fetch real-time cryptocurrency data
def fetch_crypto_data(symbol):
    try:
        response = requests.get(CRYPTO_API_URL, params={"symbol": symbol})
        response.raise_for_status()
        data = response.json()
        price = float(data["price"])
        return price
    except Exception as e:
        print(f"Error fetching cryptocurrency data: {e}")
        return None

# Save data to CSV
def save_to_csv(timestamp, price, filename="crypto_prices.csv"):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, price])

# Calculate moving average
def calculate_moving_average(prices, window_size=5):
    if len(prices) >= window_size:
        return sum(prices[-window_size:]) / window_size
    return None

# Plot prices in real time
def plot_prices(prices):
    plt.clf()
    plt.plot(prices, label="Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Real-Time Cryptocurrency Prices")
    plt.legend()
    plt.pause(0.1)

# Main bot function
def trading_bot():
    crypto_symbol = "BNBUSDT"  # Corrected symbol for Binance Coin in USD
    prices = []  # Store recent prices
    start_time = time.time()
    duration = 120  # Time in seconds to collect data

    # Set thresholds
    upper_limit = 320
    lower_limit = 300

    plt.ion()  # Enable interactive mode for real-time plotting

    while time.time() - start_time < duration:
        # Fetch cryptocurrency data
        crypto_price = fetch_crypto_data(crypto_symbol)
        if crypto_price is not None:
            print(f"Cryptocurrency ({crypto_symbol}) Price: {crypto_price}")

            # Save to CSV
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            save_to_csv(timestamp, crypto_price)

            # Track prices and calculate moving average
            prices.append(crypto_price)
            moving_average = calculate_moving_average(prices)
            if moving_average:
                print(f"Moving Average: {moving_average}")

            # Check price alert
            if crypto_price >= upper_limit:
                print("Alert: Price has crossed the upper limit!")
            elif crypto_price <= lower_limit:
                print("Alert: Price has fallen below the lower limit!")

            # Real-time plotting
            plot_prices(prices)

        time.sleep(10)  # Adjust the frequency as needed

    plt.ioff()  # Disable interactive mode
    plt.show()

if __name__ == "__main__":
    trading_bot()