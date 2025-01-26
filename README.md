# PancakeSwap Prediction Bot for BNB/USD

This is a Python-based bot for the PancakeSwap Prediction market. It uses **live BNB/USD price data** from Binance's API and applies a **simple moving average (SMA) strategy** to predict the next round's outcome as **UP** or **DOWN**.

## Features

- **Live Price Data**: Fetches real-time BNB/USD prices from Binance API.
- **Simple Moving Average Strategy**: Utilizes short-term and long-term moving averages to determine price trends.
- **Timed Predictions**: Waits until the final 30 seconds of a 5-minute round to analyze data and make predictions.
- **Adjustable Timing**: Ensures predictions are based on the freshest data, optimizing accuracy.

## How It Works

1. **Data Collection**:
   - Fetches the current BNB/USD price every 5 minutes.
   - Stores price history in memory for trend analysis.

2. **Prediction Logic**:
   - **Short-Term SMA (3 periods)** and **Long-Term SMA (5 periods)** are calculated.
   - If the short-term SMA > long-term SMA, predicts **UP**.
   - If the short-term SMA < long-term SMA, predicts **DOWN**.
   - Falls back to the latest price trend if SMAs are equal.

3. **Timing**:
   - Waits for 4 minutes and 30 seconds into the current 5-minute round.
   - Fetches data, performs analysis, and predicts the next round’s outcome.
   - Waits for 30 seconds for the round to reset and starts again.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/pancakeswap-prediction-bot.git
   cd pancakeswap-prediction-bot

2. Install dependencies:
   <br>pip install requests<br>

3. Run the bot:
<br>python prediction_bot.py<br>

## Requirements 
- Python 3.x
- requests library (for API calls)
- Binance API access for live price data

# Configuration
- The bot is configured to fetch prices for BNB/USD (BNBUSDT pair).
- Modify the round_time and buffer_time values in the script to adjust the bot’s timing:
- round_time = 5 * 60: The duration of each prediction round (5 minutes).
- buffer_time = 30: Processes data in the last 30 seconds of the round.

[2025-01-26 15:00:00] Waiting for 4 minutes 30 seconds...
[2025-01-26 15:04:30] Current Price: 318.42
[2025-01-26 15:04:30] Prediction: UP
[2025-01-26 15:05:00] Waiting for the next round...

# Strategy Improvements
- Technical Indicators:
- Add indicators like RSI (Relative Strength Index) or Bollinger Bands for more robust predictions.
- Machine Learning:
- Train a model using historical price data to predict outcomes.
- Integration with PancakeSwap:
- Use web3.py to interact with the PancakeSwap smart contract and place predictions automatically.

# Disclaimer
- This bot is for educational purposes only. The PancakeSwap Prediction market is highly speculative, and this script does not guarantee profits. Use at your own risk.

# Contributions
- Feel free to fork this repository and suggest improvements via pull requests.

# License
- This project is licensed under the MIT License. See the LICENSE file for details.

# Future Enhancements
1. Web3 Integration: Automate prediction placements on PancakeSwap.
2. Enhanced Strategies: Use more advanced technical analysis or machine learning models.
3. Logging: Save predictions and outcomes to a database for analysis.

# Author

- Created by Jonathan FlowMatic
- Inspired by the PancakeSwap Prediction market
