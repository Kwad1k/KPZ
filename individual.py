import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate random data for stock price simulation
np.random.seed(42)  # For reproducibility
num_points = 300  # Number of data points (days)
base_price = 100  # Starting price
price_changes = np.random.normal(0, 2, num_points)  # Random price changes
close_prices = base_price + np.cumsum(price_changes)  # Cumulative sum for close prices

# Create DataFrame with random data
data = pd.DataFrame({
    "Time": pd.date_range(start="2025-01-01", periods=num_points, freq="D"),
    "Close": close_prices,
    "High": close_prices + np.random.uniform(0.5, 1.5, num_points),  # High prices
    "Low": close_prices - np.random.uniform(0.5, 1.5, num_points),  # Low prices
})

# Calculate Indicators

# Calculate RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate EMA (Exponential Moving Average)
def calculate_ema(data, period=20):
    return data['Close'].ewm(span=period, adjust=False).mean()

# Calculate ATR (Average True Range)
def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# Calculate technical indicators
rsi_length = 14
ema_short_length = 20
ema_long_length = 50
atr_length = 14

data['RSI'] = calculate_rsi(data, rsi_length)
data['EMA_short'] = calculate_ema(data, ema_short_length)
data['EMA_long'] = calculate_ema(data, ema_long_length)
data['ATR'] = calculate_atr(data, atr_length)

# Risk and reward parameters (SL/TP in %)
risk = 1.5  # Stop loss in %
reward = 3.0  # Take profit in %

# Calculate Stop Loss (SL) and Take Profit (TP) levels
data['Long_StopLoss'] = data['Close'] * (1 - risk / 100)
data['Long_TakeProfit'] = data['Close'] * (1 + reward / 100)
data['Short_StopLoss'] = data['Close'] * (1 + risk / 100)
data['Short_TakeProfit'] = data['Close'] * (1 - reward / 100)

# Entry conditions for Long and Short trades
data['Long_Signal'] = (data['RSI'] < 40) & (data['EMA_short'] > data['EMA_long'])
data['Short_Signal'] = (data['RSI'] > 60) & (data['EMA_short'] < data['EMA_long'])

# Initialize balance and trade history
initial_balance = 10000  # Starting balance
balance = initial_balance
trade_history = []

# Simulate trading
for i in range(1, len(data)):
    # Long entry condition
    if data['Long_Signal'][i] and balance > data['Close'][i]:
        entry_price = data['Close'][i]
        stop_loss = data['Long_StopLoss'][i]
        take_profit = data['Long_TakeProfit'][i]
        balance -= entry_price  # Spend balance on buying
        trade_history.append({
            'Type': 'Long', 'Entry_Price': entry_price, 'Stop_Loss': stop_loss, 'Take_Profit': take_profit, 'Exit_Price': None
        })
    # Short entry condition
    elif data['Short_Signal'][i] and balance > data['Close'][i]:
        entry_price = data['Close'][i]
        stop_loss = data['Short_StopLoss'][i]
        take_profit = data['Short_TakeProfit'][i]
        balance -= entry_price  # Spend balance on selling
        trade_history.append({
            'Type': 'Short', 'Entry_Price': entry_price, 'Stop_Loss': stop_loss, 'Take_Profit': take_profit, 'Exit_Price': None
        })
    
    # Check if stop loss or take profit levels are hit
    for trade in trade_history:
        if trade['Exit_Price'] is None:  # If trade is not closed yet
            if trade['Type'] == 'Long' and (data['Low'][i] <= trade['Stop_Loss'] or data['High'][i] >= trade['Take_Profit']):
                trade['Exit_Price'] = data['Close'][i]
                balance += trade['Exit_Price']
            elif trade['Type'] == 'Short' and (data['High'][i] >= trade['Stop_Loss'] or data['Low'][i] <= trade['Take_Profit']):
                trade['Exit_Price'] = data['Close'][i]
                balance += trade['Exit_Price']

# Calculate final balance and total profit
final_balance = balance
total_profit = final_balance - initial_balance

# Print results
print(f'Initial Balance: {initial_balance}')
print(f'Final Balance: {final_balance}')
print(f'Total Profit: {total_profit}')

# Plotting the price chart with buy and sell signals
plt.figure(figsize=(12, 6))

# Plot Close price
plt.plot(data['Time'], data['Close'], label='Close Price', color='blue', alpha=0.7)

# Plot buy and sell signals
buy_signals = data[data['Long_Signal']]
sell_signals = data[data['Short_Signal']]

plt.scatter(buy_signals['Time'], buy_signals['Close'], marker='^', color='green', label='Buy Signal', alpha=1)
plt.scatter(sell_signals['Time'], sell_signals['Close'], marker='v', color='red', label='Sell Signal', alpha=1)

# Adding labels and title
plt.title("Stock Price and Trading Signals")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
