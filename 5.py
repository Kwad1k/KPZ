import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

def get_binance_data(symbol: str, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
    """Download historical data from Binance."""
    client = Client()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    candles = client.get_historical_klines(
        symbol,
        timeframe,
        start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time.strftime("%Y-%m-%d %H:%M:%S")
    )
    data = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['close'] = data['close'].astype(float)
    data['high'] = data['high'].astype(float)
    data['low'] = data['low'].astype(float)
    data['volume'] = data['volume'].astype(float)
    return data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def compute_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""
    price_diff = data['close'].diff()
    gains = price_diff.where(price_diff > 0, 0).rolling(window=period).mean()
    losses = -price_diff.where(price_diff < 0, 0).rolling(window=period).mean()
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def compute_sma(data: pd.DataFrame, period: int) -> pd.Series:
    """Compute the Simple Moving Average (SMA)."""
    return data['close'].rolling(window=period).mean()

def compute_bollinger_bands(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    sma = compute_sma(data, period)
    std_dev = data['close'].rolling(window=period).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return pd.DataFrame({'lower_band': lower_band, 'middle_band': sma, 'upper_band': upper_band})

def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR)."""
    range_high_low = data['high'] - data['low']
    range_high_close = (data['high'] - data['close'].shift()).abs()
    range_low_close = (data['low'] - data['close'].shift()).abs()
    true_range = pd.concat([range_high_low, range_high_close, range_low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def enrich_data_with_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the DataFrame."""
    data['RSI'] = compute_rsi(data)
    data['SMA_50'] = compute_sma(data, 50)
    data['SMA_200'] = compute_sma(data, 200)
    bollinger_bands = compute_bollinger_bands(data)
    data[['lower_band', 'middle_band', 'upper_band']] = bollinger_bands
    data['ATR'] = compute_atr(data)
    return data

def export_to_csv(data: pd.DataFrame, filename: str) -> None:
    """Export the DataFrame to a CSV file."""
    data.to_csv(filename, index=False)
    print(f"Data has been saved to {filename}")

if __name__ == "__main__":
    asset = "BTCUSDT"
    timeframe = Client.KLINE_INTERVAL_1HOUR
    historical_data = get_binance_data(asset, timeframe)
    historical_data = enrich_data_with_indicators(historical_data)
    export_to_csv(historical_data, "technical_analysis_output.csv")
    print(historical_data.tail())
