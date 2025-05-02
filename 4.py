import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

def calculate_rsi(data, period):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_binance_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE):
    client = Client()
    end_time = datetime.now().replace(second=0, microsecond=0)
    start_time = end_time - timedelta(days=1)

    k_lines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_str=end_time.strftime("%Y-%m-%d %H:%M:%S")
    )

    df = pd.DataFrame(k_lines)[[0, 1, 4]]
    df.columns = ['time', 'open', 'close']
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    return df

def main():
    df = fetch_binance_data()

    for period in [14, 27, 100]:
        df[f'RSI_{period}'] = calculate_rsi(df, period)

    print(df)

if __name__ == "__main__":
    main()
