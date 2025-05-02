import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from binance.client import Client

@dataclass
class TradeSignal:
    timestamp: datetime
    symbol: str
    volume: float
    action: str
    entry_price: float
    profit_target: float
    loss_limit: float
    status: str = "Active"

class TradingStrategy:
    def __init__(self, symbol: str, volume: float = 1.0):
        self.symbol = symbol
        self.volume = volume
        self.client = Client()

    def fetch_market_data(self) -> pd.DataFrame:
        kline_data = self.client.get_klines(symbol=self.symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=100)
        market_info = []
        for kline in kline_data:
            time_point = datetime.utcfromtimestamp(kline[0] / 1000)
            close_price = float(kline[4])
            high_price = float(kline[2])
            low_price = float(kline[3])
            open_price = float(kline[1])
            market_info.append([time_point, open_price, high_price, low_price, close_price])
        df = pd.DataFrame(market_info, columns=['timestamp', 'open', 'high', 'low', 'close'])
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['EMA_14'] = df['close'].ewm(span=14).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        high = df['high']
        low = df['low']
        close = df['close']
        df['TR'] = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['+DM'] = np.where((high - high.shift()) > (low.shift() - low), high - high.shift(), 0)
        df['-DM'] = np.where((low.shift() - low) > (high - high.shift()), low.shift() - low, 0)
        df['+DI'] = 100 * (df['+DM'].rolling(window=14).mean() / df['ATR'])
        df['-DI'] = 100 * (df['-DM'].rolling(window=14).mean() / df['ATR'])
        df['ADX'] = abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']) * 100
        return df

    def generate_trade_signal(self, df: pd.DataFrame) -> TradeSignal | None:
        df = self.calculate_technical_indicators(df)
        latest_data = df.iloc[-1]
        price = latest_data['close']
        rsi = latest_data['RSI']
        adx = latest_data['ADX']
        ema = latest_data['EMA_14']
        action = None
        if rsi > 70 and price > ema:
            action = "SELL"
        elif rsi < 30 and price < ema:
            action = "BUY"
        if action and adx > 35:
            target_price = round(price * 1.05 if action == "BUY" else price * 0.95, 2)
            stop_price = round(price * 0.98 if action == "BUY" else price * 1.02, 2)
            return TradeSignal(datetime.now(), self.symbol, self.volume, action, price, target_price, stop_price)
        return None

def track_trading_strategy(strategy: TradingStrategy):
    while True:
        market_data = strategy.fetch_market_data()
        signal = strategy.generate_trade_signal(market_data)
        if signal:
            print(f"[{signal.timestamp}] TRADE SIGNAL: {signal.action} {signal.symbol} @ {signal.entry_price}")
            print(f"  Target: {signal.profit_target}, Stop: {signal.loss_limit}")
        else:
            print(f"[{datetime.now()}] No new signal.")
        time.sleep(7)

if __name__ == "__main__":
    strategy = TradingStrategy("ETHUSDT", volume=1.0)
    track_trading_strategy(strategy)
