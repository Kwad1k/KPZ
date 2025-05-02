import numpy as np
import pandas as pd


class BacktestSimulator:
    def __init__(self, trading_strategy):
        self.trading_strategy = trading_strategy
        self.trades = []
        self.total_profit = 0.0

    def run(self, n_iterations: int = 100):
        for _ in range(n_iterations):
            market_data = self.trading_strategy.generate_fake_data()
            trade_signal = self.trading_strategy.create_signal(market_data)
            if trade_signal:
                self.trades.append(trade_signal)
                trade_result = self.execute_trade(trade_signal)
                trade_signal.result = trade_result
                self.total_profit += trade_result

    def execute_trade(self, trade_signal):
        if trade_signal.side == "BUY":
            exit_price = trade_signal.entry * np.random.uniform(0.95, 1.05)
        else: 
            exit_price = trade_signal.entry * np.random.uniform(0.95, 1.05)

        if (trade_signal.side == "BUY" and exit_price >= trade_signal.take_profit) or \
           (trade_signal.side == "SELL" and exit_price <= trade_signal.take_profit):
            return abs(trade_signal.take_profit - trade_signal.entry)
        elif (trade_signal.side == "BUY" and exit_price <= trade_signal.stop_loss) or \
             (trade_signal.side == "SELL" and exit_price >= trade_signal.stop_loss):
            return -abs(trade_signal.entry - trade_signal.stop_loss)
        else:
            return 0.0

    def display_summary(self):
        print(f"Total trades executed: {len(self.trades)}")
        print(f"Total profit: {round(self.total_profit, 2)}")


class TradingStrategy:
    def __init__(self, initial_balance=10000, stop_loss=5.0, take_profit=10.0):
        self.initial_balance = initial_balance
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_fake_data(self):
        return pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'close': np.random.uniform(100, 200, 100)
        })

    def create_signal(self, market_data):
        side = np.random.choice(["BUY", "SELL"])
        entry = market_data['close'].iloc[-1]
        take_profit = entry + self.take_profit if side == "BUY" else entry - self.take_profit
        stop_loss = entry - self.stop_loss if side == "BUY" else entry + self.stop_loss
        return TradeSignal(entry, side, take_profit, stop_loss)


class TradeSignal:
    def __init__(self, entry, side, take_profit, stop_loss):
        self.entry = entry
        self.side = side
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.result = None
strategy = TradingStrategy(initial_balance=10000, stop_loss=5.0, take_profit=10.0)
backtester = BacktestSimulator(trading_strategy=strategy)
backtester.run(n_iterations=100)
backtester.display_summary()
