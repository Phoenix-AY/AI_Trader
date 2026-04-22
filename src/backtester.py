import pandas as pd

def run_backtest(df, initial_capital=10000, risk_per_trade=0.02, fee=0.001):
    capital = initial_capital
    position = 0
    entry_price = 0

    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        signal = row["signal"]
        price = row["close"]

        # Position size (2% risk)
        trade_size = capital * risk_per_trade

        # BUY
        if signal == 1 and position == 0:
            position = trade_size / price
            entry_price = price
            capital -= trade_size
            capital -= trade_size * fee  # fee

        # SELL
        elif signal == -1 and position > 0:
            capital += position * price
            capital -= (position * price) * fee
            position = 0

        # Equity tracking
        equity = capital + (position * price if position > 0 else 0)
        equity_curve.append(equity)

    df["equity"] = equity_curve
    return df