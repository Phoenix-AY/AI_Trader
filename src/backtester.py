import pandas as pd

def run_backtest(df, initial_capital=10000, risk_per_trade=0.02, fee=0.001):

    capital = initial_capital
    position = 0
    entry_price = 0

    stop_loss = 0
    take_profit = 0

    equity_curve = []

    last_trade_index = -999  # cooldown tracker
    cooldown = 5  # candles between trades

    for i in range(len(df)):
        row = df.iloc[i]
        signal = row["signal"]
        price = row["close"]

        trade_size = capital * risk_per_trade

        # current equity
        equity = capital + (position * price if position > 0 else 0)

        # -----------------------
        # ENTRY RULE
        # -----------------------
        if (
            signal == 1 and
            position == 0 and
            (i - last_trade_index) > cooldown
        ):
            position = trade_size / price
            entry_price = price

            stop_loss = entry_price * 0.995
            take_profit = entry_price * 1.01

            capital -= trade_size
            capital -= trade_size * fee

            last_trade_index = i

        # -----------------------
        # EXIT RULES
        # -----------------------
        elif position > 0:

            # stop loss
            if price <= stop_loss:
                capital += position * price
                capital -= (position * price) * fee
                position = 0

            # take profit
            elif price >= take_profit:
                capital += position * price
                capital -= (position * price) * fee
                position = 0

            # signal exit
            elif signal == -1:
                capital += position * price
                capital -= (position * price) * fee
                position = 0

        equity = capital + (position * price if position > 0 else 0)
        equity_curve.append(equity)

    df["equity"] = equity_curve
    return df