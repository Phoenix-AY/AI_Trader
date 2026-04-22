import pandas as pd
from feature_engineer import add_indicators
from strategy import generate_signals
from backtester import run_backtest

import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/raw/AAPL.csv")

# Process pipeline
df = add_indicators(df)
df = generate_signals(df)
df = run_backtest(df)

df["future_return"] = df["close"].shift(-5) / df["close"] - 1

# Classification target
df["target"] = (df["future_return"] > 0).astype(int)

print(df[["close", "signal", "equity"]].tail())

print("\nFinal Equity:", df["equity"].iloc[-1])


plt.plot(df["equity"])
plt.title("Equity Curve")
plt.show()

print("\nTotal Trades:", (df["signal"] != 0).sum())

returns = df["equity"].pct_change()

print("\nFinal Equity:", df["equity"].iloc[-1])
print("Total Trades:", (df["signal"] != 0).sum())

print("Win Rate:", (returns > 0).sum() / len(returns))
print("Avg Return:", returns.mean())
print("Max Drawdown:", (df["equity"].cummax() - df["equity"]).max())