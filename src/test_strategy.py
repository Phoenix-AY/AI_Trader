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

print(df[["close", "signal", "equity"]].tail())

print("\nFinal Equity:", df["equity"].iloc[-1])


plt.plot(df["equity"])
plt.title("Equity Curve")
plt.show()