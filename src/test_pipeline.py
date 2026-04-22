import pandas as pd

from feature_engineer import add_indicators
from model import train_model
from signals import generate_ml_signals
from backtester import run_backtest

# Load data
df = pd.read_csv("data/raw/AAPL.csv")

# Step 1: features
df = add_indicators(df)

# Step 2: define target
df["future_return"] = df["close"].shift(-20) / df["close"] - 1
df["target"] = (df["future_return"] > 0.002).astype(int)

# Step 3: features list
features = ["ema_9", "ema_21", "rsi", "adx"]

# Step 4: train model
model = train_model(df, features)

# Step 5: generate ML signals
df = generate_ml_signals(df, model, features)

# Step 6: backtest
df = run_backtest(df)

# Step 7: results
print("\nFinal Equity:", df["equity"].iloc[-1])
print("Total Trades:", (df["signal"] != 0).sum())