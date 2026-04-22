import pandas as pd
import ta

def add_indicators(df):
    df = df.copy()

    # df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
    # df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)
    # df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    # df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    # df["returns"] = df["close"].pct_change()
    # df["volatility"] = df["returns"].rolling(10).std()
    # df["momentum"] = df["close"] / df["close"].shift(10) - 1

    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    df["momentum"] = df["close"].pct_change(10)
    df["range"] = (df["high"] - df["low"]) / df["close"]

    return df