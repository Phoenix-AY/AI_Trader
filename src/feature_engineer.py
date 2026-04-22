import pandas as pd
import ta

def add_indicators(df):
    df = df.copy()

    # EMA indicators
    df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)

    return df