def generate_signals(df):
    df = df.copy()

    df["signal"] = 0

    # Buy signal
    df.loc[df["ema_9"] > df["ema_21"], "signal"] = 1

    # Sell signal
    df.loc[df["ema_9"] < df["ema_21"], "signal"] = -1

    return df