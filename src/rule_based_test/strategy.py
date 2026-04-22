def generate_signals(df):
    df = df.copy()

    df["signal"] = 0
    df["prev_ema_9"] = df["ema_9"].shift(1)
    df["prev_ema_21"] = df["ema_21"].shift(1)

    last_trade_index = -50  # cooldown window

    for i in range(1, len(df)):
        if i - last_trade_index < 20:
            continue  # skip trades for 20 candles

        # BUY
        if (
            df.loc[i, "ema_9"] > df.loc[i, "ema_21"] and
            df.loc[i-1, "ema_9"] <= df.loc[i-1, "ema_21"] and
            df.loc[i, "adx"] > 25
        ):
            df.loc[i, "signal"] = 1
            last_trade_index = i

        # SELL
        elif (
            df.loc[i, "ema_9"] < df.loc[i, "ema_21"] and
            df.loc[i-1, "ema_9"] >= df.loc[i-1, "ema_21"] and
            df.loc[i, "adx"] > 25
        ):
            df.loc[i, "signal"] = -1
            last_trade_index = i

    return df