def generate_ml_signals(df, model, features):
    df = df.copy()

    probs = model.predict_proba(df[features])[:, 1]

    df["prob"] = probs

    df["signal"] = 0

    # VERY STRICT FILTER
    df.loc[df["prob"] > 0.70, "signal"] = 1
    df.loc[df["prob"] < 0.30, "signal"] = -1

    return df