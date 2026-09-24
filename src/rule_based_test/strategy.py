from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ema_signals(df: pd.DataFrame, adx_min: float = 25.0, cooldown: int = 20) -> pd.DataFrame:
    """Long-only EMA crossover baseline. ``-1`` is exit, not a short."""
    out = df.copy()
    n = len(out)
    signal = np.zeros(n, dtype=int)
    ema9 = out["ema_9"].to_numpy(dtype=float)
    ema21 = out["ema_21"].to_numpy(dtype=float)
    adx = out["adx"].to_numpy(dtype=float)
    last_trade = -cooldown

    for i in range(1, n):
        if i - last_trade < cooldown:
            continue
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(adx[i]):
            continue
        if ema9[i] > ema21[i] and ema9[i - 1] <= ema21[i - 1] and adx[i] > adx_min:
            signal[i] = 1
            last_trade = i
        elif ema9[i] < ema21[i] and ema9[i - 1] >= ema21[i - 1] and adx[i] > adx_min:
            signal[i] = -1
            last_trade = i

    out["signal"] = signal
    return out


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    return generate_ema_signals(df)
