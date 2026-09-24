from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import ta

from src.config import load_settings

FEATURE_COLUMNS = [
    "ema_9",
    "ema_21",
    "rsi",
    "adx",
    "returns",
    "volatility",
    "momentum",
    "range",
    "volume_z",
    "vwap_dist",
]


def add_indicators(df: pd.DataFrame, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    """Causal (past-only / current-bar) indicators. ``news_score`` is a chunk-2 stub."""
    settings = settings or load_settings()
    feat_cfg = settings["features"]
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    ema_fast = int(feat_cfg["ema_fast"])
    ema_slow = int(feat_cfg["ema_slow"])
    rsi_window = int(feat_cfg["rsi_window"])
    adx_window = int(feat_cfg["adx_window"])
    vol_window = int(feat_cfg["vol_window"])
    mom_window = int(feat_cfg["momentum_window"])
    vz_window = int(feat_cfg["volume_z_window"])

    out["ema_9"] = ta.trend.ema_indicator(close, window=ema_fast)
    out["ema_21"] = ta.trend.ema_indicator(close, window=ema_slow)
    out["rsi"] = ta.momentum.rsi(close, window=rsi_window)
    out["adx"] = ta.trend.adx(high, low, close, window=adx_window)

    out["returns"] = close.pct_change()
    out["volatility"] = out["returns"].rolling(vol_window, min_periods=vol_window).std()
    out["momentum"] = close.pct_change(mom_window)
    out["range"] = (high - low) / close.replace(0, np.nan)

    vol_mean = volume.rolling(vz_window, min_periods=vz_window).mean()
    vol_std = volume.rolling(vz_window, min_periods=vz_window).std()
    out["volume_z"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    typical = (high + low + close) / 3.0
    if isinstance(out.index, pd.DatetimeIndex):
        session = pd.Series(out.index.tz_convert(None).date, index=out.index)
    else:
        session = pd.Series(0, index=out.index)
    pv = typical * volume
    cum_pv = pv.groupby(session).cumsum()
    cum_vol = volume.groupby(session).cumsum().replace(0, np.nan)
    session_vwap = cum_pv / cum_vol
    out["vwap_dist"] = close / session_vwap - 1.0

    # Hook for chunk 2: join news scores here later; do not train on it until then.
    out["news_score"] = 0.0
    return out


def add_labels(df: pd.DataFrame, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    horizon = int(settings["labels"]["horizon_bars"])
    threshold = float(settings["labels"]["threshold"])
    fee_bps = float(settings["backtest"]["fee_bps"])
    slip_bps = float(settings["backtest"]["slippage_bps"])
    round_trip = 2.0 * (fee_bps + slip_bps) / 10_000.0

    out = df.copy()
    close = out["close"].astype(float)
    out["future_return"] = close.shift(-horizon) / close - 1.0
    net = out["future_return"] - round_trip
    out["target"] = np.where(
        out["future_return"].notna(),
        (net > threshold).astype(int),
        np.nan,
    )
    if horizon > 0:
        out = out.iloc[:-horizon].copy()
    return out


def build_feature_frame(df: pd.DataFrame, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    out = add_labels(add_indicators(df, settings), settings)
    out = out.dropna(subset=FEATURE_COLUMNS + ["target"])
    out["target"] = out["target"].astype(int)
    return out
