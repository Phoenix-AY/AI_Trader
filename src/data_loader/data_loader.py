from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame

from src.config import load_settings, parse_hhmm, resolve_path

OHLCV = ["open", "high", "low", "close", "volume"]


def get_rest_client(settings: dict[str, Any] | None = None) -> REST:
    settings = settings or load_settings()
    alpaca = settings["alpaca"]
    if not alpaca.get("key") or not alpaca.get("secret"):
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in config/secrets.env")
    return REST(
        alpaca["key"],
        alpaca["secret"],
        alpaca.get("base_url") or "https://paper-api.alpaca.markets",
        api_version="v2",
    )


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index(level=0, drop=True)
    if "timestamp" in out.columns:
        out = out.set_index("timestamp")
    out.index = pd.to_datetime(out.index)
    out.columns = [str(c).lower() for c in out.columns]
    return out


def to_rth_5m(df: pd.DataFrame, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    """Align to ET, keep regular hours, resample 1m OHLCV to 5m."""
    settings = settings or load_settings()
    data_cfg = settings["data"]
    tz = data_cfg["timezone"]
    start_h, start_m = parse_hhmm(data_cfg["session_start"])
    end_h, end_m = parse_hhmm(data_cfg["session_end"])
    session_start = time(start_h, start_m)
    session_end = time(end_h, end_m)

    out = _normalize_ohlcv(df)
    idx = out.index
    if idx.tz is None:
        out.index = idx.tz_localize("UTC")
    out.index = out.index.tz_convert(tz)

    times = out.index.time
    out = out[(times >= session_start) & (times < session_end)]
    if out.empty:
        return out

    work = out.copy()
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "vwap" in work.columns:
        work["_pv"] = work["vwap"].astype(float) * work["volume"].astype(float)
        agg["_pv"] = "sum"
    if "trade_count" in work.columns:
        agg["trade_count"] = "sum"

    resampled = work.resample(data_cfg["resample"], label="left", closed="left").agg(agg)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    if "_pv" in resampled.columns:
        vol = resampled["volume"].replace(0, np.nan)
        resampled["vwap"] = resampled["_pv"] / vol
        resampled = resampled.drop(columns=["_pv"])

    times = resampled.index.time
    resampled = resampled[(times >= session_start) & (times < session_end)]
    return resampled


def _raw_paths(settings: dict[str, Any], symbol: str) -> tuple[Path, Path]:
    raw_dir = resolve_path(settings, settings["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{symbol}.parquet", raw_dir / f"{symbol}.csv"


def _processed_path(settings: dict[str, Any], symbol: str) -> Path:
    processed_dir = resolve_path(settings, settings["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir / f"{symbol}_5m.parquet"


def _chunk_bounds(start: str, end: str, days: int = 7) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts < cursor:
        cursor, end_ts = end_ts, cursor
    step = pd.Timedelta(days=days)
    bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while cursor < end_ts:
        chunk_end = min(cursor + step, end_ts)
        bounds.append((cursor, chunk_end))
        cursor = chunk_end
    return bounds


def fetch_1m_bars(
    api: REST,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _chunk_bounds(start, end):
        bars = api.get_bars(
            symbol,
            TimeFrame.Minute,
            start=chunk_start.isoformat(),
            end=chunk_end.isoformat(),
            limit=10000,
        ).df
        if bars is None or bars.empty:
            continue
        frames.append(_normalize_ohlcv(bars))
    if not frames:
        return pd.DataFrame(columns=OHLCV)
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def save_raw(df: pd.DataFrame, symbol: str, settings: dict[str, Any] | None = None) -> Path:
    settings = settings or load_settings()
    parquet_path, _ = _raw_paths(settings, symbol)
    df.to_parquet(parquet_path)
    return parquet_path


def load_raw(symbol: str, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    parquet_path, csv_path = _raw_paths(settings, symbol)
    if parquet_path.exists():
        return _normalize_ohlcv(pd.read_parquet(parquet_path))
    if csv_path.exists():
        return _normalize_ohlcv(pd.read_csv(csv_path, index_col=0, parse_dates=True))
    raise FileNotFoundError(f"No raw bars for {symbol} at {parquet_path} or {csv_path}")


def load_prepared(
    symbol: str,
    settings: dict[str, Any] | None = None,
    *,
    refresh_processed: bool = False,
) -> pd.DataFrame:
    settings = settings or load_settings()
    processed = _processed_path(settings, symbol)
    if processed.exists() and not refresh_processed:
        df = pd.read_parquet(processed)
        df.index = pd.to_datetime(df.index)
        return df
    raw = load_raw(symbol, settings)
    prepared = to_rth_5m(raw, settings)
    prepared.to_parquet(processed)
    return prepared


def fetch_data(symbol: str = "AAPL", settings: dict[str, Any] | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    api = get_rest_client(settings)
    data_cfg = settings["data"]
    bars = fetch_1m_bars(api, symbol, data_cfg["start"], data_cfg["end"])
    if bars.empty:
        raise RuntimeError(f"No bars returned for {symbol}")
    path = save_raw(bars, symbol, settings)
    print(f"Saved data to {path}")
    return bars


def fetch_watchlist(
    settings: dict[str, Any] | None = None,
    symbols: list[str] | None = None,
    *,
    skip_existing: bool = True,
) -> dict[str, pd.DataFrame]:
    settings = settings or load_settings()
    api = get_rest_client(settings)
    names = symbols or list(settings["universe"])
    data_cfg = settings["data"]
    out: dict[str, pd.DataFrame] = {}
    for symbol in names:
        parquet_path, csv_path = _raw_paths(settings, symbol)
        if skip_existing and (parquet_path.exists() or csv_path.exists()):
            out[symbol] = load_raw(symbol, settings)
            print(f"{symbol}: using existing raw file")
            continue
        print(f"{symbol}: fetching 1m bars {data_cfg['start']} -> {data_cfg['end']}")
        bars = fetch_1m_bars(api, symbol, data_cfg["start"], data_cfg["end"])
        if bars.empty:
            print(f"{symbol}: no bars returned, skipping")
            continue
        save_raw(bars, symbol, settings)
        out[symbol] = bars
    return out


if __name__ == "__main__":
    fetch_data()
