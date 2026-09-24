from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any

import pandas as pd

from src.config import load_settings, parse_hhmm


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    reason: str


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    final_equity: float = 0.0
    cash: float = 0.0


def _apply_slippage(price: float, side: str, slip: float) -> float:
    if side == "buy":
        return price * (1.0 + slip)
    return price * (1.0 - slip)


def run_backtest(
    df: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    *,
    symbol: str = "",
    initial_capital: float | None = None,
) -> BacktestResult:
    """Single-name long-only backtest with costs, slippage, and EOD flatten."""
    settings = settings or load_settings()
    cfg = settings["backtest"]
    capital0 = float(cfg["initial_capital"] if initial_capital is None else initial_capital)
    risk = float(cfg["risk_per_trade"])
    fee = float(cfg["fee_bps"]) / 10_000.0
    slip = float(cfg["slippage_bps"]) / 10_000.0
    stop_pct = float(cfg["stop_pct"])
    tp_pct = float(cfg["take_profit_pct"])
    cooldown = int(cfg["cooldown_bars"])
    fh, fm = parse_hhmm(cfg["flatten_time"])
    flatten_t = time(fh, fm)

    cash = capital0
    qty = 0.0
    entry_price = 0.0
    entry_time = None
    stop_loss = 0.0
    take_profit = 0.0
    last_trade_i = -10_000
    trades: list[Trade] = []
    equity_vals: list[float] = []

    work = df.copy()
    if "signal" not in work.columns:
        raise ValueError("run_backtest requires a signal column")

    def _close_position(ts, price, reason, i):
        nonlocal cash, qty, entry_price, entry_time
        fill = _apply_slippage(price, "sell", slip)
        notional = qty * fill
        cash += notional
        cash -= notional * fee
        pnl = (fill - entry_price) * qty - (qty * entry_price * fee) - (notional * fee)
        trades.append(
            Trade(
                symbol=symbol,
                entry_time=entry_time,
                exit_time=ts,
                entry_price=entry_price,
                exit_price=fill,
                qty=qty,
                pnl=pnl,
                reason=reason,
            )
        )
        qty = 0.0
        entry_price = 0.0
        entry_time = None

    for i in range(len(work)):
        row = work.iloc[i]
        ts = work.index[i]
        signal = int(row["signal"]) if pd.notna(row["signal"]) else 0
        close = float(row["close"])
        high = float(row["high"]) if "high" in work.columns else close
        low = float(row["low"]) if "low" in work.columns else close
        bar_time = ts.time() if hasattr(ts, "time") else flatten_t

        if qty > 0:
            if bar_time >= flatten_t:
                _close_position(ts, close, "eod_flatten", i)
            elif low <= stop_loss:
                _close_position(ts, stop_loss, "stop", i)
            elif high >= take_profit:
                _close_position(ts, take_profit, "take_profit", i)
            elif signal == -1:
                _close_position(ts, close, "signal_exit", i)

        equity = cash + (qty * close if qty > 0 else 0.0)

        if qty == 0 and signal == 1 and (i - last_trade_i) > cooldown and bar_time < flatten_t:
            fill = _apply_slippage(close, "buy", slip)
            notional = equity * risk
            if notional > cash:
                notional = cash
            if fill > 0 and notional > 0:
                buy_qty = notional / fill
                cash -= notional
                cash -= notional * fee
                qty = buy_qty
                entry_price = fill
                entry_time = ts
                stop_loss = fill * (1.0 - stop_pct)
                take_profit = fill * (1.0 + tp_pct)
                last_trade_i = i

        equity = cash + (qty * close if qty > 0 else 0.0)
        equity_vals.append(equity)

    if qty > 0:
        last_ts = work.index[-1]
        last_close = float(work.iloc[-1]["close"])
        _close_position(last_ts, last_close, "final_flatten", len(work) - 1)
        equity_vals[-1] = cash

    equity = pd.Series(equity_vals, index=work.index, name="equity")
    return BacktestResult(
        equity=equity,
        trades=trades,
        final_equity=float(equity.iloc[-1]) if len(equity) else capital0,
        cash=cash,
    )


def run_portfolio_backtest(
    frames: dict[str, pd.DataFrame],
    settings: dict[str, Any] | None = None,
) -> BacktestResult:
    """Shared-capital book: one position per name, concurrent-position cap, EOD flatten."""
    settings = settings or load_settings()
    cfg = settings["backtest"]
    cash = float(cfg["initial_capital"])
    risk = float(cfg["risk_per_trade"])
    fee = float(cfg["fee_bps"]) / 10_000.0
    slip = float(cfg["slippage_bps"]) / 10_000.0
    stop_pct = float(cfg["stop_pct"])
    tp_pct = float(cfg["take_profit_pct"])
    cooldown = int(cfg["cooldown_bars"])
    max_pos = int(cfg["max_concurrent_positions"])
    fh, fm = parse_hhmm(cfg["flatten_time"])
    flatten_t = time(fh, fm)

    union = pd.Index(sorted(set().union(*(df.index for df in frames.values()))))
    positions: dict[str, dict[str, Any]] = {}
    last_entry_i: dict[str, int] = {sym: -10_000 for sym in frames}
    trades: list[Trade] = []
    equity_vals: list[float] = []
    last_close: dict[str, float] = {sym: float("nan") for sym in frames}

    def mark() -> float:
        total = cash
        for sym, pos in positions.items():
            px = last_close.get(sym)
            if px == px:
                total += pos["qty"] * px
        return total

    def close_pos(sym: str, ts, price: float, reason: str):
        nonlocal cash
        pos = positions.pop(sym)
        fill = _apply_slippage(price, "sell", slip)
        notional = pos["qty"] * fill
        cash += notional
        cash -= notional * fee
        pnl = (fill - pos["entry_price"]) * pos["qty"] - (
            pos["qty"] * pos["entry_price"] * fee
        ) - (notional * fee)
        trades.append(
            Trade(
                symbol=sym,
                entry_time=pos["entry_time"],
                exit_time=ts,
                entry_price=pos["entry_price"],
                exit_price=fill,
                qty=pos["qty"],
                pnl=pnl,
                reason=reason,
            )
        )

    for i, ts in enumerate(union):
        bar_time = ts.time() if hasattr(ts, "time") else flatten_t
        for sym, df in frames.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            close = float(row["close"])
            high = float(row["high"]) if "high" in df.columns else close
            low = float(row["low"]) if "low" in df.columns else close
            signal = int(row["signal"]) if pd.notna(row.get("signal", 0)) else 0
            last_close[sym] = close

            if sym in positions:
                pos = positions[sym]
                if bar_time >= flatten_t:
                    close_pos(sym, ts, close, "eod_flatten")
                elif low <= pos["stop"]:
                    close_pos(sym, ts, pos["stop"], "stop")
                elif high >= pos["tp"]:
                    close_pos(sym, ts, pos["tp"], "take_profit")
                elif signal == -1:
                    close_pos(sym, ts, close, "signal_exit")

            if (
                sym not in positions
                and signal == 1
                and (i - last_entry_i[sym]) > cooldown
                and bar_time < flatten_t
                and len(positions) < max_pos
            ):
                equity = mark()
                fill = _apply_slippage(close, "buy", slip)
                notional = equity * risk
                if notional > cash:
                    notional = cash
                if fill > 0 and notional > 0:
                    cash -= notional
                    cash -= notional * fee
                    positions[sym] = {
                        "qty": notional / fill,
                        "entry_price": fill,
                        "entry_time": ts,
                        "stop": fill * (1.0 - stop_pct),
                        "tp": fill * (1.0 + tp_pct),
                    }
                    last_entry_i[sym] = i

        equity_vals.append(mark())

    last_ts = union[-1] if len(union) else None
    for sym in list(positions):
        px = last_close.get(sym)
        if last_ts is not None and px == px:
            close_pos(sym, last_ts, px, "final_flatten")
    if equity_vals:
        equity_vals[-1] = cash

    equity = pd.Series(equity_vals, index=union, name="equity")
    return BacktestResult(
        equity=equity,
        trades=trades,
        final_equity=float(equity.iloc[-1]) if len(equity) else float(cfg["initial_capital"]),
        cash=cash,
    )


def buy_and_hold_equity(
    frames: dict[str, pd.DataFrame],
    settings: dict[str, Any] | None = None,
) -> pd.Series:
    settings = settings or load_settings()
    capital = float(settings["backtest"]["initial_capital"])
    fee = float(settings["backtest"]["fee_bps"]) / 10_000.0
    slip = float(settings["backtest"]["slippage_bps"]) / 10_000.0
    names = [s for s, df in frames.items() if not df.empty]
    if not names:
        return pd.Series(dtype=float)
    slice_cap = capital / len(names)
    union = pd.Index(sorted(set().union(*(frames[s].index for s in names))))
    holdings: dict[str, float] = {}
    for sym in names:
        first = float(frames[sym]["close"].iloc[0])
        fill = first * (1.0 + slip)
        notional = slice_cap * (1.0 - fee)
        holdings[sym] = notional / fill if fill else 0.0

    values = []
    for ts in union:
        total = 0.0
        for sym in names:
            df = frames[sym]
            if ts in df.index:
                px = float(df.loc[ts]["close"])
            else:
                hist = df.loc[:ts]
                px = float(hist["close"].iloc[-1]) if not hist.empty else 0.0
            total += holdings[sym] * px
        values.append(total)
    return pd.Series(values, index=union, name="buy_and_hold")
