from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtester import BacktestResult, Trade


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax().replace(0, np.nan)
    dd = (peak - equity) / peak
    return float(dd.max()) if dd.notna().any() else 0.0


def daily_sharpe(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    daily = equity.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if rets.empty or float(rets.std()) == 0:
        return 0.0
    return float(np.sqrt(252) * rets.mean() / rets.std())


def summarize_trades(trades: list[Trade]) -> dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
        }
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls > 0
    return {
        "trade_count": int(len(trades)),
        "win_rate": float(wins.mean()),
        "avg_pnl": float(pnls.mean()),
        "total_pnl": float(pnls.sum()),
    }


def summarize_backtest(
    result: BacktestResult,
    initial_capital: float,
    label: str,
    buy_hold_final: float | None = None,
) -> dict[str, Any]:
    stats = summarize_trades(result.trades)
    out: dict[str, Any] = {
        "label": label,
        "final_equity": float(result.final_equity),
        "return": float(result.final_equity / initial_capital - 1.0) if initial_capital else 0.0,
        "max_drawdown": max_drawdown(result.equity),
        "sharpe_daily": daily_sharpe(result.equity),
        **stats,
    }
    if buy_hold_final is not None:
        out["buy_and_hold_final"] = float(buy_hold_final)
        out["excess_vs_buy_hold"] = float(result.final_equity - buy_hold_final)
    return out
