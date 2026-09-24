from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from src.backtester import buy_and_hold_equity, run_backtest, run_portfolio_backtest
from src.config import load_settings, resolve_path
from src.data_loader.data_loader import fetch_watchlist, load_prepared
from src.feature_engineer import FEATURE_COLUMNS, build_feature_frame
from src.metrics import summarize_backtest
from src.model import fit_latest_window, save_model_bundle, walk_forward_predict
from src.rule_based_test.strategy import generate_ema_signals
from src.signals import generate_ml_signals


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward train, backtest, and persist models")
    parser.add_argument("--skip-fetch", action="store_true", help="Do not call Alpaca; use local raw files")
    parser.add_argument("--refresh", action="store_true", help="Re-download 1m bars even if files exist")
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated subset of the watchlist (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or list(
        settings["universe"]
    )
    initial = float(settings["backtest"]["initial_capital"])
    reports_dir = resolve_path(settings, settings["backtest"]["metrics_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_fetch:
        fetch_watchlist(settings, symbols, skip_existing=not args.refresh)

    ml_frames: dict[str, pd.DataFrame] = {}
    ema_frames: dict[str, pd.DataFrame] = {}
    per_symbol: dict[str, Any] = {}
    fold_log: dict[str, Any] = {}

    for symbol in symbols:
        print(f"\n=== {symbol} ===")
        try:
            bars_5m = load_prepared(symbol, settings, refresh_processed=True)
        except FileNotFoundError as exc:
            print(exc)
            continue
        if bars_5m.empty:
            print(f"{symbol}: no RTH 5m bars")
            continue

        featured = build_feature_frame(bars_5m, settings)
        featured["symbol"] = symbol
        oos, folds, _last = walk_forward_predict(featured, settings, FEATURE_COLUMNS)
        oos = generate_ml_signals(oos, settings=settings)
        scored = oos.dropna(subset=["prob"]).copy()
        if scored.empty:
            print(f"{symbol}: no out-of-fold probabilities")
            continue

        prod_model = fit_latest_window(featured, settings, FEATURE_COLUMNS)
        model_path = save_model_bundle(symbol, prod_model, settings, FEATURE_COLUMNS)
        print(f"{symbol}: saved {model_path} ({len(folds)} walk-forward folds, {len(scored)} OOS bars)")

        ml_result = run_backtest(scored, settings, symbol=symbol)
        ema_df = generate_ema_signals(scored.copy())
        ema_result = run_backtest(ema_df, settings, symbol=symbol)
        bh = buy_and_hold_equity({symbol: scored}, settings)

        ml_frames[symbol] = scored
        ema_frames[symbol] = ema_df
        per_symbol[symbol] = {
            "ml": summarize_backtest(ml_result, initial, "ml", float(bh.iloc[-1]) if len(bh) else None),
            "ema": summarize_backtest(ema_result, initial, "ema"),
            "oos_start": scored.index.min().isoformat(),
            "oos_end": scored.index.max().isoformat(),
            "folds": len(folds),
        }
        fold_log[symbol] = [
            {
                "train_start": f.train_start.isoformat(),
                "train_end": f.train_end.isoformat(),
                "test_start": f.test_start.isoformat(),
                "test_end": f.test_end.isoformat(),
                "n_train": f.n_train,
                "n_test": f.n_test,
            }
            for f in folds
        ]
        ml_result.equity.to_csv(reports_dir / f"equity_ml_{symbol}.csv")
        print(
            f"{symbol} ML final={ml_result.final_equity:.2f} trades={len(ml_result.trades)} | "
            f"EMA final={ema_result.final_equity:.2f} trades={len(ema_result.trades)}"
        )

    if not ml_frames:
        raise SystemExit("No symbols produced walk-forward results")

    ml_port = run_portfolio_backtest(ml_frames, settings)
    ema_port = run_portfolio_backtest(ema_frames, settings)
    bh_port = buy_and_hold_equity(ml_frames, settings)
    bh_final = float(bh_port.iloc[-1]) if len(bh_port) else None

    metrics = {
        "ml_portfolio": summarize_backtest(ml_port, initial, "ml_portfolio", bh_final),
        "ema_portfolio": summarize_backtest(ema_port, initial, "ema_portfolio", bh_final),
        "buy_and_hold": {
            "final_equity": bh_final,
            "return": (bh_final / initial - 1.0) if bh_final and initial else None,
        },
        "per_symbol": per_symbol,
        "folds": fold_log,
        "universe": list(ml_frames),
        "features": FEATURE_COLUMNS,
        "note": "Signals are walk-forward / OOS probabilities only; accuracy is not the objective.",
    }
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    ml_port.equity.to_csv(reports_dir / "equity_ml_portfolio.csv")
    ema_port.equity.to_csv(reports_dir / "equity_ema_portfolio.csv")
    if len(bh_port):
        bh_port.to_csv(reports_dir / "equity_buy_and_hold.csv")

    print("\n=== Portfolio ===")
    print(json.dumps(_json_ready(metrics["ml_portfolio"]), indent=2))
    print("EMA:", json.dumps(_json_ready(metrics["ema_portfolio"]), indent=2))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
