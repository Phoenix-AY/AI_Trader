from __future__ import annotations

import argparse
import csv
import time as time_mod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca_trade_api.rest import APIError, TimeFrame

from src.config import load_settings, resolve_path
from src.data_loader.data_loader import get_rest_client, to_rth_5m
from src.feature_engineer import FEATURE_COLUMNS, add_indicators
from src.model import load_model_bundle


def assert_paper(settings: dict[str, Any]) -> str:
    url = (settings.get("alpaca") or {}).get("base_url") or ""
    if "paper" not in url.lower():
        raise SystemExit(
            "Refusing to run: ALPACA_BASE_URL is not an Alpaca paper endpoint. "
            "Chunk 1 has no live order path."
        )
    return url


def _now_et(tz: str) -> pd.Timestamp:
    return pd.Timestamp.now(tz=tz)


def _append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "symbol",
                "prob",
                "action",
                "fill",
                "qty",
                "reason",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _round_px(price: float) -> float:
    return round(price, 2)


def _position_symbols(api) -> set[str]:
    return {p.symbol for p in api.list_positions()}


def _account_equity(api) -> float:
    account = api.get_account()
    return float(account.equity)


def _latest_closed_5m(api, symbol: str, settings: dict[str, Any]) -> pd.DataFrame:
    lookback = int(settings["paper"]["lookback_days"])
    tz = settings["data"]["timezone"]
    end = _now_et(tz)
    start = end - pd.Timedelta(days=lookback)
    bars = api.get_bars(
        symbol,
        TimeFrame.Minute,
        start=start.tz_convert("UTC").isoformat(),
        end=end.tz_convert("UTC").isoformat(),
        limit=10000,
    ).df
    if bars is None or bars.empty:
        return pd.DataFrame()
    prepared = to_rth_5m(bars, settings)
    if prepared.empty:
        return prepared
    last_ts = prepared.index[-1]
    bar_delta = pd.Timedelta(settings["data"]["resample"])
    if end < last_ts + bar_delta:
        prepared = prepared.iloc[:-1]
    return prepared


def _cancel_symbol_orders(api, symbol: str) -> None:
    try:
        orders = api.list_orders(status="open")
    except APIError:
        return
    for order in orders:
        if getattr(order, "symbol", "") == symbol:
            try:
                api.cancel_order(order.id)
            except APIError:
                pass


def _flatten_all(api, settings: dict[str, Any], log_path: Path, reason: str) -> None:
    tz = settings["data"]["timezone"]
    now = _now_et(tz)
    try:
        api.cancel_all_orders()
    except APIError:
        pass
    for pos in api.list_positions():
        qty = abs(float(pos.qty))
        if qty <= 0:
            continue
        side = "sell" if float(pos.qty) > 0 else "buy"
        try:
            order = api.submit_order(
                symbol=pos.symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            fill = getattr(order, "filled_avg_price", None) or pos.current_price
            _append_log(
                log_path,
                {
                    "time": now.isoformat(),
                    "symbol": pos.symbol,
                    "prob": "",
                    "action": "flatten",
                    "fill": fill,
                    "qty": qty,
                    "reason": reason,
                },
            )
            print(f"Flattened {pos.symbol} qty={qty}")
        except APIError as exc:
            print(f"Flatten failed {pos.symbol}: {exc}")


def _place_bracket(api, symbol: str, qty: float, price: float, settings: dict[str, Any]):
    stop_pct = float(settings["backtest"]["stop_pct"])
    tp_pct = float(settings["backtest"]["take_profit_pct"])
    take_profit = _round_px(price * (1.0 + tp_pct))
    stop_loss = _round_px(price * (1.0 - stop_pct))
    if take_profit <= price or stop_loss >= price:
        raise ValueError("Invalid bracket prices")
    return api.submit_order(
        symbol=symbol,
        qty=round(qty, 4),
        side="buy",
        type="market",
        time_in_force="day",
        order_class="bracket",
        take_profit={"limit_price": take_profit},
        stop_loss={"stop_price": stop_loss},
    )


def run_once(settings: dict[str, Any]) -> None:
    assert_paper(settings)
    api = get_rest_client(settings)
    tz = settings["data"]["timezone"]
    now = _now_et(tz)
    paper_cfg = settings["paper"]
    log_path = resolve_path(settings, paper_cfg["log_path"])
    flatten_t = datetime.strptime(paper_cfg["flatten_time"], "%H:%M").time()
    session_start = datetime.strptime(settings["data"]["session_start"], "%H:%M").time()
    session_end = datetime.strptime(settings["data"]["session_end"], "%H:%M").time()

    clock = api.get_clock()
    if now.time() >= flatten_t or now.time() >= session_end:
        _flatten_all(api, settings, log_path, "eod_flatten")
        return
    if not getattr(clock, "is_open", False) or now.time() < session_start:
        _append_log(
            log_path,
            {
                "time": now.isoformat(),
                "symbol": "",
                "prob": "",
                "action": "skip",
                "fill": "",
                "qty": "",
                "reason": "market_closed",
            },
        )
        print(f"{now}: market closed or pre-open, skip")
        return

    held = _position_symbols(api)
    equity = _account_equity(api)
    max_pos = int(paper_cfg["max_concurrent_positions"])
    notional_cap = float(paper_cfg["notional_cap"])
    risk = float(paper_cfg["risk_per_trade"])
    buy_threshold = float(settings["model"]["buy_threshold"])
    exit_threshold = float(settings["model"]["exit_threshold"])

    for symbol in settings["universe"]:
        try:
            bundle = load_model_bundle(symbol, settings)
        except FileNotFoundError:
            print(f"{symbol}: no saved model, skip")
            continue

        bars = _latest_closed_5m(api, symbol, settings)
        if bars.empty or len(bars) < 40:
            _append_log(
                log_path,
                {
                    "time": now.isoformat(),
                    "symbol": symbol,
                    "prob": "",
                    "action": "skip",
                    "fill": "",
                    "qty": "",
                    "reason": "insufficient_bars",
                },
            )
            continue

        featured = add_indicators(bars, settings).dropna(subset=FEATURE_COLUMNS)
        if featured.empty:
            continue
        last = featured.iloc[-1]
        model = bundle["model"]
        features = bundle.get("features") or FEATURE_COLUMNS
        prob = float(model.predict_proba(featured[features].iloc[[-1]])[0, 1])
        action = "hold"
        fill = last["close"]
        qty = ""
        reason = "below_gate"

        in_pos = symbol in held
        if in_pos and prob < exit_threshold:
            _cancel_symbol_orders(api, symbol)
            pos = next(p for p in api.list_positions() if p.symbol == symbol)
            sell_qty = abs(float(pos.qty))
            try:
                order = api.submit_order(
                    symbol=symbol,
                    qty=sell_qty,
                    side="sell",
                    type="market",
                    time_in_force="day",
                )
                action = "sell"
                qty = sell_qty
                fill = getattr(order, "filled_avg_price", None) or last["close"]
                reason = "prob_exit"
                held.discard(symbol)
            except APIError as exc:
                action = "error"
                reason = str(exc)
        elif (not in_pos) and prob > buy_threshold and len(held) < max_pos:
            notional = min(equity * risk, notional_cap)
            order_qty = notional / float(last["close"]) if last["close"] else 0
            if order_qty < 0.01:
                action = "skip"
                reason = "qty_too_small"
            else:
                try:
                    order = _place_bracket(api, symbol, order_qty, float(last["close"]), settings)
                    action = "buy"
                    qty = round(order_qty, 4)
                    fill = getattr(order, "filled_avg_price", None) or last["close"]
                    reason = "prob_entry"
                    held.add(symbol)
                except APIError as exc:
                    action = "error"
                    reason = str(exc)

        _append_log(
            log_path,
            {
                "time": now.isoformat(),
                "symbol": symbol,
                "prob": f"{prob:.6f}",
                "action": action,
                "fill": fill,
                "qty": qty,
                "reason": reason,
            },
        )
        print(f"{symbol} p={prob:.3f} action={action} reason={reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca paper poll / signal / order loop")
    parser.add_argument("--once", action="store_true", help="Evaluate once and exit")
    args = parser.parse_args()
    settings = load_settings()
    assert_paper(settings)
    poll = int(settings["paper"]["poll_seconds"])

    if args.once:
        run_once(settings)
        return

    print("Paper trader running (Ctrl+C to stop). No live orders.")
    while True:
        try:
            run_once(settings)
        except KeyboardInterrupt:
            print("Stopped")
            return
        except Exception as exc:
            print(f"Loop error: {exc}")
        time_mod.sleep(poll)


if __name__ == "__main__":
    main()
