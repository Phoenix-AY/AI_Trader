from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import load_settings, resolve_path
from src.feature_engineer import FEATURE_COLUMNS


@dataclass
class FoldResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


def fit_classifier(X: pd.DataFrame, y: pd.Series, settings: dict[str, Any] | None = None):
    settings = settings or load_settings()
    cfg = settings["model"]
    model_type = str(cfg.get("type", "lightgbm")).lower()
    n_estimators = int(cfg.get("n_estimators", 200))
    random_state = int(cfg.get("random_state", 42))

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    import lightgbm as lgb

    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    scale = (neg / pos) if pos > 0 else 1.0
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        num_leaves=int(cfg.get("num_leaves", 31)),
        min_child_samples=int(cfg.get("min_child_samples", 40)),
        subsample=float(cfg.get("subsample", 0.8)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
        scale_pos_weight=scale,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def _purged_walk_forward_windows(
    index: pd.DatetimeIndex,
    train_months: int,
    test_days: int,
    embargo_bars: int,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    if index.empty:
        return []
    index = index.sort_values()
    train_delta = pd.DateOffset(months=train_months)
    test_delta = pd.Timedelta(days=test_days)
    start = index.min()
    end = index.max()

    fold_train_end = start + train_delta
    windows: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    while fold_train_end + test_delta <= end:
        train_start = fold_train_end - train_delta
        train_idx = index[(index >= train_start) & (index <= fold_train_end)]
        if embargo_bars > 0 and len(train_idx) > embargo_bars:
            train_idx = train_idx[:-embargo_bars]
        test_idx = index[(index > fold_train_end) & (index <= fold_train_end + test_delta)]
        if len(train_idx) >= 200 and len(test_idx) >= 20:
            windows.append((train_idx, test_idx))
        fold_train_end = fold_train_end + test_delta
    return windows


def walk_forward_predict(
    df: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[FoldResult], Any]:
    """Train rolling purged folds; attach OOS probabilities only (never train-set scores)."""
    settings = settings or load_settings()
    features = features or FEATURE_COLUMNS
    cfg = settings["model"]
    embargo = int(settings["labels"]["horizon_bars"])
    work = df.sort_index().copy()
    work["prob"] = np.nan

    windows = _purged_walk_forward_windows(
        work.index,
        train_months=int(cfg["train_months"]),
        test_days=int(cfg["test_days"]),
        embargo_bars=embargo,
    )
    folds: list[FoldResult] = []
    last_model = None

    for train_idx, test_idx in windows:
        train = work.loc[train_idx]
        test = work.loc[test_idx]
        y_train = train["target"]
        if y_train.nunique() < 2:
            continue
        model = fit_classifier(train[features], y_train, settings)
        last_model = model
        work.loc[test_idx, "prob"] = model.predict_proba(test[features])[:, 1]
        folds.append(
            FoldResult(
                train_start=train_idx.min(),
                train_end=train_idx.max(),
                test_start=test_idx.min(),
                test_end=test_idx.max(),
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
        )

    if last_model is None:
        y = work["target"]
        if y.nunique() < 2:
            raise RuntimeError("Not enough class variety to train a model")
        cutoff = max(int(len(work) * 0.7), 200)
        train = work.iloc[: cutoff - embargo] if embargo < cutoff else work.iloc[:cutoff]
        test = work.iloc[cutoff:]
        if train["target"].nunique() < 2 or test.empty:
            raise RuntimeError("Walk-forward fallback split failed")
        last_model = fit_classifier(train[features], train["target"], settings)
        work.loc[test.index, "prob"] = last_model.predict_proba(test[features])[:, 1]
        folds.append(
            FoldResult(
                train_start=train.index.min(),
                train_end=train.index.max(),
                test_start=test.index.min(),
                test_end=test.index.max(),
                n_train=len(train),
                n_test=len(test),
            )
        )

    return work, folds, last_model


def fit_latest_window(
    df: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    features: list[str] | None = None,
):
    """Production fit on the most recent train window (labels embargoed)."""
    settings = settings or load_settings()
    features = features or FEATURE_COLUMNS
    embargo = int(settings["labels"]["horizon_bars"])
    train_months = int(settings["model"]["train_months"])
    work = df.sort_index()
    end = work.index.max()
    start = end - pd.DateOffset(months=train_months)
    train = work.loc[(work.index >= start) & (work.index <= end)]
    if embargo > 0 and len(train) > embargo:
        train = train.iloc[:-embargo]
    if train["target"].nunique() < 2:
        train = work.iloc[:-embargo] if embargo else work
    return fit_classifier(train[features], train["target"], settings)


def save_model_bundle(
    symbol: str,
    model,
    settings: dict[str, Any] | None = None,
    features: list[str] | None = None,
) -> str:
    settings = settings or load_settings()
    features = features or FEATURE_COLUMNS
    models_dir = resolve_path(settings, settings["model"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{symbol}.joblib"
    joblib.dump(
        {
            "model": model,
            "features": features,
            "threshold": float(settings["model"]["buy_threshold"]),
            "exit_threshold": float(settings["model"]["exit_threshold"]),
            "symbol": symbol,
        },
        path,
    )
    return str(path)


def load_model_bundle(symbol: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    path = resolve_path(settings, settings["model"]["models_dir"], f"{symbol}.joblib")
    if not path.exists():
        raise FileNotFoundError(f"No saved model for {symbol} at {path}")
    return joblib.load(path)


def train_model(df, features):
    """Legacy helper: time-ordered fit (not used by the walk-forward path)."""
    settings = load_settings()
    clean = df.dropna(subset=list(features) + ["target"])
    return fit_classifier(clean[features], clean["target"], settings)
