from __future__ import annotations

from typing import Any

import pandas as pd


def generate_ml_signals(
    df: pd.DataFrame,
    model=None,
    features: list[str] | None = None,
    buy_threshold: float | None = None,
    exit_threshold: float | None = None,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Map probabilities to long-only signals. ``-1`` is exit, not a short."""
    from src.config import load_settings

    settings = settings or load_settings()
    buy_threshold = (
        float(settings["model"]["buy_threshold"]) if buy_threshold is None else buy_threshold
    )
    exit_threshold = (
        float(settings["model"]["exit_threshold"]) if exit_threshold is None else exit_threshold
    )

    out = df.copy()
    if model is not None:
        if features is None:
            raise ValueError("features required when scoring a model")
        out["prob"] = model.predict_proba(out[features])[:, 1]

    if "prob" not in out.columns:
        raise ValueError("generate_ml_signals needs model+features or a precomputed 'prob' column")

    out["signal"] = 0
    valid = out["prob"].notna()
    out.loc[valid & (out["prob"] > buy_threshold), "signal"] = 1
    out.loc[valid & (out["prob"] < exit_threshold), "signal"] = -1
    return out
