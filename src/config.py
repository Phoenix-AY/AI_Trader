from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SECRETS_PATH = ROOT / "config" / "secrets.env"
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


def load_settings(settings_path: Path | None = None) -> dict[str, Any]:
    load_dotenv(SECRETS_PATH)
    path = settings_path or SETTINGS_PATH
    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    settings["alpaca"] = {
        "key": os.getenv("ALPACA_API_KEY"),
        "secret": os.getenv("ALPACA_SECRET_KEY"),
        "base_url": os.getenv("ALPACA_BASE_URL", ""),
    }
    settings["_root"] = ROOT
    return settings


def resolve_path(settings: dict[str, Any], *parts: str) -> Path:
    root = Path(settings["_root"])
    return root.joinpath(*parts)


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)
