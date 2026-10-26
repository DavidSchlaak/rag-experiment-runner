"""Shared utility functions for the method runners."""

from datetime import datetime, timezone
from pathlib import Path
import re
import pandas as pd

# --- Timestamps ---


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def row_value(row: pd.Series, *names: str, default: str = "") -> str:
    for name in names:
        if name in row.index:
            value = row[name]
            if not pd.isna(value):
                return str(value)
    return default
