from __future__ import annotations

from pathlib import Path

from humanizer_eval.config import get_settings


def var_dir() -> Path:
    p = Path(get_settings().var_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ledger_path() -> Path:
    return var_dir() / "spend_ledger.sqlite"
