"""Загрузка config/profile.yaml (паттерн Settings.load() из zotero-summarizer)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "profile.yaml"


def load(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: str((ROOT / v)) for k, v in cfg.get("paths", {}).items()}
    return cfg
