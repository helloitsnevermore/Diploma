from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("input", "geometry", "segmentation", "lines", "vectorize", "export")


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    missing = [key for key in REQUIRED_SECTIONS if key not in data]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Config is missing required sections: {missing_str}")

    return data

