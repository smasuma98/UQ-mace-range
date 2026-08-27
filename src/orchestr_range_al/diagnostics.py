"""Small human-readable diagnostic helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def write_run_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
