"""Configuration loading and lightweight validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy

import yaml


class ConfigError(ValueError):
    """Raised when the single workflow YAML is incomplete or inconsistent."""


_REQUIRED_TOP_LEVEL = (
    "paths",
    "models",
    "calculator",
    "sampling",
    "reference",
    "features",
    "calibration",
    "active_learning",
    "rdf_filter",
    "outputs",
)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML object must be a mapping.")
    cfg = copy.deepcopy(data)
    validate_config(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_TOP_LEVEL if key not in cfg]
    if missing:
        raise ConfigError(f"Missing top-level config sections: {missing}")

    for key in ("pool_trajectory", "train_dataset", "output_dir", "output_prefix"):
        if key not in cfg["paths"]:
            raise ConfigError(f"paths.{key} is required")

    models = cfg["models"]
    if not isinstance(models, list) or len(models) < 2:
        raise ConfigError("models must contain at least two model paths for ensemble UQ.")

    if not bool(cfg["calculator"].get("use_range", True)):
        raise ConfigError("calculator.use_range must be true for this repository.")

    stride = int(cfg["sampling"].get("stride", 1))
    if stride <= 0:
        raise ConfigError("sampling.stride must be a positive integer.")

    feature_type = str(cfg["features"].get("type", "soap")).lower()
    if feature_type != "soap":
        raise ConfigError(
            "Version 1 implements SOAP/PCA features only. "
            "MACE-RANGE latent features are reserved for a later version."
        )

    mode = str(cfg["calibration"].get("mode", "var")).lower()
    if mode not in {"var", "iso", "raw"}:
        raise ConfigError("calibration.mode must be one of: var, iso, raw.")


def path_from_config(value: str | Path) -> Path:
    """Resolve user paths relative to the working directory, not configs/."""
    return Path(value).expanduser()
