"""MACE-RANGE calculator construction.

This is intentionally the only module that knows how MACE-RANGE models are loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_mace_range_calculator(
    model_path: str | Path,
    *,
    device: str = "cuda",
    dtype: str = "float32",
    use_range: bool = True,
    enable_cueq: bool = True,
) -> Any:
    """Build ``mace.calculators.MACECalculator`` for one RANGE model.

    ``use_range=True`` is never silently dropped.  A small compatibility
    fallback is kept only for MACE versions that call the model keyword
    ``model_path`` instead of ``model_paths``.
    """
    if not use_range:
        raise ValueError(
            "This repository is specifically a MACE-RANGE adapter; "
            "calculator.use_range must remain true."
        )

    try:
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise ImportError(
            "Could not import mace.calculators.MACECalculator. Install or expose "
            "the MACE-RANGE source tree in this environment before running."
        ) from exc

    model_path = Path(model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"MACE-RANGE model not found: {model_path}")

    if str(device).lower().startswith("cpu"):
        enable_cueq = False

    common = {
        "device": device,
        "default_dtype": dtype,
        "use_range": True,
    }
    if enable_cueq:
        common["enable_cueq"] = True

    errors: list[str] = []
    for model_kw in ("model_paths", "model_path"):
        kwargs = {model_kw: str(model_path), **common}
        try:
            return MACECalculator(**kwargs)
        except Exception as exc:  # compatibility diagnostics
            errors.append(f"{model_kw}: {type(exc).__name__}: {exc}")

    joined = "\n  - ".join(errors)
    raise RuntimeError(
        f"Could not construct MACE-RANGE calculator for {model_path}.\n"
        f"  - {joined}"
    )
