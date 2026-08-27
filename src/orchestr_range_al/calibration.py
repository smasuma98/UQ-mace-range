"""Uncertainty calibration with Orchestr.AI-compatible calibrators when available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


_EPS = 1e-12


class _LocalVarianceScalingCalibrator:
    def __init__(self) -> None:
        self.s = 1.0

    def fit(self, delta, sigma):
        delta = np.asarray(delta, dtype=float)
        sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
        ratios = (delta**2) / (sigma**2)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size:
            ratios = np.sort(ratios)
            trim = int(ratios.size * 0.01)
            if trim:
                ratios = ratios[:-trim]
            self.s = float(np.sqrt(np.mean(ratios)))
        return self

    def transform(self, sigma):
        return self.s * np.maximum(np.asarray(sigma, dtype=float), _EPS)


class _LocalIsotonicCalibrator:
    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(
            y_min=0.0, increasing=True, out_of_bounds="clip"
        )
        self._abs_to_sigma = float(np.sqrt(2.0 / np.pi))

    def fit(self, delta, sigma):
        sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
        target = np.abs(np.asarray(delta, dtype=float))
        order = np.argsort(sigma)
        self._iso.fit(sigma[order], target[order])
        return self

    def transform(self, sigma):
        sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
        pred_abs = self._iso.predict(sigma)
        return np.maximum(pred_abs / self._abs_to_sigma, _EPS)


def _calibrator_class(mode: str):
    try:
        from orchestr_ai.postprocessing.uq_metrics_calculator import (
            VarianceScalingCalibrator,
            IsotonicCalibrator,
        )
        return VarianceScalingCalibrator if mode == "var" else IsotonicCalibrator
    except Exception:
        return _LocalVarianceScalingCalibrator if mode == "var" else _LocalIsotonicCalibrator


def _ence(delta: np.ndarray, sigma: np.ndarray, bins: int = 10) -> float:
    delta = np.asarray(delta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if len(delta) < bins:
        return float("nan")
    order = np.argsort(sigma)
    split = np.array_split(order, bins)
    vals = []
    for idx in split:
        if len(idx) == 0:
            continue
        rmv = float(np.sqrt(np.mean(sigma[idx] ** 2)))
        rmse = float(np.sqrt(np.mean(delta[idx] ** 2)))
        vals.append(abs(rmv - rmse) / max(rmv, _EPS))
    return float(np.mean(vals)) if vals else float("nan")


def _picp95(delta: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
    z = np.abs(np.asarray(delta, dtype=float)) / sigma
    return float(np.mean(z <= 1.959963984540054))


def _spearman(delta: np.ndarray, sigma: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mask = np.isfinite(delta) & np.isfinite(sigma)
    if mask.sum() < 2 or np.allclose(sigma[mask], sigma[mask][0]):
        return float("nan")
    return float(spearmanr(np.abs(delta[mask]), sigma[mask]).statistic)


def _metrics(delta, sigma) -> dict[str, float]:
    return {
        "spearman": _spearman(delta, sigma),
        "ence": _ence(delta, sigma),
        "picp95": _picp95(delta, sigma),
    }


def _passes(metrics: dict[str, float], cfg: dict[str, Any]) -> bool:
    sp = metrics["spearman"]
    en = metrics["ence"]
    pi = metrics["picp95"]
    checks = [
        np.isfinite(sp) and sp >= float(cfg.get("min_spearman", 0.4)),
        np.isfinite(en) and en <= float(cfg.get("max_ence", 0.20)),
        np.isfinite(pi)
        and abs(pi - 0.95) <= float(cfg.get("picp95_tol", 0.08)),
    ]
    return all(checks)


@dataclass
class CalibrationBundle:
    mode: str
    energy_model: Any | None
    force_model: Any | None
    energy_accepted: bool
    force_accepted: bool
    metrics: dict[str, Any]

    @property
    def energy_factor(self) -> float | None:
        return float(self.energy_model.s) if hasattr(self.energy_model, "s") else None

    @property
    def force_factor(self) -> float | None:
        return float(self.force_model.s) if hasattr(self.force_model, "s") else None


def fit_calibration(
    validation: dict[str, Any],
    train_stats: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> CalibrationBundle:
    """Fit calibration on the labelled training/reference dataset."""
    enabled = bool(cfg.get("enabled", True))
    mode = str(cfg.get("mode", "var")).lower()
    if not enabled or mode == "raw":
        return CalibrationBundle(
            mode="raw",
            energy_model=None,
            force_model=None,
            energy_accepted=False,
            force_accepted=False,
            metrics={},
        )

    energy_delta = np.asarray(validation["energy_error_per_atom"], dtype=float)
    energy_sigma = np.asarray(train_stats["sigma_energy_per_atom"], dtype=float)
    force_delta = np.asarray(validation["force_error"], dtype=float).reshape(-1)
    force_sigma = np.asarray(train_stats["sigma_force_components"], dtype=float).reshape(-1)

    cls = _calibrator_class(mode)
    energy_model = cls().fit(energy_delta, energy_sigma)
    force_model = cls().fit(force_delta, force_sigma)

    e_raw = _metrics(energy_delta, energy_sigma)
    e_cal = _metrics(energy_delta, energy_model.transform(energy_sigma))
    f_raw = _metrics(force_delta, force_sigma)
    f_cal = _metrics(force_delta, force_model.transform(force_sigma))

    return CalibrationBundle(
        mode=mode,
        energy_model=energy_model,
        force_model=force_model,
        energy_accepted=_passes(e_cal, cfg),
        force_accepted=_passes(f_cal, cfg),
        metrics={
            "energy_raw": e_raw,
            "energy_calibrated": e_cal,
            "force_raw": f_raw,
            "force_calibrated": f_cal,
        },
    )


def calibrate_energy_total(
    sigma_total: np.ndarray,
    atom_counts: np.ndarray,
    bundle: CalibrationBundle,
    *,
    use_only_if_accepted: bool = False,
) -> np.ndarray:
    sigma_total = np.asarray(sigma_total, dtype=float)
    counts = np.asarray(atom_counts, dtype=float)
    if bundle.energy_model is None:
        return sigma_total.copy()
    if use_only_if_accepted and not bundle.energy_accepted:
        return sigma_total.copy()
    per_atom = sigma_total / counts
    return np.asarray(bundle.energy_model.transform(per_atom), dtype=float) * counts


def calibrate_force_components(
    sigma_components: np.ndarray,
    bundle: CalibrationBundle,
    *,
    use_only_if_accepted: bool = False,
) -> np.ndarray:
    sigma = np.asarray(sigma_components, dtype=float)
    if bundle.force_model is None:
        return sigma.copy()
    if use_only_if_accepted and not bundle.force_accepted:
        return sigma.copy()
    shape = sigma.shape
    return np.asarray(bundle.force_model.transform(sigma.reshape(-1)), dtype=float).reshape(shape)


def write_calibration_outputs(
    output_dir: str | Path,
    prefix: str,
    bundle: CalibrationBundle,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for quantity in ("energy", "force"):
        for stage in ("raw", "calibrated"):
            metrics = bundle.metrics.get(f"{quantity}_{stage}", {})
            row = {"quantity": quantity, "stage": stage, **metrics}
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        output_dir / f"{prefix}_calibration_summary.csv", index=False
    )

    payload = {
        "mode": bundle.mode,
        "energy_accepted_for_selection": bundle.energy_accepted,
        "force_accepted_for_selection": bundle.force_accepted,
        "energy_factor": bundle.energy_factor,
        "force_factor": bundle.force_factor,
        "metrics": bundle.metrics,
    }
    with (output_dir / f"{prefix}_calibration_factors.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
