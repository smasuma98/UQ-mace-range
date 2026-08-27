"""Uncertainty calibration that follows the Orchestr.AI ivan2 pool-AL policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


_GAUSSIAN_SIGMA_TO_ABS = float(np.sqrt(2.0 / np.pi))


def _orchestr_imports():
    """Import the ivan2 calibration primitives instead of reimplementing them."""
    try:
        from orchestr_ai.postprocessing.uq_metrics_calculator import (
            VarianceScalingCalibrator,
            IsotonicCalibrator,
            nll,
            crps_gaussian,
            ence,
            picp,
            spearman_corr,
        )
        from orchestr_ai.postprocessing.active_learning import (
            apply_sigma_comp_calibration,
            apply_sigma_energy_calibration,
        )
        from orchestr_ai.postprocessing.evaluate import (
            _build_calibration_policy,
            _range_with_margin,
            _in_range,
        )
    except ImportError as exc:
        raise ImportError(
            "Orchestr.AI ivan2 postprocessing is required for uncertainty calibration."
        ) from exc

    return {
        "VarianceScalingCalibrator": VarianceScalingCalibrator,
        "IsotonicCalibrator": IsotonicCalibrator,
        "nll": nll,
        "crps_gaussian": crps_gaussian,
        "ence": ence,
        "picp": picp,
        "spearman_corr": spearman_corr,
        "apply_sigma_comp_calibration": apply_sigma_comp_calibration,
        "apply_sigma_energy_calibration": apply_sigma_energy_calibration,
        "build_calibration_policy": _build_calibration_policy,
        "range_with_margin": _range_with_margin,
        "in_range": _in_range,
    }


def _finite_pair(delta: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(delta, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    mask = np.isfinite(delta) & np.isfinite(sigma)
    return delta[mask], sigma[mask]


def _metric_block(
    delta: np.ndarray,
    sigma: np.ndarray,
    *,
    spearman_delta: np.ndarray | None = None,
    spearman_sigma: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the metrics used by Orchestr.AI's calibration policy."""
    oai = _orchestr_imports()
    delta, sigma = _finite_pair(delta, sigma)
    if delta.size == 0:
        return {
            "nll": float("nan"),
            "crps": float("nan"),
            "ence": float("nan"),
            "picp95": float("nan"),
            "spearman": float("nan"),
        }

    if spearman_delta is None:
        spearman_delta = delta
    if spearman_sigma is None:
        spearman_sigma = sigma
    sp_delta, sp_sigma = _finite_pair(spearman_delta, spearman_sigma)

    return {
        "nll": float(oai["nll"](delta, sigma)),
        "crps": float(oai["crps_gaussian"](delta, sigma)),
        "ence": float(oai["ence"](delta, sigma)),
        "picp95": float(oai["picp"](delta, sigma, alpha=0.05)),
        "spearman": (
            float(oai["spearman_corr"](sp_delta, sp_sigma))
            if sp_delta.size >= 2
            else float("nan")
        ),
    }


def _policy_metrics(
    *,
    energy_delta: np.ndarray,
    energy_sigma_raw: np.ndarray,
    energy_sigma_var: np.ndarray,
    energy_sigma_iso: np.ndarray,
    force_delta_comp: np.ndarray,
    force_sigma_comp_raw: np.ndarray,
    force_sigma_comp_var: np.ndarray,
    force_sigma_comp_iso: np.ndarray,
    force_delta_atom: np.ndarray,
    force_sigma_atom_raw: np.ndarray,
    force_sigma_atom_var: np.ndarray,
    force_sigma_atom_iso: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the same metric names consumed by ivan2 _build_calibration_policy."""
    f_raw = _metric_block(
        force_delta_comp,
        force_sigma_comp_raw,
        spearman_delta=force_delta_atom,
        spearman_sigma=force_sigma_atom_raw,
    )
    f_var = _metric_block(
        force_delta_comp,
        force_sigma_comp_var,
        spearman_delta=force_delta_atom,
        spearman_sigma=force_sigma_atom_var,
    )
    f_iso = _metric_block(
        force_delta_comp,
        force_sigma_comp_iso,
        spearman_delta=force_delta_atom,
        spearman_sigma=force_sigma_atom_iso,
    )
    e_raw = _metric_block(energy_delta, energy_sigma_raw)
    e_var = _metric_block(energy_delta, energy_sigma_var)
    e_iso = _metric_block(energy_delta, energy_sigma_iso)

    flat = {
        "NLL_raw": f_raw["nll"],
        "NLL_calVAR": f_var["nll"],
        "NLL_calISO": f_iso["nll"],
        "CRPS_raw": f_raw["crps"],
        "CRPS_calVAR": f_var["crps"],
        "CRPS_calISO": f_iso["crps"],
        "ENCE_raw": f_raw["ence"],
        "ENCE_calVAR": f_var["ence"],
        "ENCE_calISO": f_iso["ence"],
        "PICP95_raw": f_raw["picp95"],
        "PICP95_calVAR": f_var["picp95"],
        "PICP95_calISO": f_iso["picp95"],
        "Spearman_raw": f_raw["spearman"],
        "Spearman_calVAR": f_var["spearman"],
        "Spearman_calISO": f_iso["spearman"],
        "NLL_raw_E": e_raw["nll"],
        "NLL_calVAR_E": e_var["nll"],
        "NLL_calISO_E": e_iso["nll"],
        "CRPS_raw_E": e_raw["crps"],
        "CRPS_calVAR_E": e_var["crps"],
        "CRPS_calISO_E": e_iso["crps"],
        "ENCE_raw_E": e_raw["ence"],
        "ENCE_calVAR_E": e_var["ence"],
        "ENCE_calISO_E": e_iso["ence"],
        "PICP95_raw_E": e_raw["picp95"],
        "PICP95_calVAR_E": e_var["picp95"],
        "PICP95_calISO_E": e_iso["picp95"],
        "Spearman_raw_E": e_raw["spearman"],
        "Spearman_calVAR_E": e_var["spearman"],
        "Spearman_calISO_E": e_iso["spearman"],
    }
    friendly = {
        "force_raw": f_raw,
        "force_var": f_var,
        "force_iso": f_iso,
        "energy_raw": e_raw,
        "energy_var": e_var,
        "energy_iso": e_iso,
    }
    return {"metrics": flat}, friendly


@dataclass
class CalibrationBundle:
    """Train-fitted Orchestr.AI calibrators plus its AL selection policy."""

    mode: str
    calibrators: dict[str, Any]
    policy: dict[str, str | None]
    metrics_result: dict[str, Any]
    metrics: dict[str, Any]
    support_upper_mult: float = 1.25
    support_force_mult: float = 1.5

    @property
    def energy_mode(self) -> str | None:
        return self.policy.get("energy_mode")

    @property
    def force_mode(self) -> str | None:
        return self.policy.get("force_mode")

    @property
    def energy_accepted(self) -> bool:
        return self.energy_mode is not None

    @property
    def force_accepted(self) -> bool:
        return self.force_mode is not None

    @property
    def energy_factor(self) -> float | None:
        model = self.calibrators.get("cal_var_E")
        return float(model.s) if model is not None and hasattr(model, "s") else None

    @property
    def force_factor(self) -> float | None:
        model = self.calibrators.get("cal_var_F")
        return float(model.s) if model is not None and hasattr(model, "s") else None


def fit_calibration(
    validation: dict[str, Any],
    train_stats: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> CalibrationBundle:
    """Fit and gate calibration exactly like Orchestr.AI ivan2 pool AL.

    ivan2 always fits variance scaling first, then fits isotonic calibration on
    the variance-scaled sigma.  The requested AL mode is subsequently chosen by
    Orchestr.AI's own ``_build_calibration_policy``.
    """
    enabled = bool(cfg.get("enabled", True))
    requested = str(cfg.get("mode", "var")).lower()
    if requested not in {"var", "iso", "raw"}:
        requested = "var"

    support_upper_mult = float(cfg.get("calibration_support_upper_mult", 1.25))
    support_force_mult = float(cfg.get("calibration_support_force_mult", 1.5))

    if not enabled or requested == "raw":
        return CalibrationBundle(
            mode="raw",
            calibrators={},
            policy={"force_mode": None, "energy_mode": None},
            metrics_result={"metrics": {}},
            metrics={},
            support_upper_mult=support_upper_mult,
            support_force_mult=support_force_mult,
        )

    oai = _orchestr_imports()
    Var = oai["VarianceScalingCalibrator"]
    Iso = oai["IsotonicCalibrator"]

    energy_delta = np.asarray(validation["energy_error_per_atom"], dtype=float)
    energy_sigma_raw = np.asarray(train_stats["sigma_energy_per_atom"], dtype=float)

    force_error = np.asarray(validation["force_error"], dtype=float)
    force_delta_comp = force_error.reshape(-1)
    force_sigma_components = np.asarray(train_stats["sigma_force_components"], dtype=float)
    force_sigma_comp_raw = force_sigma_components.reshape(-1)

    # This reproduces MLFFStats.force_rmse_per_atom used by run_uq_metrics.
    force_delta_atom = np.sqrt(np.nanmean(force_error**2, axis=2)).reshape(-1)
    force_sigma_atom_raw = np.linalg.norm(force_sigma_components, axis=2).reshape(-1)

    cal_var_F = Var().fit(force_delta_comp, force_sigma_comp_raw)
    force_sigma_comp_var = cal_var_F.transform(force_sigma_comp_raw)
    force_sigma_atom_var = cal_var_F.transform(force_sigma_atom_raw)
    try:
        cal_iso_F = Iso().fit(force_delta_comp, force_sigma_comp_var)
    except Exception:
        cal_iso_F = None
    if cal_iso_F is None:
        force_sigma_comp_iso = force_sigma_comp_var
        force_sigma_atom_iso = force_sigma_atom_var
    else:
        force_sigma_comp_iso = cal_iso_F.transform(force_sigma_comp_var)
        force_sigma_atom_iso = cal_iso_F.transform(force_sigma_atom_var)

    cal_var_E = Var().fit(energy_delta, energy_sigma_raw)
    energy_sigma_var = cal_var_E.transform(energy_sigma_raw)
    try:
        cal_iso_E = Iso().fit(energy_delta, energy_sigma_var)
    except Exception:
        cal_iso_E = None
    energy_sigma_iso = (
        energy_sigma_var
        if cal_iso_E is None
        else cal_iso_E.transform(energy_sigma_var)
    )

    calibrators = {
        "cal_var_F": cal_var_F,
        "cal_iso_F": cal_iso_F,
        "cal_var_E": cal_var_E,
        "cal_iso_E": cal_iso_E,
    }

    metrics_result, metrics = _policy_metrics(
        energy_delta=energy_delta,
        energy_sigma_raw=energy_sigma_raw,
        energy_sigma_var=energy_sigma_var,
        energy_sigma_iso=energy_sigma_iso,
        force_delta_comp=force_delta_comp,
        force_sigma_comp_raw=force_sigma_comp_raw,
        force_sigma_comp_var=force_sigma_comp_var,
        force_sigma_comp_iso=force_sigma_comp_iso,
        force_delta_atom=force_delta_atom,
        force_sigma_atom_raw=force_sigma_atom_raw,
        force_sigma_atom_var=force_sigma_atom_var,
        force_sigma_atom_iso=force_sigma_atom_iso,
    )

    # Map the adapter's existing calibration section onto the ivan2 eval keys.
    policy_cfg = {
        "selection_calibration": requested,
        "calibration_min_spearman": float(
            cfg.get("calibration_min_spearman", cfg.get("min_spearman", 0.4))
        ),
        "calibration_max_ence": float(
            cfg.get("calibration_max_ence", cfg.get("max_ence", 0.20))
        ),
        "calibration_picp95_tol": float(
            cfg.get("calibration_picp95_tol", cfg.get("picp95_tol", 0.08))
        ),
    }
    # The adapter has full component-wise pool sigmas, so it is not light mode.
    policy = oai["build_calibration_policy"](
        policy_cfg,
        metrics_result,
        pool_cache_mode="raw",
    )

    # If isotonic fitting itself was unavailable, do not advertise ISO as active.
    if policy.get("force_mode") == "iso" and cal_iso_F is None:
        policy["force_mode"] = None
    if policy.get("energy_mode") == "iso" and cal_iso_E is None:
        policy["energy_mode"] = None

    return CalibrationBundle(
        mode=requested,
        calibrators=calibrators,
        policy=policy,
        metrics_result=metrics_result,
        metrics=metrics,
        support_upper_mult=support_upper_mult,
        support_force_mult=support_force_mult,
    )


def calibration_support(
    *,
    pool_frames,
    train_frames,
    pool_stats: dict[str, np.ndarray],
    train_stats: dict[str, np.ndarray],
    validation: dict[str, Any],
    bundle: CalibrationBundle,
) -> dict[str, np.ndarray]:
    """Reproduce ivan2's pool calibration-support mask from raw uncertainties."""
    oai = _orchestr_imports()

    train_counts = np.asarray([len(frame) for frame in train_frames], dtype=float)
    pool_counts = np.asarray([len(frame) for frame in pool_frames], dtype=float)

    sigma_E_train_atom_raw = np.asarray(train_stats["sigma_energy"], dtype=float) / train_counts
    sigma_E_pool_atom_raw = np.asarray(pool_stats["sigma_energy"], dtype=float) / pool_counts

    sigma_F_train_mean_raw = np.asarray(train_stats["sigma_force_mean"], dtype=float)
    sigma_F_train_max_raw = np.asarray(train_stats["sigma_force_max"], dtype=float)
    sigma_F_pool_mean_raw = np.asarray(pool_stats["sigma_force_mean"], dtype=float)
    sigma_F_pool_max_raw = np.asarray(pool_stats["sigma_force_max"], dtype=float)

    true_forces = np.asarray(validation["reference_forces"], dtype=float)
    frame_max_force_train_raw = np.max(np.linalg.norm(true_forces, axis=2), axis=1)
    pred_pool_forces = np.asarray(pool_stats["mean_forces"], dtype=float)
    frame_force_norm_pool = np.linalg.norm(pred_pool_forces, axis=2)
    frame_max_force_pool_raw = np.max(frame_force_norm_pool, axis=1)
    frame_mean_force_pool_raw = np.mean(frame_force_norm_pool, axis=1)

    support_E = oai["in_range"](
        sigma_E_pool_atom_raw,
        oai["range_with_margin"](
            sigma_E_train_atom_raw,
            upper_mult=bundle.support_upper_mult,
        ),
    )
    support_Fmean = oai["in_range"](
        sigma_F_pool_mean_raw,
        oai["range_with_margin"](
            sigma_F_train_mean_raw,
            upper_mult=bundle.support_upper_mult,
        ),
    )
    support_Fmax = oai["in_range"](
        sigma_F_pool_max_raw,
        oai["range_with_margin"](
            sigma_F_train_max_raw,
            upper_mult=bundle.support_upper_mult,
        ),
    )

    train_count_min = float(np.nanmin(train_counts))
    train_count_max = float(np.nanmax(train_counts))
    support_count = (pool_counts >= train_count_min) & (pool_counts <= train_count_max)
    size_extrapolation = ~support_count

    support_Fphys = frame_max_force_pool_raw <= (
        np.nanmax(frame_max_force_train_raw) * bundle.support_force_mult
    )

    # Matches ivan2: atom-count extrapolation is reported, but does not by itself
    # make calibration out-of-support.
    in_support = support_E & support_Fmean & support_Fmax & support_Fphys
    ood_risk_mask = ~in_support

    print(
        "[Pool-AL] Calibration support on pool: "
        f"{int(in_support.sum())}/{len(in_support)} in-domain "
        f"({np.mean(in_support):.3f}; excludes atom-count size extrapolation)."
    )
    print(
        "[Pool-AL] Calibration support failures: "
        f"sigma_E={int((~support_E).sum())}, "
        f"sigma_F_mean={int((~support_Fmean).sum())}, "
        f"sigma_F_max={int((~support_Fmax).sum())}, "
        f"force_magnitude={int((~support_Fphys).sum())}."
    )
    if np.any(size_extrapolation):
        print(
            "[Pool-AL] Size extrapolation warning: "
            f"{int(size_extrapolation.sum())}/{len(size_extrapolation)} pool frames have "
            f"atom counts outside [{train_count_min:.0f}, {train_count_max:.0f}]. "
            "They are not marked OOD by atom count alone."
        )

    return {
        "in_support": np.asarray(in_support, dtype=bool),
        "ood_risk_mask": np.asarray(ood_risk_mask, dtype=bool),
        "size_extrapolation": np.asarray(size_extrapolation, dtype=bool),
        "frame_max_force_pool": np.asarray(frame_max_force_pool_raw, dtype=float),
        "frame_mean_force_pool": np.asarray(frame_mean_force_pool_raw, dtype=float),
        "sigma_E_pool_atom_raw": np.asarray(sigma_E_pool_atom_raw, dtype=float),
        "sigma_F_pool_mean_raw": np.asarray(sigma_F_pool_mean_raw, dtype=float),
        "sigma_F_pool_max_raw": np.asarray(sigma_F_pool_max_raw, dtype=float),
    }


def calibrate_energy_total(
    sigma_total: np.ndarray,
    atom_counts: np.ndarray,
    bundle: CalibrationBundle,
    *,
    in_support: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the ivan2 selected energy mode; ISO falls back to VAR out-of-support."""
    sigma_total = np.asarray(sigma_total, dtype=float)
    counts = np.asarray(atom_counts, dtype=float)
    mode = bundle.energy_mode
    if mode is None:
        return sigma_total.copy()

    oai = _orchestr_imports()
    raw_atom = sigma_total / counts
    calibrated_atom = oai["apply_sigma_energy_calibration"](
        raw_atom,
        bundle.calibrators,
        mode,
    )
    if mode == "iso" and in_support is not None:
        var_atom = oai["apply_sigma_energy_calibration"](
            raw_atom,
            bundle.calibrators,
            "var",
        )
        calibrated_atom = np.where(
            np.asarray(in_support, dtype=bool),
            calibrated_atom,
            var_atom,
        )
    return np.asarray(calibrated_atom, dtype=float) * counts


def calibrate_force_components(
    sigma_components: np.ndarray,
    bundle: CalibrationBundle,
    *,
    in_support: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the ivan2 selected force mode; ISO falls back to VAR out-of-support."""
    sigma = np.asarray(sigma_components, dtype=float)
    mode = bundle.force_mode
    if mode is None:
        return sigma.copy()

    oai = _orchestr_imports()
    shape = sigma.shape
    flat = sigma.reshape(-1)
    calibrated = oai["apply_sigma_comp_calibration"](
        flat,
        bundle.calibrators,
        mode,
    ).reshape(shape)

    if mode == "iso" and in_support is not None:
        var = oai["apply_sigma_comp_calibration"](
            flat,
            bundle.calibrators,
            "var",
        ).reshape(shape)
        support = np.asarray(in_support, dtype=bool)
        if sigma.ndim < 2 or sigma.shape[0] != support.size:
            raise ValueError(
                "in_support must have one entry per frame when calibrating pool forces."
            )
        mask_shape = (support.size,) + (1,) * (sigma.ndim - 1)
        calibrated = np.where(support.reshape(mask_shape), calibrated, var)

    return np.asarray(calibrated, dtype=float)


def expected_absolute_errors(
    sigma_E_total: np.ndarray,
    atom_counts: np.ndarray,
    sigma_F_mean: np.ndarray,
    sigma_F_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert calibrated Gaussian sigma to E[|error|], as ivan2 diagnostics do."""
    counts = np.asarray(atom_counts, dtype=float)
    e_atom = np.asarray(sigma_E_total, dtype=float) / counts
    return (
        e_atom * _GAUSSIAN_SIGMA_TO_ABS,
        np.asarray(sigma_F_mean, dtype=float) * _GAUSSIAN_SIGMA_TO_ABS,
        np.asarray(sigma_F_max, dtype=float) * _GAUSSIAN_SIGMA_TO_ABS,
    )


def write_calibration_outputs(
    output_dir: str | Path,
    prefix: str,
    bundle: CalibrationBundle,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for quantity in ("energy", "force"):
        for stage in ("raw", "var", "iso"):
            metrics = bundle.metrics.get(f"{quantity}_{stage}", {})
            rows.append({"quantity": quantity, "stage": stage, **metrics})
    pd.DataFrame(rows).to_csv(
        output_dir / f"{prefix}_calibration_summary.csv", index=False
    )

    payload = {
        "requested_mode": bundle.mode,
        "energy_mode_for_selection": bundle.energy_mode or "raw",
        "force_mode_for_selection": bundle.force_mode or "raw",
        "energy_factor_var": bundle.energy_factor,
        "force_factor_var": bundle.force_factor,
        "support_upper_mult": bundle.support_upper_mult,
        "support_force_mult": bundle.support_force_mult,
        "metrics": bundle.metrics,
    }
    with (output_dir / f"{prefix}_calibration_factors.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
