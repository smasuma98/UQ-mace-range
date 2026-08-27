"""Thin adapter around Orchestr.AI pool active-learning selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import shutil

import numpy as np
import pandas as pd

from .calibration import (
    CalibrationBundle,
    calibration_support,
    calibrate_energy_total,
    calibrate_force_components,
    expected_absolute_errors,
)


def _orchestr_imports():
    try:
        from orchestr_ai.postprocessing.active_learning import (
            adaptive_learning_mig_pool_windowed,
            calibrate_alpha_reg_gcv,
            write_pool_al_diagnostics_csv,
        )
        from orchestr_ai.postprocessing.rdf import compute_rdf_thresholds_from_reference
    except ImportError as exc:
        raise ImportError(
            "Orchestr.AI ivan2 postprocessing is required for active-learning selection."
        ) from exc
    return (
        adaptive_learning_mig_pool_windowed,
        calibrate_alpha_reg_gcv,
        write_pool_al_diagnostics_csv,
        compute_rdf_thresholds_from_reference,
    )


def _force_summaries(sigma_components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    atom_sigma = np.linalg.norm(np.asarray(sigma_components, dtype=float), axis=2)
    return np.mean(atom_sigma, axis=1), np.max(atom_sigma, axis=1)


def _copy_if_distinct(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    if source.exists():
        shutil.copyfile(source, target)


def run_orchestr_selection(
    *,
    pool_frames,
    train_frames,
    pool_source_indices: np.ndarray,
    F_pool: np.ndarray,
    F_train: np.ndarray,
    pool_stats: dict[str, np.ndarray],
    train_stats: dict[str, np.ndarray],
    validation: dict[str, Any],
    calibration: CalibrationBundle,
    active_cfg: dict[str, Any],
    rdf_cfg: dict[str, Any],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Any]:
    """Prepare selector inputs, using Orchestr.AI ivan2 calibration semantics."""
    (
        adaptive_learning_mig_pool_windowed,
        calibrate_alpha_reg_gcv,
        write_pool_al_diagnostics_csv,
        compute_rdf_thresholds_from_reference,
    ) = _orchestr_imports()

    F_train = np.asarray(F_train, dtype=float)
    F_pool = np.asarray(F_pool, dtype=float)
    y_train = np.asarray(validation["energy_error_per_atom"], dtype=float)

    if F_train.ndim != 2 or F_pool.ndim != 2:
        raise ValueError("F_train and F_pool must be 2D feature matrices.")
    if F_train.shape[1] != F_pool.shape[1]:
        raise ValueError("Training and pool feature dimensions differ.")

    good = np.isfinite(F_train).all(axis=1) & np.isfinite(y_train)
    if int(good.sum()) < 2:
        raise ValueError("Need at least two finite training rows for GP calibration.")

    alpha_sq, _, _, _, L = calibrate_alpha_reg_gcv(F_train[good], y_train[good])

    train_counts = np.asarray([len(frame) for frame in train_frames], dtype=float)
    pool_counts = np.asarray([len(frame) for frame in pool_frames], dtype=float)

    # Orchestr.AI determines calibration support from RAW train/pool uncertainty.
    support = calibration_support(
        pool_frames=pool_frames,
        train_frames=train_frames,
        pool_stats=pool_stats,
        train_stats=train_stats,
        validation=validation,
        bundle=calibration,
    )
    calibration_in_support = support["in_support"]
    ood_risk_mask = support["ood_risk_mask"]

    # Train calibration is applied directly.  For pool ISO, ivan2 uses VAR
    # fallback outside calibration support.
    sigma_E_train = calibrate_energy_total(
        train_stats["sigma_energy"],
        train_counts,
        calibration,
    )
    sigma_E_pool = calibrate_energy_total(
        pool_stats["sigma_energy"],
        pool_counts,
        calibration,
        in_support=calibration_in_support,
    )
    sigma_F_train = calibrate_force_components(
        train_stats["sigma_force_components"],
        calibration,
    )
    sigma_F_pool = calibrate_force_components(
        pool_stats["sigma_force_components"],
        calibration,
        in_support=calibration_in_support,
    )
    sigma_F_pool_mean, sigma_F_pool_max = _force_summaries(sigma_F_pool)

    expected_abs_E_atom, expected_abs_F_mean, expected_abs_F_max = (
        expected_absolute_errors(
            sigma_E_pool,
            pool_counts,
            sigma_F_pool_mean,
            sigma_F_pool_max,
        )
    )

    rdf_enabled = bool(rdf_cfg.get("enabled", True))
    if rdf_enabled:
        rdf_thresholds = compute_rdf_thresholds_from_reference(train_frames)
        rdf_ok_mask = None  # let Orchestr.AI apply its RDF + connectivity filters
    else:
        rdf_thresholds = {}
        rdf_ok_mask = np.ones(len(pool_frames), dtype=bool)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_runs: list[dict[str, Any]] = []

    # Canonical Orchestr.AI basename.  _PoolActiveLearner appends
    # ``_per_frame_diagnostics.txt`` internally.
    base = str(output_dir / "al_pool")

    selected_frames, selected_rel = adaptive_learning_mig_pool_windowed(
        pool_frames,
        F_pool,
        F_train,
        alpha_sq,
        L,
        forces_train=[np.asarray(f, dtype=float) for f in validation["reference_forces"]],
        sigma_energy=sigma_E_train,
        sigma_force=[np.asarray(v, dtype=float) for v in sigma_F_train],
        mu_E_frame_train=np.asarray(train_stats["mean_energy"], dtype=float),
        mu_E_pool=np.asarray(pool_stats["mean_energy"], dtype=float),
        sigma_E_pool=np.asarray(sigma_E_pool, dtype=float),
        mu_F_pool=[np.asarray(v, dtype=float) for v in pool_stats["mean_forces"]],
        sigma_F_pool=[np.asarray(v, dtype=float) for v in sigma_F_pool],
        rdf_thresholds=rdf_thresholds,
        rdf_ok_mask=rdf_ok_mask,
        sigma_F_pool_mean=sigma_F_pool_mean,
        sigma_F_pool_max=sigma_F_pool_max,
        frame_max_force_pool=support["frame_max_force_pool"],
        frame_mean_force_pool=support["frame_mean_force_pool"],
        calibration_in_support=calibration_in_support,
        ood_risk_mask=ood_risk_mask,
        expected_abs_E_atom=expected_abs_E_atom,
        expected_abs_F_mean=expected_abs_F_mean,
        expected_abs_F_max=expected_abs_F_max,
        train_frames=train_frames,
        rho_eV=float(active_cfg.get("rho_eV", 0.002)),
        min_k=int(active_cfg.get("min_k", 5)),
        window_size=int(active_cfg.get("window_size", 100)),
        budget_max=int(active_cfg.get("budget_max", 50)),
        percentile_gamma=float(active_cfg.get("percentile_gamma", 99)),
        percentile_F_low=float(active_cfg.get("percentile_F_low", 95)),
        percentile_F_hi=float(active_cfg.get("percentile_F_hi", 99)),
        hard_sigma_E_atom_min=float(active_cfg.get("hard_sigma_E_atom_min", 0.001)),
        hard_sigma_F_mean_min=float(active_cfg.get("hard_sigma_F_mean_min", 0.075)),
        hard_sigma_F_max_min=float(active_cfg.get("hard_sigma_F_max_min", 0.150)),
        hard_Fmax_train_mult=float(active_cfg.get("hard_Fmax_train_mult", 1.5)),
        large_cluster_threshold=int(active_cfg.get("large_cluster_threshold", 300)),
        surface_relax_factor=active_cfg.get("surface_relax_factor", 1.5),
        hard_floors_from_calibrated_train=bool(
            calibration.energy_mode or calibration.force_mode
        ),
        base=base,
        state="mace_range",
        pool_indices=np.asarray(pool_source_indices, dtype=int),
        diagnostics_collector=diagnostics_runs,
    )

    selected_rel = np.asarray(selected_rel, dtype=int)
    selected_source = np.asarray(pool_source_indices, dtype=int)[selected_rel]

    diagnostics_txt = output_dir / "al_pool_per_frame_diagnostics.txt"
    diagnostics_csv = output_dir / "al_pool_diagnostics.csv"
    if diagnostics_runs:
        write_pool_al_diagnostics_csv(diagnostics_runs, str(diagnostics_csv))

    # Keep adapter-prefix aliases so existing scripts do not break.
    _copy_if_distinct(
        diagnostics_txt,
        output_dir / f"{prefix}_per_frame_diagnostics.txt",
    )
    _copy_if_distinct(
        diagnostics_csv,
        output_dir / f"{prefix}_pool_diagnostics.csv",
    )

    pd.DataFrame(
        {
            "selection_order": np.arange(len(selected_rel), dtype=int),
            "pool_row": selected_rel,
            "source_frame": selected_source,
        }
    ).to_csv(output_dir / f"{prefix}_selected_indices.csv", index=False)

    summary = {
        "n_pool": len(pool_frames),
        "n_selected": int(len(selected_rel)),
        "selected_pool_rows": selected_rel.tolist(),
        "selected_source_frames": selected_source.tolist(),
        "energy_calibration_used": calibration.energy_mode or "raw",
        "force_calibration_used": calibration.force_mode or "raw",
        "calibration_in_support_count": int(calibration_in_support.sum()),
        "calibration_ood_count": int(ood_risk_mask.sum()),
        "rdf_filter_enabled": rdf_enabled,
        "diagnostics_txt": str(diagnostics_txt),
        "diagnostics_csv": str(diagnostics_csv),
    }
    (output_dir / f"{prefix}_selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return {
        "selected_frames": selected_frames,
        "selected_indices": selected_rel,
        "selected_source_indices": selected_source,
        "diagnostics_runs": diagnostics_runs,
        "diagnostics_txt": diagnostics_txt,
        "diagnostics_csv": diagnostics_csv,
        "summary": summary,
    }
