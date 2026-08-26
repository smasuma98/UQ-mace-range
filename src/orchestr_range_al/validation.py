"""Validation against a labelled reference/training dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import mae, rmse


def validate_predictions(
    reference_energies: np.ndarray,
    reference_forces: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    source_frame_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare ensemble mean predictions with reference E/F labels."""
    ref_e = np.asarray(reference_energies, dtype=float)
    ref_f = np.asarray(reference_forces, dtype=float)
    pred_e = np.asarray(stats["mean_energy"], dtype=float)
    pred_f = np.asarray(stats["mean_forces"], dtype=float)

    if pred_e.shape != ref_e.shape:
        raise ValueError(f"Energy shape mismatch: predicted {pred_e.shape}, reference {ref_e.shape}")
    if pred_f.shape != ref_f.shape:
        raise ValueError(f"Force shape mismatch: predicted {pred_f.shape}, reference {ref_f.shape}")

    n_frames, n_atoms, _ = pred_f.shape
    energy_error = pred_e - ref_e
    energy_error_per_atom = energy_error / float(n_atoms)
    force_error = pred_f - ref_f

    force_rmse_per_frame = np.sqrt(np.mean(force_error**2, axis=(1, 2)))
    force_vector_error = np.linalg.norm(force_error, axis=2)
    force_max_error_per_frame = np.max(force_vector_error, axis=1)

    summary = {
        "MAE_E_eV": mae(energy_error),
        "RMSE_E_eV": rmse(energy_error),
        "MAE_E_per_atom_eV": mae(energy_error_per_atom),
        "RMSE_E_per_atom_eV": rmse(energy_error_per_atom),
        "MAE_F_eV_per_A": mae(force_error),
        "RMSE_F_eV_per_A": rmse(force_error),
    }

    if source_frame_indices is None:
        source_frame_indices = np.arange(n_frames, dtype=int)

    per_frame = pd.DataFrame(
        {
            "source_frame": np.asarray(source_frame_indices, dtype=int),
            "reference_energy_eV": ref_e,
            "predicted_energy_mean_eV": pred_e,
            "energy_error_eV": energy_error,
            "energy_error_per_atom_eV": energy_error_per_atom,
            "force_RMSE_frame_eV_per_A": force_rmse_per_frame,
            "force_max_error_frame_eV_per_A": force_max_error_per_frame,
            "sigma_E_eV": stats["sigma_energy"],
            "sigma_E_per_atom_eV": stats["sigma_energy_per_atom"],
            "sigma_F_mean_eV_per_A": stats["sigma_force_mean"],
            "sigma_F_max_eV_per_A": stats["sigma_force_max"],
        }
    )

    return {
        "summary": summary,
        "per_frame": per_frame,
        "energy_error": energy_error,
        "energy_error_per_atom": energy_error_per_atom,
        "force_error": force_error,
        "force_rmse_per_frame": force_rmse_per_frame,
        "force_max_error_per_frame": force_max_error_per_frame,
        "reference_energies": ref_e,
        "reference_forces": ref_f,
    }


def summary_dataframe(validation: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in validation["summary"].items()]
    )
