"""Validation against a labelled reference/training dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import mae, rmse


def validate_predictions(
    reference_energies: np.ndarray,
    reference_forces,
    stats: dict[str, Any],
    *,
    source_frame_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare ensemble mean predictions with reference E/F labels."""
    ref_e = np.asarray(reference_energies, dtype=float)
    pred_e = np.asarray(stats["mean_energy"], dtype=float)

    def frame_arrays(values, name, dimensions):
        if isinstance(values, np.ndarray) and values.dtype != object:
            array = np.asarray(values, dtype=float)
            if array.ndim != dimensions or array.shape[-1] != 3:
                raise ValueError(f"{name} has invalid shape {array.shape}.")
            return [array[index] for index in range(array.shape[0])]
        frames = [np.asarray(value, dtype=float) for value in list(values)]
        for index, frame in enumerate(frames):
            if frame.ndim != dimensions - 1 or frame.shape[-1] != 3:
                raise ValueError(f"{name} frame {index} has invalid shape {frame.shape}.")
        return frames

    ref_f_frames = frame_arrays(reference_forces, "reference_forces", 3)
    pred_f_frames = frame_arrays(stats["mean_forces"], "predicted_forces", 3)

    if pred_e.shape != ref_e.shape:
        raise ValueError(f"Energy shape mismatch: predicted {pred_e.shape}, reference {ref_e.shape}")
    if len(ref_f_frames) != len(pred_f_frames) or len(ref_f_frames) != len(ref_e):
        raise ValueError("Reference/predicted force frame counts do not match energies.")

    n_frames = len(ref_e)
    atom_counts = np.asarray([len(frame) for frame in ref_f_frames], dtype=float)
    energy_error = pred_e - ref_e
    energy_error_per_atom = energy_error / atom_counts
    force_errors = []
    force_rmse_per_frame = np.empty(n_frames, dtype=float)
    force_max_error_per_frame = np.empty(n_frames, dtype=float)
    for index, (pred_f, ref_f) in enumerate(zip(pred_f_frames, ref_f_frames)):
        if pred_f.shape != ref_f.shape:
            raise ValueError(
                f"Force shape mismatch in frame {index}: predicted {pred_f.shape}, "
                f"reference {ref_f.shape}."
            )
        error = pred_f - ref_f
        force_errors.append(error)
        force_rmse_per_frame[index] = np.sqrt(np.mean(error**2))
        force_max_error_per_frame[index] = np.max(np.linalg.norm(error, axis=1))
    force_error_flat = np.concatenate([error.reshape(-1) for error in force_errors])


    summary = {
        "MAE_E_eV": mae(energy_error),
        "RMSE_E_eV": rmse(energy_error),
        "MAE_E_per_atom_eV": mae(energy_error_per_atom),
        "RMSE_E_per_atom_eV": rmse(energy_error_per_atom),
        "MAE_F_eV_per_A": mae(force_error_flat),
        "RMSE_F_eV_per_A": rmse(force_error_flat),
    }

    if source_frame_indices is None:
        source_frame_indices = np.arange(n_frames, dtype=int)

    per_frame = pd.DataFrame(
        {
            "source_frame": np.asarray(source_frame_indices, dtype=int),
            "n_atoms": atom_counts.astype(int),
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
        "atom_counts": atom_counts.astype(int),
        "energy_error": energy_error,
        "energy_error_per_atom": energy_error_per_atom,
        "force_error": np.stack(force_errors, axis=0) if len(set(atom_counts)) == 1 else force_errors,
        "force_error_flat": force_error_flat,
        "force_rmse_per_frame": force_rmse_per_frame,
        "force_max_error_per_frame": force_max_error_per_frame,
        "reference_energies": ref_e,
        "reference_forces": np.stack(ref_f_frames, axis=0) if len(set(atom_counts)) == 1 else ref_f_frames,
    }


def summary_dataframe(validation: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in validation["summary"].items()]
    )
