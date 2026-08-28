"""Backend-independent ensemble uncertainty statistics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _force_frames(all_forces) -> tuple[list[np.ndarray], bool]:
    if isinstance(all_forces, np.ndarray) and all_forces.dtype != object:
        array = np.asarray(all_forces, dtype=float)
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(
                "Dense all_forces must have shape (n_models, n_frames, n_atoms, 3)."
            )
        return [array[:, index, :, :] for index in range(array.shape[1])], True

    frames = [np.asarray(frame, dtype=float) for frame in list(all_forces)]
    if not frames:
        raise ValueError("all_forces contains no frames.")
    n_models = frames[0].shape[0]
    for index, frame in enumerate(frames):
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"Ragged all_forces frame {index} must have shape "
                f"(n_models, n_atoms_i, 3); got {frame.shape}."
            )
        if frame.shape[0] != n_models:
            raise ValueError("All force frames must contain the same number of models.")
    return frames, False


def _maybe_stack(values: list[np.ndarray], dense_input: bool):
    return np.stack(values, axis=0) if dense_input else values


def compute_ensemble_statistics(
    all_energies: np.ndarray,
    all_forces,
) -> dict[str, Any]:
    """Compute Orchestr.AI-compatible energy/force ensemble summaries.

    Standard deviations use ``ddof=1`` (sample standard deviation).
    """
    all_energies = np.asarray(all_energies, dtype=float)
    force_frames, dense_input = _force_frames(all_forces)

    if all_energies.ndim != 2:
        raise ValueError("all_energies must have shape (n_models, n_frames).")
    if all_energies.shape[0] < 2:
        raise ValueError("At least two models are required for ddof=1 UQ.")
    if all_energies.shape[1] != len(force_frames):
        raise ValueError("Energy and force ensemble frame counts do not match.")
    if any(frame.shape[0] != all_energies.shape[0] for frame in force_frames):
        raise ValueError("Energy and force ensemble model counts do not match.")

    mean_energy = np.mean(all_energies, axis=0)
    sigma_energy = np.std(all_energies, axis=0, ddof=1)
    atom_counts = np.asarray([frame.shape[1] for frame in force_frames], dtype=int)
    if np.any(atom_counts <= 0):
        raise ValueError("Every frame must contain at least one atom.")
    sigma_energy_per_atom = sigma_energy / atom_counts.astype(float)
    mean_forces_list, sigma_components_list = [], []
    sigma_atom_list, mean_force_norm_list = [], []
    sigma_force_mean = np.empty(len(force_frames), dtype=float)
    sigma_force_max = np.empty(len(force_frames), dtype=float)
    max_uncertain_atom = np.empty(len(force_frames), dtype=int)
    frame_max_force = np.empty(len(force_frames), dtype=float)
    frame_mean_force = np.empty(len(force_frames), dtype=float)
    for index, frame in enumerate(force_frames):
        mean_forces = np.mean(frame, axis=0)
        sigma_components = np.std(frame, axis=0, ddof=1)
        sigma_atom = np.linalg.norm(sigma_components, axis=1)
        mean_force_norm = np.linalg.norm(mean_forces, axis=1)
        mean_forces_list.append(mean_forces)
        sigma_components_list.append(sigma_components)
        sigma_atom_list.append(sigma_atom)
        mean_force_norm_list.append(mean_force_norm)
        sigma_force_mean[index] = np.mean(sigma_atom)
        sigma_force_max[index] = np.max(sigma_atom)
        max_uncertain_atom[index] = np.argmax(sigma_atom)
        frame_max_force[index] = np.max(mean_force_norm)
        frame_mean_force[index] = np.mean(mean_force_norm)

    return {
        "mean_energy": mean_energy,
        "sigma_energy": sigma_energy,
        "sigma_energy_per_atom": sigma_energy_per_atom,
        "atom_counts": atom_counts,
        "mean_forces": _maybe_stack(mean_forces_list, dense_input),
        "sigma_force_components": _maybe_stack(sigma_components_list, dense_input),
        "sigma_force_atom": _maybe_stack(sigma_atom_list, dense_input),
        "sigma_force_mean": sigma_force_mean,
        "sigma_force_max": sigma_force_max,
        "max_uncertain_atom": max_uncertain_atom,
        "mean_force_norm": _maybe_stack(mean_force_norm_list, dense_input),
        "frame_max_force": frame_max_force,
        "frame_mean_force": frame_mean_force,
    }
