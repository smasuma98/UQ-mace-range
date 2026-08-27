"""Backend-independent ensemble uncertainty statistics."""

from __future__ import annotations

import numpy as np


def compute_ensemble_statistics(
    all_energies: np.ndarray,
    all_forces: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute Orchestr.AI-compatible energy/force ensemble summaries.

    Standard deviations use ``ddof=1`` (sample standard deviation).
    """
    all_energies = np.asarray(all_energies, dtype=float)
    all_forces = np.asarray(all_forces, dtype=float)

    if all_energies.ndim != 2:
        raise ValueError("all_energies must have shape (n_models, n_frames).")
    if all_forces.ndim != 4 or all_forces.shape[-1] != 3:
        raise ValueError(
            "all_forces must have shape (n_models, n_frames, n_atoms, 3)."
        )
    if all_energies.shape[:2] != all_forces.shape[:2]:
        raise ValueError("Energy and force ensemble dimensions do not match.")
    if all_energies.shape[0] < 2:
        raise ValueError("At least two models are required for ddof=1 UQ.")

    n_atoms = all_forces.shape[2]

    mean_energy = np.mean(all_energies, axis=0)
    sigma_energy = np.std(all_energies, axis=0, ddof=1)
    sigma_energy_per_atom = sigma_energy / float(n_atoms)

    mean_forces = np.mean(all_forces, axis=0)
    sigma_force_components = np.std(all_forces, axis=0, ddof=1)

    sigma_force_atom = np.linalg.norm(sigma_force_components, axis=2)
    sigma_force_mean = np.mean(sigma_force_atom, axis=1)
    sigma_force_max = np.max(sigma_force_atom, axis=1)
    max_uncertain_atom = np.argmax(sigma_force_atom, axis=1)

    mean_force_norm = np.linalg.norm(mean_forces, axis=2)
    frame_max_force = np.max(mean_force_norm, axis=1)
    frame_mean_force = np.mean(mean_force_norm, axis=1)

    return {
        "mean_energy": mean_energy,
        "sigma_energy": sigma_energy,
        "sigma_energy_per_atom": sigma_energy_per_atom,
        "mean_forces": mean_forces,
        "sigma_force_components": sigma_force_components,
        "sigma_force_atom": sigma_force_atom,
        "sigma_force_mean": sigma_force_mean,
        "sigma_force_max": sigma_force_max,
        "max_uncertain_atom": max_uncertain_atom,
        "mean_force_norm": mean_force_norm,
        "frame_max_force": frame_max_force,
        "frame_mean_force": frame_mean_force,
    }
