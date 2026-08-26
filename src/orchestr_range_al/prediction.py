"""Sequential ensemble prediction using the MACE-RANGE adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import gc
import re

import numpy as np

from .calculators import build_mace_range_calculator


def safe_label(path: str | Path, used: set[str]) -> str:
    path = Path(path)
    label = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem).strip("_") or "model"
    base = label
    suffix = 1
    while label in used:
        label = f"{base}_{suffix}"
        suffix += 1
    used.add(label)
    return label


def evaluate_one_model(
    frames,
    model_path: str | Path,
    calculator_config: dict[str, Any],
    *,
    model_label: str | None = None,
    progress_every: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one model on all frames and return total energies and forces."""
    if not frames:
        raise ValueError("No frames supplied for prediction.")

    atom_counts = np.asarray([len(frame) for frame in frames], dtype=int)
    if np.any(atom_counts != atom_counts[0]):
        raise ValueError(
            "Version 1 prediction arrays require a constant atom count within "
            "each evaluated trajectory/dataset."
        )

    label = model_label or Path(model_path).stem
    calculator = build_mace_range_calculator(
        model_path,
        device=calculator_config.get("device", "cuda"),
        dtype=calculator_config.get("dtype", "float32"),
        use_range=calculator_config.get("use_range", True),
        enable_cueq=calculator_config.get("enable_cueq", True),
    )

    n_frames = len(frames)
    n_atoms = int(atom_counts[0])
    energies = np.empty(n_frames, dtype=np.float64)
    forces = np.empty((n_frames, n_atoms, 3), dtype=np.float32)

    try:
        for i, source_atoms in enumerate(frames):
            atoms = source_atoms.copy()
            atoms.calc = calculator
            energies[i] = float(atoms.get_potential_energy())
            pred_forces = np.asarray(atoms.get_forces(), dtype=np.float32)
            if pred_forces.shape != (n_atoms, 3):
                raise ValueError(
                    f"Unexpected force shape for {label}, frame {i}: "
                    f"{pred_forces.shape}; expected {(n_atoms, 3)}."
                )
            forces[i] = pred_forces
            atoms.calc = None

            if (
                i == 0
                or (i + 1) % max(1, int(progress_every)) == 0
                or i + 1 == n_frames
            ):
                print(f"  {label}: evaluated {i + 1}/{n_frames} frames", flush=True)
    finally:
        del calculator
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return energies, forces


def evaluate_ensemble(
    frames,
    model_paths: list[str | Path],
    calculator_config: dict[str, Any],
    *,
    source_frame_indices: np.ndarray | None = None,
    progress_every: int = 50,
) -> dict[str, Any]:
    """Evaluate an ensemble one model at a time to limit GPU memory use."""
    if len(model_paths) < 2:
        raise ValueError("At least two models are required for ensemble UQ.")
    if not frames:
        raise ValueError("No frames supplied for ensemble prediction.")

    n_frames = len(frames)
    n_atoms = len(frames[0])
    used: set[str] = set()
    labels = [safe_label(path, used) for path in model_paths]

    all_energies = np.empty((len(model_paths), n_frames), dtype=np.float64)
    all_forces = np.empty(
        (len(model_paths), n_frames, n_atoms, 3), dtype=np.float32
    )

    for m, (model_path, label) in enumerate(zip(model_paths, labels)):
        print(f"[{m + 1}/{len(model_paths)}] Loading {label}: {model_path}")
        energies, forces = evaluate_one_model(
            frames,
            model_path,
            calculator_config,
            model_label=label,
            progress_every=progress_every,
        )
        all_energies[m] = energies
        all_forces[m] = forces
        del energies, forces
        gc.collect()

    if source_frame_indices is None:
        source_frame_indices = np.arange(n_frames, dtype=int)

    return {
        "all_energies": all_energies,
        "all_forces": all_forces,
        "model_labels": labels,
        "model_paths": [str(Path(p)) for p in model_paths],
        "source_frame_indices": np.asarray(source_frame_indices, dtype=int),
    }
