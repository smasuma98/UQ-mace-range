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


def load_ensemble_predictions(
    path: str | Path,
    frames,
    model_paths: list[str | Path],
    source_frame_indices: np.ndarray,
) -> dict[str, Any]:
    """Load and validate a previously written ensemble prediction file.

    A cache is only reusable when it contains predictions for the same models,
    sampled source-frame indices, number of frames, and number of atoms.
    """
    path = Path(path)
    required = {
        "all_energies",
        "all_forces",
        "model_labels",
        "model_paths",
        "source_frame_indices",
    }
    with np.load(path, allow_pickle=True) as cached:
        missing = required.difference(cached.files)
        if missing:
            raise ValueError(f"missing arrays: {', '.join(sorted(missing))}")
        result = {key: np.asarray(cached[key]) for key in required}

    n_models = len(model_paths)
    n_frames = len(frames)
    atom_counts = np.asarray([len(frame) for frame in frames], dtype=int)
    expected_models = [str(Path(model_path)) for model_path in model_paths]
    if result["model_paths"].shape != (n_models,):
        raise ValueError("model-path array shape does not match the model list")
    cached_models = [str(value) for value in result["model_paths"].tolist()]
    if cached_models != expected_models:
        raise ValueError("model paths do not match the current configuration")

    expected_source = np.asarray(source_frame_indices, dtype=int)
    cached_source = np.asarray(result["source_frame_indices"], dtype=int)
    if not np.array_equal(cached_source, expected_source):
        raise ValueError("sampled source-frame indices do not match")
    if result["all_energies"].shape != (n_models, n_frames):
        raise ValueError("energy array shape does not match the current inputs")
    cached_forces = result["all_forces"]
    if cached_forces.dtype == object:
        force_frames = list(cached_forces)
        if len(force_frames) != n_frames or any(
            np.asarray(frame).shape != (n_models, len(frames[index]), 3)
            for index, frame in enumerate(force_frames)
        ):
            raise ValueError("force array shape does not match the current inputs")
    elif cached_forces.shape != (
        n_models,
        n_frames,
        int(atom_counts[0]),
        3,
    ) or np.any(atom_counts != atom_counts[0]):
        raise ValueError("force array shape does not match the current inputs")
    if result["model_labels"].shape != (n_models,):
        raise ValueError("model-label array shape does not match the model list")

    result["model_labels"] = [str(value) for value in result["model_labels"].tolist()]
    result["model_paths"] = cached_models
    result["source_frame_indices"] = cached_source
    return result


def evaluate_one_model(
    frames,
    model_path: str | Path,
    calculator_config: dict[str, Any],
    *,
    model_label: str | None = None,
    progress_every: int = 50,
):
    """Evaluate one model, returning dense or ragged per-frame forces."""
    if not frames:
        raise ValueError("No frames supplied for prediction.")

    atom_counts = np.asarray([len(frame) for frame in frames], dtype=int)
    constant_atoms = bool(np.all(atom_counts == atom_counts[0]))

    label = model_label or Path(model_path).stem
    calculator = build_mace_range_calculator(
        model_path,
        device=calculator_config.get("device", "cuda"),
        dtype=calculator_config.get("dtype", "float32"),
        use_range=calculator_config.get("use_range", True),
        enable_cueq=calculator_config.get("enable_cueq", True),
    )

    n_frames = len(frames)
    energies = np.empty(n_frames, dtype=np.float64)
    forces_list: list[np.ndarray] = []

    try:
        for i, source_atoms in enumerate(frames):
            atoms = source_atoms.copy()
            atoms.calc = calculator
            energies[i] = float(atoms.get_potential_energy())
            pred_forces = np.asarray(atoms.get_forces(), dtype=np.float32)
            expected = (len(source_atoms), 3)
            if pred_forces.shape != expected:
                raise ValueError(
                    f"Unexpected force shape for {label}, frame {i}: "
                    f"{pred_forces.shape}; expected {expected}."
                )
            forces_list.append(pred_forces.copy())
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

    return energies, np.stack(forces_list, axis=0) if constant_atoms else forces_list


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

    n_models = len(model_paths)
    n_frames = len(frames)
    atom_counts = np.asarray([len(frame) for frame in frames], dtype=int)
    constant_atoms = bool(np.all(atom_counts == atom_counts[0]))
    used: set[str] = set()
    labels = [safe_label(path, used) for path in model_paths]

    all_energies = np.empty((n_models, n_frames), dtype=np.float64)
    all_forces = (
        np.empty((n_models, n_frames, int(atom_counts[0]), 3), dtype=np.float32)
        if constant_atoms
        else [
            np.empty((n_models, int(atom_count), 3), dtype=np.float32)
            for atom_count in atom_counts
        ]
    )

    for m, (model_path, label) in enumerate(zip(model_paths, labels)):
        print(f"[{m + 1}/{n_models}] Loading {label}: {model_path}")
        energies, forces = evaluate_one_model(
            frames,
            model_path,
            calculator_config,
            model_label=label,
            progress_every=progress_every,
        )
        all_energies[m] = energies
        if constant_atoms:
            all_forces[m] = np.asarray(forces, dtype=np.float32)
        else:
            for index, frame_forces in enumerate(forces):
                all_forces[index][m] = np.asarray(frame_forces, dtype=np.float32)
        del energies, forces
        gc.collect()

    if source_frame_indices is None:
        source_frame_indices = np.arange(n_frames, dtype=int)

    return {
        "all_energies": all_energies,
        "all_forces": all_forces,
        "atom_counts": atom_counts,
        "model_labels": labels,
        "model_paths": [str(Path(p)) for p in model_paths],
        "source_frame_indices": np.asarray(source_frame_indices, dtype=int),
    }
