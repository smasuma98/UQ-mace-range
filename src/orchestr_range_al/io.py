"""ASE input/output and tabular persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _ase_io():
    try:
        from ase import Atoms
        from ase.io import read, write
    except ImportError as exc:
        raise ImportError("ASE is required for trajectory/dataset I/O.") from exc
    return Atoms, read, write


def read_frames(
    path: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
):
    """Read sampled frames and return ``(frames, source_indices)``."""
    Atoms, read, _ = _ase_io()
    path = Path(path).expanduser()
    stop_text = "" if stop is None else str(stop)
    index = f"{int(start)}:{stop_text}:{int(stride)}"
    frames = read(path, index=index)
    if isinstance(frames, Atoms):
        frames = [frames]
    else:
        frames = list(frames)
    if not frames:
        raise ValueError(f"No frames read from {path} with ASE index {index!r}.")
    source = int(start) + np.arange(len(frames), dtype=int) * int(stride)
    return frames, source


def _reference_energy(frame, energy_key: str) -> float:
    if energy_key in frame.info:
        return float(frame.info[energy_key])
    calc = getattr(frame, "calc", None)
    results = getattr(calc, "results", {}) if calc is not None else {}
    if energy_key in results:
        return float(results[energy_key])
    if energy_key == "energy":
        try:
            return float(frame.get_potential_energy())
        except Exception:
            pass
    raise KeyError(
        f"Reference energy key {energy_key!r} not found in frame.info or calculator results."
    )


def _reference_forces(frame, forces_key: str) -> np.ndarray:
    if forces_key in frame.arrays:
        return np.asarray(frame.arrays[forces_key], dtype=float)
    calc = getattr(frame, "calc", None)
    results = getattr(calc, "results", {}) if calc is not None else {}
    if forces_key in results:
        return np.asarray(results[forces_key], dtype=float)
    if forces_key == "forces":
        try:
            return np.asarray(frame.get_forces(), dtype=float)
        except Exception:
            pass
    raise KeyError(
        f"Reference force key {forces_key!r} not found in frame.arrays or calculator results."
    )


def read_reference_dataset(
    path: str | Path,
    *,
    energy_key: str = "energy",
    forces_key: str = "forces",
):
    """Read all reference frames plus E/F labels.

    Standard extxyz ``energy``/``forces`` stored by ASE as SinglePointCalculator
    results are supported in addition to explicit ``info``/``arrays`` fields.
    """
    frames, source = read_frames(path, start=0, stop=None, stride=1)
    energies = np.asarray(
        [_reference_energy(frame, energy_key) for frame in frames], dtype=float
    )
    forces = [_reference_forces(frame, forces_key) for frame in frames]

    counts = np.asarray([len(frame) for frame in frames], dtype=int)
    forces_out = (
        np.stack(forces, axis=0)
        if np.all(counts == counts[0])
        else [np.asarray(force, dtype=float) for force in forces]
    )
    return frames, energies, forces_out, source


def write_predictions_npz(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            obj = np.empty(len(value), dtype=object)
            obj[:] = list(value)
            serializable[key] = obj
        else:
            serializable[key] = value
    np.savez_compressed(path, **serializable)


def frame_summary_dataframe(
    ensemble_result: dict[str, Any],
    stats: dict[str, np.ndarray],
) -> pd.DataFrame:
    source = ensemble_result["source_frame_indices"]
    n_frames = len(source)
    atom_counts = np.asarray(
        stats.get("atom_counts", [len(value) for value in stats["mean_forces"]]),
        dtype=float,
    )
    table: dict[str, Any] = {
        "sampled_frame": np.arange(n_frames, dtype=int),
        "source_frame": source,
        "energy_mean_eV": stats["mean_energy"],
        "sigma_E_eV": stats["sigma_energy"],
        "sigma_E_per_atom_eV": stats["sigma_energy_per_atom"],
        "sigma_F_mean_eV_per_A": stats["sigma_force_mean"],
        "sigma_F_max_eV_per_A": stats["sigma_force_max"],
        "max_uncertain_atom_index": stats["max_uncertain_atom"],
        "frame_max_mean_force_eV_per_A": stats["frame_max_force"],
        "frame_mean_mean_force_eV_per_A": stats["frame_mean_force"],
    }
    for i, label in enumerate(ensemble_result["model_labels"]):
        table[f"energy_{label}_eV"] = ensemble_result["all_energies"][i]
        table[f"energy_per_atom_{label}_eV"] = (
            ensemble_result["all_energies"][i] / atom_counts
        )
    return pd.DataFrame(table)


def write_frame_summary_csv(path: str | Path, dataframe: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def write_annotated_xyz(
    path: str | Path,
    frames,
    ensemble_result: dict[str, Any],
    stats: dict[str, np.ndarray],
) -> None:
    _, _, write = _ase_io()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    n_models = len(ensemble_result["model_labels"])
    for i, source_atoms in enumerate(frames):
        atoms = source_atoms.copy()
        atoms.calc = None
        atoms.info["source_frame"] = int(ensemble_result["source_frame_indices"][i])
        atoms.info["ensemble_size"] = int(n_models)
        atoms.info["energy_mean_eV"] = float(stats["mean_energy"][i])
        atoms.info["sigma_E_eV"] = float(stats["sigma_energy"][i])
        atoms.info["sigma_E_per_atom_eV"] = float(stats["sigma_energy_per_atom"][i])
        atoms.info["sigma_F_mean_eV_per_A"] = float(stats["sigma_force_mean"][i])
        atoms.info["sigma_F_max_eV_per_A"] = float(stats["sigma_force_max"][i])
        atoms.info["max_uncertain_atom_index"] = int(stats["max_uncertain_atom"][i])

        for m, label in enumerate(ensemble_result["model_labels"]):
            atoms.info[f"energy_{label}_eV"] = float(
                ensemble_result["all_energies"][m, i]
            )

        for name in ("mean_forces", "sigma_F_components", "sigma_F_norm"):
            if name in atoms.arrays:
                del atoms.arrays[name]
        atoms.arrays["mean_forces"] = np.asarray(stats["mean_forces"][i], dtype=float)
        atoms.arrays["sigma_F_components"] = np.asarray(
            stats["sigma_force_components"][i], dtype=float
        )
        atoms.arrays["sigma_F_norm"] = np.asarray(
            stats["sigma_force_atom"][i], dtype=float
        )
        write(path, atoms, format="extxyz", append=i > 0)


def write_selected_frames(path: str | Path, selected_frames) -> None:
    _, _, write = _ase_io()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if selected_frames:
        write(path, selected_frames, format="extxyz")
