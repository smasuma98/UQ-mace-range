"""Optional headless plots generated from saved numerical results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _save_line(x, y, xlabel, ylabel, title, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_standard_plots(
    pool_summary: pd.DataFrame,
    train_validation: pd.DataFrame,
    *,
    selected_source_indices: np.ndarray | None,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = pool_summary["source_frame"].to_numpy()
    _save_line(
        x, pool_summary["energy_mean_eV"], "Source frame", "Mean energy (eV)",
        "Ensemble mean energy", output_dir / "energy_mean_trace.png"
    )
    _save_line(
        x, pool_summary["sigma_E_per_atom_eV"], "Source frame", "σE / atom (eV)",
        "Energy uncertainty", output_dir / "sigma_E_per_atom_trace.png"
    )
    _save_line(
        x, pool_summary["sigma_F_mean_eV_per_A"], "Source frame", "Mean σF (eV/Å)",
        "Mean force uncertainty", output_dir / "sigma_F_mean_trace.png"
    )
    _save_line(
        x, pool_summary["sigma_F_max_eV_per_A"], "Source frame", "Max σF (eV/Å)",
        "Maximum force uncertainty", output_dir / "sigma_F_max_trace.png"
    )
    _save_line(
        x, pool_summary["frame_max_mean_force_eV_per_A"], "Source frame", "Max |F| (eV/Å)",
        "Maximum predicted force", output_dir / "max_force_trace.png"
    )

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(
        train_validation["sigma_E_per_atom_eV"],
        np.abs(train_validation["energy_error_per_atom_eV"]),
        s=16,
        alpha=0.6,
    )
    ax.set_xlabel("σE / atom (eV)")
    ax.set_ylabel("|energy error| / atom (eV)")
    ax.set_title("Energy error vs uncertainty")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_uncertainty_energy.png", dpi=160)
    plt.close(fig)

    if selected_source_indices is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, pool_summary["sigma_F_max_eV_per_A"])
        selected = np.asarray(selected_source_indices, dtype=int)
        mask = np.isin(x, selected)
        ax.scatter(
            x[mask],
            pool_summary.loc[mask, "sigma_F_max_eV_per_A"],
            marker="x",
            s=45,
        )
        ax.set_xlabel("Source frame")
        ax.set_ylabel("Max σF (eV/Å)")
        ax.set_title("Selected frames")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "selected_frames_overlay.png", dpi=160)
        plt.close(fig)
