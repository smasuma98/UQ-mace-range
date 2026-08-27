"""Optional headless plots for MACE-RANGE / Orchestr.AI postprocessing.

The default dashboard intentionally mirrors the Orchestr.AI pool-active-learning
visualization: one smoothed pool-energy trace and one four-panel uncertainty /
physical-cap dashboard driven by the selector's per-frame diagnostics file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge values retained."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    window = max(1, int(window))
    if window == 1:
        return arr.copy()
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def parse_orchestr_diagnostics_file(path: str | Path) -> dict[str, Any]:
    """Parse Orchestr.AI ``*_per_frame_diagnostics.txt`` output.

    Returns pool arrays, training arrays and the threshold metadata written by
    ``_PoolActiveLearner``.  Both Unicode sigma column names and their ASCII
    fallbacks are accepted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Active-learning diagnostics file not found: {path}")

    pool_idx: list[int] = []
    pool_sE: list[float] = []
    pool_sFmax: list[float] = []
    pool_sFmean: list[float] = []
    pool_Fmax: list[float] = []
    pool_selected: list[bool] = []
    pool_shortlist: list[bool] = []
    pool_geom_ok: list[bool] = []
    pool_caps_ok: list[bool] = []

    train_idx: list[int] = []
    train_sE: list[float] = []
    train_sFmax: list[float] = []
    train_sFmean: list[float] = []
    train_Fmax: list[float] = []
    thresholds: dict[str, float] = {}

    in_pool = False
    in_train = False
    colnames: list[str] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()

            if stripped.startswith("# TRAIN DATASET"):
                in_train, in_pool = True, False
                continue

            if not in_pool and not in_train:
                if stripped.startswith("#") and "=" in stripped:
                    left, right = stripped.lstrip("#").split("=", 1)
                    try:
                        thresholds[left.strip()] = float(right.strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                elif stripped.startswith("idx") and "shortlist" in stripped:
                    in_pool = True
                    colnames = stripped.split()
                continue

            if in_pool and not in_train:
                if stripped.startswith("#") or stripped.startswith("-") or not stripped:
                    continue
                parts = stripped.split()
                if colnames is None or len(parts) < len(colnames):
                    continue
                row = dict(zip(colnames, parts))
                try:
                    pool_idx.append(int(row["idx"]))
                    pool_sE.append(float(row.get("σE_atom", row.get("sE_atom"))))
                    pool_sFmax.append(float(row.get("σF_max", row.get("sF_max"))))
                    pool_sFmean.append(float(row.get("σF_mean", row.get("sF_mean"))))
                    pool_Fmax.append(float(row.get("Fmax", row.get("F_mag"))))
                    pool_selected.append(bool(int(row.get("selected", 0))))
                    pool_shortlist.append(bool(int(row.get("shortlist", 0))))
                    pool_geom_ok.append(bool(int(row.get("geom_ok", row.get("rdf_ok", 0)))))
                    pool_caps_ok.append(bool(int(row.get("caps_ok", row.get("pass_caps", 0)))))
                except (TypeError, ValueError, KeyError):
                    continue

            if in_train:
                if stripped.startswith("#") or stripped.startswith("-") or not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 5:
                    continue
                try:
                    train_idx.append(int(parts[0]))
                    train_sE.append(float(parts[1]))
                    train_sFmax.append(float(parts[2]))
                    train_sFmean.append(float(parts[3]))
                    train_Fmax.append(float(parts[4]))
                except ValueError:
                    continue

    return {
        "pool_idx": np.asarray(pool_idx, dtype=int),
        "pool_sigma_E_atom": np.asarray(pool_sE, dtype=float),
        "pool_sigma_F_max": np.asarray(pool_sFmax, dtype=float),
        "pool_sigma_F_mean": np.asarray(pool_sFmean, dtype=float),
        "pool_Fmax": np.asarray(pool_Fmax, dtype=float),
        "pool_selected": np.asarray(pool_selected, dtype=bool),
        "pool_shortlist": np.asarray(pool_shortlist, dtype=bool),
        "pool_geom_ok": np.asarray(pool_geom_ok, dtype=bool),
        "pool_caps_ok": np.asarray(pool_caps_ok, dtype=bool),
        "train_idx": np.asarray(train_idx, dtype=int),
        "train_sigma_E_atom": np.asarray(train_sE, dtype=float),
        "train_sigma_F_max": np.asarray(train_sFmax, dtype=float),
        "train_sigma_F_mean": np.asarray(train_sFmean, dtype=float),
        "train_Fmax": np.asarray(train_Fmax, dtype=float),
        "thresholds": thresholds,
    }


def write_pool_energy_trace_npz(
    pool_summary: pd.DataFrame,
    diagnostics: dict[str, Any],
    path: str | Path,
    *,
    smoothing_window: int = 50,
) -> Path:
    """Write the compact energy-trace cache used by the Orchestr dashboard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_pool = len(pool_summary)
    diag_idx = np.asarray(diagnostics.get("pool_idx", []), dtype=int)
    geom_ok = np.asarray(diagnostics.get("pool_geom_ok", []), dtype=bool)

    if len(diag_idx) == n_pool:
        steps = diag_idx
    elif "sampled_frame" in pool_summary:
        steps = pool_summary["sampled_frame"].to_numpy(dtype=int)
    else:
        steps = np.arange(n_pool, dtype=int)

    if len(geom_ok) != n_pool:
        geom_ok = np.ones(n_pool, dtype=bool)

    mu_raw = pool_summary["energy_mean_eV"].to_numpy(dtype=float)
    sigma_raw = pool_summary["sigma_E_eV"].to_numpy(dtype=float)
    mu = _rolling_mean(mu_raw, smoothing_window)
    sigma = _rolling_mean(sigma_raw, smoothing_window)
    bad = ~geom_ok

    np.savez_compressed(
        path,
        steps=steps,
        mu=mu,
        sigma=sigma,
        bad=bad,
        mu_raw=mu_raw,
        sigma_raw=sigma_raw,
        smoothing_window=np.asarray(int(smoothing_window)),
    )
    return path


def plot_orchestr_energy_trace(
    npz_path: str | Path,
    output_path: str | Path,
    *,
    smoothing_window: int | None = None,
) -> Path:
    """Plot the Orchestr.AI-style pool energy ± uncertainty trace."""
    npz_path = Path(npz_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    steps = np.asarray(data["steps"])
    mu = np.asarray(data["mu"], dtype=float)
    sigma = np.asarray(data["sigma"], dtype=float)
    bad = np.asarray(data["bad"], dtype=bool)
    if smoothing_window is None:
        smoothing_window = int(np.asarray(data.get("smoothing_window", 50)).item())

    lower, upper = mu - sigma, mu + sigma
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, mu, label=f"{smoothing_window}-pt MA of μE")
    ax.fill_between(steps, lower, upper, alpha=0.3, label="σ (smoothed)")
    if bad.any():
        ax.scatter(
            steps[bad],
            mu[bad],
            marker="x",
            color="r",
            alpha=0.5,
            label="Failed Geometry",
        )
    ax.set_xlabel("Pool frame index")
    ax.set_ylabel("Predicted energy")
    ax.set_title("Pool energy ± uncertainty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_orchestr_uncertainty_panels(
    diagnostics: dict[str, Any],
    output_path: str | Path,
    *,
    title: str = "MACE-RANGE / Orchestr.AI pool AL",
) -> Path:
    """Plot the four-panel Orchestr.AI AL dashboard."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pool_idx = diagnostics["pool_idx"]
    pool_sE = diagnostics["pool_sigma_E_atom"]
    pool_sFmax = diagnostics["pool_sigma_F_max"]
    pool_sFmean = diagnostics["pool_sigma_F_mean"]
    pool_Fmax = diagnostics["pool_Fmax"]
    pool_shortlist = diagnostics["pool_shortlist"]
    pool_geom_ok = diagnostics["pool_geom_ok"]
    pool_caps_ok = diagnostics["pool_caps_ok"]
    train_idx = diagnostics["train_idx"]
    train_sE = diagnostics["train_sigma_E_atom"]
    train_sFmax = diagnostics["train_sigma_F_max"]
    train_sFmean = diagnostics["train_sigma_F_mean"]
    train_Fmax = diagnostics["train_Fmax"]
    thr = diagnostics["thresholds"]

    if pool_idx.size == 0:
        raise ValueError("No pool rows found in active-learning diagnostics file.")

    plt = _plt()
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 12), sharex=True, constrained_layout=True
    )

    def draw_lines(ax, low, hi, hard):
        if np.isfinite(low):
            ax.axhline(low, ls="--", lw=1.0, color="gray")
        if np.isfinite(hi):
            ax.axhline(hi, ls="--", lw=1.0, color="gray")
        if np.isfinite(hard):
            ax.axhline(hard, ls="--", lw=1.0, color="red")

    shortlist_x = pool_idx[pool_shortlist]
    failed_geom = ~pool_geom_ok
    failed_caps = pool_geom_ok & ~pool_caps_ok

    def plot_panel(
        ax,
        metric_pool,
        metric_train,
        ylabel,
        thr_low,
        thr_hi,
        thr_hard,
        *,
        mult=1.0,
    ):
        # Match the Orchestr.AI dashboard: full pool trace + explicit failure markers.
        ax.plot(pool_idx, metric_pool * mult, lw=1, color="blue", alpha=0.5, label="Valid Pool")
        if failed_caps.any():
            ax.scatter(
                pool_idx[failed_caps],
                metric_pool[failed_caps] * mult,
                color="orange",
                marker="x",
                alpha=0.6,
                s=15,
                label="Failed Caps (Phys)",
            )
        if failed_geom.any():
            ax.scatter(
                pool_idx[failed_geom],
                metric_pool[failed_geom] * mult,
                color="red",
                marker="x",
                alpha=0.3,
                s=15,
                label="Failed Geom (RDF)",
            )
        if pool_shortlist.any():
            ax.scatter(
                shortlist_x,
                metric_pool[pool_shortlist] * mult,
                color="k",
                marker="o",
                s=35,
                label="Shortlist",
                zorder=5,
            )
        if train_idx.size > 0:
            ax.plot(train_idx, metric_train * mult, lw=1, color="green", label="Train Max")

        draw_lines(
            ax,
            float(thr.get(thr_low, np.nan)) * mult,
            float(thr.get(thr_hi, np.nan)) * mult,
            float(thr.get(thr_hard, np.nan)) * mult,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plot_panel(
        axes[0],
        pool_sE,
        train_sE,
        r"$\sigma_E$/atom (meV)",
        "thr_sigma_E_low",
        "thr_sigma_E_hi_eff",
        "hard_sigma_E_atom_min",
        mult=1000.0,
    )
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    plot_panel(
        axes[1],
        pool_sFmax,
        train_sFmax,
        r"$\sigma F_{\max}$ (eV/$\AA$)",
        "thr_sigma_F",
        "thr_sigma_F_hi_eff",
        "hard_sigma_F_max_min",
    )
    plot_panel(
        axes[2],
        pool_sFmean,
        train_sFmean,
        r"$\sigma F_{\mathrm{mean}}$ (eV/$\AA$)",
        "thr_sigma_Fmean",
        "thr_sigma_Fmean_hi_eff",
        "hard_sigma_F_mean_min",
    )
    plot_panel(
        axes[3],
        pool_Fmax,
        train_Fmax,
        r"$|F|_{\max}$ (eV/$\AA$)",
        "thr_Fmag",
        "thr_Fmag_hi_eff",
        "train_Fmax_hard_cap",
    )
    axes[3].set_xlabel("Frame Index")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_orchestr_dashboard(
    pool_summary: pd.DataFrame,
    *,
    diagnostics_path: str | Path,
    output_dir: str | Path,
    title: str = "MACE-RANGE / Orchestr.AI pool AL",
    smoothing_window: int = 50,
) -> dict[str, Path]:
    """Generate the Orchestr.AI-compatible dashboard and cache files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = parse_orchestr_diagnostics_file(diagnostics_path)
    energy_npz = write_pool_energy_trace_npz(
        pool_summary,
        diagnostics,
        output_dir / "pool_energy_trace.npz",
        smoothing_window=smoothing_window,
    )
    energy_png = plot_orchestr_energy_trace(
        energy_npz,
        output_dir / "al_dashboard_energy_trace.png",
        smoothing_window=smoothing_window,
    )
    panels_png = plot_orchestr_uncertainty_panels(
        diagnostics,
        output_dir / "al_dashboard_panels.png",
        title=title,
    )
    return {
        "energy_trace_npz": energy_npz,
        "energy_trace_png": energy_png,
        "panels_png": panels_png,
    }


# ---------------------------------------------------------------------------
# Legacy compact plots retained as an optional debugging view.
# ---------------------------------------------------------------------------

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
    """Generate the older compact plots; disabled by default in the YAML."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = pool_summary["source_frame"].to_numpy()
    _save_line(
        x,
        pool_summary["energy_mean_eV"],
        "Source frame",
        "Mean energy (eV)",
        "Ensemble mean energy",
        output_dir / "energy_mean_trace.png",
    )
    _save_line(
        x,
        pool_summary["sigma_E_per_atom_eV"],
        "Source frame",
        "σE / atom (eV)",
        "Energy uncertainty",
        output_dir / "sigma_E_per_atom_trace.png",
    )
    _save_line(
        x,
        pool_summary["sigma_F_mean_eV_per_A"],
        "Source frame",
        "Mean σF (eV/Å)",
        "Mean force uncertainty",
        output_dir / "sigma_F_mean_trace.png",
    )
    _save_line(
        x,
        pool_summary["sigma_F_max_eV_per_A"],
        "Source frame",
        "Max σF (eV/Å)",
        "Maximum force uncertainty",
        output_dir / "sigma_F_max_trace.png",
    )
    _save_line(
        x,
        pool_summary["frame_max_mean_force_eV_per_A"],
        "Source frame",
        "Max |F| (eV/Å)",
        "Maximum predicted force",
        output_dir / "max_force_trace.png",
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
