#!/usr/bin/env python3
"""Regenerate Orchestr.AI-style AL dashboard plots from saved outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from orchestr_range_al.plotting import (
    parse_orchestr_diagnostics_file,
    plot_orchestr_energy_trace,
    plot_orchestr_uncertainty_panels,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestr.AI-style AL dashboard plotter")
    parser.add_argument("--npz", default="pool_energy_trace.npz")
    parser.add_argument("--txt", default="al_pool_per_frame_diagnostics.txt")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--title",
        default="MACE-RANGE / Orchestr.AI pool AL",
        help="Title above the four-panel dashboard",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    diagnostics = parse_orchestr_diagnostics_file(args.txt)
    plot_orchestr_energy_trace(
        args.npz,
        out / "al_dashboard_energy_trace.png",
    )
    plot_orchestr_uncertainty_panels(
        diagnostics,
        out / "al_dashboard_panels.png",
        title=args.title,
    )
    print(f"Saved {out / 'al_dashboard_energy_trace.png'}")
    print(f"Saved {out / 'al_dashboard_panels.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
