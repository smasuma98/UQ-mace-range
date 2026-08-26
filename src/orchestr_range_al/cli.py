"""Single-config command-line pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

from .calibration import fit_calibration, write_calibration_outputs
from .config import load_config, path_from_config
from .diagnostics import write_run_manifest
from .ensemble import compute_ensemble_statistics
from .features import build_soap_pca_features
from .io import (
    frame_summary_dataframe,
    read_frames,
    read_reference_dataset,
    write_annotated_xyz,
    write_frame_summary_csv,
    write_predictions_npz,
    write_selected_frames,
)
from .plotting import generate_standard_plots
from .prediction import evaluate_ensemble
from .selection import run_orchestr_selection
from .validation import summary_dataframe, validate_predictions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MACE-RANGE -> Orchestr.AI active-learning postprocessing adapter"
    )
    parser.add_argument("--config", required=True, help="Single workflow YAML file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print planned workflow without loading data/models",
    )
    return parser


def _planned_workflow(cfg):
    print("Planned workflow:")
    print("  1. read precomputed pool trajectory")
    print("  2. read labelled training/reference dataset")
    print("  3. evaluate MACE-RANGE ensemble on pool")
    print("  4. evaluate MACE-RANGE ensemble on reference data")
    print("  5. compute ensemble UQ + validation")
    print("  6. fit/gate uncertainty calibration")
    print("  7. build SOAP/PCA train+pool features")
    print("  8. call Orchestr.AI adaptive_learning_mig_pool_windowed")
    print("  9. write selected frames + diagnostics")
    print(f"Output directory: {cfg['paths']['output_dir']}")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    _planned_workflow(cfg)
    if args.dry_run:
        return 0

    paths = cfg["paths"]
    output_dir = path_from_config(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(paths["output_prefix"])

    pool_path = path_from_config(paths["pool_trajectory"])
    train_path = path_from_config(paths["train_dataset"])
    model_paths = [path_from_config(p) for p in cfg["models"]]

    sampling = cfg["sampling"]
    pool_frames, pool_source = read_frames(
        pool_path,
        start=int(sampling.get("start", 0)),
        stop=sampling.get("stop"),
        stride=int(sampling.get("stride", 1)),
    )
    train_frames, ref_E, ref_F, train_source = read_reference_dataset(
        train_path,
        energy_key=cfg["reference"].get("energy_key", "energy"),
        forces_key=cfg["reference"].get("forces_key", "forces"),
    )

    print("\n=== Pool ensemble prediction ===")
    pool_ensemble = evaluate_ensemble(
        pool_frames,
        model_paths,
        cfg["calculator"],
        source_frame_indices=pool_source,
        progress_every=int(cfg.get("progress_every", 50)),
    )
    pool_stats = compute_ensemble_statistics(
        pool_ensemble["all_energies"], pool_ensemble["all_forces"]
    )

    print("\n=== Reference ensemble prediction ===")
    train_ensemble = evaluate_ensemble(
        train_frames,
        model_paths,
        cfg["calculator"],
        source_frame_indices=train_source,
        progress_every=int(cfg.get("progress_every", 50)),
    )
    train_stats = compute_ensemble_statistics(
        train_ensemble["all_energies"], train_ensemble["all_forces"]
    )

    pool_payload = {**pool_ensemble, **pool_stats}
    train_payload = {**train_ensemble, **train_stats}
    if cfg["outputs"].get("save_npz", True):
        write_predictions_npz(
            output_dir / f"{prefix}_predictions_pool.npz", pool_payload
        )
        write_predictions_npz(
            output_dir / f"{prefix}_predictions_train.npz", train_payload
        )

    pool_df = frame_summary_dataframe(pool_ensemble, pool_stats)
    if cfg["outputs"].get("save_csv", True):
        write_frame_summary_csv(
            output_dir / f"{prefix}_pool_summary.csv", pool_df
        )

    if cfg["outputs"].get("save_annotated_xyz", True):
        write_annotated_xyz(
            output_dir / f"{prefix}_annotated_pool.xyz",
            pool_frames,
            pool_ensemble,
            pool_stats,
        )

    print("\n=== Reference validation ===")
    validation = validate_predictions(
        ref_E,
        ref_F,
        train_stats,
        source_frame_indices=train_source,
    )
    validation["per_frame"].to_csv(
        output_dir / f"{prefix}_train_validation.csv", index=False
    )
    summary_dataframe(validation).to_csv(
        output_dir / f"{prefix}_validation_summary.csv", index=False
    )

    print("\n=== Uncertainty calibration ===")
    calibration = fit_calibration(validation, train_stats, cfg["calibration"])
    write_calibration_outputs(output_dir, prefix, calibration)
    print(
        "Calibration accepted for AL: "
        f"energy={calibration.energy_accepted}, force={calibration.force_accepted}"
    )

    print("\n=== SOAP/PCA feature construction ===")
    feature_result = build_soap_pca_features(
        train_frames,
        pool_frames,
        cfg["features"],
        output_dir=output_dir,
    )

    selected = {
        "selected_frames": [],
        "selected_indices": np.asarray([], dtype=int),
        "selected_source_indices": np.asarray([], dtype=int),
        "summary": {"n_selected": 0},
    }
    if cfg["active_learning"].get("enabled", True):
        print("\n=== Orchestr.AI pool active learning ===")
        selected = run_orchestr_selection(
            pool_frames=pool_frames,
            train_frames=train_frames,
            pool_source_indices=pool_source,
            F_pool=feature_result["F_pool"],
            F_train=feature_result["F_train"],
            pool_stats=pool_stats,
            train_stats=train_stats,
            validation=validation,
            calibration=calibration,
            active_cfg=cfg["active_learning"],
            rdf_cfg=cfg["rdf_filter"],
            output_dir=output_dir,
            prefix=prefix,
        )
        if cfg["outputs"].get("save_selected_xyz", True):
            write_selected_frames(
                output_dir / f"{prefix}_selected_frames.xyz",
                selected["selected_frames"],
            )

    if cfg["outputs"].get("save_plots", True):
        print("\n=== Optional plots ===")
        generate_standard_plots(
            pool_df,
            validation["per_frame"],
            selected_source_indices=selected["selected_source_indices"],
            output_dir=output_dir / "plots",
        )

    write_run_manifest(
        output_dir / f"{prefix}_run_manifest.json",
        {
            "config": cfg["_config_path"],
            "pool_trajectory": str(pool_path),
            "train_dataset": str(train_path),
            "models": [str(p) for p in model_paths],
            "n_pool_frames": len(pool_frames),
            "n_train_frames": len(train_frames),
            "n_models": len(model_paths),
            "validation_summary": validation["summary"],
            "calibration_mode": calibration.mode,
            "energy_calibration_used_for_selection": calibration.energy_accepted,
            "force_calibration_used_for_selection": calibration.force_accepted,
            "selection_summary": selected["summary"],
        },
    )

    print(f"\nFinished. Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
