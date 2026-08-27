# Orchestr.AI MACE-RANGE Active-Learning Postprocessing

## Purpose

This repository provides a **MACE-RANGE adapter** for Orchestr.AI-compatible
active-learning postprocessing of **precomputed** ML-MD trajectories.

It supports:

- sequential ensemble prediction with MACE-RANGE models;
- energy/force ensemble uncertainty quantification;
- validation against a labelled reference/training dataset;
- variance-scaling or isotonic uncertainty calibration;
- SOAP/PCA novelty features;
- active-learning frame selection by importing Orchestr.AI's
  `adaptive_learning_mig_pool_windowed` selector;
- Orchestr.AI-style diagnostics and selected-frame outputs.

The design rule is simple: **custom code is used for MACE-RANGE prediction;
generic postprocessing and selection are reused from Orchestr.AI wherever the
current API permits it.**

## Non-goals

Version 1 does **not** run:

- molecular dynamics;
- geometry optimization;
- vibrational analysis;
- NAMD;
- DFT labelling/submission;
- model retraining.

## Required environment

This repository deliberately does not install a second copy of MACE, because
the prediction backend must be your working **MACE-RANGE** implementation.

Before running, ensure these imports resolve in the same environment:

```bash
python -c "from mace.calculators import MACECalculator; print('MACE OK')"
python -c "from orchestr_ai.postprocessing.active_learning import adaptive_learning_mig_pool_windowed; print('Orchestr.AI OK')"
```

For the Orchestr.AI behavior targeted by this repository, use the `ivan2`
branch of `nlesc-nano/Orchestr.AI`.

A typical development install is:

```bash
git clone -b ivan2 https://github.com/nlesc-nano/Orchestr.AI.git
pip install -e ./Orchestr.AI

# Install/expose your MACE-RANGE source checkout according to that project.
# Then install this adapter:
pip install -e .
```

## Input files

The single YAML config points to:

1. one existing ASE-readable pool trajectory (`.xyz`/`.extxyz`);
2. one labelled training/reference dataset;
3. at least two (normally four) trained MACE-RANGE models.

Reference `energy` and `forces` may be explicit extxyz fields or ASE
`SinglePointCalculator` results.

Version 1 stores ensemble forces in dense arrays, so atom count must be
constant **within** each evaluated file. The pool and training dataset may have
different atom counts from each other.

## Single-config workflow

Edit:

```text
configs/config_range_al.yaml
```

Then run:

```bash
python scripts/run_range_al.py --config configs/config_range_al.yaml
```

or after installation:

```bash
orchestr-range-al --config configs/config_range_al.yaml
```

To validate the YAML without touching models/data:

```bash
python scripts/run_range_al.py --config configs/config_range_al.yaml --dry-run
```

Pipeline:

```text
pool trajectory + labelled training data + MACE-RANGE ensemble
        ↓
MACE-RANGE ensemble prediction (one model loaded at a time)
        ↓
ensemble UQ arrays
        ↓
reference validation
        ↓
uncertainty calibration + quality gates
        ↓
SOAP/PCA feature space
        ↓
Orchestr.AI calibrate_alpha_reg_gcv
        ↓
Orchestr.AI adaptive_learning_mig_pool_windowed
        ↓
selected frames + Orchestr.AI diagnostics + dashboard
```

## Important shape conventions

The adapter preserves both scalar summaries and component-wise uncertainty.

- `all_energies`: `(n_models, n_frames)`
- `all_forces`: `(n_models, n_frames, n_atoms, 3)`
- `sigma_force_components`: `(n_frames, n_atoms, 3)`
- `sigma_force_atom`: `(n_frames, n_atoms)`
- `sigma_force_mean`: `(n_frames,)`
- `sigma_force_max`: `(n_frames,)`

The Orchestr.AI pool selector receives **component-wise** force sigma for
training and pool frames. Scalar `sigma_force_mean`/`sigma_force_max` are passed
as additional pool summaries, not substituted for the component arrays.

The selector also receives the **reference training forces** for its
train-derived force magnitude caps.

## Calibration

Energy calibration is fitted on:

```text
energy error per atom  vs  sigma_E per atom
```

Force calibration is fitted component-wise on:

```text
(predicted - reference) force components  vs  sigma_force_components
```

The configured calibration is still written even if it fails the selection
quality gates. In that case, active learning falls back to raw uncertainty for
that quantity.

## Feature space

Version 1 uses Orchestr.AI's `compute_soap_features` helper, then fits a
`StandardScaler` and PCA on training frames only. The transformed matrices are
used as `F_train` and `F_pool`.

The current Orchestr.AI helper is non-periodic and uses averaged SOAP for
quantum-dot-style systems, therefore version 1 requires:

```yaml
features:
  periodic: false
  average: outer
```

MACE-RANGE latent features are intentionally deferred.

## Output files

For `output_prefix: range_al`, the pipeline writes into `range_al_output/`:

```text
range_al_predictions_pool.npz
range_al_predictions_train.npz
range_al_pool_summary.csv
range_al_train_validation.csv
range_al_validation_summary.csv
range_al_calibration_summary.csv
range_al_calibration_factors.json
range_al_annotated_pool.xyz
range_al_selected_frames.xyz
range_al_selected_indices.csv
# Canonical Orchestr.AI-style diagnostics/dashboard
al_pool_per_frame_diagnostics.txt
al_pool_diagnostics.csv
pool_energy_trace.npz
al_dashboard_energy_trace.png
al_dashboard_panels.png

# Adapter-prefix aliases / other outputs
range_al_pool_diagnostics.csv
range_al_per_frame_diagnostics.txt
range_al_selection_summary.json
range_al_run_manifest.json
feature_metadata.json
soap_scaler.joblib
soap_pca.joblib
```

The dashboard is generated from the **native Orchestr.AI per-frame diagnostics**,
so the plotted train thresholds, hard floors, geometry failures, physical-cap
failures and shortlist markers are the same values used by the selector.

Dashboard settings are controlled in the same single YAML:

```yaml
plots:
  dashboard_title: "MACE-RANGE / Orchestr.AI pool AL"
  energy_smoothing_window: 50
  save_legacy_plots: false
```

Set `save_legacy_plots: true` only if you also want the older compact debug
plots under `range_al_output/plots/`.

The dashboard can also be regenerated without rerunning MACE inference:

```bash
python scripts/plot_al_dashboard.py \
  --npz range_al_output/pool_energy_trace.npz \
  --txt range_al_output/al_pool_per_frame_diagnostics.txt \
  --output-dir range_al_output \
  --title "My pool trajectory"
```

Existing pool and training prediction NPZ files are reused by default, so a
repeat run does not evaluate those frames with the ensemble again. Each cache
is checked against the configured model paths, sampled source-frame indices,
and expected array shapes before use. To force fresh predictions, set:

```yaml
outputs:
  reuse_predictions: false
```

## CPU vs CUDA

CUDA:

```yaml
calculator:
  device: cuda
  enable_cueq: true
  use_range: true
```

CPU debugging:

```yaml
calculator:
  device: cpu
  enable_cueq: false
  use_range: true
```

`use_range: true` is mandatory and is never silently removed by the adapter.

## Notes for HPC

Models are evaluated **sequentially**, one model across all requested frames at
a time. This follows the working notebook design and avoids keeping the entire
ensemble resident on GPU.

For a smoke test, reduce the sampled trajectory range in YAML:

```yaml
sampling:
  start: 0
  stop: 5
  stride: 1
```

Plotting is optional and uses a headless Matplotlib backend.

## Relation to Orchestr.AI

This repository is not a rewrite of Orchestr.AI. It imports the current
Orchestr.AI implementation for:

- SOAP descriptor generation;
- GP/latent calibration (`calibrate_alpha_reg_gcv`);
- RDF/reference geometry thresholds;
- pool selection (`adaptive_learning_mig_pool_windowed`);
- pool diagnostic CSV writing.

The prediction backend is the intentional replacement: structures are
evaluated through `mace.calculators.MACECalculator(..., use_range=True)`.
