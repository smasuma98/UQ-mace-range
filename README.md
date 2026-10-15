# MACE-RANGE + Orchestr.AI Postprocessing

This repository provides a small postprocessing workflow that connects **MACE-RANGE** predictions with **Orchestr.AI** uncertainty quantification and active-learning selection.

## Before you start

You should already have a working **MACE-RANGE environment**.

**Orchestr.AI must be installed in the same environment.** This repository does not install or configure MACE-RANGE for you.

For the current workflow, use the `ivan2` branch of Orchestr.AI:

```bash
conda activate <your-mace-range-env>

git clone -b ivan2 https://github.com/nlesc-nano/Orchestr.AI.git
pip install -e ./Orchestr.AI
```

Then install this repository:

```bash
cd orchestrai-mace-range-al
pip install -e .
```

## What this repository does

It postprocesses already calculated data using an ensemble of trained MACE-RANGE models.

You provide:

- trained MACE-RANGE model files;
- an existing ML-MD trajectory;
- a labelled training/reference dataset;
- one YAML config file.

The workflow performs:

- ensemble energy and force prediction;
- uncertainty quantification;
- validation and uncertainty calibration;
- Orchestr.AI-style active-learning frame selection;
- selected-frame and diagnostic outputs.

## What it does not do

This repository does **not**:

- train MACE-RANGE models;
- run molecular dynamics;
- run geometry optimisation;
- perform DFT labelling;
- retrain models.

Your models and MD trajectories must therefore already exist before running the workflow.

## Run

Edit:

```text
configs/config_range_al.yaml
```

Then run:

```bash
python scripts/run_range_al.py --config configs/config_range_al.yaml
```

For a quick config check without running model inference:

```bash
python scripts/run_range_al.py --config configs/config_range_al.yaml --dry-run
```

Results are written to the output directory defined in the YAML config, including selected frames, diagnostics, uncertainty data, and Orchestr.AI-style dashboard plots. Good luck!
