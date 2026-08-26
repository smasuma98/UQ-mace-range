# Implementation notes: alignment with Orchestr.AI `ivan2`

This scaffold was written against the current `ivan2` postprocessing API.

## Reused directly

- `orchestr_ai.postprocessing.active_learning.compute_soap_features`
- `orchestr_ai.postprocessing.active_learning.calibrate_alpha_reg_gcv`
- `orchestr_ai.postprocessing.active_learning.adaptive_learning_mig_pool_windowed`
- `orchestr_ai.postprocessing.active_learning.write_pool_al_diagnostics_csv`
- `orchestr_ai.postprocessing.rdf.compute_rdf_thresholds_from_reference`
- `orchestr_ai.postprocessing.uq_metrics_calculator.VarianceScalingCalibrator`
- `orchestr_ai.postprocessing.uq_metrics_calculator.IsotonicCalibrator`

## Why `features.compute_features` is not called directly

The current Orchestr.AI `compute_features` function expects its own nested
`config["eval"]["SOAP"]` schema, combines SOAP with bond length/min-distance
features, and writes scaler/PCA artifacts to fixed working-directory paths.

For this adapter, SOAP descriptor generation is still imported from
Orchestr.AI, while scaler/PCA fitting and artifact paths are kept in the
adapter so the single YAML schema and output directory remain deterministic.

## Selector shape contract

The pool wrapper maps positional arguments to:

1. `pool_frames`
2. `F_pool`
3. `F_train`
4. `alpha_sq`
5. `L`
6. `forces_train`
7. `sigma_energy`
8. `sigma_force`
9. `mu_E_frame_train`
10. `mu_E_pool`
11. `sigma_E_pool`
12. `mu_F_pool`
13. `sigma_F_pool`
14. `rdf_thresholds`

The force sigma inputs are per-frame arrays of component standard deviations
with shape `(n_atoms, 3)`. `sigma_F_pool_mean` and `sigma_F_pool_max` are
additional scalar summaries.

`forces_train` is populated with **reference forces**, matching Orchestr.AI's
current evaluator, because it is used to derive training force-magnitude caps.

## Feature-space GP calibration

Before pool selection, Orchestr.AI's evaluator fits:

```python
calibrate_alpha_reg_gcv(F_train, energy_error_per_atom)
```

The adapter follows the same pattern and passes the returned Cholesky factor
`L` plus `alpha_sq` to the pool learner.

## Threshold defaults

The YAML in this scaffold preserves the project values requested in the design
brief (`percentile_F_low: 95`, `percentile_F_hi: 99`, hard floors
0.001/0.075/0.150). The current `ivan2/evaluate.py` has different fallback
percentiles when those keys are absent. Because this adapter passes the YAML
values explicitly, selection uses the requested policy while retaining the
same Orchestr.AI selector implementation.
