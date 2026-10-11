# Corrected wildfire baseline

Starting point: `kyleenliao/PM2.5_Forecasting_GNN`, commit
`ab63ceddfc612631d31bcb58cef78b2800a4adbe`. This is the prescribed-fire
adaptation of PM2.5-GNN. These fixes change training and evaluation behavior;
results should be described as a corrected implementation, not an exact
reproduction of published scores.

## Fixes

- Clear gradients for each training batch. Disable gradient tracking during
  validation, testing, and simulation.
- Fit feature and PM2.5 normalization on the training split and reuse it for
  validation/test. Save `normalization.npz` beside each checkpoint. Constant
  features receive a nonzero scale.
- Evaluate only the forecast horizon. Saved prediction/label arrays retain
  historical context for compatibility; slice `[:, hist_len:]` when scoring.
- Preserve the upstream mean of per-window/per-station RMSEs and label it
  explicitly. Also report pooled RMSE in each run's `metrics.json`. MAE is
  pooled. Undefined CSI/POD/FAR denominators produce NaN in text and null in JSON.
- Restore the missing `get_mean_std` helper that previously crashed the final
  experiment summary.
- Include the first complete time window and every final partial batch.
  Average losses by sample count, not by batch count.
- Use validation for checkpoint selection, then evaluate the best checkpoint
  on test data once. Reject nonfinite losses and empty eligible loaders.
- Correct graph wind reconstruction: the final two features are speed and
  direction, so use their normalization statistics rather than the u/v scales.
  Wind direction now points **toward** the destination, matching graph bearings.
  Convert angular differences from degrees to radians before cosine.
- Correct GC-LSTM batched edge offsets to use the number of stations; otherwise
  edges can connect unrelated samples. All recurrent models accept actual batch
  sizes, and single-node recurrent cells retain their batch dimension.
- Fix ambient preparation returning inside its first loop iteration. Load raw
  FRP automatically, retain clean windows within mixed batches, divide loss by
  the retained sample count, and remove the debugger breakpoint.
- Make training imports independent of expensive graph/data initialization.
  Memory-map the raw dataset and copy only the requested split.
- Simulation loads a corrected checkpoint and its training normalization,
  supports CPU execution, and includes partial batches. Experiment 1 loads the
  precomputed simulation inputs directly and reconstructs wind speed/direction
  from their raw u/v columns.

## Run locally

The adjacent `PM2.5_Forecasting_GNN-main` folder is preserved. Its dataset and
Python environment are reused by the ignored `config.local.yaml` and
`run_local.ps1` in this checkout. From this checkout in PowerShell:

```powershell
.\run_local.ps1
```

The local configuration retains batch size 4, 50 epochs, and 10 repetitions.
Review those settings before a full training run. Outputs go into this
checkout's `results` directory; the existing downloaded copy is not modified.

For another machine, set the paths in a config or use environment overrides:
`PM25_CONFIG` selects a YAML config, `PM25_DATA` overrides its input dataset,
and `PM25_RESULTS` overrides its output directory. Relative data/output paths
are resolved against the checkout. `train_ambient.py` defaults to
`config_ambient.yaml` unless `PM25_CONFIG` is set.

Run regression tests with the existing environment:

```powershell
..\PM2.5_Forecasting_GNN-main\.venv37\Scripts\python.exe -B -m unittest test_regressions -v
```

## Checkpoints and simulations

Start corrected training from a fresh model. The bundled `model.pth` and old
checkpoints were trained under different normalization/transport behavior and
must not be used with this implementation.

For simulation, set `PM25_CHECKPOINT` to a new run's `model.pth`; its sibling
`normalization.npz` is required. Set `PM25_SIM_EXPERIMENT` to `1` or `2`, and
select the appropriate config with `PM25_CONFIG`. Experiment 1 requires the
external files `data/input_exp1/{feature,pm25,time}.npy` and expects raw,
unnormalized features in the configured order. Experiment 2 requires its
separately prepared scenario dataset via `PM25_DATA`. Simulation input
preparation remains a separate research workflow.

