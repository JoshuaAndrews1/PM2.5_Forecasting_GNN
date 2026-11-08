# Dynamic fire-source PM2.5 GNN

Implements the saved `fire_source_model_math/main.tex` equations using the existing
processed VIIRS/NLCD/weather tables. Defaults selected for this model:

* Maximum lag: **24 hours**, including lags 0 through 24.
* Fire-to-monitor cutoff: **300 km**, great-circle distance from actual coordinates.
* Observed history: **240 hours**; autoregressive prediction: **48 hours**.
* Receptors: all **112 monitors**, preserving their dataset order.

## Model

Fire nodes are active cell/source-hour records with positive FRP. An episode's
age is computed from the first-detection timestamp in `fire_id`. Each source
contains age, seven weather variables, raw FRP, a learned categorical NLCD
embedding, midnight Unix timestamp, and UTC hour. Unknown NLCD is category zero.

For each lag and candidate receptor, the transport network receives distance,
sine/cosine of relative wind direction, speed, lag, source weather and current
receptor weather. The wind direction is the direction of travel derived from
east/north components. Both angular features are zero in calm winds.

The message is **source-time FRP × softplus(transport network) × message network**.
Messages are summed over sources and lags, then transformed with a bias-free
linear layer and tanh, ensuring exactly zero fire contribution for an empty
neighborhood. The learned message itself can have either sign; this is not a
monotonic or physically mass-conserving dispersion model.

Receptors use previous PM2.5, seven weather variables, date/hour, and elevation.
The existing 5,840-edge monitor topology is retained. Edge distances and bearings
are recomputed from the unrounded monitor coordinates. A learned monitor-message
network aggregates incoming minus outgoing messages. A GRU combines monitor and
fire messages and predicts PM2.5; its predictions feed the next forecast hour.

A separate GRU reads all 240 historical PM/weather/calendar observations to
initialize recurrent state. Historical fire messages are not replayed during
this encoding. This is a deliberate architectural extension: the old baseline
accepted a 240-hour history but used only its final PM2.5 observation.

No future PM2.5 is read during prediction. However, future reanalysis and
retrospectively reconstructed fire power are used as agreed: **this is a
retrospective prediction model, not an operational real-time forecast**.

## Data and normalization

`prepare.py` packs existing files; it does not redo satellite reconstruction,
land-cover lookup, or meteorological interpolation. Sources are read-only.
The baseline monitor split endpoints are preserved for comparable evaluation:

| Split | Inclusive UTC interval | Retained active fire-hours |
|---|---|---:|
| Training | 2017-01-02 00:00 to 2018-12-31 00:00 | 2,003,581 |
| Validation | 2020-01-01 00:00 to 2020-12-31 00:00 | 1,846,161 |
| Testing | 2021-01-01 00:00 to 2021-12-31 00:00 | 1,455,973 |

Of 5,307,497 exported rows, 976 fall outside those monitor intervals, 614 have
missing weather, and 192 have zero FRP. These disjoint exclusions are recorded
in the cache manifest. Missing weather is not silently imputed. Unknown NLCD
rows remain included. No lag window imports records from another split.

Monitor/source continuous statistics and PM2.5 scaling are fitted on training
only and shared across splits. FRP remains in raw MW as the message multiplier;
its copy within the node feature vector is standardized. Geometry uses fixed
physical scales. Sparse arrays and nested activation checkpointing avoid a dense
time × fire-cell × receptor × lag tensor. No top-k fire pruning is used.

Fire precipitation is the prepared preceding-hour accumulation. Existing
monitor precipitation is retained; its upstream accumulation interval has not
been independently established here. Confirm that alignment before interpreting
the model as having fully harmonized meteorological preprocessing.

## Run from the research project root

The local training environment already has PyTorch 1.13.1 with CUDA. Preparation
can run with the normal Python environment containing NumPy and pandas.

```powershell
python PM2.5_Forecasting_GNN/fire_source_gnn/prepare.py --project-root . --output fire_source_cache

& .\PM2.5_Forecasting_GNN-main\.venv37\Scripts\python.exe PM2.5_Forecasting_GNN/fire_source_gnn/test_model.py

# Small engineering verification only: one 48-hour window per split.
& .\PM2.5_Forecasting_GNN-main\.venv37\Scripts\python.exe PM2.5_Forecasting_GNN/fire_source_gnn/train.py --cache fire_source_cache --output PM2.5_Forecasting_GNN/results/fire_source_smoke --epochs 1 --limit-windows 1

# Full training: every window, validation checkpoint selection, then held-out test.
& .\PM2.5_Forecasting_GNN-main\.venv37\Scripts\python.exe PM2.5_Forecasting_GNN/fire_source_gnn/train.py --cache fire_source_cache --output PM2.5_Forecasting_GNN/results/fire_source_full

# Export one checkpoint rollout with observed values for comparison.
& .\PM2.5_Forecasting_GNN-main\.venv37\Scripts\python.exe PM2.5_Forecasting_GNN/fire_source_gnn/predict.py --cache fire_source_cache --checkpoint PM2.5_Forecasting_GNN/results/fire_source_full/best.pt --output forecast.csv
```

Use a fresh output path for each preparation or new training run. Continue a
paused run with the same `--output` and `--resume`; do not change its config,
cache or window limit. `last.pt` atomically records weights, optimizer state,
random state, shuffled order, partial-batch gradients and evaluation progress.
Saves occur every optimizer step, every 32 evaluation windows, and at window
boundaries after 60 seconds. SIGUSR1/SIGTERM/SIGINT request a safe pause after
the current window, with exit code 75. A hard kill may lose work since the last
save. `--max-seconds` provides an optional per-invocation time budget.
`progress.json` reports the saved position. Use only one writer per run directory.
The original version saved no checkpoint before validation; its timed-out
first epoch cannot be recovered. `test_resume.py` compares interrupted and
uninterrupted runs, including partial batches and both evaluation phases.

The prepared cache
is approximately 875 MB. Its monitor/fire arrays are memory-mapped. The training
default is batch size 1; larger values accumulate gradients across variable-size
graphs and take their mean. Set model/training choices in `config.json`. Rebuild
the candidate cache if the distance cutoff changes. Lag changes need no rebuild.

The default full run has 17,186 training, 8,474 validation, and 8,450 testing
windows. Training uses RMSprop, gradient clipping, and validation early stopping.
Evaluation reports clipped and unclipped errors in original units, without
feeding test labels into model selection. Limited runs are explicitly marked
`verification_only` and do not establish model accuracy.

The source package does not require PyG or torch-scatter. Required libraries:
NumPy, pandas for preparation, and PyTorch for training/inference. Verified in
the existing Python 3.7 / PyTorch 1.13.1 environment; do not replace that working
environment merely to use this package.

## Verification

`test_model.py` covers lag inclusion/expiry, split isolation, geometry, calm wind,
exact zero-fire behavior, FRP weighting, chunk invariance, checkpoint gradients,
48-hour output, no future-PM leakage, history use, and boundary rejection.

`verify_real_data.py --cache fire_source_cache --output peak_check.json` selects
the busiest training hour by candidate-edge count and checks a 48-hour
forward/backward pass, finite gradients, and fire ablation. Its results are
engineering diagnostics, not trained predictive performance.
