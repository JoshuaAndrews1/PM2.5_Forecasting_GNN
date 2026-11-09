# Dynamic fire-source PM2.5 GNN

The model predicts all 112 monitors for 48 autoregressive forecast hours after
encoding 240 observed hours. Fire sources cover lags 0 through 24 inclusive
and receptors within a 300 km great-circle cutoff.

## Architecture

A fire source is an active cell/source-hour record with positive FRP.
Its numeric features are episode age, seven source-weather fields, FRP, and
calendar features; a learned NLCD embedding represents land cover. Episode age
comes from the first-detection timestamp in `fire_id`. Unknown NLCD is category
zero. Fresh runs use UTC-hour and annual-phase sine/cosine pairs rather than
Unix seconds; see [WEATHER_TIME.md](WEATHER_TIME.md).

For each source/lag/receptor edge, the transport network receives distance,
relative-wind sine/cosine, speed, lag, source weather and current receptor
weather. Wind direction is the direction of travel derived from east/north
components. Both angular features are zero in calm winds.

Each message is source-time FRP multiplied by softplus(learned transport) and
the learned message vector. Messages are summed over sources and lags, then
passed through a bias-free linear layer and tanh. An empty fire neighborhood
therefore contributes exactly zero. Messages can have either sign; the model
does not enforce monotonicity or physical mass conservation.

Each monitor is encoded from previous PM2.5, seven weather fields, calendar
features and elevation. The existing 5,840-edge monitor topology is retained,
with geometry from unrounded coordinates. Incoming minus outgoing learned
messages provide monitor interactions. A shared GRUCell updates one independent
hidden state per monitor, and a shared linear layer maps its 32 hidden values
to one PM2.5 prediction. There are no separately trained per-monitor readouts.

A separate history GRU initializes those states from each monitor's complete
240-hour observed PM/weather/calendar sequence. Historical fire messages are
not replayed. At each forecast hour both the updated hidden state and previous
predicted PM are carried forward. Unix time is not fed to the cyclical model.

No future PM observations are read. Future reanalysis weather and retrospectively
reconstructed FRP are supplied: this remains a retrospective prediction model,
rather than an operational forecast with unknown future covariates.

## Concurrent windows

Fresh configurations process up to four disjoint windows together and accumulate
32 windows per optimizer update. Histories, monitor networks and packed fire
messages are vectorized across windows. Forecast hours stay sequential.
Workload limits and CUDA OOM recovery can reduce concurrency without changing
the effective batch or graph membership. See [BATCHING.md](BATCHING.md).

The scalar model interface returns [forecast_hour, monitor]; the batched
interface returns [window, forecast_hour, monitor]. Validation/testing currently
use the scalar interface. Batching adds no learned parameters.

## Prepared data and normalization

`prepare.py` packs already processed satellite/NLCD/weather data into read-only,
memory-mapped arrays. It does not rerun those upstream processing stages.
The disk cache keeps original timestamps for provenance/export; model-boundary
conversion supplies cyclical features, so no cache rebuild is needed for that
encoding. Rebuild candidate edges if the distance cutoff changes; changing the
maximum lag alone needs no geometry rebuild.

| Split | Inclusive UTC interval | Retained active fire-hours |
|---|---|---:|
| Training | 2017-01-02 00:00 to 2018-12-31 00:00 | 2,003,581 |
| Validation | 2020-01-01 00:00 to 2020-12-31 00:00 | 1,846,161 |
| Testing | 2021-01-01 00:00 to 2021-12-31 00:00 | 1,455,973 |

The existing manifest records 5,307,497 exported fire rows, excluding 976 outside
the monitor intervals, 614 with missing weather, and 192 with zero FRP.
Unknown NLCD remains included. No lag window imports another split's records.

Continuous physical features and PM scaling use training-only statistics.
FRP stays in raw MW as the message multiplier; its source-feature copy is
standardized. Calendar sine/cosine values remain bounded and unstandardized.
Geometry has fixed physical scales. Sparse chunks avoid a dense
time x fire-cell x receptor x lag tensor, without top-k pruning.

Source precipitation is the prepared preceding-hour accumulation. Monitor
precipitation is retained from the existing dataset; its upstream accumulation
interval has not been independently established.

## Training and evaluation

Fresh defaults use RMSprop at 0.0005 with weight decay 0.0005, gradient clipping,
up to 15 epochs and patience 3. The complete data provide 17,186 training,
8,474 validation and 8,450 test windows. The final partial effective batch is
included. `best.pt` and early stopping use unclipped normalized validation MSE.
Held-out testing follows model selection.

HPC metrics include pooled RMSE/MAE, unclipped errors and the mean of each
window/station's forecast-horizon RMSE. Reported clipped metrics use
max(prediction, 0) with unchanged observations. Clipping is not fed back into
autoregression. HPC `best_window_station.pt` independently tracks that
validation metric, and `epochs/epoch-NNNN.pt` retains completed epoch models.
These snapshots are for evaluation; resume training exclusively from `last.pt`.

`last.pt` atomically records weights, optimizer, shuffled progress, accumulated
gradients, random states and batching runtime limits. Training saves at optimizer
steps and periodically at complete-group boundaries; evaluation saves every
32 windows or at the time interval. Pauses exit 75. See [BATCHING.md](BATCHING.md)
for memory recovery, resume behavior and remaining failure limits.

Training/inference require NumPy and PyTorch. Preparation additionally requires
pandas. Neither PyG nor torch-scatter is required. The HPC environment uses
PyTorch 2.5.1/CUDA 12.1; local compatibility tests use the existing
Python 3.7/PyTorch 1.13.1 environment. Cached weather is described in
[REUSABLE_INPUTS.md](REUSABLE_INPUTS.md).

## Research entry points

From the Windows research project root, the local Python executable is
`PM2.5_Forecasting_GNN-main/.venv37/Scripts/python.exe`.
The model scripts live under `PM2.5_Forecasting_GNN/fire_source_gnn`.

Fresh `train.py --cache PATH --output NEW_DIRECTORY` selects the cyclical,
weather-cached, concurrent configuration. Existing explicit config files retain
their saved choices. To continue a run use the same cache/output and `--resume`;
omitting `--config` restores the checkpoint configuration.

Use `predict.py --cache PATH --checkpoint CHECKPOINT --output NEW_CSV` for a
retrospective rollout. Use a fresh output path and only trusted checkpoints.
For cluster operations follow the active WSL `hpc/README.md`.
