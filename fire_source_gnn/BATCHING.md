# Concurrent graph batching

Fresh runs use up to four independent windows concurrently, with an effective
optimizer batch of 32 windows. The H100 default is
`config.h100-weather-cyclical.json`; the Windows research default is
`config.weather-cyclical.json`. Older explicit configurations without
`concurrent_windows` retain concurrency one.

## Computation

`model.forward_batch(data, split, starts)` returns
`[window, forecast_hour, monitor]`. The scalar `model(..., start)` interface
still returns `[forecast_hour, monitor]` and uses the same implementation.

The history GRU evaluates `B * 112` independent 240-hour monitor histories in
one call. For each of 48 sequential forecast hours, monitor graphs are copied
with offsets of `window_index * 112`. Fire receptors receive the same offsets.
There are no edges between windows. Variable fire inputs are coalesced into
bounded chunks spanning multiple windows; no padding to the largest fire graph
or new graph-library dependency is required. Shared cached tensors are never
modified by these offsets. Each monitor retains its own hidden state, and its
previous prediction supplies the next hour's PM input.

All monitors/windows share the same learned parameters. Batching adds no
parameters and does not change a checkpoint's layer shapes. Cyclical versus
legacy time encodings still have different layer shapes.

## Scheduling and loss

`batch_size` counts windows per optimizer update, not concurrent groups.
`concurrent_windows` is the maximum windows in a forward/backward pass.
With the default limits and no reductions, eight groups of four produce one
update. The final group and final effective batch may be smaller.

`max_batch_edges` caps the sum of lag-expanded fire-to-monitor edges over each
group's complete forecast rollouts. Zero disables this estimate. Windows stay
in their shuffled order, and groups never cross an effective-batch boundary.
A single oversized window is processed alone. The estimate is not a GPU-memory
guarantee and does not prune edges or change the forecast horizon.

For an effective batch containing N windows, each group's backward loss is
`sum(per_window_MSE) / N`. Thus small final groups receive the correct weight.
Gradients are clipped and the optimizer steps only after all N windows have
contributed. Vectorization can change floating-point reduction order;
comparisons with sequential execution use numerical tolerances rather than
requiring bitwise equality. Validation/testing remain sequential and keep their existing
metric aggregation.

## CUDA memory recovery

Before each group, the trainer snapshots accumulated gradients to host memory
and saves Python, NumPy, CPU-Torch and CUDA RNG states. A CUDA OOM during
forward/backward discards the failed graph, clears input caches, restores the
snapshot, and retries from the same shuffled position. Progress/loss totals
are advanced only after a complete successful group.

Recovery reduces actual group size (normally 4 -> 2 -> 1). If one window still
fails, caches are disabled and nested activation checkpointing is enabled.
Edge chunks are capped at 32,768, then halved on subsequent failures down to
`min_edge_chunk_size` (default 1,024, or the configured chunk size if smaller).
The reduced settings stay active and are saved in `batching_runtime` inside
`last.pt` and `progress.json`; resumes restore them.

If the minimum single-window attempt still fails, the trainer atomically saves
the restored pre-group state and raises `BatchMemoryError`. Earlier windows
and their partial-batch gradients remain resumable. Changing hardware can help;
a resumed run retains its conservative settings. Recovery handles
forward/backward CUDA OOM, including a partially executed backward pass.
Non-memory exceptions are propagated. Optimizer-step or checkpoint-writing
failures are not retried; the preceding atomic checkpoint remains the recovery
point. No windows are silently dropped.

## Resume compatibility

New checkpoints keep format version 2 and add optional runtime fields.
Old checkpoints without these fields derive concurrency one from their saved
configuration. Omit `--config` on resume to restore that saved configuration.
Explicit configuration changes remain rejected. Start a new run to change
configured concurrency or time encoding.

Signal/time-budget pauses and periodic training checkpoints occur at successful
group boundaries. Partial gradients and the exact next unprocessed window are
saved; incomplete groups are not committed. A hard kill can lose work since the
last completed atomic save. Only one process may write a run directory.

## Verification

`test/test_concurrent_batching.py` checks packed index isolation, cached-input
immutability, workload estimates, partial loss weighting, CPU/CUDA comparison
with independent windows, rollback after simulated partial backward,
single-window exhaustion, and interrupted concurrent accumulation. Existing
model, cache, metric and resume tests cover compatibility.

Tests use small synthetic datasets and temporary run directories. They are not
full training runs or throughput/accuracy benchmarks. Current verification
results are in `nmt/verification/concurrent_batching`. No production training
or cluster deployment is part of this implementation.
