# Reusable fire and weather inputs

`FireInputCache` is a bounded device-side LRU keyed by data object, split,
target hour, lag/chunk settings, device/dtype, calendar encoding and prepared
layout. A positive `fire_input_cache_mb` enables retention; zero streams
inputs without retaining them.

The raw layout stores source features, category, FRP, receptor indices,
distance, bearing and lag. With `cache_weather_features: true`, it instead
stores normalized source features, category, FRP, receptor indices and the
12 fixed transport fields. No learned embeddings, transport weights, messages,
hidden states or predictions are cached.

`WeatherInputCache` separately retains split-level physical monitor weather
and normalized monitor/calendar features, bounded by `weather_cache_mb`.
Oversized splits fall back to window conversion. See
[WEATHER_TIME.md](WEATHER_TIME.md) for cyclical conversion and normalization.

Concurrent graph packing offsets receptor indices into new tensors without
modifying cached entries. Sources stay isolated by split and window graph.
Inputs must remain immutable; call `model.clear_input_caches()` after manual
changes. `clear_fire_input_cache()` remains an alias for older callers.
Device/dtype changes and `load_state_dict` invalidate both caches.

Cache `info()` exposes bytes, budget, hits, misses and evictions. Budgets cover
cache-owned storage, not live autograd references, temporary packed graphs or
allocator reservation. Oversized fire hours stream without retention. The
fire cache also has a 1,024-entry cap for quiet/empty hours. Normal tensors are
created even during inference mode so entries can be used later for training.
Caches are transient and excluded from checkpoints.

Configured budgets are device-specific. Fresh H100 settings retain 6 GiB of
fire inputs and 256 MiB of monitor weather; the Windows research default uses
512 MiB and 256 MiB respectively. CUDA OOM recovery can disable caches and
persist those reduced runtime limits. See [BATCHING.md](BATCHING.md).

Legacy cache examples `config.h100-cached.json` (HPC) and
`config.cached.json` (Windows) preserve their original training choices.
New defaults additionally enable weather preparation, cyclical time and
concurrent batching. Resume without `--config` to restore saved choices;
explicit config changes remain rejected.

`test/test_reusable_inputs.py`, `test/test_weather_time.py` and
`test/test_concurrent_batching.py` cover cache lifecycle, immutable packing and
training equivalence. The benchmark scripts are optional engineering checks:
`benchmark_reusable_inputs.py` and `benchmark_weather_time.py` measure
single-window accumulation and should not be presented as concurrent-batching
throughput results. Historical benchmark artifacts describe the source version
used at their creation.
