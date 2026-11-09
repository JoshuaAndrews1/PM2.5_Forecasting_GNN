# Weather caching and cyclical UTC features

Fresh research training defaults to `config.weather-cyclical.json`, including
up to four concurrent windows and an effective batch of 32; see [BATCHING.md](BATCHING.md).
The equivalent research configuration is `config.weather-cyclical.json` in
the Windows model project. Existing explicit configurations retain their
original behavior. A resume without `--config` restores the configuration
embedded in `last.pt`; explicit mismatches remain rejected.

## Model time inputs

Monitor and fire-source Unix date/hour fields are replaced at the model
boundary by four bounded inputs: UTC-hour sine/cosine and calendar-year-phase
sine/cosine. Annual phase includes the fractional day and uses the actual
365/366-day year. January 1 is phase zero. Fire episode age and transport
lag remain elapsed hours, not cyclic values.

Conversion uses the original float64 timestamps before GPU float32 conversion.
The four time inputs stay in [-1, 1]; training-only statistics continue to
normalize the seven weather fields and nine non-time source fields. Monitor
inputs have 11 features and fire inputs 13. The history GRU takes 12 inputs,
including observed PM. Cyclical and legacy checkpoints have different layer
shapes and require separate training runs.

The prepared disk cache remains unchanged, including its timestamps for
provenance and export. No satellite, NLCD, or weather processing needs rerunning.
The cyclical model does not receive Unix seconds.

## Cached weather work

`weather_cache_mb` bounds an LRU of complete split-level monitor weather
and normalized calendar/weather inputs on the model device. A split larger
than the budget falls back to window conversion. Shared monitor timestamps
are encoded once per hour. Only exogenous covariates are retained; future
PM observations never enter this cache.

With `cache_weather_features: true`, the existing fire-input LRU retains
normalized source features and the 12 fixed transport fields (geometry,
wind angle/speed, lag and source weather), replacing redundant raw tensors.
Receptor weather is normalized once per aggregation before its receptor gather.
Land-cover embeddings, learned transport weights, fire messages and hidden
states are always recomputed so gradients and parameter updates remain current.

Cache keys isolate data objects, splits, devices, dtypes, time encodings and
graph chunk settings. Device/dtype changes and `load_state_dict` clear both
caches because normalization depends on checkpoint statistics. Inputs/statistics
must remain immutable; after manual changes call `clear_input_caches()`.
The older `clear_fire_input_cache()` name remains an alias. Neither cache is saved in checkpoints.
Budgets bound cache-owned tensor storage, not live autograd references or
allocator reservation.

The H100 fresh-run configuration uses 6,144 MiB for fire inputs and 256 MiB
for monitor weather. The busiest saved rollout contains about 37.5 million
edges, requiring about 4.19 GiB of the new 120-byte prepared edge inputs.
A 512 MiB budget can cause substantial eviction. Budget tuning is device
specific; the local GPU benchmark uses smaller overrides.

## Run and verify

Fresh Python training (from this package):

```bash
python train.py --cache PATH_TO_PREPARED_CACHE --output results/weather-cyclical-new
```

Fresh Slurm submissions also select the new configuration. For an explicit
selection, export `FIRE_SOURCE_CONFIG=$PWD/config.h100-weather-cyclical.json`.
Resuming existing Slurm runs without that override retains their saved
configuration. Changing time encoding cannot upgrade a trained legacy model.

Checks:

```bash
python -m unittest discover -s test -p 'test_*.py' -v
python test/test_batch32.py
python benchmark_weather_time.py --cache cache --fire-cache-mb 2048 \
  --starts 15479 2008 4426 15479 2008 4426 \
  --output ../verification/weather_time_update/gpu_benchmark_2g.json
```

The benchmark compares identical cyclical weights, cache-off versus cache-on
weather features, with the same raw/prepared fire-cache budget and two
alternating-order trials. Losses are accumulated at effective batch size 32
before one optimizer step. This is an engineering check, not trained accuracy
or full-epoch throughput evidence. Test/report artifacts live under
`nmt/verification/weather_time_update`. Local changes do not deploy source
or submit cluster jobs.

Historical weather-cache benchmarks used single-window accumulation. They do
not measure concurrent-batching throughput or accuracy.
