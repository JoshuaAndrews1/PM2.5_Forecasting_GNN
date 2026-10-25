"""Forecast-only metrics in original PM2.5 units."""
import numpy as np


def forecast_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 4 or not prediction.size:
        raise ValueError('Expected matching nonempty [window, horizon, station, channel] arrays')
    error = prediction - target
    hit = np.count_nonzero((prediction >= 75) & (target >= 75))
    miss = np.count_nonzero((prediction < 75) & (target >= 75))
    false_alarm = np.count_nonzero((prediction >= 75) & (target < 75))
    def ratio(a, b):
        return float(a) / b if b else float('nan')
    # Retain the upstream window/station averaging definition, explicitly named.
    return {
        'rmse_window_station': float(np.sqrt(np.mean(error ** 2, axis=1)).mean()),
        'rmse_pooled': float(np.sqrt(np.mean(error ** 2))),
        'mae': float(np.abs(error).mean()),
        'csi': ratio(hit, hit + miss + false_alarm),
        'pod': ratio(hit, hit + miss),
        'far': ratio(false_alarm, hit + false_alarm),
    }
