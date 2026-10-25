"""Shared forecast batches, model construction and streaming evaluation."""
import importlib
from pathlib import Path

import numpy as np
import torch

MODELS = ('PM25_GNN', 'PM25_GNN_nosub', 'LSTM', 'MLP', 'GRU', 'GC_LSTM', 'nodesFC_GRU')


def build_model(name, hist_len, pred_len, in_dim, cities, batch_size, device, graph, normalization):
    if name not in MODELS:
        raise ValueError('Unknown neural model: '+name)
    cls = getattr(importlib.import_module('model.'+name), name)
    args = [hist_len, pred_len, in_dim]
    if name != 'MLP':
        args += [cities, batch_size, device]
    if name == 'GC_LSTM':
        args += [graph.edge_index]
    elif name.startswith('PM25_GNN'):
        args += [graph.edge_index, graph.edge_attr,
                 normalization['feature_mean'][-2:], normalization['feature_std'][-2:]]
    return cls(*args).to(device)


def window_batch(data, starts, hist_len, pred_len):
    """Return history, full covariates and forecast labels for window starts."""
    idx = np.asarray(starts)[:, None] + np.arange(hist_len+pred_len)[None, :]
    pm = data.pm25_full[idx]
    return pm[:, :hist_len], data.feature_full[idx], pm[:, hist_len:]


class ForecastScores:
    """Pool errors without storing overlapping history or all errors."""
    def __init__(self):
        self.absolute = self.squared = self.group_rmse = 0.0
        self.count = self.groups = self.windows = 0

    def add(self, prediction, target):
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape or prediction.ndim != 4 or not prediction.size:
            raise ValueError('Expected matching [window, horizon, station, channel] arrays')
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError('Nonfinite predictions or targets')
        error = prediction-target
        self.absolute += np.abs(error).sum()
        self.squared += np.square(error).sum()
        self.group_rmse += np.sqrt(np.square(error).mean(axis=1)).sum()
        self.count += error.size
        self.groups += error.shape[0]*error.shape[2]*error.shape[3]
        self.windows += error.shape[0]

    def result(self):
        if not self.count:
            raise ValueError('No forecast samples were evaluated')
        return dict(mae=float(self.absolute/self.count),
                    rmse_pooled=float(np.sqrt(self.squared/self.count)),
                    rmse_window_station=float(self.group_rmse/self.groups),
                    windows=self.windows, forecast_values=self.count)


@torch.no_grad()
def evaluate(model, data, hist_len, pred_len, batch_size, device, output=None,
             max_windows=None, random_forest=False):
    """Score original-unit forecasts; optionally export forecast-only arrays."""
    if batch_size < 1 or (max_windows is not None and max_windows < 1):
        raise ValueError('Batch size and window limit must be positive')
    count = len(data) if max_windows is None else min(len(data), max_windows)
    if not count:
        raise ValueError('No eligible forecast windows')
    if not random_forest:
        model.eval()
    scores = ForecastScores()
    predictions = targets = None
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        shape = (count, pred_len, data.pm25_full.shape[1], 1)
        predictions = np.lib.format.open_memmap(str(output/'predictions.npy'), mode='w+', dtype='float32', shape=shape)
        targets = np.lib.format.open_memmap(str(output/'targets.npy'), mode='w+', dtype='float64', shape=shape)
        starts = data.time_arr_full[np.arange(count)+hist_len]
        np.save(str(output/'forecast_times.npy'), starts[:, None]+np.arange(pred_len)[None, :]*3600)
    for first in range(0, count, batch_size):
        ids = np.arange(first, min(first+batch_size, count))
        history, features, truth = window_batch(data, ids, hist_len, pred_len)
        if random_forest:
            shape = (len(ids), pred_len, data.pm25_full.shape[1], 1)
            pm = np.broadcast_to(history[:, -1:, :, :], shape)
            lead = np.broadcast_to(np.arange(1,pred_len+1,dtype=np.float32)[None,:,None,None]/pred_len, shape)
            x = np.concatenate([pm, features[:, hist_len:], lead], axis=-1)
            pred = model.predict(x.reshape(-1,x.shape[-1])).reshape(shape)
        else:
            pred = model(torch.as_tensor(history,device=device),
                         torch.as_tensor(features,device=device)).cpu().numpy()
        pred = np.maximum(pred*data.pm25_std+data.pm25_mean, 0)
        truth = truth.astype(np.float64)*data.pm25_std+data.pm25_mean
        scores.add(pred,truth)
        if predictions is not None:
            predictions[first:first+len(ids)] = pred
            targets[first:first+len(ids)] = truth
        if first % (100*batch_size) == 0:
            print('Evaluated {}/{} windows'.format(first+len(ids),count),flush=True)
    if predictions is not None:
        predictions.flush()
        targets.flush()
    result = scores.result()
    result.update(horizon_hours=pred_len, stations=data.pm25_full.shape[1],
                  negative_predictions_clipped_to_zero=True, units='ug/m3',
                  partial_evaluation=count<len(data))
    return result
