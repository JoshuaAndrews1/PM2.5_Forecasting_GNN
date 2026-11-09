"""Read-only access to prepared hourly arrays and sparse fire-source records."""
import json
from pathlib import Path
import numpy as np

WEATHER = ['boundary_layer_height_m', 'wind_u_100m_m_s', 'wind_v_100m_m_s',
           'temperature_2m_k', 'dewpoint_2m_k', 'surface_pressure_pa', 'precipitation_1h_m']
MONITOR_COLUMNS = [4, 0, 1, 3, 2, 6, 5, 15, 16]
NLCD = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]


def geometry(source, target):
    """Pairwise great-circle kilometres and initial bearing clockwise from north."""
    a, b = np.radians(source), np.radians(target)
    lat1, lon1 = a[:, 0, None], a[:, 1, None]
    lat2, lon2 = b[None, :, 0], b[None, :, 1]
    dl = lon2 - lon1
    h = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dl/2)**2
    dist = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    bearing = np.arctan2(np.sin(dl)*np.cos(lat2),
                        np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dl))
    return dist.astype('float32'), bearing.astype('float32')


class PreparedData:
    def __init__(self, directory):
        self.root = Path(directory)
        self.meta = json.loads((self.root / 'manifest.json').read_text())
        self.stats = dict(np.load(str(self.root / 'statistics.npz')))
        self.graph = dict(np.load(str(self.root / 'graph.npz')))
        self.splits = {}
        for split in ('training', 'validation', 'testing'):
            self.splits[split] = {name: np.load(str(self.root / split / (name+'.npy')), mmap_mode='r')
                for name in ('monitor', 'pm25', 'fire_features', 'fire_frp', 'fire_nlcd',
                             'fire_cell', 'fire_hour', 'hour_ptr')}

    def starts(self, split, history, horizon):
        return np.arange(history, len(self.splits[split]['pm25'])-horizon+1, dtype=np.int64)

    def forecast_edge_counts(self, split, horizon, lag):
        """Count all lag-expanded edges for every valid rollout start.

        The count includes lags 0..lag and every forecast hour. It is a
        scheduling estimate, not a guarantee about peak CUDA allocation.
        """
        if horizon < 1 or lag < 0:
            raise ValueError('Invalid horizon or lag')
        cache = self.__dict__.setdefault('_forecast_edge_counts', {})
        key = (split, horizon, lag)
        if key not in cache:
            item = self.splits[split]
            degree = np.diff(self.graph['candidate_ptr'])
            hourly = np.bincount(item['fire_hour'], weights=degree[item['fire_cell']],
                                 minlength=len(item['pm25'])).astype(np.int64)
            sums = np.r_[0, np.cumsum(hourly)]
            hours = np.arange(len(hourly))
            lagged = sums[hours+1]-sums[np.maximum(0, hours-lag)]
            totals = np.r_[0, np.cumsum(lagged)]
            cache[key] = totals[horizon:]-totals[:-horizon]
        return cache[key]

    def close(self):
        """Release memory maps after a training invocation, including safe pauses."""
        for item in self.splits.values():
            for array in item.values():
                mapping = getattr(array, '_mmap', None)
                if mapping is not None:
                    mapping.close()

    def edge_chunks(self, split, hour, lag, chunk_size):
        """Yield bounded source-row/receptor/lag arrays, never a dense fire tensor.

        Fire records are unique (cell, source hour). All lags 0..L are included.
        Rows absent from this split stay absent; another split is never consulted.
        """
        item = self.splits[split]
        if lag < 0 or chunk_size < 1:
            raise ValueError('Invalid lag or chunk size')
        if not 0 <= hour < len(item['monitor']):
            raise IndexError(hour)
        lo, hi = item['hour_ptr'][max(0, hour-lag)], item['hour_ptr'][hour+1]
        ptr, dst = self.graph['candidate_ptr'], self.graph['candidate_target']
        # Blocks of records cap temporary expansion to chunk_size * receptor_count.
        nmonitor = item['monitor'].shape[1]
        for first in range(int(lo), int(hi), max(1, chunk_size // nmonitor)):
            rows = np.arange(first, min(first+max(1, chunk_size//nmonitor), hi), dtype=np.int64)
            cells = item['fire_cell'][rows]
            counts = ptr[cells+1]-ptr[cells]
            total = int(counts.sum())
            if not total:
                continue
            source = np.repeat(rows, counts)
            offsets = np.repeat(ptr[cells], counts)
            local = np.arange(total)-np.repeat(np.cumsum(counts)-counts, counts)
            edges = offsets+local
            for p in range(0, total, chunk_size):
                r, e = source[p:p+chunk_size], edges[p:p+chunk_size]
                yield r, dst[e], (hour-item['fire_hour'][r]).astype('float32'), e
