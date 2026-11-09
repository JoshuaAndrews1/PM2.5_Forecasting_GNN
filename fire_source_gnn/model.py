"""Dynamic directed fire -> monitor messages and autoregressive PM2.5 GNN."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from data import NLCD
from reusable_inputs import FireInputCache
from weather_inputs import WeatherInputCache
from time_features import cyclical_time
from graph_batch import pack_fire_chunks


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class FireSourceGNN(nn.Module):
    def __init__(self, config, statistics, graph):
        super().__init__()
        self.config = dict(config)
        self.time_encoding = config.get('time_encoding', 'legacy')
        if self.time_encoding not in ('legacy', 'cyclical'):
            raise ValueError('time_encoding must be legacy or cyclical')
        time_dim = 4 if self.time_encoding == 'cyclical' else 2
        self.weather_input_cache = WeatherInputCache(
            int(config.get('weather_cache_mb', 0) * 1024**2))
        self.fire_input_cache = FireInputCache(
            int(config.get("fire_input_cache_mb", 0) * 1024**2))
        h, m, e = config['hidden_dim'], config['message_dim'], config['nlcd_embedding_dim']
        self.monitor_encoder = mlp(9+time_dim, h, h)  # previous PM, weather, calendar, elevation
        self.history_gru = nn.GRU(8+time_dim, h, batch_first=True)
        self.land_cover = nn.Embedding(len(NLCD), e)
        self.transport = mlp(19, h, 1)
        self.fire_message = mlp(9+time_dim+e+19+h, h, m)
        # Small initial messages temper the raw-MW weighting without changing FRP.
        nn.init.normal_(self.fire_message[-1].weight, std=1e-4)
        nn.init.zeros_(self.fire_message[-1].bias)
        self.fire_transform = nn.Sequential(nn.Linear(m, m, bias=False), nn.Tanh())
        self.monitor_message = mlp(2*h+3, h, m)
        self.recurrent = nn.GRUCell(h+2*m, h)
        self.output = nn.Linear(h, 1)
        for key, value in statistics.items():
            self.register_buffer(key, torch.as_tensor(np.asarray(value), dtype=torch.float32))
        self.register_buffer('monitor_edges', torch.as_tensor(graph['monitor_edges'], dtype=torch.long))
        self.register_buffer('monitor_geometry', torch.as_tensor(graph['monitor_geometry'], dtype=torch.float32))
        self.register_buffer('elevation', torch.as_tensor(graph['elevation'], dtype=torch.float32))

    def clear_input_caches(self):
        self.fire_input_cache.clear()
        self.weather_input_cache.clear()

    def clear_fire_input_cache(self):
        # Compatibility with existing callers; both derived caches must clear.
        self.clear_input_caches()

    def _apply(self, fn):
        # Cached tensors are not buffers; invalidate them on device/dtype changes.
        self.clear_input_caches()
        return super()._apply(fn)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Normalized cached features depend on the checkpoint's statistics buffers.
        self.clear_input_caches()
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def tensor(self, array, integer=False):
        return torch.as_tensor(np.array(array, copy=True), device=self.pm_mean.device,
                               dtype=torch.long if integer else torch.float32)

    def prepare_monitor_inputs(self, array):
        # Convert Unix seconds in float64 on CPU, before uploading model inputs.
        if self.time_encoding == 'cyclical':
            weather = self.tensor(array[..., :7])
            # Every monitor normally shares UTC timestamps. Encode once per hour,
            # rather than repeating calendar arithmetic across all 112 monitors.
            shared = (np.array_equal(array[..., 7], np.broadcast_to(array[:, :1, 7], array.shape[:2]))
                      and np.array_equal(array[..., 8], np.broadcast_to(array[:, :1, 8], array.shape[:2])))
            if shared:
                calendar = self.tensor(cyclical_time(array[:, 0, 7], array[:, 0, 8]))[:, None, :]
                calendar = calendar.expand(-1, array.shape[1], -1)
            else:
                calendar = self.tensor(cyclical_time(array[..., 7], array[..., 8]))
            normalized = torch.cat(((weather-self.monitor_mean[:7])/self.monitor_std[:7], calendar), dim=-1)
            return weather, normalized
        covariates = self.tensor(array)
        normalized = (covariates-self.monitor_mean)/self.monitor_std
        # A separate compact tensor keeps cache accounting independent of views.
        return covariates[..., :7].clone(), normalized

    def fire_tensor(self, array):
        if self.time_encoding == 'cyclical':
            array = np.concatenate((array[..., :9], cyclical_time(array[..., 9], array[..., 10])), axis=-1)
        return self.tensor(array)

    def normalize_fire_features(self, z):
        if self.time_encoding == 'cyclical':
            return torch.cat(((z[:, :9]-self.fire_mean[:9])/self.fire_std[:9], z[:, 9:]), dim=-1)
        return (z-self.fire_mean)/self.fire_std

    def transport_prefix(self, z, distance, bearing, lag):
        u, v = z[:, 2], z[:, 3]  # source wind components; independent of calendar encoding
        speed = torch.sqrt(u*u+v*v)
        theta = bearing-torch.atan2(u, v)
        active = (speed > 0).to(z.dtype)
        angular = torch.stack((torch.sin(theta)*active, torch.cos(theta)*active), dim=-1)
        # Fixed physical scales for geometry; weather uses TRAINING-only statistics.
        return torch.cat((distance[:, None]/self.config['fire_cutoff_km'], angular,
                          speed[:, None]/10.0, lag[:, None]/max(1, self.config['max_lag_hours']),
                          (z[:, 1:8]-self.fire_mean[1:8])/self.fire_std[1:8]), dim=-1)

    def transport_features(self, z, distance, bearing, lag, receptor_weather):
        return torch.cat((self.transport_prefix(z, distance, bearing, lag),
                          (receptor_weather-self.monitor_mean[:7])/self.monitor_std[:7]), dim=-1)

    def fire_chunk_prepared(self, h, normalized_weather, source, category, frp, target, prefix):
        transport = torch.cat((prefix, normalized_weather[target]), dim=-1)
        return self.fire_messages(h, source, category, frp, target, transport)

    def fire_messages(self, h, normalized_source, category, frp, target, transport):
        # Only immutable inputs are cached; all learned operations run every time.
        k = F.softplus(self.transport(transport))
        source = torch.cat((normalized_source, self.land_cover(category)), dim=-1)
        phi = self.fire_message(torch.cat((source, transport, h[target]), dim=-1))
        message = frp[:, None] * k * phi
        return h.new_zeros((len(h), self.config['message_dim'])).index_add(0, target, message)

    def fire_chunk(self, h, weather, z, category, frp, target, distance, bearing, lag):
        transport = self.transport_features(z, distance, bearing, lag, weather[target])
        return self.fire_messages(h, self.normalize_fire_features(z), category, frp, target, transport)

    def _fire_aggregate_batch(self, h, weather, data, split, hours):
        monitors = len(h)//len(hours)
        prepared = self.config.get('cache_weather_features', False)
        if prepared:
            weather = (weather-self.monitor_mean[:7])/self.monitor_std[:7]
        result = h.new_zeros((len(h), self.config['message_dim']))

        def inputs():
            for window, hour in enumerate(hours):
                for tensors in self.fire_input_cache.chunks(self, data, split, hour, prepared=prepared):
                    yield window*monitors, tensors

        for tensors in pack_fire_chunks(inputs(), self.config['edge_chunk_size']):
            # Bind immutable tensors to this chunk, including backward recomputation.
            def chunk(hh, ww, tensors=tensors):
                if prepared:
                    return self.fire_chunk_prepared(hh, ww, *tensors)
                return self.fire_chunk(hh, ww, *tensors)
            if (self.training and torch.is_grad_enabled()
                    and self.config.get('activation_checkpointing', 'nested') == 'nested'):
                result = result + checkpoint(chunk, h, weather, use_reentrant=True)
            else:
                result = result + chunk(h, weather)
        return self.fire_transform(result)

    def fire_aggregate(self, h, weather, data, split, hour):
        return self.fire_aggregate_batch(h, weather, data, split, (int(hour),))

    def fire_aggregate_batch(self, h, weather, data, split, hours):
        # Each hour's packed graphs are recomputed independently during backward.
        def aggregate(hh, ww):
            return self._fire_aggregate_batch(hh, ww, data, split, hours)
        if (self.training and torch.is_grad_enabled()
                and self.config.get('activation_checkpointing', 'nested') in ('nested', 'outer')):
            return checkpoint(aggregate, h, weather, use_reentrant=True)
        return aggregate(h, weather)

    def forward(self, data, split, start, horizon=None, use_fire=True):
        """Return [forecast_hour, monitor] for one window."""
        return self.forward_batch(data, split, (start,), horizon=horizon, use_fire=use_fire)[0]

    def forward_batch(self, data, split, starts, horizon=None, use_fire=True):
        """Return [window, forecast_hour, monitor] for disjoint graph windows."""
        starts = tuple(starts)
        horizon = self.config['forecast_hours'] if horizon is None else horizon
        history = self.config['history_hours']
        item = data.splits[split]
        if (not starts or horizon < 1
                or any(not isinstance(start, (int, np.integer)) or start < history
                       or start+horizon > len(item['pm25']) for start in starts)):
            raise ValueError('Window outside this split')
        starts = tuple(int(start) for start in starts)
        windows, monitors = len(starts), item['pm25'].shape[1]
        inputs = [self.weather_input_cache.window(
            self, data, split, start-history, start+horizon) for start in starts]
        covariates = torch.stack([raw for raw, _ in inputs])
        normalized = torch.stack([normalized for _, normalized in inputs])
        observed = self.tensor(np.stack([item['pm25'][start-history:start] for start in starts]))
        observed = (observed-self.pm_mean)/self.pm_std
        # Histories are independent sequences, all evaluated in the same GRU call.
        hist = torch.cat((observed[..., None], normalized[:, :history]), dim=-1)
        hist = hist.permute(0, 2, 1, 3).reshape(windows*monitors, history, -1)
        _, hidden = self.history_gru(hist)
        hidden, previous = hidden[0], observed[:, -1].reshape(-1)
        elevation = ((self.elevation-self.elevation_mean)/self.elevation_std)[:, None].repeat(windows, 1)
        # Offset complete monitor graphs. No edge joins different windows.
        offsets = torch.arange(windows, device=self.monitor_edges.device)*monitors
        edges = self.monitor_edges[None] + offsets[:, None, None]
        src, dst = edges.permute(1, 0, 2).reshape(2, -1)
        geometry = self.monitor_geometry.repeat(windows, 1)
        predictions = []
        for offset in range(horizon):
            current = normalized[:, history+offset].reshape(windows*monitors, -1)
            h = self.monitor_encoder(torch.cat((previous[:, None], current, elevation), dim=-1))
            messages = self.monitor_message(torch.cat((h[src], h[dst], geometry), dim=-1))
            pm_message = h.new_zeros((len(h), self.config['message_dim'])).index_add(0, dst, messages)
            pm_message = pm_message-h.new_zeros(pm_message.shape).index_add(0, src, messages)
            weather = covariates[:, history+offset].reshape(windows*monitors, 7)
            hours = tuple(start+offset for start in starts)
            fire = (self.fire_aggregate_batch(h, weather, data, split, hours)
                    if use_fire else h.new_zeros(pm_message.shape))
            hidden = self.recurrent(torch.cat((h, pm_message, fire), dim=-1), hidden)
            previous = self.output(hidden).squeeze(-1)
            predictions.append(previous.reshape(windows, monitors))
        return torch.stack(predictions, dim=1)
