"""Dynamic directed fire -> monitor messages and autoregressive PM2.5 GNN."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from data import NLCD


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class FireSourceGNN(nn.Module):
    def __init__(self, config, statistics, graph):
        super().__init__()
        self.config = dict(config)
        h, m, e = config['hidden_dim'], config['message_dim'], config['nlcd_embedding_dim']
        self.monitor_encoder = mlp(11, h, h)  # previous PM, seven weather, date/hour, elevation
        self.history_gru = nn.GRU(10, h, batch_first=True)
        self.land_cover = nn.Embedding(len(NLCD), e)
        self.transport = mlp(19, h, 1)
        self.fire_message = mlp(11+e+19+h, h, m)
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

    def tensor(self, array, integer=False):
        return torch.as_tensor(np.array(array, copy=True), device=self.pm_mean.device,
                               dtype=torch.long if integer else torch.float32)

    def transport_features(self, z, distance, bearing, lag, receptor_weather):
        u, v = z[:, 2], z[:, 3]  # z = age, H, u, v, T, Td, P, R, FRP, J, hour
        speed = torch.sqrt(u*u+v*v)
        theta = bearing-torch.atan2(u, v)
        active = (speed > 0).to(z.dtype)
        angular = torch.stack((torch.sin(theta)*active, torch.cos(theta)*active), dim=-1)
        # Fixed physical scales for geometry; weather uses TRAINING-only statistics.
        return torch.cat((distance[:, None]/self.config['fire_cutoff_km'], angular,
                          speed[:, None]/10.0, lag[:, None]/max(1, self.config['max_lag_hours']),
                          (z[:, 1:8]-self.fire_mean[1:8])/self.fire_std[1:8],
                          (receptor_weather-self.monitor_mean[:7])/self.monitor_std[:7]), dim=-1)

    def fire_chunk(self, h, weather, z, category, frp, target, distance, bearing, lag):
        transport = self.transport_features(z, distance, bearing, lag, weather[target])
        k = F.softplus(self.transport(transport))
        source = torch.cat(((z-self.fire_mean)/self.fire_std, self.land_cover(category)), dim=-1)
        phi = self.fire_message(torch.cat((source, transport, h[target]), dim=-1))
        message = frp[:, None] * k * phi
        return h.new_zeros((len(h), self.config['message_dim'])).index_add(0, target, message)

    def _fire_aggregate(self, h, weather, data, split, hour):
        item, graph = data.splits[split], data.graph
        result = h.new_zeros((len(h), self.config['message_dim']))
        for rows, targets, lags, edges in data.edge_chunks(split, hour,
                self.config['max_lag_hours'], self.config['edge_chunk_size']):
            # Bind each chunk to its own closure: checkpoint recomputes it in backward.
            def chunk(hh, ww, rows=rows, targets=targets, lags=lags, edges=edges):
                return self.fire_chunk(hh, ww, self.tensor(item['fire_features'][rows]),
                    self.tensor(item['fire_nlcd'][rows], True), self.tensor(item['fire_frp'][rows]),
                    self.tensor(targets, True), self.tensor(graph['candidate_distance'][edges]),
                    self.tensor(graph['candidate_bearing'][edges]), self.tensor(lags))
            if self.training and torch.is_grad_enabled():
                result = result + checkpoint(chunk, h, weather, use_reentrant=True)
            else:
                result = result + chunk(h, weather)
        return self.fire_transform(result)

    def fire_aggregate(self, h, weather, data, split, hour):
        # Outer checkpoint prevents keeping every chunk's graph across 48 forecast hours.
        def aggregate(hh, ww):
            return self._fire_aggregate(hh, ww, data, split, hour)
        if self.training and torch.is_grad_enabled():
            return checkpoint(aggregate, h, weather, use_reentrant=True)
        return aggregate(h, weather)

    def forward(self, data, split, start, horizon=None, use_fire=True):
        horizon = self.config['forecast_hours'] if horizon is None else horizon
        history = self.config['history_hours']
        item = data.splits[split]
        if start < history or start+horizon > len(item['pm25']) or horizon < 1:
            raise ValueError('Window outside this split')
        covariates = self.tensor(item['monitor'][start-history:start+horizon])
        normalized = (covariates-self.monitor_mean)/self.monitor_std
        observed = (self.tensor(item['pm25'][start-history:start])-self.pm_mean)/self.pm_std
        # All 240 observed hours initialize temporal state. Future PM is never read.
        hist = torch.cat((observed[..., None], normalized[:history]), dim=-1).transpose(0, 1)
        _, hidden = self.history_gru(hist)
        hidden, previous = hidden[0], observed[-1]
        elevation = ((self.elevation-self.elevation_mean)/self.elevation_std)[:, None]
        predictions = []
        src, dst = self.monitor_edges
        for offset in range(horizon):
            h = self.monitor_encoder(torch.cat((previous[:, None], normalized[history+offset], elevation), dim=-1))
            messages = self.monitor_message(torch.cat((h[src], h[dst], self.monitor_geometry), dim=-1))
            pm_message = h.new_zeros((len(h), self.config['message_dim'])).index_add(0, dst, messages)
            pm_message = pm_message-h.new_zeros(pm_message.shape).index_add(0, src, messages)
            fire = (self.fire_aggregate(h, covariates[history+offset, :, :7], data, split, start+offset)
                    if use_fire else h.new_zeros(pm_message.shape))
            hidden = self.recurrent(torch.cat((h, pm_message, fire), dim=-1), hidden)
            previous = self.output(hidden).squeeze(-1)
            predictions.append(previous)
        return torch.stack(predictions)
