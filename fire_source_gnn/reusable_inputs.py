"""Bounded LRU of immutable fire graph/input tensors, never learned messages.

PreparedData arrays must remain read-only. Clear the cache if inputs change.
The byte budget counts tensors owned by the cache, not autograd's live references.
"""
from collections import OrderedDict
import torch


class FireInputCache:
    def __init__(self, budget_bytes, max_entries=1024):
        if budget_bytes < 0 or max_entries < 1:
            raise ValueError("Invalid fire input cache limits")
        self.budget_bytes = int(budget_bytes)
        self.max_entries = int(max_entries)
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.oversized = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def info(self):
        return dict(bytes=self.bytes, budget_bytes=self.budget_bytes,
                    entries=len(self.entries), hits=self.hits, misses=self.misses,
                    evictions=self.evictions, oversized=self.oversized)

    def chunks(self, model, data, split, hour, prepared=False):
        cfg = model.config
        # Retain the data object in the key: object-id reuse must never return stale data.
        key = (data, split, int(hour), cfg['max_lag_hours'], cfg['edge_chunk_size'],
               str(model.pm_mean.device), str(model.pm_mean.dtype), model.time_encoding, prepared)
        if key in self.entries:
            chunks, size = self.entries.pop(key)
            self.entries[key] = (chunks, size)
            self.hits += 1
            yield from chunks
            return
        self.misses += 1
        item, graph = data.splits[split], data.graph
        pending, size = [], 0
        keep = self.budget_bytes > 0
        for rows, targets, lags, edges in data.edge_chunks(
                split, hour, cfg['max_lag_hours'], cfg['edge_chunk_size']):
            # Inference-created entries must remain usable by later training.
            with torch.inference_mode(False):
                chunk = (model.fire_tensor(item['fire_features'][rows]),
                         model.tensor(item['fire_nlcd'][rows], True),
                         model.tensor(item['fire_frp'][rows]), model.tensor(targets, True),
                         model.tensor(graph['candidate_distance'][edges]),
                         model.tensor(graph['candidate_bearing'][edges]), model.tensor(lags))
            if prepared:
                # Weather/geometry/normalization contain no learned parameters.
                with torch.inference_mode(False), torch.no_grad():
                    z, category, frp, target, distance, bearing, lag = chunk
                    chunk = (model.normalize_fire_features(z), category, frp, target,
                             model.transport_prefix(z, distance, bearing, lag))
            chunk_bytes = sum(t.numel()*t.element_size() for t in chunk)
            if keep:
                size += chunk_bytes
                if size > self.budget_bytes:
                    pending.clear()
                    keep = False
                    self.oversized += 1
                else:
                    # Make room while building, rather than exceeding the budget at insertion.
                    while self.entries and self.bytes + size > self.budget_bytes:
                        _, (_, removed) = self.entries.popitem(last=False)
                        self.bytes -= removed
                        self.evictions += 1
                    pending.append(chunk)
            yield chunk
        if keep:
            while len(self.entries) >= self.max_entries:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
                self.evictions += 1
            self.entries[key] = (tuple(pending), size)
            self.bytes += size
