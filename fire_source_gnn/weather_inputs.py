"""Bounded device cache of immutable monitor weather/calendar covariates."""
from collections import OrderedDict
import torch


class WeatherInputCache:
    def __init__(self, budget_bytes):
        if budget_bytes < 0:
            raise ValueError('Invalid weather cache budget')
        self.budget_bytes = int(budget_bytes)
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

    def window(self, model, data, split, first, last):
        key = (data, split, model.time_encoding, str(model.pm_mean.device), str(model.pm_mean.dtype))
        if key in self.entries:
            raw, normalized, size = self.entries.pop(key)
            self.entries[key] = (raw, normalized, size)
            self.hits += 1
            return raw[first:last], normalized[first:last]
        self.misses += 1
        array = data.splits[split]['monitor']
        # Retain only seven raw physical weather fields plus normalized model features.
        features = 11 if model.time_encoding == 'cyclical' else 9
        size = array.shape[0] * array.shape[1] * (7 + features) * model.pm_mean.element_size()
        if not self.budget_bytes or size > self.budget_bytes:
            if self.budget_bytes:
                self.oversized += 1
            return model.prepare_monitor_inputs(array[first:last])
        # Evict before allocating; live autograd references are outside this budget.
        while self.entries and self.bytes + size > self.budget_bytes:
            _, (_, _, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        with torch.inference_mode(False), torch.no_grad():
            raw, normalized = model.prepare_monitor_inputs(array)
        size = sum(t.numel() * t.element_size() for t in (raw, normalized))
        self.entries[key] = (raw, normalized, size)
        self.bytes += size
        return raw[first:last], normalized[first:last]
