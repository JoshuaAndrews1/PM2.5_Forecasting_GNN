"""Concurrent window selection and transactional forward/backward OOM recovery."""
import gc
import random
import numpy as np
import torch
from torch.nn import functional as F


class BatchMemoryError(RuntimeError):
    """One complete window still cannot fit after bounded memory reductions."""


def validate_batching_config(config):
    for name, default in (('concurrent_windows', 1), ('min_edge_chunk_size', 1024)):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name+' must be a positive integer')
    budget = config.get('max_batch_edges', 0)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError('max_batch_edges must be a nonnegative integer')
    if config['edge_chunk_size'] < 1:
        raise ValueError('edge_chunk_size must be positive')
    if config.get('activation_checkpointing', 'nested') not in ('nested', 'outer', 'none'):
        raise ValueError('Unknown activation checkpoint policy')


def select_group(order, position, end, concurrency, edge_counts, edge_budget):
    """Choose an ordered prefix; never cross an optimizer batch boundary."""
    group = []
    edges = 0
    for start in order[position:min(position+concurrency, end)]:
        start = int(start)
        count = int(edge_counts[start]) if edge_counts is not None else 0
        if group and edge_budget and edges+count > edge_budget:
            break
        group.append(start)
        edges += count
    if not group:
        raise ValueError('Cannot select an empty training group')
    return group


def is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and any(
            marker in str(error).lower() for marker in ('cuda out of memory', 'cuda error: out of memory')))


def snapshot_accumulation(model):
    """Only immutable weights are used within an attempt; snapshot gradients/RNG."""
    return dict(
        gradients={name: p.grad.detach().cpu().clone()
                   for name, p in model.named_parameters() if p.grad is not None},
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if model.pm_mean.is_cuda else None)


def restore_accumulation(model, snapshot):
    for name, parameter in model.named_parameters():
        gradient = snapshot['gradients'].get(name)
        parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    random.setstate(snapshot['python_rng'])
    np.random.set_state(snapshot['numpy_rng'])
    torch.set_rng_state(snapshot['torch_rng'])
    if snapshot['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(snapshot['cuda_rng'])


class ConcurrentTrainer:
    """Accumulate complete groups; update progress/weights only in the caller."""

    def __init__(self, model, config, progress):
        validate_batching_config(config)
        self.model = model
        self.edge_budget = config.get('max_batch_edges', 0)
        self.minimum_chunk = min(config.get('min_edge_chunk_size', 1024), config['edge_chunk_size'])
        self.runtime = progress.setdefault('batching_runtime', dict(
            concurrent_windows=min(config.get('concurrent_windows', 1), config['batch_size']),
            edge_chunk_size=config['edge_chunk_size'],
            activation_checkpointing=config.get('activation_checkpointing', 'nested'),
            fire_cache_bytes=model.fire_input_cache.budget_bytes,
            weather_cache_bytes=model.weather_input_cache.budget_bytes,
            oom_retries=0))
        self._apply_runtime()

    def _apply_runtime(self):
        self.model.config['edge_chunk_size'] = self.runtime['edge_chunk_size']
        self.model.config['activation_checkpointing'] = self.runtime['activation_checkpointing']
        self.model.fire_input_cache.budget_bytes = self.runtime['fire_cache_bytes']
        self.model.weather_input_cache.budget_bytes = self.runtime['weather_cache_bytes']

    def _attempt(self, data, split, group, denominator):
        # Keep all failed autograd tensors/tracebacks local to this function.
        try:
            prediction = (self.model(data, split, group[0])[None]
                          if len(group) == 1 else self.model.forward_batch(data, split, group))
            horizon = prediction.shape[1]
            truth = self.model.tensor(np.stack([
                data.splits[split]['pm25'][start:start+horizon] for start in group]))
            truth = (truth-self.model.pm_mean)/self.model.pm_std
            losses = F.mse_loss(prediction, truth, reduction='none').reshape(len(group), -1).mean(dim=1)
            if not bool(torch.isfinite(losses).all()):
                raise ValueError('Nonfinite loss')
            (losses.sum()/denominator).backward()
            return float(losses.detach().sum()), None
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            return None, str(error)

    def _release_failed_attempt(self):
        self.model.zero_grad(set_to_none=True)
        self.model.clear_input_caches()
        gc.collect()
        if self.model.pm_mean.is_cuda:
            torch.cuda.empty_cache()

    def _reduce_memory(self, group_size):
        if group_size > 1:
            self.runtime['concurrent_windows'] = max(1, group_size//2)
            return True
        # A final partial group can already have size one while its configured
        # concurrency is larger. Keep later groups/resumes conservative too.
        self.runtime['concurrent_windows'] = 1
        chunk = self.runtime['edge_chunk_size']
        if (self.runtime['fire_cache_bytes'] or self.runtime['weather_cache_bytes']
                or self.runtime['activation_checkpointing'] != 'nested'):
            self.runtime.update(fire_cache_bytes=0, weather_cache_bytes=0,
                                activation_checkpointing='nested',
                                edge_chunk_size=max(self.minimum_chunk, min(chunk, 32768)))
            return True
        if chunk > self.minimum_chunk:
            self.runtime['edge_chunk_size'] = max(self.minimum_chunk, chunk//2)
            return True
        return False

    def accumulate(self, data, split, order, position, end, denominator, edge_counts=None):
        snapshot = snapshot_accumulation(self.model)
        while True:
            group = select_group(order, position, end, self.runtime['concurrent_windows'],
                                 edge_counts, self.edge_budget)
            total, error = self._attempt(data, split, group, denominator)
            if error is None:
                return len(group), total
            # The failed function has returned: no exception traceback pins its graph.
            self._release_failed_attempt()
            restore_accumulation(self.model, snapshot)
            if not self._reduce_memory(len(group)):
                raise BatchMemoryError(
                    'CUDA OOM at window {} even with one concurrent window, caches disabled, '
                    'nested checkpointing and edge chunk {}. The failed group was rolled back.'
                    .format(group[0], self.runtime['edge_chunk_size']))
            self.runtime['oom_retries'] += 1
            self._apply_runtime()
            print('CUDA OOM: retrying windows from {} with concurrency={}, edge_chunk={}, '
                  'checkpointing={}, fire_cache_mib={:.0f}, weather_cache_mib={:.0f}'.format(
                      group[0], self.runtime['concurrent_windows'], self.runtime['edge_chunk_size'],
                      self.runtime['activation_checkpointing'], self.runtime['fire_cache_bytes']/1024**2,
                      self.runtime['weather_cache_bytes']/1024**2), flush=True)
