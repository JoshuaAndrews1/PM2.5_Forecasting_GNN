"""Concurrent graph isolation, loss weighting, OOM rollback and resumability."""
import sys
from pathlib import Path

# Keep source imports working from the test subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from batching import (BatchMemoryError, ConcurrentTrainer, select_group,
                      snapshot_accumulation, validate_batching_config)
from graph_batch import pack_fire_chunks
from model import FireSourceGNN
from test_model import fixture
from test_resume import create_fixture, assert_tree_equal
from test_weather_time import cyclical_fixture
import train


class ConcurrentBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_packed_indices_are_disjoint_and_cached_indices_unchanged(self):
        target = torch.tensor([0, 1, 0])
        tensors = (torch.ones(3, 2), torch.zeros(3, dtype=torch.long),
                   torch.ones(3), target, torch.zeros(3, 12))
        packed = list(pack_fire_chunks(iter([(0, tensors), (2, tensors)]), 4))
        self.assertEqual([len(chunk[0]) for chunk in packed], [4, 2])
        self.assertEqual(torch.cat([chunk[3] for chunk in packed]).tolist(), [0, 1, 0, 2, 3, 2])
        self.assertEqual(target.tolist(), [0, 1, 0])

    def test_workload_budget_preserves_order_and_effective_batch_boundary(self):
        counts = np.array([3, 20, 4, 2, 1])
        order = np.array([2, 0, 1, 4, 3])
        self.assertEqual(select_group(order, 0, 5, 4, counts, 10), [2, 0])
        self.assertEqual(select_group(order, 2, 5, 4, counts, 10), [1])
        self.assertEqual(select_group(order, 3, 4, 4, counts, 0), [4])
        for updates in ({'concurrent_windows': 0}, {'max_batch_edges': -1},
                        {'activation_checkpointing': 'unknown'}):
            model, _ = fixture()
            with self.assertRaises(ValueError):
                validate_batching_config(dict(model.config, **updates))

    def test_edge_estimates_match_explicit_lag_expansion(self):
        model, data = fixture()
        for lag in (0, 24):
            counts = data.forecast_edge_counts('training', 4, lag)
            for start in (240, 241, 260, 270):
                actual = sum(len(rows) for hour in range(start, start+4)
                             for rows, _, _, _ in data.edge_chunks('training', hour, lag, 2))
                self.assertEqual(int(counts[start]), actual)

    def test_batched_predictions_and_gradients_match_independent_windows(self):
        for encoding in ('legacy', 'cyclical'):
            for policy in ('nested', 'outer', 'none'):
                for cached in (False, True):
                    with self.subTest(encoding=encoding, policy=policy, cached=cached):
                        batched, data = cyclical_fixture(cached=cached, policy=policy)
                        config = dict(batched.config, time_encoding=encoding)
                        batched = FireSourceGNN(config, data.stats, data.graph)
                        sequential = FireSourceGNN(config, data.stats, data.graph)
                        sequential.load_state_dict(batched.state_dict())
                        starts = [240, 241, 244, 270]
                        expected = []
                        for start in starts:
                            prediction = sequential(data, 'training', start, horizon=4)
                            expected.append(prediction.detach())
                            (prediction.square().mean()/len(starts)).backward()
                        result = batched.forward_batch(data, 'training', starts, horizon=4)
                        result.square().mean().backward()
                        self.assertEqual(tuple(result.shape), (4, 4, 2))
                        torch.testing.assert_close(result, torch.stack(expected), rtol=1e-4, atol=1e-6)
                        for (name, p), (_, reference) in zip(batched.named_parameters(), sequential.named_parameters()):
                            self.assertEqual(p.grad is None, reference.grad is None, name)
                            if p.grad is not None:
                                torch.testing.assert_close(p.grad, reference.grad, rtol=1e-4, atol=1e-6)
                        # Adding/removing another graph cannot change a window's neighbors.
                        with torch.no_grad():
                            isolated = batched(data, 'training', 240, horizon=4)
                        torch.testing.assert_close(result[0], isolated, rtol=1e-4, atol=1e-6)

    def test_partial_group_loss_is_weighted_by_windows(self):
        batched, data = cyclical_fixture()
        batched.config['forecast_hours'] = 3
        config = dict(batched.config, batch_size=5, concurrent_windows=4)
        batched = FireSourceGNN(config, data.stats, data.graph)
        sequential = FireSourceGNN(config, data.stats, data.graph)
        sequential.load_state_dict(batched.state_dict())
        order = np.arange(240, 245)
        runner = ConcurrentTrainer(batched, config, {})
        position = 0
        total = 0.
        sizes = []
        while position < len(order):
            count, loss = runner.accumulate(data, 'training', order, position, len(order), len(order))
            sizes.append(count)
            total += loss
            position += count
        expected = 0.
        for start in order:
            prediction = sequential(data, 'training', int(start))
            truth = sequential.tensor(data.splits['training']['pm25'][start:start+3])
            truth = (truth-sequential.pm_mean)/sequential.pm_std
            loss = torch.nn.functional.mse_loss(prediction, truth)
            expected += float(loss.detach())
            (loss/len(order)).backward()
        self.assertEqual(sizes, [4, 1])
        self.assertAlmostEqual(total, expected, places=5)
        for p, q in zip(batched.parameters(), sequential.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)

    def test_oom_restores_partial_gradients_and_all_rng_states(self):
        model, data = cyclical_fixture()
        model.config['forecast_hours'] = 3
        config = dict(model.config, concurrent_windows=4, batch_size=32)
        reference = FireSourceGNN(config, data.stats, data.graph)
        reference.load_state_dict(model.state_dict())
        for a, b in zip(model.parameters(), reference.parameters()):
            a.grad = torch.full_like(a, .25)
            b.grad = torch.full_like(b, .25)
        runner = ConcurrentTrainer(model, config, {})
        expected_rng = snapshot_accumulation(model)
        original = runner._attempt
        calls = []
        def fail_large(data, split, group, denominator):
            calls.append(len(group))
            if len(group) > 2:
                for p in model.parameters():
                    p.grad.add_(123)
                random.random(); np.random.rand(); torch.rand(1)
                return None, 'CUDA out of memory: simulated partial backward'
            return original(data, split, group, denominator)
        with patch.object(runner, '_attempt', side_effect=fail_large):
            count, _ = runner.accumulate(data, 'training', np.arange(240, 244), 0, 4, 32)
        self.assertEqual(calls, [4, 2])
        self.assertEqual(count, 2)
        self.assertEqual(runner.runtime['concurrent_windows'], 2)
        self.assertEqual(runner.runtime['oom_retries'], 1)
        reference_runner = ConcurrentTrainer(reference, dict(config, concurrent_windows=2), {})
        reference_runner.accumulate(data, 'training', np.arange(240, 242), 0, 2, 32)
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        current = snapshot_accumulation(model)
        for key in ('python_rng', 'numpy_rng', 'torch_rng'):
            assert_tree_equal(self, expected_rng[key], current[key])

    def test_exhausted_single_window_retries_leave_saved_accumulation(self):
        model, data = cyclical_fixture()
        config = dict(model.config, concurrent_windows=1, edge_chunk_size=4096,
                      min_edge_chunk_size=1024)
        runner = ConcurrentTrainer(model, config, {})
        for p in model.parameters():
            p.grad = torch.full_like(p, .25)
        before = snapshot_accumulation(model)
        def always_fail(*args):
            for p in model.parameters():
                p.grad.mul_(2)
            return None, 'CUDA out of memory: simulated'
        with patch.object(runner, '_attempt', side_effect=always_fail):
            with self.assertRaises(BatchMemoryError):
                runner.accumulate(data, 'training', [240], 0, 1, 1)
        self.assertEqual(runner.runtime['edge_chunk_size'], 1024)
        self.assertEqual(runner.runtime['fire_cache_bytes'], 0)
        self.assertEqual(runner.runtime['weather_cache_bytes'], 0)
        self.assertEqual(runner.runtime['activation_checkpointing'], 'nested')
        assert_tree_equal(self, before['gradients'], snapshot_accumulation(model)['gradients'])

    def test_concurrent_resume_keeps_accumulation_and_runtime_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = create_fixture(root)
            config = json.loads(args.config.read_text())
            config.update(concurrent_windows=4, time_encoding='cyclical',
                          fire_input_cache_mb=1, weather_cache_mb=1, cache_weather_features=True)
            args.config.write_text(json.dumps(config))
            args.epochs = 1
            args.limit_windows = 5
            self.assertEqual(train.run(args), 'completed')
            reference = train.load_checkpoint(args.output/'last.pt')
            args.output = root/'resumed'
            original = train.save_checkpoint
            class Pause(Exception):
                pass
            def pause(path, model, optimizer, state, *rest):
                original(path, model, optimizer, state, *rest)
                if state['stage'] == 'training' and state['train_next'] == 4:
                    raise Pause()
            with patch.object(train, 'save_checkpoint', side_effect=pause):
                with self.assertRaises(Pause):
                    train.run(args)
            paused = train.load_checkpoint(args.output/'last.pt')
            self.assertTrue(paused['gradients'])
            self.assertEqual(paused['progress']['batching_runtime']['concurrent_windows'], 4)
            args.resume = True
            self.assertEqual(train.run(args), 'completed')
            resumed = train.load_checkpoint(args.output/'last.pt')
            for key in ('model', 'optimizer', 'python_rng', 'numpy_rng', 'torch_rng'):
                assert_tree_equal(self, reference[key], resumed[key])

    def test_attempt_catches_cuda_oom_but_propagates_other_failures(self):
        model, data = cyclical_fixture()
        runner = ConcurrentTrainer(model, dict(model.config, concurrent_windows=2), {})
        with patch.object(model, 'forward_batch', side_effect=torch.cuda.OutOfMemoryError('simulated')):
            total, error = runner._attempt(data, 'training', [240, 241], 32)
        self.assertIsNone(total)
        self.assertIn('simulated', error)
        with patch.object(model, 'forward_batch', side_effect=RuntimeError('shape mismatch')):
            with self.assertRaisesRegex(RuntimeError, 'shape mismatch'):
                runner._attempt(data, 'training', [240, 241], 32)

    def test_batched_rollout_never_reads_future_pm(self):
        model, data = cyclical_fixture()
        model.eval()
        with torch.no_grad():
            before = model.forward_batch(data, 'training', [240, 241], horizon=4)
            data.splits['training']['pm25'][241:] = 1e6
            after = model.forward_batch(data, 'training', [240, 241], horizon=4)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_terminal_oom_saves_previous_group_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = create_fixture(Path(temporary))
            config = json.loads(args.config.read_text())
            config.update(concurrent_windows=4, time_encoding='cyclical', min_edge_chunk_size=2,
                          fire_input_cache_mb=1, weather_cache_mb=1, cache_weather_features=True,
                          activation_checkpointing='outer')
            args.config.write_text(json.dumps(config))
            args.epochs, args.limit_windows = 1, 5
            original = ConcurrentTrainer._attempt
            accepted = []
            def fail_after_first(runner, data, split, group, denominator):
                if not accepted:
                    total, error = original(runner, data, split, group, denominator)
                    accepted.append(total)
                    return total, error
                for parameter in runner.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.add_(7)
                torch.rand(1)
                return None, 'CUDA out of memory: simulated terminal failure'
            with patch.object(ConcurrentTrainer, '_attempt', new=fail_after_first):
                with self.assertRaises(BatchMemoryError):
                    train.run(args)
            saved = train.load_checkpoint(args.output/'last.pt')
            self.assertEqual(saved['progress']['train_next'], 4)
            self.assertEqual(saved['progress']['training_total'], accepted[0])
            self.assertTrue(saved['gradients'])
            self.assertEqual(saved['progress']['batching_runtime']['concurrent_windows'], 1)
            self.assertEqual(saved['progress']['batching_runtime']['fire_cache_bytes'], 0)
            args.resume = True
            self.assertEqual(train.run(args), 'completed')
            finished = train.load_checkpoint(args.output/'last.pt')
            self.assertEqual(finished['progress']['stage'], 'done')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable')
    def test_cuda_packed_graph_predictions_and_gradients(self):
        batched, data = cyclical_fixture()
        batched = batched.cuda()
        sequential = FireSourceGNN(batched.config, data.stats, data.graph).cuda()
        sequential.load_state_dict(batched.state_dict())
        expected = []
        for start in (240, 241):
            pred = sequential(data, 'training', start, horizon=3)
            expected.append(pred.detach())
            (pred.square().mean()/2).backward()
        actual = batched.forward_batch(data, 'training', [240, 241], horizon=3)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-4, atol=1e-5)
        for p, q in zip(batched.parameters(), sequential.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-3, atol=1e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
