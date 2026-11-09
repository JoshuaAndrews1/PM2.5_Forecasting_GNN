"""Calendar boundaries, immutable weather reuse, and training compatibility."""
import sys
from pathlib import Path

# Keep source imports working from the test subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from model import FireSourceGNN
from weather_inputs import WeatherInputCache
from time_features import cyclical_time
from test_model import fixture
from test_resume import create_fixture, assert_tree_equal
import train


def timestamp(day):
    return np.datetime64(day, 's').astype(np.int64)


def cyclical_fixture(cached=True, policy='outer'):
    original, data = fixture()
    for item in data.splits.values():
        hours = np.arange(len(item['monitor']))
        item['monitor'][:, :, 7] = (timestamp('2020-01-01') + (hours//24)*86400)[:, None]
        item['monitor'][:, :, 8] = (hours % 24)[:, None]
        item['fire_features'][:, 9] = timestamp('2020-01-01') + (item['fire_hour']//24)*86400
        item['fire_features'][:, 10] = item['fire_hour'] % 24
    config = dict(original.config, time_encoding='cyclical', activation_checkpointing=policy,
                  weather_cache_mb=1 if cached else 0, fire_input_cache_mb=1 if cached else 0,
                  cache_weather_features=cached)
    torch.manual_seed(42)
    return FireSourceGNN(config, data.stats, data.graph), data


class WeatherTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_daily_and_yearly_wraparound_and_leap_year(self):
        midnight = timestamp('2020-01-01')
        features = cyclical_time(np.array([midnight, midnight]), [0, 6])
        np.testing.assert_allclose(features[:, :2], [[0, 1], [1, 0]], atol=1e-15)
        dec31 = cyclical_time(timestamp('2020-12-31'), 23)
        jan1 = cyclical_time(timestamp('2021-01-01'), 0)
        self.assertLess(np.linalg.norm(dec31[2:] - jan1[2:]), .001)
        self.assertLess(np.linalg.norm(dec31[:2] - jan1[:2]), .27)
        leap_day = cyclical_time(timestamp('2020-02-29'), 0)
        np.testing.assert_allclose(leap_day[2:], [np.sin(2*np.pi*59/366), np.cos(2*np.pi*59/366)])
        np.testing.assert_array_equal(cyclical_time(timestamp('2020-01-01'), 0),
                                      cyclical_time(timestamp('2021-01-01'), 0))
        with self.assertRaises(ValueError):
            cyclical_time(midnight, 24)

    def test_model_inputs_are_bounded_calendar_features(self):
        model, data = cyclical_fixture()
        raw, normalized = model.prepare_monitor_inputs(data.splits['training']['monitor'])
        self.assertEqual(normalized.shape[-1], 11)
        self.assertEqual(raw.shape[-1], 7)
        self.assertTrue(torch.all(normalized[..., 7:].abs() <= 1))
        source = model.fire_tensor(data.splits['training']['fire_features'])
        self.assertEqual(source.shape[-1], 13)
        self.assertTrue(torch.all(source[:, 9:].abs() <= 1))
        self.assertEqual(model.history_gru.input_size, 12)
        self.assertEqual(model.monitor_encoder[0].in_features, 13)
        with self.assertRaisesRegex(ValueError, 'time_encoding'):
            FireSourceGNN(dict(model.config, time_encoding='invalid'), data.stats, data.graph)

    def test_cached_weather_preserves_updates_for_all_policies(self):
        for encoding in ('legacy', 'cyclical'):
            for policy in ('nested', 'outer', 'none'):
                cached, data = cyclical_fixture(policy=policy)
                config = dict(cached.config, time_encoding=encoding)
                torch.manual_seed(42)
                cached = FireSourceGNN(config, data.stats, data.graph)
                plain = FireSourceGNN(dict(config, fire_input_cache_mb=0, weather_cache_mb=0,
                                          cache_weather_features=False), data.stats, data.graph)
                plain.load_state_dict(cached.state_dict())
                optimizers = [torch.optim.RMSprop(m.parameters(), lr=.0005) for m in (plain, cached)]
                for start in (240, 241):
                    results, gradients = [], []
                    for model, optimizer in zip((plain, cached), optimizers):
                        optimizer.zero_grad()
                        result = model(data, 'training', start, horizon=4)
                        result.square().mean().backward()
                        results.append(result.detach())
                        gradients.append({k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None})
                        optimizer.step()
                    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
                    assert_tree_equal(self, gradients[0], gradients[1])
                    assert_tree_equal(self, plain.state_dict(), cached.state_dict())
                self.assertGreater(cached.weather_input_cache.hits, 0)
                self.assertGreater(cached.fire_input_cache.hits, 0)

    def test_fixed_work_reused_but_learned_messages_recomputed(self):
        model, data = cyclical_fixture()
        model.eval()
        with patch.object(model, 'prepare_monitor_inputs', wraps=model.prepare_monitor_inputs) as weather:
            with patch.object(model, 'transport_prefix', wraps=model.transport_prefix) as prefix:
                with patch.object(model, 'fire_messages', wraps=model.fire_messages) as messages:
                    with torch.no_grad():
                        first = model(data, 'training', 240, horizon=3)
                        before = prefix.call_count
                        calls = messages.call_count
                        model.fire_message[-1].bias.add_(.1)
                        second = model(data, 'training', 240, horizon=3)
                    self.assertEqual(weather.call_count, 1)
                    self.assertEqual(prefix.call_count, before)
                    self.assertGreater(messages.call_count, calls)
                    self.assertFalse(torch.equal(first, second))

    def test_split_budget_device_and_inference_lifecycle(self):
        model, data = cyclical_fixture()
        model.weather_input_cache = WeatherInputCache(50000)
        for split in ('training', 'validation', 'testing'):
            with torch.inference_mode():
                model(data, split, 240, horizon=2)
            self.assertLessEqual(model.weather_input_cache.bytes, 50000)
        self.assertGreater(model.weather_input_cache.evictions, 0)
        model.train()
        model(data, 'testing', 240, horizon=2).sum().backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())
        model.to('cpu')
        self.assertEqual(model.weather_input_cache.bytes, 0)
        self.assertEqual(model.fire_input_cache.bytes, 0)
        model.weather_input_cache = WeatherInputCache(1)
        model(data, 'training', 240, horizon=1)
        self.assertEqual(model.weather_input_cache.bytes, 0)
        self.assertGreater(model.weather_input_cache.oversized, 0)

    def test_fresh_defaults_and_legacy_resume_configuration(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = SimpleNamespace(config=None, output=output, resume=False)
            fresh = train.load_run_config(args)
            self.assertEqual(fresh['time_encoding'], 'cyclical')
            self.assertTrue(fresh['cache_weather_features'])
            self.assertGreater(fresh['weather_cache_mb'], 0)
            legacy, _ = fixture()
            train.atomic_save(output/'last.pt', dict(config=legacy.config))
            args.resume = True
            self.assertEqual(train.load_run_config(args), legacy.config)

    def test_checkpoint_statistics_invalidate_derived_caches(self):
        model, data = cyclical_fixture()
        with torch.no_grad():
            model(data, 'training', 240, horizon=2)
        self.assertGreater(model.weather_input_cache.bytes, 0)
        self.assertGreater(model.fire_input_cache.bytes, 0)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        state['monitor_mean'][:7] += 1
        state['fire_mean'][:9] += 1
        model.load_state_dict(state)
        self.assertEqual(model.weather_input_cache.bytes, 0)
        self.assertEqual(model.fire_input_cache.bytes, 0)
        plain = FireSourceGNN(dict(model.config, weather_cache_mb=0, fire_input_cache_mb=0),
                              data.stats, data.graph)
        plain.load_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(model(data, 'training', 240, horizon=2),
                                       plain(data, 'training', 240, horizon=2), rtol=0, atol=0)

    def test_cyclical_cached_resume_preserves_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            args = create_fixture(Path(directory))
            config = json.loads(args.config.read_text())
            config.update(time_encoding='cyclical', weather_cache_mb=1, fire_input_cache_mb=1,
                          cache_weather_features=True, activation_checkpointing='outer')
            args.config.write_text(json.dumps(config))
            args.epochs = 1
            args.limit_windows = 2
            self.assertEqual(train.run(args), 'completed')
            reference = train.load_checkpoint(args.output/'last.pt')
            args.output = Path(directory)/'resumed'
            original_save = train.save_checkpoint
            class Pause(Exception):
                pass
            def pause(path, model, optimizer, state, *rest):
                original_save(path, model, optimizer, state, *rest)
                if state['stage'] == 'training' and state['train_next'] == 1:
                    raise Pause()
            with patch.object(train, 'save_checkpoint', pause):
                with self.assertRaises(Pause):
                    train.run(args)
            args.resume = True
            self.assertEqual(train.run(args), 'completed')
            resumed = train.load_checkpoint(args.output/'last.pt')
            for key in ('model', 'optimizer', 'python_rng', 'numpy_rng', 'torch_rng'):
                assert_tree_equal(self, reference[key], resumed[key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
