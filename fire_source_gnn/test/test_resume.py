"""Compare interrupted/resumed runs with uninterrupted training, including gradients."""
import sys
from pathlib import Path

# Keep source imports working from the test subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import json
import signal
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import train
from test_model import fixture


class SimulatedTimeout(Exception):
    pass


def create_fixture(root):
    model, data = fixture()
    cache = root / 'cache'
    cache.mkdir()
    (cache / 'manifest.json').write_text(json.dumps({'fire_cutoff_km': 300.0}))
    np.savez(str(cache / 'statistics.npz'), **data.stats)
    np.savez(str(cache / 'graph.npz'), **data.graph)
    for split, arrays in data.splits.items():
        (cache / split).mkdir()
        for name in ('monitor', 'pm25'):
            arrays[name] = np.concatenate((arrays[name], arrays[name][-10:]), axis=0)
        arrays['hour_ptr'] = np.r_[arrays['hour_ptr'], np.repeat(arrays['hour_ptr'][-1], 10)]
        for name, array in arrays.items():
            np.save(str(cache / split / (name+'.npy')), array)
    config = dict(model.config, batch_size=32)
    config_path = root / 'config.json'
    config_path.write_text(json.dumps(config))
    return argparse.Namespace(cache=cache, config=config_path, output=root / 'reference', epochs=2,
        limit_windows=33, threads=1, device='cpu', resume=False, checkpoint_seconds=0., max_seconds=0.)


def assert_tree_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_tree_equal(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree_equal(test, x, y)
    else:
        test.assertEqual(a, b)


class ResumeTests(unittest.TestCase):
    def test_restarts_preserve_training_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = create_fixture(root)
            self.assertEqual(train.run(args), 'completed')
            reference = train.load_checkpoint(args.output / 'last.pt')
            reference_metrics = json.loads((args.output / 'metrics.json').read_text())
            reference_best = train.load_checkpoint(args.output / 'best.pt')
            args.output = root / 'interrupted'
            original_save = train.save_checkpoint
            # Includes a partial accumulated batch, optimizer boundary, epoch shuffle,
            # both evaluation phases, and the final one-window partial batch.
            targets = [('training', 1, 7), ('training', 1, 32), ('validation', 1, 5),
                       ('training', 2, 7), ('training', 2, 33), ('testing', 2, 5)]
            for stage, epoch, position in targets:
                def interrupt(path, model, optimizer, state, *rest):
                    original_save(path, model, optimizer, state, *rest)
                    actual = state['train_next'] if stage == 'training' else state['evaluation']['next']
                    if state['stage'] == stage and state['epoch'] == epoch and actual == position:
                        raise SimulatedTimeout()
                with patch.object(train, 'save_checkpoint', interrupt):
                    with self.assertRaises(SimulatedTimeout):
                        train.run(args)
                saved = train.load_checkpoint(args.output / 'last.pt')
                if stage == 'training' and position == 7:
                    self.assertTrue(saved['gradients'])
                args.resume = True
            self.assertEqual(train.run(args), 'completed')
            resumed = train.load_checkpoint(args.output / 'last.pt')
            self.assertEqual(json.loads((args.output / 'metrics.json').read_text()), reference_metrics)
            for key in ('model', 'optimizer', 'python_rng', 'numpy_rng', 'torch_rng'):
                assert_tree_equal(self, reference[key], resumed[key])
            assert_tree_equal(self, reference_best['model'], train.load_checkpoint(args.output / 'best.pt')['model'])
            for a, b in zip(reference['progress']['history'], resumed['progress']['history']):
                a, b = copy.deepcopy(a), copy.deepcopy(b)
                a.pop('elapsed_seconds'); b.pop('elapsed_seconds')
                self.assertEqual(a, b)
            # Completed resumes are idempotent; no duplicate test or optimizer steps.
            with patch.object(torch.optim.RMSprop, 'step', side_effect=AssertionError('extra update')):
                self.assertEqual(train.run(args), 'completed')
            # Reject mismatched selections/config before changing any run files.
            args.limit_windows = 32
            with self.assertRaisesRegex(ValueError, 'Resume requires'):
                train.run(args)
            args.limit_windows = 33
            config = json.loads(args.config.read_text()); config['batch_size'] = 16
            args.config.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'Resume requires'):
                train.run(args)

    def test_budget_and_signal_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            args = create_fixture(Path(directory))
            args.max_seconds = 1e-9
            self.assertEqual(train.run(args), 'paused')
            saved = train.load_checkpoint(args.output / 'last.pt')
            # Windows monotonic-clock resolution may permit the first window.
            self.assertIn(saved['progress']['train_next'], (0, 1))
            self.assertTrue(json.loads((args.output / 'progress.json').read_text())['paused'])
        previous = signal.getsignal(signal.SIGTERM)
        with train.RunControl(0.) as control:
            control.stop(signal.SIGTERM, None)
            self.assertTrue(control.stopping())
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_failed_atomic_save_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            train.atomic_save(path, {'sentinel': 42})
            def fail(value, stream):
                stream.write(b'partial')
                raise OSError('simulated storage failure')
            with patch.object(torch, 'save', fail):
                with self.assertRaises(OSError):
                    train.atomic_save(path, {'sentinel': 99})
            self.assertEqual(train.load_checkpoint(path), {'sentinel': 42})


if __name__ == '__main__':
    unittest.main(verbosity=2)
