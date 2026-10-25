import copy
import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import dataset
import train
import train_ambient
from metrics import forecast_metrics


class RegressionTests(unittest.TestCase):
    def test_forecast_metrics_exclude_history(self):
        target = np.zeros((2, train.seq_len, 1, 1))
        prediction = target.copy()
        prediction[0, train.hist_len:] = 2
        prediction[1, train.hist_len:] = 4
        rmse, mae, csi, pod, far = train.get_metric(prediction, target)
        self.assertEqual(rmse, 3)
        self.assertEqual(mae, 3)
        self.assertTrue(np.isnan([csi, pod, far]).all())
        scores = forecast_metrics(prediction[:, train.hist_len:], target[:, train.hist_len:])
        self.assertAlmostEqual(scores['rmse_pooled'], np.sqrt(10))

    def test_dataset_normalization_and_window_endpoints(self):
        raw = np.ones((30, 2, 18), dtype=np.float32)
        raw[:, :, -1] = np.arange(30)[:, None]
        cfg = copy.deepcopy(dataset.config)
        cfg['dataset'] = {
            'data_start': [[2020, 1, 1, 0], 'GMT'],
            'data_end': [[2020, 1, 2, 5], 'GMT'],
            1: {'train_start': [[2020, 1, 1, 0], 'GMT'],
                'train_end': [[2020, 1, 1, 9], 'GMT'],
                'val_start': [[2020, 1, 1, 10], 'GMT'],
                'val_end': [[2020, 1, 1, 19], 'GMT'],
                'test_start': [[2020, 1, 1, 20], 'GMT'],
                'test_end': [[2020, 1, 2, 5], 'GMT']}}
        with tempfile.TemporaryDirectory() as tmp:
            fp = str(Path(tmp) / 'data.npy')
            np.save(fp, raw)
            with patch.object(dataset, 'config', cfg), patch.object(dataset, 'file_dir', {'knowair_fp': fp}):
                graph = SimpleNamespace(node_num=2)
                training = dataset.HazeData(graph, 2, 2)
                validation = dataset.HazeData(graph, 2, 2, flag='Val', normalization=training.normalization)
                self.assertEqual(len(training), 7)
                self.assertEqual(training.time_index[training.time_arr[0]], 3)
                self.assertEqual(training.time_index[training.time_arr[-1]], 9)
                self.assertEqual(validation.pm25_mean, 4.5)
                self.assertTrue(np.isfinite(training.feature_full).all())
                np.testing.assert_array_equal(training.wind_mean, training.feature_mean[-2:])
                restored = validation.pm25_full * training.pm25_std + training.pm25_mean
                np.testing.assert_allclose(restored[:, 0, 0], np.arange(10, 20), atol=1e-5)
                with self.assertRaises(ValueError):
                    dataset.HazeData(graph, 2, 2, flag='Test')

    def test_models_accept_partial_batches_without_cross_sample_edges(self):
        edges = np.array([[0, 1], [1, 0]])
        attrs = np.array([[10, 90], [10, 270]], dtype=np.float32)
        for name in ('MLP', 'GRU', 'LSTM', 'nodesFC_GRU', 'GC_LSTM', 'PM25_GNN', 'PM25_GNN_nosub'):
            with self.subTest(model=name):
                cls = getattr(importlib.import_module('model.' + name), name)
                args = [2, 2, 5]
                if name != 'MLP':
                    args += [2, 4, torch.device('cpu')]
                if name == 'GC_LSTM':
                    args += [edges]
                elif name.startswith('PM25_GNN'):
                    args += [edges, attrs, np.zeros(2), np.ones(2)]
                model = cls(*args).eval()
                hist = torch.randn(3, 2, 2, 1)
                feat = torch.randn(3, 4, 2, 4)
                output = model(hist, feat)
                separate = torch.cat([model(hist[i:i+1], feat[i:i+1]) for i in range(3)])
                self.assertEqual(tuple(output.shape), (3, 2, 2, 1))
                torch.testing.assert_close(output, separate, atol=1e-6, rtol=1e-5)
                output.sum().backward()
                self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None))

    def test_single_node_recurrent_cells(self):
        from model.cells import GRUCell, LSTMCell
        x = torch.zeros(1, 1, 3)
        h = torch.zeros(1, 4)
        self.assertEqual(GRUCell(3, 4)(x, h).shape, h.shape)
        self.assertEqual(LSTMCell(3, 4)(x, (h, h))[0].shape, h.shape)

    def test_transport_uses_degrees_and_speed_direction_statistics(self):
        from model.PM25_GNN import GraphGNN
        graph = GraphGNN(torch.device('cpu'), [[0, 0], [1, 2]], [[10, 90], [10, 270]], 3, 1,
                         np.array([10, 90]), np.array([2, 30]))
        captured = []
        handle = graph.edge_mlp.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
        graph(torch.zeros(1, 3, 3))
        handle.remove()
        np.testing.assert_allclose(captured[0][0, :, -1].numpy(), [3, 0], atol=1e-6)

    def test_training_clears_old_gradients_and_evaluation_is_weighted(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.grad_modes = []
            def forward(self, hist, feature):
                self.grad_modes.append(torch.is_grad_enabled())
                return self.weight * hist

        def prepare(pm, feature, times, flag):
            return pm, torch.zeros_like(pm), feature, pm.numpy()

        loader = [(torch.ones(2, 1, 1, 1), torch.zeros(2), torch.zeros(2)),
                  (torch.full((1, 1, 1, 1), 3.0), torch.zeros(1), torch.zeros(1))]
        with patch.object(train, 'prepare', prepare), patch.object(train, 'device', torch.device('cpu')):
            clean, dirty = TinyModel(), TinyModel()
            dirty.weight.grad = torch.tensor(1000.0)
            for model in (clean, dirty):
                train.train(loader, model, torch.optim.SGD(model.parameters(), lr=0.01))
            torch.testing.assert_close(clean.weight, dirty.weight)
            model = TinyModel()
            self.assertAlmostEqual(train.val(loader, model), 11 / 3)
            self.assertEqual(model.grad_modes, [False, False])
            with self.assertRaises(ValueError):
                train.val([], model)

    def test_ambient_prepare_fills_every_sample(self):
        length = train_ambient.seq_len + 2
        fake = SimpleNamespace(pm25_full=np.arange(length)[:, None, None],
                               feature_full=np.arange(length)[:, None, None],
                               frp500=np.arange(length)[:, None],
                               time_index={0: length-2, 1: length-1})
        with patch.object(train_ambient, 'train_data', fake, create=True):
            hist, labels, features, pm, fire = train_ambient.prepare(
                np.zeros((2, 1, 1)), np.zeros((2, 1, 1)), np.array([0, 1]), 'Train')
        self.assertEqual(labels[1, -1, 0, 0], length-1)
        self.assertEqual(fire[1, -1, 0], length-1)

    def test_saved_simulation_inputs_use_checkpoint_scaling(self):
        import dataset_exp1
        features = np.zeros((2, 4, 2, 20), dtype=np.float32)
        features[..., 7] = 1  # eastward wind: TO direction is 90 degrees
        pm = np.full((2, 4, 2, 1), 20, dtype=np.float32)
        normalization = dict(feature_mean=np.zeros(20), feature_std=np.ones(20),
                             pm25_mean=10., pm25_std=2.)
        arrays = {dataset_exp1.pm25_input_fp: pm,
                  dataset_exp1.feature_input_fp: features,
                  dataset_exp1.time_input_fp: np.zeros((2, 4))}
        with patch.object(dataset_exp1.np, 'load', side_effect=lambda path: arrays[path].copy()):
            data = dataset_exp1.HazeData(SimpleNamespace(node_num=2), 2, 2,
                                         normalization=normalization)
        np.testing.assert_allclose(data.pm25, 5)
        np.testing.assert_allclose(data.feature[..., -2], 3.6)
        np.testing.assert_allclose(data.feature[..., -1], 90)

    def test_ambient_filter_retains_clean_samples_in_mixed_batch(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, hist, feature):
                return hist * self.weight
        def prepare(pm, features, times, flag):
            return pm, torch.zeros_like(pm), features, pm.numpy(), np.array([[[0.]], [[1.]]])
        loader = [(torch.ones(2, 1, 1, 1), torch.zeros(2), torch.zeros(2))]
        model = TinyModel()
        with patch.object(train_ambient, 'prepare', prepare), patch.object(train_ambient, 'device', torch.device('cpu')):
            loss = train_ambient.train(loader, model, torch.optim.SGD(model.parameters(), lr=0.1))
        self.assertEqual(loss, 1.)
        self.assertAlmostEqual(model.weight.item(), 0.8)


if __name__ == '__main__':
    unittest.main()
