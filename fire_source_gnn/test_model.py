import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from data import PreparedData, geometry
from model import FireSourceGNN


def fixture():
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    config.update(hidden_dim=8, message_dim=4, nlcd_embedding_dim=3, edge_chunk_size=2)
    stats = dict(monitor_mean=np.zeros(9), monitor_std=np.ones(9),
                 fire_mean=np.zeros(11), fire_std=np.ones(11), pm_mean=0., pm_std=1.,
                 elevation_mean=0., elevation_std=1.)
    graph = dict(monitor_edges=np.array([[0, 1], [1, 0]]), monitor_geometry=np.zeros((2, 3)),
                 elevation=np.zeros(2), candidate_ptr=np.array([0, 2, 4]),
                 candidate_target=np.array([0, 1, 0, 1]), candidate_distance=np.ones(4),
                 candidate_bearing=np.zeros(4))
    item = dict(monitor=np.zeros((310, 2, 9)), pm25=np.ones((310, 2)),
                fire_hour=np.array([215, 216, 240, 241]), fire_cell=np.array([0, 0, 1, 0]),
                fire_features=np.ones((4, 11)), fire_nlcd=np.ones(4, dtype=int),
                fire_frp=np.array([1., 2., 3., 4.]))
    item['hour_ptr'] = np.r_[0, np.cumsum(np.bincount(item['fire_hour'], minlength=310))]
    data = PreparedData.__new__(PreparedData)
    data.stats, data.graph = stats, graph
    data.splits = {k: copy.deepcopy(item) for k in ['training', 'validation', 'testing']}
    torch.manual_seed(42)
    return FireSourceGNN(config, stats, graph), data


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bearing_and_distance(self):
        d, b = geometry(np.array([[0., 0.]]), np.array([[1., 0.], [0., 1.], [0., 0.]]))
        np.testing.assert_allclose(b[0, :2], [0, np.pi/2], atol=1e-6)
        self.assertAlmostEqual(float(d[0, 0]), 111.195, places=2)
        self.assertEqual(float(d[0, 2]), 0)

    def test_lags_are_inclusive_and_future_excluded(self):
        _, data = fixture()
        pieces = list(data.edge_chunks('training', 240, 24, 2))
        rows = np.concatenate([p[0] for p in pieces])
        lags = np.concatenate([p[2] for p in pieces])
        np.testing.assert_array_equal(rows, [1, 1, 2, 2])
        np.testing.assert_array_equal(lags, [24, 24, 0, 0])

    def test_expired_fire_and_split_isolation(self):
        _, data = fixture()
        self.assertEqual(list(data.edge_chunks('validation', 0, 24, 2)), [])
        self.assertEqual(list(data.edge_chunks('training', 266, 24, 2)), [])

    def test_calm_wind_and_alignment(self):
        model, _ = fixture()
        z = torch.zeros(2, 11); z[1, 3] = 5
        features = model.transport_features(z, torch.ones(2), torch.zeros(2), torch.zeros(2), torch.zeros(2, 7))
        torch.testing.assert_close(features[0, 1:3], torch.zeros(2))
        torch.testing.assert_close(features[1, 1:3], torch.tensor([0., 1.]))

    def test_no_fire_is_exact_zero(self):
        model, data = fixture(); model.eval()
        result = model.fire_aggregate(torch.ones(2, 8), torch.ones(2, 7), data, 'training', 100)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_linear_frp_weight(self):
        model, _ = fixture()
        args = [torch.ones(2, 8), torch.ones(2, 7), torch.ones(2, 11), torch.ones(2, dtype=torch.long),
                torch.ones(2), torch.tensor([0, 1]), torch.ones(2), torch.zeros(2), torch.zeros(2)]
        one = model.fire_chunk(*args)
        args[4] *= 2
        torch.testing.assert_close(model.fire_chunk(*args), 2*one)

    def test_chunk_invariance_and_checkpoint_gradients(self):
        model, data = fixture(); model.train()
        h = torch.ones(2, 8, requires_grad=True)
        a = model.fire_aggregate(h, torch.ones(2, 7), data, 'training', 240)
        a.square().sum().backward()
        self.assertGreater(float(model.transport[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.land_cover.weight.grad.abs().sum()), 0)
        gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(); model.config['edge_chunk_size'] = 100
        b = model.fire_aggregate(h, torch.ones(2, 7), data, 'training', 240)
        b.square().sum().backward()
        torch.testing.assert_close(a, b)
        for k, p in model.named_parameters():
            if k in gradients:
                torch.testing.assert_close(p.grad, gradients[k], atol=1e-6, rtol=1e-4)

    def test_48_hour_rollout_and_no_future_pm_leakage(self):
        model, data = fixture(); model.eval()
        with torch.no_grad():
            a = model(data, 'training', 240)
            data.splits['training']['pm25'][240:] = 1e6
            b = model(data, 'training', 240)
        self.assertEqual(tuple(a.shape), (48, 2))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_earlier_history_affects_prediction(self):
        model, data = fixture(); model.eval()
        with torch.no_grad():
            a = model(data, 'training', 240, horizon=1)
            data.splits['training']['pm25'][235:239] = 100
            b = model(data, 'training', 240, horizon=1)
        self.assertFalse(torch.allclose(a, b))

    def test_bad_window_rejected(self):
        model, data = fixture()
        with self.assertRaises(ValueError):
            model(data, 'training', 239)
        with self.assertRaises(ValueError):
            model(data, 'training', 300)


if __name__ == '__main__':
    unittest.main(verbosity=2)
