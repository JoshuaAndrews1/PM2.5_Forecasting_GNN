"""Verify activation checkpoint policies against the original nested policy."""
import sys
from pathlib import Path

# Keep source imports working from the test subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import torch
from test_model import fixture


class PolicyTests(unittest.TestCase):
    def test_policies_preserve_predictions_and_gradients(self):
        torch.set_num_threads(1)
        reference, data = fixture()
        weights = {name: value.clone() for name, value in reference.state_dict().items()}
        expected = expected_grad = None
        for policy in ("nested", "outer", "none"):
            model, _ = fixture()
            model.load_state_dict(weights)
            model.config["activation_checkpointing"] = policy
            model.train()
            prediction = model(data, "training", 240, horizon=3)
            prediction.square().mean().backward()
            gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            if expected is None:
                expected, expected_grad = prediction.detach(), gradients
            else:
                torch.testing.assert_close(prediction, expected)
                self.assertEqual(gradients.keys(), expected_grad.keys())
                for name, value in gradients.items():
                    torch.testing.assert_close(value, expected_grad[name], rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
