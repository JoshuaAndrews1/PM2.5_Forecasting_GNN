"""Cache correctness, invalidation and bounded-memory regression checks."""
import sys
from pathlib import Path

# Keep source imports working from the test subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import torch
from model import FireSourceGNN
from reusable_inputs import FireInputCache
from test_model import fixture

class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_training_updates_match_for_all_checkpoint_policies(self):
        for policy in ('nested', 'outer', 'none'):
            baseline, data = fixture()
            config = dict(baseline.config, activation_checkpointing=policy)
            baseline.config.update(config)
            cached = FireSourceGNN(dict(config, fire_input_cache_mb=1), data.stats, data.graph)
            cached.load_state_dict(baseline.state_dict())
            optimizers = [torch.optim.RMSprop(m.parameters(), lr=.0005) for m in (baseline,cached)]
            for start in (240, 241, 240):
                predictions, gradients = [], []
                for model, optimizer in zip((baseline,cached),optimizers):
                    optimizer.zero_grad()
                    p=model(data,'training',start,horizon=4)
                    p.square().mean().backward()
                    predictions.append(p.detach())
                    gradients.append({k:v.grad.clone() for k,v in model.named_parameters() if v.grad is not None})
                    optimizer.step()
                torch.testing.assert_close(predictions[0],predictions[1],rtol=0,atol=0)
                self.assertEqual(gradients[0].keys(),gradients[1].keys())
                for name in gradients[0]:
                    torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=0,atol=0)
                for a,b in zip(baseline.parameters(),cached.parameters()):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertGreater(cached.fire_input_cache.hits,0)
            self.assertEqual(baseline.state_dict().keys(),cached.state_dict().keys())

    def test_split_data_and_config_keys(self):
        model,data=fixture();model.fire_input_cache=FireInputCache(100000)
        other,otherdata=fixture()
        otherdata.splits['training']['fire_frp'] *= 2
        def chunks(d,split='training'):
            return list(model.fire_input_cache.chunks(model,d,split,240))
        a=chunks(data);chunks(data)
        self.assertEqual(model.fire_input_cache.hits,1)
        b=chunks(otherdata)
        torch.testing.assert_close(b[0][2],a[0][2]*2)
        chunks(data,'validation')
        model.config['max_lag_hours']=0;chunks(data)
        model.config['edge_chunk_size']=100;chunks(data)
        self.assertEqual(model.fire_input_cache.misses,5)

    def test_eviction_and_oversized_hours_preserve_chunks(self):
        model,data=fixture(); cache=FireInputCache(350,max_entries=2)
        for hour in (240,241,242,240):
            actual=list(cache.chunks(model,data,'training',hour))
            expected=list(FireInputCache(0).chunks(model,data,'training',hour))
            for a,b in zip(actual,expected):
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertEqual(len(actual),len(expected))
            self.assertLessEqual(cache.bytes,cache.budget_bytes)
        self.assertGreater(cache.evictions,0)
        tiny=FireInputCache(200)
        actual=list(tiny.chunks(model,data,'training',240))
        expected=list(FireInputCache(0).chunks(model,data,'training',240))
        self.assertGreater(tiny.oversized,0)
        self.assertEqual(tiny.bytes,0)
        for a,b in zip(actual,expected):
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_empty_hours_entry_limit_and_device_invalidation(self):
        model,data=fixture();model.fire_input_cache=FireInputCache(10000,max_entries=2)
        for hour in (1,2,3):list(model.fire_input_cache.chunks(model,data,'training',hour))
        self.assertEqual(len(model.fire_input_cache.entries),2)
        model.to('cpu')
        self.assertEqual(len(model.fire_input_cache.entries),0)

    def test_inference_cache_can_be_reused_for_training(self):
        model,data=fixture(); model.fire_input_cache=FireInputCache(10000)
        model.eval()
        with torch.inference_mode():model(data,'training',240,horizon=3)
        model.train()
        model(data,'training',240,horizon=3).square().mean().backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())

    def test_no_future_fire_and_disabled_default(self):
        model,data=fixture()
        self.assertEqual(model.fire_input_cache.budget_bytes,0)
        model.fire_input_cache=FireInputCache(10000)
        chunks=list(model.fire_input_cache.chunks(model,data,'training',240))
        self.assertEqual(sum(len(c[0]) for c in chunks),4)
        self.assertEqual(torch.cat([c[-1] for c in chunks]).tolist(),[24.,24.,0.,0.])

if __name__=='__main__':unittest.main(verbosity=2)
