"""Stress the busiest training fire window and verify gradients/ablation."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from data import PreparedData
from model import FireSourceGNN


def run(args):
    torch.set_num_threads(2); torch.manual_seed(42)
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    data = PreparedData(args.cache)
    item, graph = data.splits['training'], data.graph
    degrees = np.diff(graph['candidate_ptr'])
    hourly = np.bincount(item['fire_hour'], weights=degrees[item['fire_cell']], minlength=len(item['monitor']))
    rolling = np.convolve(hourly, np.ones(config['max_lag_hours']+1), mode='full')[:len(hourly)]
    rolling[:config['history_hours']] = -1
    start = int(np.argmax(rolling[:-config['forecast_hours']+1]))
    model = FireSourceGNN(config, data.stats, graph).to(args.device)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    prediction = model(data, 'training', start, horizon=args.horizon)
    target = model.tensor(item['pm25'][start:start+args.horizon])
    loss = torch.nn.functional.mse_loss(prediction, (target-model.pm_mean)/model.pm_std)
    loss.backward()
    norms = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None}
    if not np.isfinite(float(loss)) or not all(np.isfinite(v) for v in norms.values()):
        raise AssertionError('Nonfinite peak-load training')
    if norms['transport.0.weight'] <= 0 or norms['fire_message.0.weight'] <= 0:
        raise AssertionError('Fire networks received no gradient')
    model.eval()
    with torch.no_grad():
        without = model(data, 'training', start, horizon=args.horizon, use_fire=False)
    delta = float((prediction.detach()-without).abs().max()*model.pm_std)
    if delta <= 0:
        raise AssertionError('Active fires had no effect')
    result = dict(training_hour=start, forecast_hours=args.horizon,
                  active_source_lag_edges_first_hour=int(rolling[start]),
                  active_source_lag_edges_rollout=int(rolling[start:start+args.horizon].sum()),
                  normalized_loss=float(loss), gradient_norms=norms,
                  max_fire_ablation_difference_ug_m3=delta,
                  elapsed_seconds=time.monotonic()-begin,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                  interpretation='Engineering stress check on untrained weights, not accuracy evidence')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--horizon', type=int, default=48)
    run(p.parse_args())
