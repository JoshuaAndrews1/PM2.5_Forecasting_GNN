"""Compare weather caching with identical cyclical weights on real graph windows."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from data import PreparedData
from model import FireSourceGNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.weather-cyclical.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fire-cache-mb', type=int, default=512)
    parser.add_argument('--weather-cache-mb', type=int, default=256)
    parser.add_argument('--starts', type=int, nargs='+', default=[15479, 2008, 4426])
    parser.add_argument('--trials', type=int, default=2)
    args = parser.parse_args()
    if min(args.fire_cache_mb, args.weather_cache_mb) < 0 or args.trials < 1:
        raise ValueError('Invalid cache budget or trials')
    torch.set_num_threads(2)
    data = PreparedData(args.cache)
    config = json.loads(args.config.read_text())
    config['fire_input_cache_mb'] = args.fire_cache_mb
    torch.manual_seed(config['seed'])
    original = FireSourceGNN(config, data.stats, data.graph)
    weights = {k: v.clone() for k, v in original.state_dict().items()}
    del original
    report = dict(verification_only=True, gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                  torch=torch.__version__, starts=args.starts, trials=args.trials, config=config,
                  note='Identical cyclical models; fixed fire-input budget; fresh caches per trial. Includes forward/backward with loss accumulated at effective batch 32, then one optimizer update. No accuracy claims.',
                  results=[], correctness=[])
    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    try:
        for trial in range(args.trials):
            reference = {}
            for enabled in ([False, True] if trial % 2 == 0 else [True, False]):
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                settings = dict(config, weather_cache_mb=args.weather_cache_mb if enabled else 0,
                                cache_weather_features=enabled)
                model = FireSourceGNN(settings, data.stats, data.graph).to(args.device)
                model.load_state_dict(weights)
                optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'],
                                               weight_decay=config['weight_decay'])
                # Warm kernels on the actual architecture without filling caches.
                model(data, 'training', args.starts[0], use_fire=False).square().mean().backward()
                model.zero_grad(set_to_none=True)
                model.clear_input_caches()
                sync()
                if args.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                record = dict(trial=trial, weather_caching=enabled, window_seconds=[])
                predictions = []
                optimizer.zero_grad(set_to_none=True)
                begin = time.perf_counter()
                for start in args.starts:
                    tick = time.perf_counter()
                    pred = model(data, 'training', start)
                    truth = model.tensor(data.splits['training']['pm25'][start:start+len(pred)])
                    loss = torch.nn.functional.mse_loss(pred, (truth-model.pm_mean)/model.pm_std)
                    (loss/config['batch_size']).backward()
                    sync()
                    record['window_seconds'].append(time.perf_counter()-tick)
                    predictions.append(pred.detach().cpu())
                gradients = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
                assert all(torch.isfinite(g).all() for g in gradients.values())
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
                sync()
                record.update(seconds=time.perf_counter()-begin, fire_cache=model.fire_input_cache.info(),
                              weather_cache=model.weather_input_cache.info(),
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device == 'cuda' else None)
                reference[enabled] = (torch.stack(predictions), gradients,
                                      {k: p.detach().cpu().clone() for k, p in model.named_parameters()})
                report['results'].append(record)
                save()
                print(json.dumps(record), flush=True)
                del model, optimizer, pred, loss, gradients, predictions
            a, b = reference[False], reference[True]
            torch.testing.assert_close(a[0], b[0], rtol=1e-4, atol=1e-5)
            for idx in (1, 2):
                for name in a[idx]:
                    torch.testing.assert_close(a[idx][name], b[idx][name], rtol=1e-3, atol=1e-4)
            report['correctness'].append(dict(trial=trial, prediction_max_abs=float((a[0]-b[0]).abs().max()),
                gradient_max_abs=max(float((a[1][k]-b[1][k]).abs().max()) for k in a[1]),
                parameter_max_abs=max(float((a[2][k]-b[2][k]).abs().max()) for k in a[2])))
            save()
        baseline = np.median([r['seconds'] for r in report['results'] if not r['weather_caching']])
        cached = np.median([r['seconds'] for r in report['results'] if r['weather_caching']])
        report['summary'] = dict(baseline_seconds=float(baseline), cached_seconds=float(cached),
                                speedup=float(baseline/cached), time_reduction_percent=float(100*(1-cached/baseline)))
        save()
        print(json.dumps(report['summary']), flush=True)
    finally:
        data.close()


if __name__ == '__main__':
    main()
