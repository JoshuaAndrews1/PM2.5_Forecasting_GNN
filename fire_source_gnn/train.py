"""Resumable chronological training, validation selection and held-out testing."""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from data import PreparedData
from model import FireSourceGNN
from batching import BatchMemoryError, ConcurrentTrainer, validate_batching_config


def save_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(str(temporary), str(path))


def atomic_save(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def load_checkpoint(path):
    # Includes optimizer/RNG state; load only trusted run files.
    return torch.load(str(path), map_location='cpu', weights_only=False)


def select_starts(data, split, config, limit=0):
    starts = data.starts(split, config['history_hours'], config['forecast_hours'])
    if not len(starts):
        raise ValueError('No complete windows in '+split)
    if limit and limit < len(starts):
        starts = starts[np.linspace(0, len(starts)-1, limit, dtype=int)]
    return starts


def evaluation_state():
    return dict(next=0, squared=0., absolute=0., normalized=0., raw_squared=0., count=0)


@torch.no_grad()
def evaluation_window(model, data, split, start, state):
    pred = model(data, split, int(start))
    truth = np.asarray(data.splits[split]['pm25'][start:start+len(pred)], dtype=np.float64)
    raw = pred.cpu().numpy().astype(np.float64)*float(data.stats['pm_std'])+float(data.stats['pm_mean'])
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite evaluation prediction')
    error = np.maximum(raw, 0)-truth
    state['squared'] += float(np.square(error).sum())
    state['absolute'] += float(np.abs(error).sum())
    state['raw_squared'] += float(np.square(raw-truth).sum())
    state['normalized'] += float(np.square((raw-truth)/float(data.stats['pm_std'])).sum())
    state['count'] += truth.size
    state['next'] += 1


def evaluation_metrics(state):
    count = state['count']
    return dict(mae=state['absolute']/count, rmse=(state['squared']/count)**0.5,
                rmse_unclipped=(state['raw_squared']/count)**0.5,
                mse_normalized=state['normalized']/count, windows=state['next'],
                values=count, units='ug/m3')


@torch.no_grad()
def evaluate(model, data, split, starts):
    model.eval()
    state = evaluation_state()
    for start in starts:
        evaluation_window(model, data, split, start, state)
    return evaluation_metrics(state)


def save_checkpoint(path, model, optimizer, state, config, data_manifest, limit_windows):
    atomic_save(path, dict(format_version=2, model=model.state_dict(),
        optimizer=optimizer.state_dict(), progress=state, config=config,
        data_manifest=data_manifest, limit_windows=limit_windows,
        gradients={k: p.grad.detach().cpu() for k, p in model.named_parameters() if p.grad is not None},
        python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if model.pm_mean.is_cuda else None))


class RunControl:
    def __init__(self, max_seconds):
        self.started = time.monotonic()
        self.max_seconds = max_seconds
        self.requested = False
        self.previous = {}

    def __enter__(self):
        for name in ('SIGTERM', 'SIGINT', 'SIGUSR1'):
            if hasattr(signal, name):
                number = getattr(signal, name)
                self.previous[number] = signal.signal(number, self.stop)
        return self

    def stop(self, number, frame):
        self.requested = True

    def stopping(self):
        return self.requested or (self.max_seconds > 0 and time.monotonic()-self.started >= self.max_seconds)

    def __exit__(self, *unused):
        for number, handler in self.previous.items():
            signal.signal(number, handler)


def run(args):
    data = PreparedData(args.cache)
    try:
        return _run(args, data)
    finally:
        data.close()


def load_run_config(args):
    if args.config is not None:
        return json.loads(args.config.read_text())
    if getattr(args, 'resume', False):
        return load_checkpoint(args.output/'last.pt')['config']
    return json.loads(Path(__file__).with_name('config.weather-cyclical.json').read_text())


def _run(args, data):
    config = load_run_config(args)
    validate_batching_config(config)
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if config['epochs'] < 1 or config['batch_size'] < 1:
        raise ValueError('Invalid epochs/batch_size')
    checkpoint_seconds = getattr(args, 'checkpoint_seconds', 60.)
    max_seconds = getattr(args, 'max_seconds', 0.)
    if args.limit_windows < 0 or checkpoint_seconds < 0 or max_seconds < 0:
        raise ValueError('Window limit and time intervals must be nonnegative')
    resume = getattr(args, 'resume', False)
    if config['fire_cutoff_km'] != data.meta['fire_cutoff_km']:
        raise ValueError('Rebuild candidates after changing fire cutoff')
    if args.output.exists() and not resume:
        raise ValueError('Use a new output directory, or --resume for its last.pt')
    saved = None
    if resume:
        saved = load_checkpoint(args.output/'last.pt')
        if (saved.get('format_version') != 2 or saved['config'] != config
                or saved['data_manifest'] != data.meta or saved['limit_windows'] != args.limit_windows):
            raise ValueError('Resume requires the same config, data manifest and window selection')
    else:
        args.output.mkdir(parents=True)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    model = FireSourceGNN(config, data.stats, data.graph).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    starts = {split: select_starts(data, split, config, args.limit_windows) for split in data.splits}
    if saved is None:
        provenance = dict(config=config, data_manifest=data.meta, device=str(device), torch=torch.__version__,
                          verification_only=bool(args.limit_windows),
                          selected_windows={k: len(v) for k, v in starts.items()},
                          history_encoder='GRU over all observed PM/weather/calendar history',
                          future_pm_policy='autoregressive; no future truth passed to model', resume_format_version=2,
                          time_encoding=model.time_encoding,
                          time_features=(['utc_hour_sin', 'utc_hour_cos', 'year_phase_sin', 'year_phase_cos']
                                         if model.time_encoding == 'cyclical' else ['midnight_unix_seconds', 'utc_hour']),
                          weather_cache_mb=config.get('weather_cache_mb', 0),
                          cache_weather_features=config.get('cache_weather_features', False),
                          concurrent_windows=config.get('concurrent_windows', 1),
                          max_batch_edges=config.get('max_batch_edges', 0),
                          batching='Disjoint monitor graphs and packed fire edges; sequential forecast hours')
        save_json(args.output/'run.json', provenance)
        state = dict(stage='training', epoch=1, order=np.random.permutation(starts['training']),
                     train_next=0, training_total=0., epoch_seconds=0.,
                     best=float('inf'), best_epoch=0, history=[], evaluation=evaluation_state())
    else:
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        for name, parameter in model.named_parameters():
            if name in saved['gradients']:
                parameter.grad = saved['gradients'][name].to(device)
        state = saved['progress']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print('Resuming epoch {} stage {} training {}/{} evaluation {}'.format(
            state['epoch'], state['stage'], state['train_next'], len(state['order']),
            state['evaluation']['next']), flush=True)

    concurrent = ConcurrentTrainer(model, config, state)
    edge_counts = (data.forecast_edge_counts('training', config['forecast_hours'], config['max_lag_hours'])
                   if config.get('max_batch_edges', 0) else None)
    last_saved = last_tick = time.monotonic()
    with RunControl(max_seconds) as control:
        def checkpoint(force=False):
            nonlocal last_saved, last_tick
            now = time.monotonic()
            state['epoch_seconds'] += now-last_tick
            last_tick = now
            stopping = control.stopping()
            if force or stopping or now-last_saved >= checkpoint_seconds:
                save_checkpoint(args.output/'last.pt', model, optimizer, state, config, data.meta, args.limit_windows)
                last_saved = now
                save_json(args.output/'progress.json', dict(stage=state['stage'], epoch=state['epoch'],
                    training_windows_completed=state['train_next'], training_windows_total=len(state['order']),
                    evaluation_windows_completed=state['evaluation']['next'], paused=bool(stopping),
                    checkpoint='last.pt', batch_size=config['batch_size'], batching_runtime=dict(concurrent.runtime)))
            return stopping

        def pause():
            print('PAUSED safely; resume with --output {} --resume (exit 75)'.format(args.output), flush=True)
            return 'paused'

        if state['stage'] == 'done':
            print('Run already completed', str(args.output), flush=True)
            return 'completed'
        if checkpoint(force=True):
            return pause()
        while state['stage'] != 'done':
            if state['stage'] == 'training':
                model.train()
                order = state['order']
                while state['train_next'] < len(order):
                    position = state['train_next']
                    first = (position//config['batch_size'])*config['batch_size']
                    batch_end = min(first+config['batch_size'], len(order))
                    if position == first:
                        optimizer.zero_grad(set_to_none=True)
                    try:
                        count, loss_total = concurrent.accumulate(
                            data, 'training', order, position, batch_end, batch_end-first, edge_counts)
                    except BatchMemoryError:
                        # Failed group was rolled back; prior accumulation is safe.
                        checkpoint(force=True)
                        raise
                    state['training_total'] += loss_total
                    state['train_next'] += count
                    boundary = state['train_next'] == batch_end
                    if boundary:
                        for p in model.parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                raise ValueError('Nonfinite gradient')
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                        print('epoch {} windows {}/{} loss {:.6g}'.format(
                            state['epoch'], state['train_next'], len(order), loss_total/count), flush=True)
                    # Save complete concurrent groups, including partial effective batches.
                    if checkpoint(force=boundary):
                        return pause()
                state['stage'] = 'validation'
                state['evaluation'] = evaluation_state()
                if checkpoint(force=True):
                    return pause()

            if state['stage'] in ('validation', 'testing'):
                split = state['stage']
                model.eval()
                scores = state['evaluation']
                while scores['next'] < len(starts[split]):
                    evaluation_window(model, data, split, starts[split][scores['next']], scores)
                    if scores['next'] % 32 == 0:
                        print('epoch {} {} windows {}/{}'.format(state['epoch'], split,
                            scores['next'], len(starts[split])), flush=True)
                    if checkpoint(force=scores['next'] % 32 == 0):
                        return pause()
                metrics = evaluation_metrics(scores)
                if split == 'validation':
                    record = dict(epoch=state['epoch'], training_mse=state['training_total']/len(state['order']),
                                  validation=metrics, elapsed_seconds=state['epoch_seconds'])
                    state['history'].append(record)
                    save_json(args.output/'history.json', state['history'])
                    print(json.dumps(record), flush=True)
                    if metrics['mse_normalized'] < state['best']:
                        state['best'], state['best_epoch'] = metrics['mse_normalized'], state['epoch']
                        atomic_save(args.output/'best.pt', dict(model=model.state_dict(), config=config,
                            epoch=state['epoch'], optimizer=optimizer.state_dict(), validation=metrics))
                    if state['epoch'] >= config['epochs'] or state['epoch']-state['best_epoch'] >= config['patience']:
                        selected = load_checkpoint(args.output/'best.pt')
                        model.load_state_dict(selected['model'])
                        state['selected_validation'] = selected['validation']
                        state['stage'] = 'testing'
                    else:
                        state.update(stage='training', epoch=state['epoch']+1,
                            order=np.random.permutation(starts['training']), train_next=0,
                            training_total=0., epoch_seconds=0.)
                    state['evaluation'] = evaluation_state()
                else:
                    save_json(args.output/'metrics.json', dict(best_epoch=state['best_epoch'],
                        validation=state['selected_validation'], testing=metrics,
                        verification_only=bool(args.limit_windows)))
                    state['stage'] = 'done'
                if checkpoint(force=True) and state['stage'] != 'done':
                    return pause()
    print('Completed', str(args.output), flush=True)
    return 'completed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Fresh default: concurrent/weather/cyclical config; resume default: saved config')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--limit-windows', type=int, default=0, help='Verification only; 0 means ALL windows')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='Resume last.pt in the same output directory')
    parser.add_argument('--checkpoint-seconds', type=float, default=60., help='Also save at window boundaries this often')
    parser.add_argument('--max-seconds', type=float, default=0., help='Graceful pause after invocation budget; 0 disables')
    raise SystemExit(75 if run(parser.parse_args()) == 'paused' else 0)
