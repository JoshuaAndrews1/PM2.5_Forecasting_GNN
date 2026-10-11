"""Test a saved forecasting checkpoint; optionally export its PM2.5 forecasts."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import util
from dataset import HazeData
from forecasting import MODELS, build_model, evaluate


def resolve_run(run, config_path=None, model_name=None):
    """Recover effective features/settings from current or earlier run layouts."""
    run = Path(run).resolve()
    metadata = {}
    for candidate in (run/'manifest.json', run.parent/'manifest.json'):
        if candidate.is_file():
            metadata = json.loads(candidate.read_text(encoding='utf-8'))
            break
    if config_path:
        util.load_config(config_path)
    elif (run/'config.yaml').is_file():
        util.load_config(run/'config.yaml')
    elif 'config' in metadata:
        util.configure(metadata['config'])
    else:
        raise ValueError('This older run needs --config pointing to its original configuration')
    if model_name is None:
        if (run/'model.joblib').is_file():
            model_name = 'RF'
        elif run.name in MODELS:
            model_name = run.name
        else:
            model_name = util.config['forecast']['model']
    names = metadata.get('feature_names')
    if names:
        configured = util.config['forecast']['metero_use']
        if names[:-4] != configured:
            raise ValueError('Configuration feature order differs from checkpoint metadata')
    normalization_path = run/'normalization.npz'
    if not normalization_path.is_file():
        normalization_path = run.parent/'normalization.npz'
    with np.load(str(normalization_path)) as saved:
        normalization = {key:saved[key].copy() for key in saved.files}
    expected = len(util.config['forecast']['metero_use'])+4
    if normalization['feature_mean'].shape != (expected,):
        raise ValueError('Checkpoint normalization does not match configured features')
    return run, model_name, normalization


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Directory containing model.pth or model.joblib')
    parser.add_argument('--config', type=Path, help='Original config for legacy checkpoints without metadata')
    parser.add_argument('--model', choices=MODELS+('RF',))
    parser.add_argument('--split', choices=('Test','Val'), default='Test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=Path, help='New output directory; existing directories are refused')
    parser.add_argument('--save-predictions', action='store_true', help='Export predictions, targets and UTC timestamps')
    parser.add_argument('--max-windows', type=int, help='Evaluate only this many windows for a smoke check')
    args = parser.parse_args(argv)
    if args.batch_size<1 or (args.max_windows is not None and args.max_windows<1):
        parser.error('batch-size and max-windows must be positive')
    run,name,normalization = resolve_run(args.run,args.config,args.model)
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = None
    if name in ('PM25_GNN','PM25_GNN_nosub','GC_LSTM'):
        if (run/'graph.npz').is_file():
            with np.load(str(run/'graph.npz')) as saved:
                graph = SimpleNamespace(edge_index=saved['edge_index'].copy(),edge_attr=saved['edge_attr'].copy())
        else:
            from graph import Graph
            graph = Graph()
    raw = np.load(util.file_dir['knowair_fp'],mmap_mode='r')
    cities = raw.shape[1]
    del raw
    hist = util.config['train']['hist_len']
    horizon = util.config['train']['pred_len']
    data = HazeData(SimpleNamespace(node_num=cities),hist,horizon,
                    util.config['forecast']['dataset_num'],flag=args.split,normalization=normalization)
    if name=='RF':
        import joblib
        model = joblib.load(str(run/'model.joblib'))
    else:
        model = build_model(name,hist,horizon,data.feature.shape[-1]+1,cities,
                            args.batch_size,device,graph,normalization)
        model.load_state_dict(torch.load(str(run/'model.pth'),map_location=device))
    output = args.output or run/'evaluations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    scores = evaluate(model,data,hist,horizon,args.batch_size,device,
                      output=output if args.save_predictions else None,
                      max_windows=args.max_windows,random_forest=name=='RF')
    scores.update(model=name,checkpoint=str(run),split=args.split,
                  split_start=str(data.start_time),split_end=str(data.end_time))
    (output/'metrics.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
    print(json.dumps(scores,indent=2))
    print('Saved evaluation: '+str(output.resolve()))


if __name__=='__main__':
    main()
