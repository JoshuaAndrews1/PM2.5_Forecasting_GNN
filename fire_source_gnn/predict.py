"""Export a checkpoint's complete 48-hour retrospective rollout."""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import torch
from data import PreparedData
from model import FireSourceGNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['training', 'validation', 'testing'], default='testing')
    parser.add_argument('--start-index', type=int, default=240, help='Prediction start hour within selected split')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists')
    torch.set_num_threads(2)
    data = PreparedData(args.cache)
    checkpoint = torch.load(str(args.checkpoint), map_location=args.device)
    if checkpoint['config']['fire_cutoff_km'] != data.meta['fire_cutoff_km']:
        raise ValueError('Checkpoint/candidate cutoff mismatch')
    model = FireSourceGNN(checkpoint['config'], data.stats, data.graph).to(args.device)
    model.load_state_dict(checkpoint['model']); model.eval()
    with torch.no_grad():
        pred = model(data, args.split, args.start_index)*model.pm_std+model.pm_mean
    origin = datetime.fromisoformat(data.meta['splits'][args.split]['start'])
    with args.output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hour_utc', 'monitor_index', 'prediction_ug_m3', 'prediction_unclipped', 'observed_ug_m3'])
        for offset, row in enumerate(pred.cpu().numpy()):
            index = args.start_index+offset
            for station, value in enumerate(row):
                writer.writerow([(origin+timedelta(hours=index)).isoformat(), station, max(0., float(value)),
                                 float(value), float(data.splits[args.split]['pm25'][index, station])])
