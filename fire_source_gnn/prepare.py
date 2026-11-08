"""Pack already processed data; never rerun satellite, NLCD, or weather processing."""
import argparse
import json
import hashlib
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from data import MONITOR_COLUMNS, NLCD, WEATHER, geometry


def moments(x):
    x = np.asarray(x, dtype=np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    return mean.astype('float32'), np.where(std < 1e-8, 1, std).astype('float32')


def prepare(root, output, config):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Use a new cache directory; existing data will not be overwritten')
    output.mkdir(parents=True)
    (output / 'INCOMPLETE').write_text('Preparation in progress\n')
    fire_dir = root/'VIIRS/hourly_1km_vegetation_nh_6h_gap24h_5hl_min1_nlcd_weather'
    metadata = json.loads((fire_dir/'metadata.json').read_text())
    raw_path = root/'PM2.5_Forecasting_GNN-main/data/pm25gnn/data/dataset_fire_wind_aligned.npy'
    raw = np.load(str(raw_path), mmap_mode='r')
    baseline = json.loads((root/'gnn_data_exploration/metadata.json').read_text())
    origin = pd.Timestamp(baseline['time_start'])
    locations = pd.read_csv(root/'PM2.5_Forecasting_GNN/data/locations.txt', sep=r'\s+',
                            header=None, names=['index', 'name', 'lat', 'lon'])
    if not np.array_equal(locations['index'], np.arange(raw.shape[1])):
        raise ValueError('Monitor order does not match dataset axes')
    coords = locations[['lat', 'lon']].values
    cells = pd.read_csv(fire_dir/'cells.csv')
    cell_index = pd.Index(cells['cell_id'])
    if not cell_index.is_unique:
        raise ValueError('Duplicate cell geometry')
    with (root/'PM2.5_Forecasting_GNN/data/alt.pkl').open('rb') as f:
        altitude = pickle.load(f)
    elevation = np.array([altitude[(int(lat*10)/10, int(lon*10)/10)] for lat, lon in coords], dtype='float32')
    pm_edges = np.load(str(root/'hpc/nmt/fire_gnn/data/graph.npz'))['edge_index']
    distance, bearing = geometry(coords, coords)
    src, dst = pm_edges
    pm_geometry = np.stack((distance[src, dst]/300, np.sin(bearing[src, dst]),
                            np.cos(bearing[src, dst])), axis=-1)
    counts, targets, distances, bearings = [], [], [], []
    for first in range(0, len(cells), 2048):
        d, b = geometry(cells[['latitude', 'longitude']].values[first:first+2048], coords)
        keep = d <= config['fire_cutoff_km']
        counts.append(keep.sum(axis=1))
        r, c = np.nonzero(keep)
        targets.append(c.astype('int32')); distances.append(d[r, c]); bearings.append(b[r, c])
    graph = dict(candidate_ptr=np.r_[0, np.cumsum(np.concatenate(counts))],
                 candidate_target=np.concatenate(targets), candidate_distance=np.concatenate(distances),
                 candidate_bearing=np.concatenate(bearings), monitor_edges=pm_edges,
                 monitor_geometry=pm_geometry, elevation=elevation, monitor_coordinates=coords)
    np.savez(output/'graph.npz', **graph)
    report = dict(format_version=1, source_root=str(root), fire_source=str(fire_dir),
                  monitor_source=str(raw_path), fire_cutoff_km=config['fire_cutoff_km'],
                  monitor_features=WEATHER+['midnight_unix_seconds', 'utc_hour'],
                  fire_features=['episode_age_hours']+WEATHER+['frp_mw', 'midnight_unix_seconds', 'utc_hour'],
                  nlcd_categories=NLCD, missing_weather_policy='exclude and report',
                  source_weather_policy=metadata['weather']['precipitation_interval'],
                  monitor_weather_policy='Existing processed monitor covariates retained; upstream precipitation interval not independently established',
                  mode='Retrospective: reconstructed FRP and reanalysis at prediction/source times',
                  splits={}, candidate_edges=len(graph['candidate_target']))
    inputs = [raw_path, fire_dir/'cells.csv', fire_dir/'metadata.json',
              root/'PM2.5_Forecasting_GNN/data/locations.txt',
              root/'PM2.5_Forecasting_GNN/data/alt.pkl',
              root/'hpc/nmt/fire_gnn/data/graph.npz', root/'gnn_data_exploration/metadata.json']
    inputs += [fire_dir/(s+'_fire_hours.csv') for s in ['training', 'validation', 'testing']]
    report['input_inventory'] = [dict(path=str(p), bytes=p.stat().st_size,
        mtime_ns=p.stat().st_mtime_ns,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < 1000000 else None)
        for p in inputs]
    report['large_input_hash_policy'] = 'Files >= 1 MB recorded by size/mtime, not content hash'
    stats = {}
    for split, old in [('training', 'train'), ('validation', 'validation'), ('testing', 'test')]:
        spec = baseline['splits'][old]
        start, end = pd.Timestamp(spec['start']), pd.Timestamp(spec['end'])
        lo, hi = int((start-origin).total_seconds()/3600), int((end-origin).total_seconds()/3600)+1
        monitor = np.array(raw[lo:hi, :, MONITOR_COLUMNS], dtype='float64')
        pm25 = np.array(raw[lo:hi, :, -1], dtype='float32')
        dates = pd.date_range(start, end, freq='h')
        expected_date = dates.floor('D').asi8/1e9
        if not (np.all(monitor[:, :, -2] == expected_date[:, None]) and
                np.all(monitor[:, :, -1] == dates.hour.values[:, None])):
            raise ValueError('Monitor timestamp mismatch')
        if not np.isfinite(monitor).all() or not np.isfinite(pm25).all():
            raise ValueError('Nonfinite monitor input')
        pieces = {k: [] for k in ['features', 'frp', 'nlcd', 'cell', 'hour']}
        audit = dict(csv_rows=0, outside_monitor_period=0, missing_weather=0, inactive_frp=0, unknown_nlcd=0)
        columns = ['hour_utc', 'cell_id', 'fire_id', 'frp_mw', 'nlcd_class']+WEATHER
        for frame in pd.read_csv(fire_dir/(split+'_fire_hours.csv'), usecols=columns, chunksize=200000):
            audit['csv_rows'] += len(frame)
            times = pd.to_datetime(frame.hour_utc, utc=True)
            hours = ((times-start).dt.total_seconds()/3600).values
            if np.any(hours != np.floor(hours)):
                raise ValueError('Nonhourly fire timestamp')
            period = (hours >= 0) & (hours < len(monitor))
            audit['outside_monitor_period'] += int((~period).sum())
            complete = np.isfinite(frame[WEATHER].values).all(axis=1)
            audit['missing_weather'] += int((period & ~complete).sum())
            if not np.isfinite(frame.frp_mw).all() or (frame.frp_mw < 0).any():
                raise ValueError('Invalid FRP in source export')
            active = frame.frp_mw.values > 0
            audit['inactive_frp'] += int((period & complete & ~active).sum())
            keep = period & complete & active
            frame, times, hours = frame.loc[keep], times.loc[keep], hours[keep].astype('int32')
            index = cell_index.get_indexer(frame.cell_id)
            if np.any(index < 0):
                raise ValueError('Fire cell absent from geometry')
            episode_start = pd.to_datetime(frame.fire_id.str.rsplit('@', n=1).str[-1],
                                           format='%Y%m%dT%H%MZ', utc=True)
            age = (times-episode_start).dt.total_seconds().values/3600
            frp = frame.frp_mw.values
            if np.any(age < 0) or not np.isfinite(frp).all() or np.any(frp <= 0):
                raise ValueError('Invalid episode age or active FRP')
            category = frame.nlcd_class.map({code: i for i, code in enumerate(NLCD)}).fillna(0).values.astype('int16')
            audit['unknown_nlcd'] += int((category == 0).sum())
            features = np.column_stack((age, frame[WEATHER].values, frp,
                                        times.dt.floor('D').astype('int64').values/1e9, times.dt.hour.values))
            if not np.isfinite(features).all():
                raise ValueError('Invalid source features')
            for key, value in dict(features=features, frp=frp.astype('float32'), nlcd=category,
                                   cell=index.astype('int32'), hour=hours).items():
                pieces[key].append(value)
        packed = {k: np.concatenate(v) for k, v in pieces.items()}
        order = np.lexsort((packed['cell'], packed['hour']))
        packed = {k: v[order] for k, v in packed.items()}
        if np.any((np.diff(packed['hour']) == 0) & (np.diff(packed['cell']) == 0)):
            raise ValueError('Duplicate cell/hour records')
        directory = output/split
        directory.mkdir()
        np.save(directory/'monitor.npy', monitor)  # retain exact Unix timestamps
        np.save(directory/'pm25.npy', pm25)
        for key, values in packed.items():
            np.save(directory/('fire_'+key+'.npy'), values)
        np.save(directory/'hour_ptr.npy', np.r_[0, np.cumsum(np.bincount(packed['hour'], minlength=len(monitor)))])
        audit.update(start=str(start), end_inclusive=str(end), monitor_hours=len(monitor),
                     retained_fire_rows=len(packed['hour']))
        report['splits'][split] = audit
        if split == 'training':
            stats['monitor_mean'], stats['monitor_std'] = moments(monitor.reshape(-1, 9))
            stats['fire_mean'], stats['fire_std'] = moments(packed['features'])
            stats['pm_mean'], stats['pm_std'] = moments(pm25.reshape(-1))
            stats['elevation_mean'], stats['elevation_std'] = moments(elevation)
        print(split, json.dumps(audit), flush=True)
    np.savez(output/'statistics.npz', **stats)
    (output/'manifest.json').write_text(json.dumps(report, indent=2))
    (output/'INCOMPLETE').unlink()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    args = parser.parse_args()
    prepare(args.project_root, args.output, json.loads(args.config.read_text()))
