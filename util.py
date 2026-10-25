"""Forecast configuration, including compatibility with saved older runs."""
import copy
import os
from pathlib import Path
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
FIRE_FEATURES = ('frp_25km_idw', 'frp_50km_idw', 'frp_100km_idw', 'frp_500km_idw', 'numfires')
config = {}
file_dir = {}


def configure(values):
    """Update shared dictionaries in place so dataset imports remain valid."""
    values = copy.deepcopy(values)
    forecast = values.pop('experiments', values.get('forecast', {}))
    values['forecast'] = forecast
    # Compatibility for older saved comparison/reproduction scripts.
    values['experiments'] = forecast
    config.clear()
    config.update(values)
    file_dir.clear()
    file_dir.update(config['filepath']['GPU-Server'])
    for key, env in [('knowair_fp', 'PM25_DATA'), ('results_dir', 'PM25_RESULTS')]:
        path = Path(os.environ.get(env, file_dir[key]))
        file_dir[key] = str(path if path.is_absolute() else PROJECT_DIR / path)


def load_config(path):
    with open(path, encoding='utf-8') as source:
        configure(yaml.safe_load(source))


def snapshot():
    """Run metadata with effective paths and no legacy aliases."""
    values = copy.deepcopy(config)
    values.pop('experiments', None)
    values['filepath']['GPU-Server'] = dict(file_dir)
    values['train'].pop('exp_repeat', None)
    return values


conf_fp = os.environ.get('PM25_CONFIG')
if conf_fp is None:
    local = PROJECT_DIR / 'config.local.yaml'
    conf_fp = str(local if local.is_file() else PROJECT_DIR / 'config.yaml')
load_config(conf_fp)
