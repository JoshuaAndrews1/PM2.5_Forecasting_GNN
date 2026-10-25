import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from util import config, file_dir
from datetime import datetime
import numpy as np
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data

class HazeData(data.Dataset):
    def __init__(self, graph,
                       hist_len=1,
                       pred_len=24,
                       dataset_num=1,
                       flag='Train',
                       normalization=None,
                       ):
        if flag != 'Train' and normalization is None:
            raise ValueError('Validation/test require training normalization statistics')
        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')
          
        self.start_time = self._get_time(config['dataset'][dataset_num][start_time_str])
        self.end_time = self._get_time(config['dataset'][dataset_num][end_time_str])
        self.data_start = self._get_time(config['dataset']['data_start'])
        self.data_end = self._get_time(config['dataset']['data_end'])
        self.knowair_fp = file_dir['knowair_fp']
        self.graph = graph
        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        del self.knowair
        self._process_feature()
        self.feature, self.pm25 = np.float32(self.feature), np.float32(self.pm25)
        if normalization is None:
            self._calc_mean_std()
        else:
            for name in ('feature_mean', 'feature_std', 'pm25_mean', 'pm25_std'):
                setattr(self, name, np.asarray(normalization[name], dtype=np.float32).copy())
        self.wind_mean = self.feature_mean[-2:]
        self.wind_std = self.feature_std[-2:]
        self.normalization = {name: getattr(self, name) for name in
                              ('feature_mean', 'feature_std', 'pm25_mean', 'pm25_std')}
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)                 
        self._norm()
        self._dictionary()

    def _dictionary(self):
        self.time_index = {}
        for i in range(self.time_arr_full.shape[0]):
            self.time_index[self.time_arr_full[i]] = i

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std
        self.feature_full = (self.feature_full - self.feature_mean) / self.feature_std
        self.pm25_full = (self.pm25_full - self.pm25_mean) / self.pm25_std

    def _add_time_dim(self, seq_len):
        self.pm25_full = np.empty_like(self.pm25)
        self.pm25_full[:] = self.pm25[:]
        if seq_len < 2 or len(self.pm25) < seq_len:
            raise ValueError('Split must contain at least hist_len + pred_len hours')
        self.pm25 = self.pm25[seq_len-1:]
        self.feature_full = np.empty_like(self.feature)
        self.feature_full[:] = self.feature[:]
        self.feature = self.feature[seq_len-1:]
        self.time_arr_full = np.empty_like(self.time_arr)
        self.time_arr_full[:] = self.time_arr[:]
        self.time_arr = self.time_arr[seq_len-1:]

    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = np.maximum(self.feature.std(axis=(0,1)), 1e-8)
        self.wind_mean = self.feature_mean[-2:] # final speed/direction features
        self.wind_std = self.feature_std[-2:] # final speed/direction features
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = max(float(self.pm25.std()), 1e-8)

    def _process_feature(self):
        metero_var = config['data']['metero_var']
        metero_use = config['forecast']['metero_use']
        metero_idx = [metero_var.index(var) for var in metero_use]
        self.feature = self.feature[:,:,metero_idx]
        u = self.feature[:, :, metero_use.index('u_component_of_wind+950')] * units.meter / units.second
        v = self.feature[:, :, metero_use.index('v_component_of_wind+950')] * units.meter / units.second
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
        direc = mpcalc.wind_direction(u, v, convention='to')._magnitude
        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)
        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direc[:, :, None]
                                       ], axis=-1)

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        if not 0 <= start_idx <= end_idx < len(self.pm25):
            raise ValueError('Configured split lies outside the dataset')
        self.pm25 = np.array(self.pm25[start_idx: end_idx+1, :], copy=True)
        self.feature = np.array(self.feature[start_idx: end_idx+1, :], copy=True)
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]

    def _gen_time_arr(self):
        expected = int((self.data_end - self.data_start).total_seconds() / 3600) + 1
        if len(self.pm25) != expected:
            raise ValueError('Dataset length does not match configured hourly date range')
        self.time_arrow = [self.data_start.shift(hours=i) for i in range(expected)]
        self.time_arr = np.asarray([t.datetime.timestamp() for t in self.time_arrow])

    def _load_npy(self):
        self.knowair = np.load(self.knowair_fp, mmap_mode='r')
        self.feature = self.knowair[:,:,:-1]
        self.pm25 = self.knowair[:,:,-1:]

    def _get_idx(self, t):
        t0 = self.data_start
        # determines time granularity
        return int((t - t0).total_seconds() / 3600)

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index], self.time_arr[index]

if __name__ == '__main__':
    from graph import Graph
    graph = Graph()
    train_data = HazeData(graph, flag='Train')
    val_data = HazeData(graph, flag='Val', normalization=train_data.normalization)
    test_data = HazeData(graph, flag='Test', normalization=train_data.normalization)
