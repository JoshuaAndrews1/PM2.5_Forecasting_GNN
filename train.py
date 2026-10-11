import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
from util import config, file_dir
from metrics import forecast_metrics

from graph import Graph
import pdb
from dataset import HazeData

from model.MLP import MLP
from model.LSTM import LSTM
from model.GRU import GRU
from model.GC_LSTM import GC_LSTM
from model.nodesFC_GRU import nodesFC_GRU
from model.PM25_GNN import PM25_GNN
from model.PM25_GNN_nosub import PM25_GNN_nosub

import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(device)


batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
seq_len = hist_len + pred_len
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
criterion = nn.MSELoss()

def initialize():
    global graph, city_num, train_data, val_data, test_data
    global in_dim, wind_mean, wind_std, pm25_mean, pm25_std
    graph = Graph()
    city_num = graph.node_num
    train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
    val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val',
                        normalization=train_data.normalization)
    test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test',
                         normalization=train_data.normalization)
    in_dim = train_data.feature.shape[-1] + 1
    wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
    pm25_mean, pm25_std = train_data.pm25_mean, train_data.pm25_std


def get_metric(predict_epoch, label_epoch):
    scores = forecast_metrics(predict_epoch[:, hist_len:], label_epoch[:, hist_len:])
    return tuple(scores[key] for key in ('rmse_window_station', 'mae', 'csi', 'pod', 'far'))


def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr + \
                '========================================\n'
    return exp_info

def get_model():
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'PM25_GNN':
        return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_nosub':
        return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    else:
        raise Exception('Wrong model name!')

def prepare(pm25, feature, time_arr, flag):
    pm25_hist = np.full((pm25.shape[0], hist_len, pm25.shape[1], pm25.shape[2]), -1.0, dtype=np.float32)#double)
    pm25_label = np.full((pm25.shape[0], pred_len, pm25.shape[1], pm25.shape[2]), -1.0, dtype=np.float32)#double)
    feature = np.full((feature.shape[0], hist_len+pred_len, feature.shape[1], feature.shape[2]), -1.0, dtype=np.float32)#double)
    pm = np.full((pm25.shape[0], hist_len+pred_len, pm25.shape[1], pm25.shape[2]), -1.0, dtype=np.float32)#double)

    for i in range(np.asarray(time_arr).shape[0]):
        if flag == "Train":
            end = train_data.time_index[np.asarray(time_arr)[i]]
            pm25_hist[i,:,:,:] = train_data.pm25_full[end-seq_len+1:end-pred_len+1, :, :]
            pm25_label[i,:,:,:] = train_data.pm25_full[end-pred_len+1:end+1, :, :]
            feature[i,:,:,:] = train_data.feature_full[end-seq_len+1:end+1, :, :]
            pm[i,:,:,:] = train_data.pm25_full[end-seq_len+1:end+1, :, :]
        elif flag == "Val":
            end = val_data.time_index[np.asarray(time_arr)[i]]
            pm25_hist[i,:,:,:] = val_data.pm25_full[end-seq_len+1:end-pred_len+1, :, :]
            pm25_label[i,:,:,:] = val_data.pm25_full[end-pred_len+1:end+1, :, :]
            feature[i,:,:,:] = val_data.feature_full[end-seq_len+1:end+1, :, :]
            pm[i,:,:,:] = val_data.pm25_full[end-seq_len+1:end+1, :, :]
        else:
            end = test_data.time_index[np.asarray(time_arr)[i]]
            pm25_hist[i,:,:,:] = test_data.pm25_full[end-seq_len+1:end-pred_len+1, :, :]
            pm25_label[i,:,:,:] = test_data.pm25_full[end-pred_len+1:end+1, :, :]
            feature[i,:,:,:] = test_data.feature_full[end-seq_len+1:end+1, :, :]
            pm[i,:,:,:] = test_data.pm25_full[end-seq_len+1:end+1, :, :]

    return torch.tensor(pm25_hist, dtype=torch.float), torch.tensor(pm25_label, dtype=torch.float), torch.tensor(feature, dtype=torch.float), pm.astype('float')


def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    sample_count = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
        pm25,feature, time_arr = data
        pm25_hist, pm25_label, feature, pm25 = prepare(pm25, feature, time_arr, "Train")
        feature = feature.to(device)
        pm25_hist = pm25_hist.to(device)
        pm25_label = pm25_label.to(device)
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * pm25_label.size(0)
        sample_count += pm25_label.size(0)
    if sample_count == 0:
        raise ValueError('No eligible samples in train loader')
    train_loss /= sample_count
    return train_loss

@torch.no_grad()
def val(val_loader, model):
    model.eval()
    val_loss = 0
    sample_count = 0
    for batch_idx, data in tqdm(enumerate(val_loader)):
        pm25, feature, time_arr = data
        pm25_hist, pm25_label, feature, pm25 = prepare(pm25, feature, time_arr, "Val")
        feature = feature.to(device)
        pm25_hist = pm25_hist.to(device)
        pm25_pred = model(pm25_hist, feature)
        pm25_label = pm25_label.to(device)
        loss = criterion(pm25_pred, pm25_label)
        val_loss += loss.item() * pm25_label.size(0)
        sample_count += pm25_label.size(0)
    if sample_count == 0:
        raise ValueError('No eligible samples in val loader')
    val_loss /= sample_count
    return val_loss


@torch.no_grad()
def test(test_loader, model):
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    sample_count = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25_hist, pm25_label, feature, pm25 = prepare(pm25, feature, time_arr, "Test")
        feature = feature.to(device)
        pm25_hist = pm25_hist.to(device)
        pm25_label = pm25_label.to(device)
        pm25_pred = model(pm25_hist, feature)

        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item() * pm25_label.size(0)
        sample_count += pm25_label.size(0)

        pm25_pred = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label = pm25 * pm25_std + pm25_mean
        predict_list.append(pm25_pred)
        label_list.append(pm25_label)
        time_list.append(time_arr.cpu().detach().numpy())

    if sample_count == 0:
        raise ValueError('No eligible samples in test loader')
    test_loss /= sample_count
    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    return test_loss, predict_epoch, label_epoch, time_epoch


def get_mean_std(values):
    return np.mean(values), np.std(values)


def main():
    initialize()
    exp_info = get_exp_info()
    print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], []

    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__

        print(str(model))

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        np.savez(os.path.join(exp_model_dir, 'normalization.npz'), **train_data.normalization)
        val_loss_min = float('inf')
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0

        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))

            train_loss = train(train_loader, model, optimizer)
            val_loss = val(val_loader, model)

            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)

            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise RuntimeError('Non-finite training/validation loss')

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                train_loss_, val_loss_ = train_loss, val_loss
            elif epoch - best_epoch >= early_stop:
                break

        if not os.path.isfile(model_fp):
            raise RuntimeError('No valid checkpoint was produced')
        model.load_state_dict(torch.load(model_fp, map_location=device))
        test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model)
        rmse, mae, csi, pod, far = get_metric(predict_epoch, label_epoch)
        scores = forecast_metrics(predict_epoch[:, hist_len:], label_epoch[:, hist_len:])
        import json
        # JSON null denotes undefined event scores (zero denominator).
        with open(os.path.join(exp_model_dir, 'metrics.json'), 'w') as output:
            json.dump({key: value if np.isfinite(value) else None
                       for key, value in scores.items()}, output, indent=2)
        if save_npy:
            np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
            np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
            np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)

        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)

        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE(window/station): %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))

    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE(window/station) | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list))

    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)


if __name__ == '__main__':
    main()
