from model.cells import LSTMCell
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import torch
from torch import nn


class GC_LSTM(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index):
        super(GC_LSTM, self).__init__()
        self.edge_index = torch.LongTensor(edge_index)
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = 1
        self.gcn_out = 1
        self.conv = ChebConv(self.in_dim, self.gcn_out, K=2)
        self.lstm_cell = LSTMCell(self.in_dim + self.gcn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):
        self.edge_index = self.edge_index.to(self.device)
        batch_size = pm25_hist.size(0)
        edge_index = (self.edge_index[:, None, :] +
                      torch.arange(batch_size, device=self.device)[None, :, None] * self.city_num).reshape(2, -1)
        pm25_pred = []
        h0 = torch.zeros(batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.zeros(batch_size * self.city_num, self.hid_dim).to(self.device)
        cn = c0
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            x_gcn = x.contiguous()
            x_gcn = x_gcn.view(batch_size * self.city_num, -1)
            x_gcn = F.sigmoid(self.conv(x_gcn, edge_index))
            x_gcn = x_gcn.view(batch_size, self.city_num, -1)
            x = torch.cat((x, x_gcn), dim=-1)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred
