import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from layers.Embed import DataEmbedding
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return F.gelu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        d_model = 128 # configs.d_model
        n_layer = 3  #configs.e_layers + configs.d_layers
        # Embedding
        self.dec_embedding = DataEmbedding(configs.dec_in, d_model, configs.embed, configs.freq,
                                           configs.dropout)

        num_chans = [d_model] * (n_layer - 1) + [d_model]
        kernel_size = 10 #self.seq_len // len(num_chans)
        self.tcn = TemporalConvNet(d_model, num_chans, kernel_size=kernel_size, dropout=configs.dropout)
        self.linear_dec = nn.Linear(d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x = torch.cat([x_enc[:, :-self.label_len, :], x_dec], dim=1)
        x_mark = torch.cat([x_mark_enc[:, :-self.label_len, :], x_mark_dec], dim=1)
        # (B, L, D) > (B, D, L)
        dec_out = self.dec_embedding(x, x_mark).permute(0, 2, 1)
        dec_out = self.tcn(dec_out)
        # (B, D, L) > (B, L, D)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = F.gelu(dec_out)
        dec_out = self.linear_dec(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], None
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
