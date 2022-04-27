import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from statsmodels.tsa.api import acf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang


def z_test(p, s):
    p_m = p.mean(axis=1)
    p_n = p.shape[1]
    p_v = p.std(axis=1)**2

    s_m = s.mean(axis=1)
    s_n = s.shape[1]
    s_v = s.std(axis=1)**2
    z = (p_m - s_m) / np.sqrt(p_v/p_n + s_v/s_n + 1e-4)
    return z


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def cal_weights(labels, reweight, max_target=51, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    value_dict = {x: 0 for x in range(max_target)}
    # mbr
    for label in labels:
        value_dict[min(max_target - 1, int(label))] += 1
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    print(f"Using re-weighting: [{reweight.upper()}]")

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    # weights = [np.float32(1 / x) for x in num_per_label]
    weights = [np.float32(x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights


def get_periodic_diff(inputs, targets):
    period_diff = []
    for idx in range(inputs.shape[0]):
        diffs = []
        for dim in range(inputs.shape[2]):
            x = inputs[idx,:,dim]
            y = targets[idx, :, dim]
            auto_x = torch.from_numpy(acf(x)).float()
            auto_y = torch.from_numpy(acf(y)).float()
            diff = torch.abs(auto_x - auto_y).mean()

            diffs.append(diff)
        diffs = torch.stack(diffs)
        period_diff.append(diffs)
    return torch.stack(period_diff)

def get_decomposed_data(inputs, targets):
    B, IL, D = inputs.shape
    B, TL, D = targets.shape
    # (B, L, D)
    # (IL,D) cat (B-1, D) cat (TL, D)> (TL + B - 1 + TL, D)
    series = np.concatenate([inputs[0], inputs[1:, -1, :], targets[-1]], axis=0)
    print(inputs[0].shape, inputs[1:, -1, :].shape, targets[-1].shape)
    decomposed_series = seasonal_decompose(series, model='additive', period=IL + TL)
    is1d = (len(decomposed_series.observed.shape) == 1)

    observed = np.expand_dims(decomposed_series.observed, axis=1) if is1d else decomposed_series.observed
    trend = np.expand_dims(decomposed_series.trend, axis=1) if is1d else decomposed_series.trend
    seasonal = np.expand_dims(decomposed_series.seasonal, axis=1) if is1d else decomposed_series.seasonal

    trend_series = pd.DataFrame(trend).fillna(method='bfill').fillna(method='ffill').values
    print(observed.shape, trend_series.shape, seasonal.shape)

    remainder = observed - seasonal # observed - trend_series - seasonal
    # remainder (IL + B - 1 + TL, D)
    remainder_windows = torch.from_numpy(np.lib.stride_tricks.sliding_window_view(remainder, (IL + TL, D)))
    remainder_windows = remainder_windows.squeeze(1) # (B, 1, IL, TL) > (B, IL, TL)
    remainder_inputs, remainder_targets = remainder_windows[:, :IL, :],  remainder_windows[:, -TL:, :]

    assert inputs.shape == remainder_inputs.shape
    assert targets.shape == remainder_targets.shape
    return remainder_inputs, remainder_targets

def digitize_gap(inputs, targets, max_val=None, n_bins=200):
    B, XL, D = inputs.shape
    B, YL, D = targets.shape

    CL = YL if XL > YL else XL
    # inputs = inputs[:, -CL:, :]
    # targets = targets[:, :CL, :]

    # pdiff = get_periodic_diff(inputs, targets).numpy()
    diff_mus = z_test(inputs, targets).mean(axis=1)
    #diff_mus = (np.abs(inputs.mean(axis=1) - targets.mean(axis=1))).mean(axis=1)
    # print(pdiff.min(), pdiff.max(),  diff_mus.min(), diff_mus.max())

    if max_val is None:
        max_val = diff_mus.max()
        min_val = diff_mus.min()
        print(diff_mus.shape, max_val, min_val)
    bins = np.linspace(min_val, max_val, n_bins)
    bin_labels = np.digitize(diff_mus, bins=bins) - 1
    weights = cal_weights(bin_labels, 'sqrt_inv', n_bins, lds=True)
    print(f'max gap:{max_val:.4f} min w:{min(weights):.4f} max w:{max(weights):.4f}')
    return weights, max_val

def digitize_gap_mul(inputs, targets, max_val=None, n_bins=200):
    B, XL, D = inputs.shape
    B, YL, D = targets.shape

    CL = YL if XL > YL else XL
    diff_mus = z_test(inputs, targets)

    weights = []
    for d in range(D):
        max_val = diff_mus[:, d].max()
        min_val = diff_mus[:, d].min()
        print(diff_mus.shape, max_val, min_val)
        bins = np.linspace(min_val, max_val, n_bins)
        bin_labels = np.digitize(diff_mus[:, d], bins=bins) - 1
        weight = cal_weights(bin_labels, 'sqrt_inv', n_bins, lds=True)
        weights.append(weight)
        print(f'max gap:{max_val:.4f} min w:{min(weight):.4f} max w:{max(weight):.4f}')
    return torch.tensor(weights).transpose(0, 1).float(), max_val

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', reweight=None, max_val=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.reweight = reweight
        self.max_val = max_val
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.weights = torch.ones((self.__len__(), 1))
        if (self.reweight == 'lds') and (flag =='train'):
            decompose = False
            self._prepare_weights(decompose)

    def _prepare_weights(self, decompose=False):
        train_inputs, train_targets = [], []
        for index in range(self.__len__()):
            seq_x, seq_y, seq_x_mark, seq_y_mark, _ = self.__getitem__(index)
            train_inputs.append(seq_x)
            train_targets.append(seq_y[-self.pred_len:])

        train_inputs = np.stack(train_inputs, axis=0)
        train_targets = np.stack(train_targets, axis=0)
        print(train_inputs.shape, train_targets.shape)

        if decompose:
            print('Local Discrepancy on Remainder from TS decomposition')
            remainder_inputs, remainder_targets = get_decomposed_data(train_inputs, train_targets)
            self.weights, self.max_val = digitize_gap_mul(remainder_inputs, remainder_targets, max_val=self.max_val)
        else:
            print('Local Discrepancy on Raw Series')
            self.weights, self.max_val = digitize_gap_mul(train_inputs, train_targets, max_val=self.max_val)


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.weights[index]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', reweight=None, max_val=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.reweight = reweight
        self.max_val = max_val
        self.weights = torch.ones((self.__len__(), 1))
        if (self.reweight == 'lds') and (flag =='train'):
            decompose = False
            self._prepare_weights(decompose)

    def _prepare_weights(self, decompose=False):
        train_inputs, train_targets = [], []
        for index in range(self.__len__()):
            seq_x, seq_y, seq_x_mark, seq_y_mark, _ = self.__getitem__(index)
            train_inputs.append(seq_x)
            train_targets.append(seq_y[-self.pred_len:])
        train_inputs = np.stack(train_inputs, axis=0)
        train_targets = np.stack(train_targets, axis=0)
        print(train_inputs.shape, train_targets.shape)

        if decompose:
            print('Local Discrepancy on Remainder from TS decomposition')
            remainder_inputs, remainder_targets = get_decomposed_data(train_inputs, train_targets)
            self.weights, self.max_val = digitize_gap_mul(remainder_inputs, remainder_targets, max_val=self.max_val)
        else:
            print('Local Discrepancy on Raw Series')
            self.weights, self.max_val = digitize_gap_mul(train_inputs, train_targets, max_val=self.max_val)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.weights[index]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', reweight=None, max_val=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.reweight = reweight
        self.max_val = max_val
        self.weights = torch.ones((self.__len__(), 1))
        if (self.reweight == 'lds') and (flag =='train'):
            decompose = False
            self._prepare_weights(decompose)

    def _prepare_weights(self, decompose=False):
        train_inputs, train_targets = [], []
        for index in range(self.__len__()):
            seq_x, seq_y, seq_x_mark, seq_y_mark, _ = self.__getitem__(index)
            train_inputs.append(seq_x)
            train_targets.append(seq_y[-self.pred_len:])
        train_inputs = np.stack(train_inputs, axis=0)
        train_targets = np.stack(train_targets, axis=0)
        print(train_inputs.shape, train_targets.shape)

        if decompose:
            print('Local Discrepancy on Remainder from TS decomposition')
            remainder_inputs, remainder_targets = get_decomposed_data(train_inputs, train_targets)
            self.weights, self.max_val = digitize_gap_mul(remainder_inputs, remainder_targets, max_val=self.max_val)
        else:
            print('Local Discrepancy on Raw Series')
            self.weights, self.max_val = digitize_gap_mul(train_inputs, train_targets, max_val=self.max_val)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0: # don't use time features
            df_stamp['month'] = 0.0 # df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = 0.0 # df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = 0.0 # df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = 0.0 # df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.weights[index]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
