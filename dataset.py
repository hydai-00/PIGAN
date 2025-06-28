import os
import torch.utils.data as data
import pandas as pd
import warnings
import time
import numpy as np
import torch
import pickle
from datetime import datetime

warnings.filterwarnings("ignore")


def calculate_ndvi(row, epsilon=1e-10):
    return (row['B8'] - row['B4']) / (row['B8'] + row['B4'] + epsilon)


def calculate_evi(row, epsilon=1e-10):
    return 2.5 * (row['B8'] - row['B4']) / (row['B8'] + 6 * row['B4'] - 7.5 * row['B2'] + 1e4 + epsilon)


def calculate_savi(row, epsilon=1e-10):
    L = 0.5
    return ((row['B8'] - row['B4']) * (1 + L)) / (row['B8'] + row['B4'] + L * 1e4 + epsilon)


def calculate_ndbi(row, epsilon=1e-10):
    return (row['B11'] - row['B8']) / (row['B11'] + row['B8'] + epsilon)


def calculate_ndwi(row, epsilon=1e-10):
    return (row['B3'] - row['B8']) / (row['B3'] + row['B8'] + epsilon)


def calculate_ndmi(row, epsilon=1e-10):
    return (row['B8'] - row['B11']) / (row['B8'] + row['B11'] + epsilon)


def calculate_ndre(row, epsilon=1e-10):
    return (row['B8A'] - row['B5']) / (row['B8A'] + row['B5'] + epsilon)


def calculate_rvi(row, epsilon=1e-10):
    return row['B8'] / (row['B4'] + epsilon)


def calculate_ngrdi(row, epsilon=1e-10):
    return (row['B3'] - row['B4']) / (row['B3'] + row['B4'] + epsilon)


def calculate_cvi(row, epsilon=1e-10):
    return (row['B8'] * row['B4']) / (row['B3'] * row['B3'] + epsilon)


def calculate_ci(row, epsilon=1e-10):
    return row['B8'] / (row['B5'] + epsilon) - 1


def calculate_mtci(row, epsilon=1e-10):
    return (row['B6'] - row['B5']) / (row['B5'] - row['B4'] + epsilon)


def calculate_cr(row, epsilon=1e-10):
    return (row['VH'] + 35) / (row['VV'] + 25 + epsilon) / 1.4


def calculate_Span(row):
    return (row['VH'] + 35) / 35 + (row['VV'] + 25) / 25


def calculate_nrpb(row, epsilon=1e-10):
    return ((row['VH'] + 35) / 35 - (row['VV'] + 25) / 25) / ((row['VH'] + 35) / 35 + (row['VV'] + 25) / 25 + epsilon)


def calculate_time(row):
    day_of_year = row["date"].timetuple().tm_yday  # 获取一年中的第几天
    days_in_year = 366 if (
            row["date"].year % 4 == 0 and (row["date"].year % 100 != 0 or row["date"].year % 400 == 0)) else 365
    sin_feature = np.sin(2 * np.pi * day_of_year / days_in_year)
    return sin_feature


class Dataset(data.Dataset):
    def __init__(self, file_path='./data', length=360, is_train=True, normal=True, ind=None, is_test=False):
        self.start_time = time.time()
        self.is_train = is_train
        if self.is_train:
            self.s1_path = os.path.join(file_path, 'sentinel1_data_new_train_rm.csv')
            self.s2_path = os.path.join(file_path, 'sentinel2_data_new_train_rm.csv')
            self.cache_file = './dataset_cache_rm.pkl'
        else:
            self.s1_path = os.path.join(file_path, 'sentinel1_data_new_test_2_rm.csv')
            self.s2_path = os.path.join(file_path, 'sentinel2_data_new_test_2_rm.csv')
            self.cache_file = './dataset_cache_test_rm.pkl'
        if is_test:
            self.s1_path = os.path.join(file_path, 'sentinel1_data_test_for_test.csv')
            self.s2_path = os.path.join(file_path, 'sentinel2_data_test_for_test.csv')
            self.cache_file = './dataset_cache_test_for_test.pkl'
            seed = 42
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.s1_length = int(length / 12)
        self.s2_length = int(length / 1)
        self.th = 0.85  # 至少保证s2数据中25%的点可用
        self.th_s2 = 0.2  # 至少保证s1数据中20%的点可用
        self.gap_day = 360
        # self.fake_ratio_s2 = 0.95  # 至少保证数据中5%的点可用
        self.fake_num = 10
        self.fake_ratio_s1 = 0.2
        if ind:
            self.indicator_S2 = ['NDVI'] + ind[0] + ['time']
            self.indicator_S1 = ind[1] + ['time']
        else:
            self.indicator_S2 = ['NDVI'] + ['CI', 'RVI', 'NGRDI', 'B1', 'CVI', 'B3'] + ['time']
            self.indicator_S1 = ['VV', 'VH'] + ['time']

        self.norm = normal
        self.indicator_functions_S1 = {
            'CR': calculate_cr,
            'Span': calculate_Span,
            'NRPB': calculate_nrpb
        }

        self.indicator_functions_S2 = {
            'NDVI': calculate_ndvi,
            'EVI': calculate_evi,
            'SAVI': calculate_savi,
            'NDBI': calculate_ndbi,
            'NDWI': calculate_ndwi,
            'NDMI': calculate_ndmi,
            'NDRE': calculate_ndre,
            'RVI': calculate_rvi,
            'NGRDI': calculate_ngrdi,
            'CVI': calculate_cvi,
            'CI': calculate_ci,
            'MTCI': calculate_mtci
        }

        if os.path.exists(self.cache_file):
            print('Loading cached dataset...')
            self.dataset = self.load_dataset()
        else:
            keep_columns = ['point_id', 'cropland', 'date']
            self.s1_data = pd.read_csv(self.s1_path)
            self.s1_data['date'] = pd.to_datetime(self.s1_data['date'])
            nan_rows = self.s1_data[self.s1_data.isna().any(axis=1)]
            self.s1_data.loc[nan_rows.index, ~self.s1_data.columns.isin(keep_columns)] = self.s1_data.loc[
                nan_rows.index, ~self.s1_data.columns.isin(keep_columns)].fillna(0)
            self.s1_data.loc[nan_rows.index, 'QA60'] = 1

            self.s2_data = pd.read_csv(self.s2_path)
            self.s2_data['date'] = pd.to_datetime(self.s2_data['date'])
            nan_rows = self.s2_data[self.s2_data.isna().any(axis=1)]
            self.s2_data.loc[nan_rows.index, ~self.s2_data.columns.isin(keep_columns)] = self.s2_data.loc[
                nan_rows.index, ~self.s2_data.columns.isin(keep_columns)].fillna(0)
            self.s2_data.loc[nan_rows.index, 'QA60'] = 1

            self.dataset = []
            self.make_dataset()
            self.save_dataset()
            print('Dataset compilation completed!')

        if self.norm:
            self.norm_list = []
            while len(self.norm_list) < len(self):
                samples = np.random.normal(0.87, 0.06, len(self))
                valid_samples = samples[(samples >= 0.78) & (samples <= 0.97)]
                self.norm_list.extend(valid_samples.tolist())
            self.norm_list = np.array(self.norm_list[:len(self)])
            self.norm_list = np.sort(self.norm_list)[::-1]
            self.miss_list = []
            for index in range(len(self.dataset)):
                self.miss_list.append((index, self.dataset[index]['s2']['QA60'].values.mean()))
            self.miss_list = sorted(self.miss_list, key=lambda x: x[1], reverse=True)
            self.miss_list = [(self.miss_list[index][0], self.norm_list[index] - self.miss_list[index][1]) for index in
                              range(len(self.miss_list))]
            self.miss_list = sorted(self.miss_list, key=lambda x: x[0])

        print('Dataset successfully build!')

    def make_dataset(self):
        grouped_s2 = self.s2_data.groupby('point_id')
        grouped_s1 = self.s1_data.groupby('point_id')

        for point_id, group_s2 in grouped_s2:
            if point_id in grouped_s1.groups:
                group_s1 = grouped_s1.get_group(point_id)
            print(point_id)
            if point_id % 100 == 0:
                cost = time.time() - self.start_time
                print('It cost {} s,length {}'.format(cost, len(self)))
            i = 0
            while i <= len(group_s2) - self.s2_length - 1:
                window_s2 = group_s2.iloc[i: i + self.s2_length]
                if window_s2['target'].eq(2).any() or sum(window_s2['QA60']) >= self.th * self.s2_length:
                    i = i + 1
                    continue
                date_range = window_s2['date']
                middle_date = date_range.iloc[len(date_range) // 2]
                group_s1.loc[:, 'date_diff'] = (group_s1['date'] - middle_date).abs()
                group_s1_sorted = group_s1.sort_values('date_diff')
                window_s1 = group_s1_sorted.head(self.s1_length).sort_values('date')
                window_s1 = window_s1.drop(columns=['date_diff'])

                df1_min_date = window_s1['date'].min()
                df1_max_date = window_s1['date'].max()

                df2_min_date = window_s2['date'].min()
                df2_max_date = window_s2['date'].max()

                min_date_diff = abs((df1_min_date - df2_min_date).days)
                max_date_diff = abs((df1_max_date - df2_max_date).days)
                max_diff = max(min_date_diff, max_date_diff)

                if window_s1['target'].eq(2).any() or sum(
                        window_s1['target']) >= self.th_s2 * self.s1_length or max_diff >= 10:
                    i = i + 1
                    continue

                ret = {}
                ret['s1'] = window_s1
                ret['s2'] = window_s2
                self.dataset.append(ret)
                i = i + self.gap_day

    def save_dataset(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.dataset, f)
        print(f'Dataset saved to {self.cache_file}')

    def load_dataset(self):
        with open(self.cache_file, 'rb') as f:
            dataset = pickle.load(f)
        print(f'Dataset loaded from {self.cache_file}')
        return dataset

    def _normalize(self, data, optical):
        if optical:
            min_v = 0
            max_v = 10000
        else:
            min_v = [-35, -25]
            max_v = 0
        channels = data.shape[0]
        for channel in range(channels):
            if isinstance(min_v, list):
                data[channel] = np.clip(data[channel], min_v[channel], max_v)
                data[channel] = (data[channel] - min_v[channel]) / (max_v - min_v[channel])
            else:
                data[channel] = np.clip(data[channel], min_v, max_v)
                data[channel] = (data[channel] - min_v) / (max_v - min_v)
        # data = (data - min_v) / (max_v - min_v)
        # data = data * 2 - 1
        return data

    def _generate_fake(self, data):
        # 生成s2的fake_mask
        # remain = int(self.s2_length * self.fake_ratio_s2 - data['s2_mask_real'].sum().item())
        ava = torch.where(data['s2_mask_real'] == 0)[0]
        # num_ones_tensor2 = torch.randint(0, remain + 1, (1,)).item()
        if self.norm:
            num = int(self.miss_list[self.index][1] * self.gap_day)
            num = num if num > 0 else self.fake_num
        else:
            num = self.fake_num
        indices_tensor2 = ava[torch.randperm(len(ava))[:num]]
        fake_mask = torch.zeros(self.s2_length, dtype=torch.float32)
        fake_mask[indices_tensor2] = 1
        data['s2_mask_fake'] = fake_mask
        data['s2_mask'] = data['s2_mask_real'] + data['s2_mask_fake']
        data['s2_fake'] = data['s2'].clone()
        data['ind_fake'] = data['ind'].clone()

        data['s2_fake'][data['s2_mask'].expand(data['s2'].shape[0], self.s2_length).bool()] = 0
        if 'time' in self.indicator_S1:
            data['ind_fake'][:-1, :][data['s2_mask'].expand(data['ind'].shape[0] - 1, self.s2_length).bool()] = 0
        else:
            data['ind_fake'][data['s2_mask'].expand(data['ind'].shape[0], self.s2_length).bool()] = 0

        data['NDVI_fake'] = data['ind_fake'][0, :].unsqueeze(0)

        # 生成s1的fake_mask
        remain = int(self.s1_length * self.fake_ratio_s1 - data['s1_mask_real'].sum().item())
        ava = torch.where(data['s1_mask_real'] == 0)[0]
        num_ones_tensor2 = torch.randint(0, remain + 1, (1,)).item()
        indices_tensor2 = ava[torch.randperm(len(ava))[:num_ones_tensor2]]
        fake_mask = torch.zeros(self.s1_length, dtype=torch.float32)
        fake_mask[indices_tensor2] = 1
        data['s1_mask_fake'] = fake_mask
        data['s1_mask'] = data['s1_mask_real'] + data['s1_mask_fake']
        data['s1_fake'] = data['s1'].clone()
        data['s1_fake'][data['s1_mask'].expand(data['s1'].shape[0], self.s1_length).bool()] = 0

        return data

    def _generate_ind_s2(self, data):
        results = {}
        for indicator in self.indicator_S2:
            if indicator in self.indicator_functions_S2:
                results[indicator] = data.apply(self.indicator_functions_S2[indicator], axis=1)
            elif indicator == 'time':
                results[indicator] = data.apply(calculate_time, axis=1)
            else:
                results[indicator] = data[indicator] / 10000

        tensor = torch.tensor(np.stack(list(results.values()), axis=1), dtype=torch.float32).transpose(0, 1)
        tensor = torch.where((tensor > 10) | (tensor < -10), torch.tensor(0.0), tensor)
        NDVI = torch.tensor(results['NDVI'].values, dtype=torch.float32).reshape(1, self.s2_length)
        NDVI = torch.where((NDVI > 10) | (NDVI < -10), torch.tensor(0.0), NDVI)
        return tensor, NDVI

    def _generate_ind_s1(self, data):
        results = {}
        for indicator in self.indicator_S1:
            if indicator in self.indicator_functions_S1:
                results[indicator] = data.apply(self.indicator_functions_S1[indicator], axis=1)
            elif indicator == 'VH':
                VH = torch.tensor(data['VH'].values, dtype=torch.float32)
                VH = np.clip(VH, -35, 0)
                VH = (VH + 35) / 35
                results[indicator] = VH
            elif indicator == 'VV':
                VV = torch.tensor(data['VV'].values, dtype=torch.float32)
                VV = np.clip(VV, -25, 0)
                VV = (VV + 25) / 25
                results[indicator] = VV
            elif indicator == 'time':
                results[indicator] = data.apply(calculate_time, axis=1)

        tensor = torch.tensor(np.stack(list(results.values()), axis=1), dtype=torch.float32).transpose(0, 1)
        tensor = torch.where((tensor > 10) | (tensor < -10), torch.tensor(0.0), tensor)
        return tensor

    def __getitem__(self, item):
        item = item % len(self)
        self.index = item
        obj = self.dataset[item]
        ret = {}
        ret['cropland'] = obj['s2']['cropland'].iloc[1]
        ret['s1'] = self._generate_ind_s1(obj['s1'])
        ret['ind'], ret['NDVI'] = self._generate_ind_s2(obj['s2'])
        s2 = obj['s2'].iloc[:, :12]
        ret['s2'] = self._normalize(torch.tensor(s2.values, dtype=torch.float32).transpose(0, 1), True)
        ret['s1_mask_real'] = torch.tensor(obj['s1']['target'].values, dtype=torch.float32)
        ret['s1'][ret['s1_mask_real'].expand(ret['s1'].shape[0], self.s1_length).bool()] = 0
        ret['s2_mask_real'] = torch.tensor(obj['s2']['QA60'].values, dtype=torch.float32)
        ret = self._generate_fake(ret)
        return ret  # C*N

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self):
            value = self.__getitem__(self.index)
            self.index += 1
            return value
        else:
            raise StopIteration


if __name__ == '__main__':
    a = Dataset()
    data = a.__getitem__(20)
