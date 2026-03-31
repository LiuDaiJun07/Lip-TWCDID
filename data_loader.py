"""
data_loader.py  （修改版）
========================
在原始版本基础上，新增对以下两种格式的兼容支持：

  格式A（原始仓库）：
      cube_fft = np.load(path)['data'].item()['lip']['raw']
      shape: (frames, D, D, D)  D >= 4，用 [:, i,i,i] 取 range bin

  格式B（你的 data.npz）：
      cube_fft = np.load(path)['data'].item()['lip']['raw']
      shape: (frames, 1, 1, range_bins)  range_bins 在最后一维
      取法改为 [:, 0, 0, i]

  格式C（你的 data.mat，已转换为 npz）：
      cube_fft = np.load(path)['data'].item()['lip']['raw']
      shape: (frames, freq_bins)  已是谱图，跳过 STFT
"""

import csv
import os
import numpy as np
import torch
from scipy.signal import stft
from torch.utils.data import Dataset

source_domain = '15-0'
target_domain = '15+30'

dataset_dict = {
    source_domain: {
        "source": {
            "train_path":    f'data_path_csv/{source_domain}_source/train.csv',
            "val_path":      f'data_path_csv/{source_domain}_source/val.csv',
            "test_path":     f'data_path_csv/{source_domain}_source/test.csv',
            'database_path': f'data_path_csv/{source_domain}_source/database.csv'
        }
    },
    target_domain: {
        "target": {
            "train_path":    f'data_path_csv/{target_domain}_target/train.csv',
            "val_path":      f'data_path_csv/{target_domain}_target/val.csv',
            "test_path":     f'data_path_csv/{target_domain}_target/test.csv',
            'database_path': f'data_path_csv/{target_domain}_target/database.csv',
            'class_names':   ['U0', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9']
        }
    }
}


def get_rd_class_name(dataset_name):
    return dataset_dict[dataset_name]['target']['class_names']


# ── 格式检测与 range-bin 提取 ─────────────────────────────────────────────────

def _detect_format(cube_fft: np.ndarray) -> str:
    """
    根据 shape 自动判断数据格式：
      'A' : (frames, D, D, D)          D >= 4  原始仓库格式
      'B' : (frames, 1, 1, range_bins) range_bins >= 4  你的 npz 格式
      'C' : (frames, freq_bins)        已是谱图
    """
    if cube_fft.ndim == 4:
        if cube_fft.shape[1] >= 4 and cube_fft.shape[1] == cube_fft.shape[2] == cube_fft.shape[3]:
            return 'A'
        if cube_fft.shape[1] == 1 and cube_fft.shape[2] == 1:
            return 'B'
    if cube_fft.ndim == 2:
        return 'C'
    raise ValueError(f"未知数据格式，shape={cube_fft.shape}")


def _extract_range_bins(cube_fft: np.ndarray, index: int = 2):
    """
    提取 3 个相邻 range bin，返回 shape (frames, 1, 1, 3) 的 ndarray。
    兼容格式 A 和 B。
    """
    fmt = _detect_format(cube_fft)

    if fmt == 'A':
        # 原始仓库：立方体格式，索引 (i-1,i,i+1) 三个对角 bin
        c0 = cube_fft[:, index - 1, index - 1, index - 1]
        c1 = cube_fft[:, index,     index,     index]
        c2 = cube_fft[:, index + 1, index + 1, index + 1]
    elif fmt == 'B':
        # 你的 npz：range_bins 在最后一维
        c0 = cube_fft[:, 0, 0, index - 1]
        c1 = cube_fft[:, 0, 0, index]
        c2 = cube_fft[:, 0, 0, index + 1]
    else:
        raise ValueError("格式C（谱图）不需要调用此函数")

    # stack -> (frames, 3) -> (frames, 1, 1, 3)
    cube = np.stack([c0, c1, c2], axis=1)
    cube = np.expand_dims(cube, axis=(1, 2))   # (frames, 1, 1, 3)
    return cube


# ── Dataset ───────────────────────────────────────────────────────────────────

class RD_dataset(Dataset):

    def __init__(self, root, index_file, transform, strong_transform=None, argument=True):
        with open(index_file, 'r') as fh:
            reader = csv.reader(fh)
            img_paths, label_list = [], []
            for line in reader:
                img_path = os.path.join(root, line[0]) if root else line[0]
                img_paths.append(img_path)
                label_list.append(int(line[1]))

        self.data_path_list  = img_paths
        self.label_list      = label_list
        self.transform       = transform
        self.strong_transform = strong_transform
        self.argument        = argument
        self.range_bin       = 3          # 用于 index=2 时选 bin 1,2,3

    def __len__(self):
        return len(self.data_path_list)

    # ── 核心读取逻辑 ──────────────────────────────────────────────────────────

    def get_item(self, item):
        path     = self.data_path_list[item]
        raw_npz  = np.load(path, allow_pickle=True)
        cube_fft = raw_npz['data'].item()['lip']['raw']

        fmt = _detect_format(cube_fft)

        # ── 格式 C：data 已是谱图，直接转 tensor ─────────────────────────────
        if fmt == 'C':
            spec = cube_fft                          # (frames, freq_bins)
            spec = spec.real.astype(np.float32)      # 取实部或模
            spec = self.standardize(spec)
            x = torch.tensor(spec.copy()).float()
            y = self.label_list[item]
            if self.argument:
                # 复制出 4 路（格式统一，后续 train 代码拆分 x,x1,x2,x3）
                return (x, x, x, x, y)
            return (x, y)

        # ── 格式 A/B：原始 IQ，需要提取 range bin 并做 STFT ─────────────────
        index = 2
        cube_raw = _extract_range_bins(cube_fft, index)   # (frames, 1, 1, 3)

        if self.argument:
            # 4 路数据增强（复用同一帧，增强由 transform['IQ'] 实现）
            cubes = [self.transform['IQ'](cube_raw.copy()) for _ in range(4)]
            results = []
            for c in cubes:
                c -= np.average(c, 0)
                _, _, c_stft = stft(c, fs=100, nperseg=128, noverlap=124, nfft=128,
                                    padded=False, detrend=False,
                                    return_onesided=False, axis=0)
                c_stft = c_stft.transpose(-1, 0, 1, 2, 3)
                c_stft = np.fft.fftshift(c_stft, axes=1)
                c_stft = c_stft.squeeze()
                c_stft = self.standardize(c_stft, is_complex=False)
                results.append(torch.tensor(c_stft.copy()).float())
            y = self.label_list[item]
            return (*results, y)

        else:
            cube_raw -= np.average(cube_raw, 0)
            _, _, cube_stft = stft(cube_raw, fs=100, nperseg=128, noverlap=124, nfft=128,
                                   padded=False, detrend=False,
                                   return_onesided=False, axis=0)
            cube_stft = cube_stft.transpose(-1, 0, 1, 2, 3)
            cube_stft = np.fft.fftshift(cube_stft, axes=1)
            cube_stft = cube_stft.squeeze()
            cube_stft = self.standardize(cube_stft, is_complex=False)
            x = torch.tensor(cube_stft.copy()).float()
            y = self.label_list[item]
            return (x, y)

    def __getitem__(self, item):
        return self.get_item(item)

    def standardize(self, matrix, is_complex=False):
        real_matrix = matrix.real
        mean_real   = real_matrix.mean()
        std_real    = real_matrix.std() + 1e-8    # 防止除零
        real_matrix = (real_matrix - mean_real) / std_real
        if is_complex:
            imag_matrix = matrix.imag
            mean_imag   = imag_matrix.mean()
            std_imag    = imag_matrix.std() + 1e-8
            imag_matrix = (imag_matrix - mean_imag) / std_imag
            real_matrix = torch.from_numpy(real_matrix) if isinstance(real_matrix, np.ndarray) else real_matrix
            imag_matrix = torch.from_numpy(imag_matrix) if isinstance(imag_matrix, np.ndarray) else imag_matrix
            return torch.complex(real_matrix, imag_matrix)
        return real_matrix


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

def get_rd_dataset(root, source, dataset_name, transform, appli):
    if source:
        paths = dataset_dict[dataset_name]['source']
    else:
        paths = dataset_dict[dataset_name]['target']

    key_map = {'train': 'train_path', 'val': 'val_path',
               'test': 'test_path',   'database': 'database_path'}
    data_csv = paths[key_map.get(appli, 'train_path')]

    argument = appli not in ('val', 'test')
    return RD_dataset(root, index_file=data_csv, transform=transform, argument=argument)
