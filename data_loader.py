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
            "train_path": f'data_path_csv/{source_domain}_source/train.csv',
            "val_path": f'data_path_csv/{source_domain}_source/val.csv',
            "test_path": f'data_path_csv/{source_domain}_source/test.csv',
            'database_path': f'data_path_csv/{source_domain}_source/database.csv'
        }
    },
    target_domain: {
        "target": {
            "train_path": f'data_path_csv/{target_domain}_target/train.csv',
            "val_path": f'data_path_csv/{target_domain}_target/val.csv',
            "test_path": f'data_path_csv/{target_domain}_target/test.csv',
            'database_path': f'data_path_csv/{target_domain}_target/database.csv',
            'class_names': ['U0', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9']
        }
    }
}

def get_rd_class_name(dataset_name):
    class_names = dataset_dict[dataset_name]['target']['class_names']
    return class_names

class RD_dataset(Dataset):

    def __init__(self, root, index_file, transform, strong_transform=None, argument=True):
        with open(index_file, 'r') as fh:
            reader = csv.reader(fh)
            img_paths = []
            label_list = []
            for line in reader:
                img_path = os.path.join(root, line[0])
                img_paths.append(img_path)
                label_list.append(int(line[1]))

        self.data_path_list = img_paths
        self.label_list = label_list
        self.transform = transform
        self.strong_transform = strong_transform

        self.IQ = False

        self.argument = argument
        # self.data_transforms = False
        self.range_bin = 3

    def __len__(self):
        return len(self.data_path_list)

    def get_item(self, item):

        path = self.data_path_list[item]
        cube_fft = np.load(path, allow_pickle=True)['data'].item()['lip']['raw']
        index = 2
        cube, cube1, cube2 = (cube_fft[:, index-1,index-1,index-1],cube_fft[:, index,index,index],
                              cube_fft[:, index+1,index+1,index+1])
        cube = np.stack([cube, cube1, cube2], axis=1)
        cube = np.expand_dims(cube, axis=(1, 2))

        if self.argument:
            cube,cube1,cube2,cube3 = self.transform['IQ'](cube)
            cube -= np.average(cube, 0)
            cube1 -= np.average(cube1, 0)
            cube2 -= np.average(cube2, 0)
            cube3 -= np.average(cube3, 0)

            f, t, cube = stft(cube, fs=100, nperseg=128, noverlap=124, nfft=128,
                             padded=False, detrend=False, return_onesided=False, axis=0)
            f1, t1, cube1 = stft(cube1, fs=100, nperseg=128, noverlap=124, nfft=128,
                             padded=False, detrend=False, return_onesided=False, axis=0)
            f2, t2, cube2 = stft(cube2, fs=100, nperseg=128, noverlap=124, nfft=128,
                             padded=False, detrend=False, return_onesided=False, axis=0)
            f3, t3, cube3 = stft(cube3, fs=100, nperseg=128, noverlap=124, nfft=128,
                             padded=False, detrend=False, return_onesided=False, axis=0)
            cube = cube.transpose(-1, 0, 1, 2, 3)
            cube = np.fft.fftshift(cube, axes=1)
            cube1 = cube1.transpose(-1, 0, 1, 2, 3)
            cube1 = np.fft.fftshift(cube1, axes=1)
            cube2 = cube2.transpose(-1, 0, 1, 2, 3)
            cube2 = np.fft.fftshift(cube2, axes=1)
            cube3 = cube3.transpose(-1, 0, 1, 2, 3)
            cube3 = np.fft.fftshift(cube3, axes=1)

            cube = cube.squeeze()
            cube1 = cube1.squeeze()
            cube2 = cube2.squeeze()
            cube3 = cube3.squeeze()

            #标准化
            cube = self.standardize(cube, is_complex=False)
            cube1 = self.standardize(cube1, is_complex=False)
            cube2 = self.standardize(cube2, is_complex=False)
            cube3 = self.standardize(cube3, is_complex=False)

            x = torch.tensor(cube.copy())
            x = x.type(torch.float32)
            x1 = torch.tensor(cube1.copy())
            x1 = x1.type(torch.float32)
            x2 = torch.tensor(cube2.copy())
            x2 = x2.type(torch.float32)
            x3 = torch.tensor(cube3.copy())
            x3 = x3.type(torch.float32)

        else:
            cube -= np.average(cube, 0)
            f, t, cube = stft(cube, fs=100, nperseg=128, noverlap=124, nfft=128,
                             padded=False, detrend=False, return_onesided=False, axis=0)
            cube = cube.transpose(-1, 0, 1, 2, 3)
            cube = np.fft.fftshift(cube, axes=1)
            cube = cube.squeeze()
            cube = self.standardize(cube, is_complex=False)
            # cube = normalize(cube)
            x = torch.tensor(cube)
            x = x.type(torch.float32)

        y = self.label_list[item]

        return (x,x1,x2,x3,y) if self.argument else (x,y)

    def __getitem__(self, item):
        return self.get_item(item)

    def standardize(self, matrix, is_complex=False):
        real_matrix = matrix.real
        if is_complex:
            imag_matrix = matrix.imag

        mean_real = real_matrix.mean()
        std_real = real_matrix.std()
        real_matrix = (real_matrix - mean_real) / std_real
        if is_complex:
            mean_imag = imag_matrix.mean()
            std_imag = imag_matrix.std()
            imag_matrix = (imag_matrix - mean_imag) / std_imag
            if isinstance(real_matrix, np.ndarray) and isinstance(imag_matrix, np.ndarray):
                real_matrix = torch.from_numpy(real_matrix)
                imag_matrix = torch.from_numpy(imag_matrix)

            standardized_matrix = torch.complex(real_matrix, imag_matrix)
        else:
            standardized_matrix = real_matrix
        return standardized_matrix

def get_rd_dataset(root, source, dataset_name, transform, appli):

    if source:
        if appli == 'train':
            data_csv = dataset_dict[dataset_name]['source']['train_path']
        elif appli == 'val':
            data_csv = dataset_dict[dataset_name]['source']['val_path']
        elif appli == 'test':
            data_csv = dataset_dict[dataset_name]['source']['test_path']
        else:
            data_csv = dataset_dict[dataset_name]['source']['database_path']
    else:
        if appli == 'train':
            data_csv = dataset_dict[dataset_name]['target']['train_path']
        elif appli == 'val':
            data_csv = dataset_dict[dataset_name]['target']['val_path']
        elif appli == 'test':
            data_csv = dataset_dict[dataset_name]['target']['test_path']
        else:
            data_csv = dataset_dict[dataset_name]['target']['database_path']

    if appli in ['val', 'test']:
        dataset = RD_dataset(root, index_file=data_csv, transform=transform, argument=False)
    else:
        dataset = RD_dataset(root, index_file=data_csv, transform=transform, argument=True)

    return dataset


