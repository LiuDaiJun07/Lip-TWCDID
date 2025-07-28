import numpy as np
import random

from time import time
from scipy.signal import resample
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

def get_train_transform(x):
    original = IQ_transforms(x, is_add_noise=False, is_rot=False, is_scaling=False)
    weak = IQ_transforms(x, is_add_noise=True, is_rot=False, is_scaling=True)
    strong1 = IQ_transforms(x, is_add_noise=True, is_rot=True, is_scaling=True)
    strong1 = time_slip_windows(strong1)
    strong2 = IQ_transforms(x, is_add_noise=True, is_rot=True, is_scaling=True)
    strong2 = time_slip_windows(strong2)

    return original, weak, strong1, strong2

def get_val_transform(x):
    return IQ_transforms(x, is_add_noise=False, is_rot=False, is_scaling=False)

def IQ_transforms(x, is_add_noise=True, is_rot=True, is_scaling=True):

    if is_rot:
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x_new = x * np.exp(1j * theta)
    else:
        x_new = x

    mu, sigma, N = 0, 0.005, x_new.shape[0]
    if is_add_noise:
        noise = np.random.normal(mu, sigma, N) + 1j * np.random.normal(mu, sigma, N)
        x_new += noise[:, np.newaxis, np.newaxis, np.newaxis]

    if is_scaling:
        scale = np.random.uniform(0.5, 2)
        x_new = scale * x_new

    return x_new

# def time_slip_windows(cube, window_length=2000, step_size=10):
def time_slip_windows(cube, window_length=335, step_size=5):

    num_windows = (cube.shape[0] - window_length) // step_size + 1
    if num_windows > 0:
        random_i = np.random.randint(0, num_windows)
        start_index = random_i * step_size
        end_index = start_index + window_length
        window_spec = cube[start_index:end_index, ...]
        prepend_data = cube[:start_index, ...]
        tail_data = cube[end_index:, ...]
        window_padded = np.concatenate((window_spec, prepend_data), axis=0)
        window_padded = np.concatenate((window_padded, tail_data), axis=0)

    return window_padded

if __name__ == '__main__':

    matrix = np.random.rand(3, 256, 251)



