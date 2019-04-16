import numpy as np
from skimage import transform, io
import torch

def _get_last_mul_N(num, N):
    while(num % N != 0):
        num -= 1
    return num

def scale_to_N_mult(x, N):
    h, w = x.shape[:2]
    nh = _get_last_mul_N(h, N)
    nw = _get_last_mul_N(w, N)
    return resize(x, (nh, nw))

def default_loader(pth):
    arr = np.array(io.imread(pth))
    assert arr.ndim == 3 and arr.shape[-1] == 3
    return arr

def to_tensor(x):
    return torch.from_numpy(np.moveaxis(x, -1, -3))

def to_array(x):
    return np.moveaxis(np.array(x), -3, -1)

def resize(x, size):
    # Bilinear interpolation
    return transform.resize(x.astype(np.float), size, order=1).astype(x.dtype)

