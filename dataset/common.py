import numpy as np
from skimage import transform, io, color
import torch

from constants import INTERP_ORDER, GAUSSIAN_BLUR


def mod_crop(x, N):
    h, w = x.shape[:2]
    nh = h - h % N
    nw = w - w % N
    return x[:nh, :nw]

def default_loader(pth):
    arr = np.array(io.imread(pth))
    # Only for gray and RGB images
    if arr.ndim == 2:
        # Convert to RGB
        arr = color.gray2rgb(arr)
    assert arr.ndim == 3 and arr.shape[-1] == 3
    # uint8 -> double
    return arr.astype(np.float)

def npz_loader(pth):
    pass

def to_tensor(x):
    return torch.from_numpy(np.moveaxis(x, -1, -3))

def to_float_tensor(x):
    return to_tensor(x).float()

def to_array(x):
    return np.moveaxis(np.asarray(x), -3, -1)

def resize(x, size, no_blur=False):
    # Seems that this operation fails to promise as good
    # performance as those done by MATLAB imresize, with
    # the PSNR decreased by roughly 1dB.
    #
    # Double-in-double-out
    blur = GAUSSIAN_BLUR and not no_blur
    return transform.resize(
        x.astype(np.float), 
        size,
        order=INTERP_ORDER, 
        anti_aliasing=blur, 
        preserve_range=True, 
        clip=True
    )
