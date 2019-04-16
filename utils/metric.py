from skimage.measure import compare_psnr, compare_ssim
from skimage import color
import numpy as np

class Metric:
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        self.reset()

    def compute(self, *args):
        if self.callback is not None:
            return self.callback(*args) 
        else:
            if len(args) == 1:
                return args[0]
            else:
                raise NotImplementedError

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, *args, n=1):
        if not self._check_type(args):
            raise TypeError('invalid type')
        self.val = self.compute(*args)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def _check_type(self, args):
        return True


class DualMetric(Metric):
    def __init__(self, callback=None):
        super().__init__(callback)

    def _check_type(self, args):
        return super()._check_type(args) and len(args) == 2


class FRIQA(DualMetric):
    def __init__(self):
        super().__init__(callback=None)

    def _check_type(self, args):
        flag = True
        for obj in args:
            if not isinstance(obj, np.ndarray) or not self._check_image(obj):
                flag = False
                break
        return super()._check_type(args) and flag

    def _check_image(self, arr):
        # Only for RGB images
        flag_dim = arr.ndim == 3 and arr.shape[-1] == 3
        flag_type = arr.dtype == np.uint8
        return flag_dim & flag_type

    def compute(self, *rgb_pair):
        # Compute the metric only on Y channel
        y_pair = [color.rgb2gray(rgb) for rgb in rgb_pair]
        return self._compute(*y_pair)

    def _compute(self, true, test):
        raise NotImplementedError


class PSNR(FRIQA):
    def __init__(self):
        super().__init__()

    def _compute(self, true, test):
        return compare_psnr(true, test)


class SSIM(FRIQA):
    def __init__(self):
        super().__init__()

    def _compute(self, true, test):
        return compare_ssim(true, test)

