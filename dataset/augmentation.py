import random

import numpy as np
from .common import resize

rand = random.random
randi = random.randint
choice = random.choice
uniform = random.uniform

def _istuple(x): return isinstance(x, (tuple, list))


class Transform:
    def __init__(self, random_state=False):
        self.random_state = random_state
    def _transform(self, x):
        raise NotImplementedError
    def __call__(self, *args):
        if self.random_state: self._set_rand_param()
        assert len(args) > 0
        return self._transform(args[0]) if len(args) == 1 else tuple(map(self._transform, args))
    def _set_rand_param(self):
        raise NotImplementedError

class Compose:
    def __init__(self, *tf):
        assert len(tf) > 0
        self.tfs = tf
    def __call__(self, *x):
        if len(x) > 1:
            for tf in self.tfs: x = tf(*x)
        else:
            x = x[0]
            for tf in self.tfs: x = tf(x)
        return x
        
class Scale(Transform):
    def __init__(self, scale=(0.5,1.0)):
        if _istuple(scale):
            assert len(scale) == 2
            self.scale_range = scale #sorted(scale)
            self.scale = scale[0]
            super(Scale, self).__init__(random_state=True)
        else:
            super(Scale, self).__init__(random_state=False)
            self.scale = scale
    def _transform(self, x):
        # assert x.ndim == 3
        h, w = x.shape[:2]
        size = (int(h*self.scale), int(w*self.scale))
        if size == (h,w):
            return x
        return resize(x, size)
    def _set_rand_param(self):
        self.scale = uniform(*self.scale_range)
        
class DiscreteScale(Scale):
    def __init__(self, bins=(0.5, 0.75), keep_prob=0.5):
        super(DiscreteScale, self).__init__(scale=(min(bins), 1.0))
        self.bins = bins
        self.keep_prob = keep_prob
    def _set_rand_param(self):
        self.scale = 1.0 if rand()<self.keep_prob else choice(self.bins)


class Flip(Transform):
    # Flip or rotate
    _directions = ('ud', 'lr', 'no', '90', '180', '270')
    def __init__(self, direction=None):
        super(Flip, self).__init__(random_state=(direction is None))
        self.direction = direction
        if direction is not None: assert direction in self._directions
    def _transform(self, x):
        if self.direction == 'ud':
            ## Current torch version doesn't support negative stride of numpy arrays
            return np.ascontiguousarray(x[::-1])
        elif self.direction == 'lr':
            return np.ascontiguousarray(x[:,::-1])
        elif self.direction == 'no':
            return x
        elif self.direction == '90':
            # Clockwise
            return np.ascontiguousarray(self._T(x)[:,::-1])
        elif self.direction == '180':
            return np.ascontiguousarray(x[::-1,::-1])
        elif self.direction == '270':
            return np.ascontiguousarray(self._T(x)[::-1])
        else:
            raise ValueError('invalid flipping direction')

    def _set_rand_param(self):
        self.direction = choice(self._directions)

    @staticmethod
    def _T(x):
        return np.swapaxes(x, 0, 1)
        

class HorizontalFlip(Flip):
    _directions = ('lr', 'no')
    def __init__(self, flip=None):
        if flip is not None: flip = self._directions[~flip]
        super(HorizontalFlip, self).__init__(direction=flip)
    
class VerticalFlip(Flip):
    _directions = ('ud', 'no')
    def __init__(self, flip=None):
        if flip is not None: flip = self._directions[~flip]
        super(VerticalFlip, self).__init__(direction=flip)
        
class Crop(Transform):
    _inner_bounds = ('bl', 'br', 'tl', 'tr', 't', 'b', 'l', 'r')
    def __init__(self, crop_size=None, bounds=None):
        __no_bounds = (bounds is None)
        super(Crop, self).__init__(random_state=__no_bounds)
        if __no_bounds:
            assert crop_size is not None
        else:
            if not((_istuple(bounds) and len(bounds)==4) or (isinstance(bounds, str) and bounds in self._inner_bounds)):
                raise ValueError('invalid bounds')
        self.bounds = bounds
        self.crop_size = crop_size if _istuple(crop_size) else (crop_size, crop_size)
    def _transform(self, x):
        h, w = x.shape[:2]
        if self.bounds == 'bl':
            return x[h//2:,:w//2]
        elif self.bounds == 'br':
            return x[h//2:,w//2:]
        elif self.bounds == 'tl':
            return x[:h//2,:w//2]
        elif self.bounds == 'tr':
            return x[:h//2,w//2:]
        elif self.bounds == 't':
            return x[:h//2]
        elif self.bounds == 'b':
            return x[h//2:]
        elif self.bounds == 'l':
            return x[:,:w//2]
        elif self.bounds == 'r':
            return x[:,w//2:]
        elif len(self.bounds) == 2:
            assert self.crop_size < (h, w)
            ch, cw = self.crop_size
            cx, cy = int((w-cw)*self.bounds[0]), int((h-ch)*self.bounds[1])
            return x[cy:cy+ch, cx:cx+cw]
        else:
            left, top, right, lower = self.bounds
            return x[top:lower, left:right]
    def _set_rand_param(self):
        self.bounds = (rand(), rand())
   

class MSCrop(Crop):
    def __init__(self, scale, crop_size=None, bounds=None):
        super(MSCrop, self).__init__(crop_size, bounds)
        assert crop_size % scale == 0
        self.scale = scale  # Scale factor

    def __call__(self, lr, hr):
        if self.random_state:
            self._set_rand_param()
        lr_crop = self._transform(lr)
        self.crop_size = tuple(int(cs*self.scale) for cs in self.crop_size)
        hr_crop = self._transform(hr)
        self.crop_size = tuple(cs//self.scale for cs in self.crop_size)

        return lr_crop, hr_crop


def __test():
    a = np.arange(100).reshape((10,10)).astype(np.uint8)
    b = a.copy()
    tf = Compose(Flip(), Crop(3))
    c,d = tf(a,b)
    print(c)
    print(d)
    
    
if __name__ == '__main__':
    __test()
    