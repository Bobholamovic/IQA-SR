import torch
import torch.utils.data
import numpy as np

from .common import (
    default_loader, npz_loader, is_img, is_npz, 
    to_float_tensor as to_tensor, to_array, mod_crop
)
from .augmentation import Scale
from constants import IMAGE_LIST_PATTERN, LR_LIST_PATTERN

from os.path import (join, basename, exists, isdir)


class SRDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir, phase, scale, subset='test', 
        list_dir='', transform=None, repeats=1
    ):
        super().__init__()
        self.list_dir = data_dir if not list_dir else list_dir
        self.data_dir = data_dir
        assert phase in ('train', 'val', 'test')
        self.phase = phase
        self.transform = transform
        self.subset = subset if phase == 'val' else phase
        self.scale = scale
        self.scaler = Scale(1.0/scale)
        self.repeats = repeats

        self._read_lists()

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        index %= self.num

        name = self._get_name(index)
        hr_img = self._fetch_hr(index)

        if self.phase == 'test':
            # This is special cuz hr labels can not
            # be accessed during the test phase.
            # Fetch the large images from {phase}_list.txt
            # or from a folder as the lr inputs. 
            return name, self.to_tensor_lr(hr_img)
        else:
            # hr_img = mod_crop(hr_img, self.scale)
            if self.lr_avai:
                lr_img = self._fetch_lr(index)
            else:
                lr_img = self._make_lr(hr_img)

            if self.transform is not None:
                lr_img, hr_img = self.transform(lr_img, hr_img)

            lr_tensor = self.to_tensor_lr(lr_img)
            hr_tensor = self.to_tensor_hr(hr_img)

            if self.phase == 'train':
                return lr_tensor, hr_tensor
            elif self.phase == 'val':
                return name, lr_tensor, hr_tensor
            else:
                raise ValueError('invalid phase')

    def __len__(self):
        return self.num * self.repeats
        
    def _read_lists(self):
        assert isdir(self.list_dir)
        self.lr_avai = False
        list_path = join(self.list_dir, IMAGE_LIST_PATTERN.format(ph=self.subset))  
        if exists(list_path):
            self.image_list = self._read_single_list(list_path)
            self.image_list = [join(self.data_dir, p) for p in self.image_list]
            lr_path = join(self.list_dir, LR_LIST_PATTERN.format(ph=self.subset))
            if exists(lr_path):
                self.lr_list = self._read_single_list(lr_path)
                self.lr_list = [join(self.data_dir, p) for p in self.lr_list]
                self.lr_avai = True
        else:
            # Handle a directory
            from glob import glob
            file_list = glob(join(self.data_dir, '*'))

            self.image_list = [f for f in file_list if is_img(f)]
            # assert len(self.image_list) > 0

        self.num = len(self.image_list)

    def load(self, pth):
        return default_loader(pth)
        # Block it for the moment as 
        # the if-else wastes time in most cases
        if is_img(pth):
            loader = default_loader
        elif is_npz(pth):
            loader = npz_loader
        else:
            raise ValueError('no applicable loader for this type')
        return loader(pth)

    def _make_lr(self, hr):
        return self.scaler(hr)

    def _fetch_lr(self, index):
        return self.load(self.lr_list[index])

    def _fetch_hr(self, index):
        return self.load(self.image_list[index])

    @staticmethod
    def _read_single_list(pth):
        return [line.strip() for line in open(pth, 'r')]

    def _get_name(self, index):
        return basename(self.image_list[index])
        # return self.image_list[index]

    @classmethod
    def normalize(cls, x, mode='lr'):
        raise NotImplementedError

    @classmethod
    def denormalize(cls, x, mode='hr'):
        raise NotImplementedError

    @classmethod
    def tensor_to_image(cls, tensor, mode='hr'):
        assert tensor.ndimension() == 3
        t2a = getattr(cls, 'to_array_'+mode)
        return cls._quantize(t2a(tensor))

    @classmethod
    def array_to_image_lr(cls, arr):
        assert arr.ndim == 3
        return cls._quantize(cls.denormalize(arr, 'lr'))

    @classmethod
    def array_to_image_hr(cls, arr):
        assert arr.ndim == 3
        return cls._quantize(cls.denormalize(arr, 'hr'))

    @classmethod
    def to_tensor_lr(cls, arr):
        return to_tensor(cls.normalize(arr, 'lr'))

    @classmethod
    def to_tensor_hr(cls, arr):
        return to_tensor(cls.normalize(arr, 'hr'))

    @classmethod
    def to_array_lr(cls, tensor):
        return cls.denormalize(to_array(tensor), 'lr')

    @classmethod
    def to_array_hr(cls, tensor):
        return cls.denormalize(to_array(tensor), 'hr')

    @staticmethod
    def _quantize(arr):
        return np.clip(arr, 0, 255).astype('uint8')


class DefaultDataset(SRDataset):
    _mean = np.asarray([124.46190829, 115.98740693, 104.40056142])
    _std = np.asarray([63.25935824, 61.56368587, 63.60759291])
    
    @classmethod
    def normalize(cls, x, mode='lr'):
        return x/255.0

    @classmethod
    def denormalize(cls, x, mode='hr'):
        return x*255.0


class WaterlooDataset(SRDataset):
    _mean = np.asarray([124.46190829, 115.98740693, 104.40056142])
    _std = np.asarray([63.25935824, 61.56368587, 63.60759291])
    
    @classmethod
    def normalize(cls, x, mode='lr'):
        x_norm = x/255.0
        return x_norm if mode == 'lr' else 2*x_norm - 1.0

    @classmethod
    def denormalize(cls, x, mode='hr'):
        if mode == 'hr':
            return (x+1.0)/2.0*255.0
        else:
            return x*255.0


class DIV2KDataset(SRDataset):
    _mean = 255.0 * np.asarray([0.4488, 0.4371, 0.4040])
    _std = np.asarray([1.0, 1.0, 1.0])#*127.5

    @classmethod
    def normalize(cls, x, mode='lr'):
        return (x-cls._mean)/cls._std

    @classmethod
    def denormalize(cls, x, mode='hr'):
        # Compatible for numpy arrays and 4-D torch tensors
        # This is done specially for IQA loss
        if isinstance(x, torch.Tensor):
            nc = cls._mean.size
            _mean = torch.from_numpy(cls._mean).type_as(x)
            _std = torch.from_numpy(cls._std).type_as(x)
            return x*_std.view(1,nc,1,1) + _mean.view(1,nc,1,1)
        else:
            return x*cls._std + cls._mean


def get_dataset(name):
    return globals().get(name+'Dataset', None)


def build_dataset(name, *opts, **kopts):
    dataset = get_dataset(name)
    if not dataset:
        raise ValueError('{} is not supported'.format(name))
    return dataset(*opts, **kopts)
