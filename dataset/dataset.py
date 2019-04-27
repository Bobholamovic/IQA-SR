import torch
import torch.utils.data
import numpy as np

from .common import (default_loader, to_float_tensor as to_tensor, to_array, mod_crop)
from .augmentation import Scale

from os.path import (join, basename, exists, isdir)


class SRDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, scale, subset='test', list_dir='', transform=None):
        super().__init__()
        self.list_dir = data_dir if not list_dir else list_dir
        self.data_dir = data_dir
        assert phase in ('train', 'val', 'test')
        self.phase = phase
        self.transform = transform
        self.subset = subset if phase == 'val' else phase
        self.scale = scale
        self.scaler = Scale(1.0/scale)

        self._read_lists()

    def __getitem__(self, index):
        hr_img = self._fetch_hr(index)
        name = self._get_name(index)

        if self.phase == 'test':
            name, self.to_tensor_hr(hr_img)
        else:
            if self.transform is not None:
                hr_img = self.transform(hr_img)
            lr_img = self._make_lr(hr_img)
            lr_tensor = self.to_tensor_lr(lr_img)
            hr_tensor = self.to_tensor_hr(hr_img)

            if self.phase == 'train':
                return lr_tensor, hr_tensor
            elif self.phase == 'val':
                return name, lr_tensor, hr_tensor
            else:
                raise ValueError('invalid phase')

    def __len__(self):
        return self.num
        
    def _read_lists(self):   
        assert isdir(self.list_dir)
        list_path = join(self.list_dir, '{}_list.txt'.format(self.subset))  
        if exists(list_path):  
            self._fetch_hr = self._fetch_hr_from_list
            self.image_list = self._read_single_list(list_path)
        else:
            # Handle a directory
            self._fetch_hr = self._fetch_hr_from_folder
            from glob import glob
            from constants import IMAGE_POSTFIXES as IPF
            file_list = glob(join(self.list_dir, '*'))

            def isimg(fn):
                for ipf in IPF:
                    if fn.endswith(ipf): 
                        return True
                return False
            self.image_list = [f for f in file_list if isimg(f)]

        self.num = len(self.image_list)

    def _make_lr(self, hr):
        return self.scaler(hr)

    def _fetch_hr_from_list(self, index):
        data_path = join(self.data_dir, self.image_list[index])
        data = default_loader(data_path)   
        return mod_crop(data, self.scale)

    def _fetch_hr_from_folder(self, index):
        data = default_loader(self.image_list[index])
        return mod_crop(data, self.scale)

    @staticmethod
    def _read_single_list(pth):
        with open(pth, 'r') as lst:
            return [line.strip() for line in lst]

    def _get_name(self, index):
        return basename(self.image_list[index])
        # return self.image_list[index]

    def normalize(self, x, mode='lr'):
        raise NotImplementedError

    def denormalize(self, x, mode='hr'):
        raise NotImplementedError

    def tensor_to_image(self, tensor, mode='hr'):
        assert tensor.ndimension() == 3
        t2a = getattr(self, 'to_array_'+mode)
        return self._clamp(t2a(tensor)).astype(np.uint8)

    def array_to_image_lr(self, arr):
        assert arr.ndim == 3
        return self._clamp(self.denormalize(arr, 'lr')).astype(np.uint8)

    def array_to_image_hr(self, arr):
        assert arr.ndim == 3
        return self._clamp(self.denormalize(arr, 'hr')).astype(np.uint8)

    def to_tensor_lr(self, arr):
        return to_tensor(self.normalize(arr, 'lr'))

    def to_tensor_hr(self, arr):
        return to_tensor(self.normalize(arr, 'hr'))

    def to_array_lr(self, tensor):
        return self.denormalize(to_array(tensor), 'lr')

    def to_array_hr(self, tensor):
        return self.denormalize(to_array(tensor), 'hr')

    @staticmethod
    def _clamp(arr):
        return np.clip(arr, 0, 255)


class WaterlooDataset(SRDataset):
    def __init__(self, data_dir, phase, scale, subset='test', list_dir=None, transform=None):
        super().__init__(
            data_dir, phase, scale, subset=subset, 
            list_dir=list_dir, transform=transform
        )
        self._mean = np.asarray([124.46190829, 115.98740693, 104.40056142])
        self._std = np.asarray([63.25935824, 61.56368587, 63.60759291])

    def normalize(self, x, mode='lr'):
        return ((x-self._mean)/self._std).astype(np.float32)
        '''
        x_norm = x/255.0
        return x_norm if mode == 'lr' else 2*x_norm - 1.0
        '''

    def denormalize(self, x, mode='hr'):
        return (x*self._std+self._mean).astype(np.float32)
        '''
        if mode == 'hr':
            return (x+1.0)/2.0*255.0
        else:
            return x*255.0
        '''

