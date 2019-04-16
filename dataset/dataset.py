import torch
import torch.utils.data
import numpy as np

from .common import default_loader, to_tensor, to_array
from .augmentation import Scale

from os.path import join, basename, exists
      

class SRDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, scale, subset='test', list_dir=None, transform=None):
        super().__init__()
        self.list_dir = data_dir if list_dir is None else list_dir
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
            name, self.to_tensor(hr_img)
        else:
            if self.transform is not None:
                hr_img = self.transform(hr_img)
            lr_img = self._make_lr(hr_img)
            lr_tensor, hr_tensor = self.to_tensor(lr_img), self.to_tensor(hr_img)

            if self.phase == 'train':
                return lr_tensor, hr_tensor
            elif self.phase == 'val':
                return name, lr_tensor, hr_tensor
            else:
                raise ValueError('invalid phase')

    def __len__(self):
        return self.num
        
    def _read_lists(self):   
        list_path = join(self.list_dir, '{}_list.txt'.format(self.subset))  
        assert exists(list_path)       
        self.image_list = self._read_single_list(list_path)
        self.num = len(self.image_list)

    def _make_lr(self, hr):
        return self.scaler(hr)

    def _fetch_hr(self, index):
        data_path = join(self.data_dir, self.image_list[index])
        return default_loader(data_path)   

    def normalize(self, x):
        # For inputs
        raise NotImplementedError

    def denormalize(self, x):
        # For outputs
        raise NotImplementedError

    @staticmethod
    def _read_single_list(pth):
        with open(pth, 'r') as lst:
            return [line.strip() for line in lst]

    def _get_name(self, index):
        return basename(self.image_list[index])

    def tensor_to_image(self, tensor):
        assert tensor.ndimension() == 3
        return self.to_array(tensor).astype(np.uint8)

    def array_to_image(self, arr):
        assert arr.ndim == 3
        return self.denormalize(arr).astype(np.uint8)

    def to_tensor(self, arr):
        return to_tensor(self.normalize(arr))
    
    def to_array(self, tensor):
        return self.denormalize(to_array(tensor))


class WaterlooDataset(SRDataset):
    def __init__(self, data_dir, phase, scale, subset='test', list_dir=None, transform=None):
        super().__init__(
            data_dir, phase, scale, subset=subset, 
            list_dir=list_dir, transform=transform
        )

    def normalize(self, x):
        return (x/255.0).astype(np.float32)

    def denormalize(self, x):
        return (x*255.0).astype(np.float32)
