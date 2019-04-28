import math
from collections import namedtuple
import torch
import torch.nn as nn


def register(module, name, features):
    _forward = module.forward
    def hooked_forward(*args, **kwargs):
        out = _forward(*args, **kwargs)
        features.update({name: out})
        return out
    # Monkey patch here
    module.forward = hooked_forward
    return module


def register_du(module, name, features):
    _forward = module.forward
    def hooked_forward(*args, __buffer=[], **kwargs):
        out = _forward(*args, **kwargs)
        if len(__buffer) < 2:
            __buffer.append(out)
            if len(__buffer) == 2:
                features.update({name: tuple(__buffer)})
                __buffer.clear()
        return out
    # Patch it
    module.forward = hooked_forward
    return module


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=(1,1), padding=(1,1), bias=True), 
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MaxPool2x2(nn.Module):
    def __init__(self):
        super(MaxPool2x2, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=(2,2), padding=(0,0))
    
    def forward(self, x):
        return self.pool(x)

class DoubleConv(nn.Module):
    """
    Double convolution as a basic block for the net

    Actually this is from a VGG16 block
    """
    def __init__(self, in_dim, out_dim):
        super(DoubleConv, self).__init__()
        self.conv1 = Conv3x3(in_dim, out_dim)
        self.conv2 = Conv3x3(out_dim, out_dim)
        self.pool = MaxPool2x2()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.pool(y)
        return y

class SingleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleConv, self).__init__()
        self.conv = Conv3x3(in_dim, out_dim)
        self.pool = MaxPool2x2()

    def forward(self, x):
        y = self.conv(x)
        y = self.pool(y)
        return y


class IQANet(nn.Module):
    """
    The CNN model for full-reference image quality assessment
    
    Implements a siamese network at first and then there is regression
    """
    def __init__(self, weighted=False):
        super(IQANet, self).__init__()

        self.weighted = weighted
        self.features = dict()

        # Feature extraction layers
        self.fl1 = DoubleConv(3, 32)
        self.fl2 = DoubleConv(32, 64)
        self.fl3 = DoubleConv(64, 128)

        # Fusion layers
        self.cl1 = SingleConv(384, 128)
        self.cl2 = nn.Conv2d(128, 64, kernel_size=2)

        # Regression layers
        self.rl1 = nn.Linear(64, 32)
        self.rl2 = nn.Linear(32, 1)

        if self.weighted:
            self.wl1 = nn.Linear(64, 32)
            self.wl2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.5)

        self._initialize_weights()

        self.ret_tuple = namedtuple('rets', ['score', 'features'])

    def __setattr__(self, name, value):
        if isinstance(value, nn.Module):
            if name[:2] == 'fl':
                register_du(value, name, self.features)
            else:
                register(value, name, self.features)
        return super().__setattr__(name, value)

    def extract_feature(self, x):
        """ Forward function for feature extraction of each branch of the siamese net """
        y = self.fl1(x)
        y = self.fl2(y)
        y = self.fl3(y)

        return y
        
    def forward(self, x1, x2):
        """ x1 as distorted and x2 as reference """
        n_imgs, n_ptchs_per_img = x1.shape[0:2]
        
        # Reshape
        x1 = x1.view(-1,*x1.shape[-3:])
        x2 = x2.view(-1,*x2.shape[-3:])

        f1 = self.extract_feature(x1)
        f2 = self.extract_feature(x2)

        f_com = torch.cat([f2, f1, f2-f1], dim=1)  # Concat the features
        f_com = self.cl1(f_com)
        f_com = self.cl2(f_com)

        flatten = f_com.view(f_com.shape[0], -1)

        y = self.rl1(flatten)
        y = self.rl2(y)

        if self.weighted:
            w = self.wl1(flatten)
            w = self.wl2(w)
            w = torch.nn.functional.relu(w) + 1e-8
            # Weighted average
            y_by_img = y.view(n_imgs, n_ptchs_per_img)
            w_by_img = w.view(n_imgs, n_ptchs_per_img)
            score = torch.sum(y_by_img*w_by_img, dim=1) / torch.sum(w_by_img, dim=1)
        else:
            # Calculate average score for each image
            score = torch.mean(y.view(n_imgs, n_ptchs_per_img), dim=1)

        return self.ret_tuple(score.squeeze(), self.features)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            else:
                pass
