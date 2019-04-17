import torch.nn as nn


def relu():
    return nn.LeakyReLU(0.2)


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, bn=False, act=False, **extra):
        super().__init__()
        self._seq = nn.Sequential()
        self._seq.add_module('_conv', nn.Conv2d(
            in_ch, out_ch, kernel, 
            stride=1, padding=kernel//2, 
            **extra
        ))
        if bn:
            self._seq.add_module('_bn', nn.BatchNorm2d(out_ch))
        if act:
            self._seq.add_module('_act', relu())

    def forward(self, x):
        return self._seq(x)


class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, bn=False, act=False, **extra):
        super().__init__(in_ch, out_ch, 3, bn, act, **extra)