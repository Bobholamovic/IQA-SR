# Modified from https://github.com/thstkdgus35/EDSR-PyTorch

import torch
import torch.nn as nn
import math

__MM__ = 'MTEDSR'


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ConcatResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ConcatResBlock, self).__init__()
        m = []

        def _add_conv(chn_in, chn_out, with_act):
            m.append(conv(chn_in, chn_out, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if with_act:
                m.append(act)

        _add_conv(n_feats, n_feats, with_act=True)
        _add_conv(n_feats, n_feats//2, with_act=False)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x1, x2):
        res = self.body(torch.cat([x1, x2], 1)).mul(self.res_scale)
        res += x1

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(
        self, scale, n_resblocks=16, n_feats=64, res_scale=1.0,
        n_colors=3, conv=default_conv, blk_type='default'
    ):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        if blk_type == 'default':
            # define head module
            m_head = [conv(n_colors, n_feats, kernel_size)]
            # define body module
            m_body = [
                ResBlock(
                    conv, n_feats, kernel_size, act=act, res_scale=res_scale
                ) for _ in range(n_resblocks)
            ]
        elif blk_type == 'concat':
            m_head = [conv(n_colors, n_feats//2, kernel_size)]
            m_body = [
                ConcatResBlock(
                    conv, n_feats, kernel_size, act=act, res_scale=res_scale
                ) for _ in range(n_resblocks-1) 
            ]
            m_body.append(ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale))
        else:
            raise NotImplementedError
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 


class MTEDSR(nn.Module):
    def __init__(
        self, scale, n_resblocks=16, n_feats=64, res_scale=1.0,
        n_colors=3, conv=default_conv
    ):
        super(MTEDSR, self).__init__()

        self.n_resblocks = n_resblocks

        self.sr_branch = EDSR(scale, n_resblocks, n_feats, res_scale, n_colors, conv, 'concat')
        self.iqa_branch = EDSR(scale, n_resblocks, n_feats//2, res_scale, n_colors, conv)
        # Replace the tail of the IQA branch
        self.iqa_branch.tail = nn.Sequential(
            conv(n_feats//2, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_sr = x_sr = self.sr_branch.head(x)
        res_iqa = self.iqa_branch.head(x)

        for sr_blk, iqa_blk in zip(self.sr_branch.body[:-2], self.iqa_branch.body[:-2]):
            res_sr = sr_blk(res_sr, res_iqa)
            res_iqa = iqa_blk(res_iqa)

        res_sr = self.sr_branch.body[-2](torch.cat([res_sr, x_sr], 1))
        res_sr = self.sr_branch.body[-1](res_sr)

        for idx in range(-2,0):
            res_iqa = self.iqa_branch.body[idx](res_iqa)

        iqa_map = self.iqa_branch.tail(res_iqa)

        # Weighting
        res_sr *= iqa_map
        sr_result = self.sr_branch.tail(res_sr)

        return sr_result

    def iqa_forward(self, x):
        return self.iqa_branch(x)

