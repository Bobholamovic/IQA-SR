# Modified from https://github.com/thstkdgus35/EDSR-PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__MM__ = 'MTEDSR'


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class SeparableConv2d(nn.Module):
    r"""
        Modified from https://github.com/wuhuikai/FastFCN/blob/master/encoding/nn/customize.py
    """
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SRJPU(nn.Module):
    r"""
        Joint Pyramid Upsampling
        Paper: FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation

        Modified from https://github.com/wuhuikai/FastFCN/blob/master/encoding/nn/customize.py
        Take away all norm layers and change the inputs and outputs

    """
    def __init__(self, conv, scale, n_feats, width=64):
        super(SRJPU, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                conv(n_feats, width, 3, bias=True),
                nn.ReLU(inplace=True) 
            )
        ])

        if (scale & (scale-1)) == 0:
            self.upsample = nn.Sequential(
                conv(n_feats, 4 * n_feats, 3, bias=True),
                nn.PixelShuffle(2)
            )
            # Stack weight-sharing upsampling modules on every scale
            self.convs.extend([
                nn.Sequential(
                    *[self.upsample]*(s+1), 
                    conv(n_feats, width, 3, bias=True),
                    nn.ReLU(inplace=True)
                ) for s in range(int(math.log(scale, 2)))
            ])
        elif scale == 3:
            self.convs.extend([
                nn.Sequential(
                    conv(n_feats, 9 * n_feats, 3, bias=True),
                    nn.PixelShuffle(3),
                    conv(n_feats, width, 3, bias=True),
                    nn.ReLU(inplace=True)
                )
            ])
        else:
            raise NotImplementedError

        n = len(self.convs)

        self.dilation1 = nn.Sequential(SeparableConv2d(n*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(n*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(n*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(n*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       nn.ReLU(inplace=True))

        self.out_conv = conv(4*width, n_feats, 3, bias=True)

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        _, _, h, w = feats[-1].size()

        for i in range(len(feats)-1):
            feats[i] = F.upsample(feats[i], (h, w), mode='bilinear', align_corners=True)

        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return self.out_conv(feat)


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

        _add_conv(n_feats+n_feats//2, n_feats, with_act=True)
        _add_conv(n_feats, n_feats, with_act=False)

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
        n_colors=3, conv=default_conv, blk_types=('R',)*16
    ):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = []
        for i in range(n_resblocks):
            if blk_types[i] == 'C':
                m_body.append(
                    ConcatResBlock(
                        conv, n_feats, kernel_size, act=act, res_scale=res_scale
                    )
                )
            else:
                m_body.append(
                    ResBlock(
                        conv, n_feats, kernel_size, act=act, res_scale=res_scale
                    ) 
                )

        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            # Upsampler(conv, scale, n_feats, act=False),
            SRJPU(conv, scale, n_feats),
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
        n_colors=3, conv=default_conv, 
        blk_types=('R',)*4+('C',)*8+('R',)*4
    ):
        super(MTEDSR, self).__init__()

        assert len(blk_types) == n_resblocks

        self.sr_branch = EDSR(
            scale, n_resblocks, n_feats, res_scale, n_colors, conv, 
            blk_types
        )
        self.iqa_branch = EDSR(
            scale, n_resblocks, n_feats//2, res_scale, n_colors, conv
        )
        # Replace the tail of the IQA branch
        self.iqa_branch.tail = nn.Sequential(
            conv(n_feats//2, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_sr = x_sr = self.sr_branch.head(x)
        res_iqa = self.iqa_branch.head(x)

        for sr_blk, iqa_blk in zip(self.sr_branch.body, self.iqa_branch.body):
            if isinstance(sr_blk, ConcatResBlock):
                res_sr = sr_blk(res_sr, res_iqa)
            elif isinstance(sr_blk, nn.Conv2d):
                res_sr = x_sr + sr_blk(res_sr)
            else:
                res_sr = sr_blk(res_sr)

            res_iqa = iqa_blk(res_iqa)

        iqa_map = self.iqa_branch.tail(res_iqa)

        # Weighting
        res_sr *= iqa_map
        sr_result = self.sr_branch.tail(res_sr)

        return sr_result, iqa_map

    def iqa_forward(self, x):
        return self.iqa_branch(x)

