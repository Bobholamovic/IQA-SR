import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pool(x):
    return F.max_pool2d(x, 2, stride=(2,2))


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=(1,1), padding=(1,1), bias=True), 
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MaxPool2x2(nn.MaxPool2d):
    def __init__(self):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0))


class DoubleConv(nn.Module):
    """
    Double convolution as a basic block for the net

    Actually this is from a VGG16 block
    """
    def __init__(self, in_dim, out_dim):
        super(DoubleConv, self).__init__()
        self.conv1 = Conv3x3(in_dim, out_dim)
        self.conv2 = Conv3x3(out_dim, out_dim)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y


class SingleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleConv, self).__init__()
        self.conv = Conv3x3(in_dim, out_dim)

    def forward(self, x):
        y = self.conv(x)
        return y


class IQANet(nn.Module):
    def __init__(self, weighted=False):
        super(IQANet, self).__init__()

        self.weighted = weighted

        # Feature extraction layers
        self.fl1 = DoubleConv(3, 32)
        self.fl2 = DoubleConv(32, 64)
        self.fl3 = DoubleConv(64, 128)
        self.fl4 = DoubleConv(128, 256)
        self.fl5 = DoubleConv(256, 512)

        # Regression layers
        self.rl1 = nn.Linear(512, 512)
        self.rl2 = nn.Linear(512, 1)

        if self.weighted:
            self.wl1 = nn.Linear(512, 512)
            self.wl2 = nn.Linear(512, 1)

        self._initialize_weights()

    def extract_feature(self, x):
        y = pool(self.fl1(x))
        y = pool(self.fl2(y))
        y = pool(self.fl3(y))
        y = pool(self.fl4(y))
        y = pool(self.fl5(y))

        return y
        
    def forward(self, x):
        n_imgs, n_ptchs_per_img = x.shape[0:2]
        
        # Reshape
        x = x.view(-1, *x.shape[-3:])

        f = self.extract_feature(x)

        flatten = f.view(f.shape[0], -1)

        y = F.dropout(F.relu(self.rl1(flatten)), 0.5, self.training)
        y = self.rl2(y)

        if self.weighted:
            w = F.dropout(F.relu(self.wl1(flatten)), 0.5, self.training)
            w = self.wl2(w)
            w = F.relu(w) + 1e-8
            # Weighted averaging
            y_by_img = y.view(n_imgs, n_ptchs_per_img)
            w_by_img = w.view(n_imgs, n_ptchs_per_img)
            score = torch.sum(y_by_img*w_by_img, dim=1) / torch.sum(w_by_img, dim=1)
        else:
            # Calculate average score for each image
            score = torch.mean(y.view(n_imgs, n_ptchs_per_img), dim=1)

        return score.squeeze()

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
                