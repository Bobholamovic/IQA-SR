import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.2


class MyConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, **kwargs):
        super(MyConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel, padding=(kernel//2, kernel//2), **kwargs), 
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(ALPHA, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Conv3x3(MyConv):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv3x3, self).__init__(in_dim, out_dim, 3, **kwargs)

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DoubleConv, self).__init__()
        self.conv1 = Conv3x3(in_dim, out_dim, stride=(1,1))
        self.conv2 = Conv3x3(out_dim, out_dim, stride=(1,1))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y


class IQANet(nn.Module):
    def __init__(self, weighted=False, freeze=False):
        super(IQANet, self).__init__()

        self.weighted = weighted

        # Feature extraction layers
        self.fl1 = DoubleConv(3, 32)
        self.sc1 = MyConv(32, 32, 5, stride=(4,4))    # Strided convolution
        self.fl2 = DoubleConv(32, 64)
        self.sc2 = MyConv(64, 64, 5, stride=(4,4))
        self.fl3 = DoubleConv(64, 128)
        self.sc3 = MyConv(128, 128, 3, stride=(2,2))
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
        self.fl4 = DoubleConv(128, 256)
        self.sc4 = MyConv(256, 256, 3, stride=(2,2))
        self.fl5 = DoubleConv(256, 512)
        self.sc5 = MyConv(512, 512, 3, stride=(2,2))

        # Regression layers
        self.rl1 = nn.Linear(2048, 512)
        self.rl2 = nn.Linear(512, 1)

        if self.weighted:
            self.wl1 = nn.Linear(2048, 512)
            self.wl2 = nn.Linear(512, 1)

        self._initialize_weights()

    def extract_feature(self, x):
        x = self.sc1(self.fl1(x))
        x = self.sc2(self.fl2(x))
        x = self.sc3(self.fl3(x))
        x = self.sc4(self.fl4(x))
        x = self.sc5(self.fl5(x))

        return x
        
    def forward(self, x):
        # n_imgs, n_ptchs_per_img = x.shape[0:2]
        
        # # Reshape
        # x = x.view(-1, *x.shape[-3:])

        f = self.extract_feature(x)

        flatten = f.view(f.shape[0], -1)

        # y = F.dropout(F.leaky_relu(self.rl1(flatten), ALPHA), 0.5, self.training)
        y = F.leaky_relu(self.rl1(flatten), ALPHA, inplace=True)
        y = self.rl2(y)

        # if self.weighted:
        #     w = F.dropout(F.relu(self.wl1(flatten)), 0.5, self.training)
        #     w = self.wl2(w)
        #     w = F.relu(w) + 1e-8
        #     # Weighted averaging
        #     y_by_img = y.view(n_imgs, n_ptchs_per_img)
        #     w_by_img = w.view(n_imgs, n_ptchs_per_img)
        #     score = torch.sum(y_by_img*w_by_img, dim=1) / torch.sum(w_by_img, dim=1)
        # else:
        #     # Calculate average score for each image
        #     score = torch.mean(y.view(n_imgs, n_ptchs_per_img), dim=1)

        # score = torch.mean(y.view(n_imgs, n_ptchs_per_img), dim=1)
        # score = y.view(n_imgs, n_ptchs_per_img)
        # score = torch.sigmoid(score)

        score = torch.sigmoid(y)

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
                