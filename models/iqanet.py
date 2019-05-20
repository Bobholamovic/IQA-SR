import math
import torch
import torch.nn as nn


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

        # Feature extraction layers
        self.fl1 = DoubleConv(3, 32)
        self.fl2 = DoubleConv(32, 64)
        self.fl3 = DoubleConv(64, 128)
        self.fl4 = DoubleConv(128, 256)
        self.fl5 = DoubleConv(256, 512)

        # Regression layers
        self.rl1 = nn.Linear(512, 128)
        self.rl2 = nn.Linear(128, 64)
        self.rl3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.5)

        self._initialize_weights()

    def extract_feature(self, x):
        """ Forward function for feature extraction of each branch of the siamese net """
        y = self.fl1(x)
        y = self.fl2(y)
        y = self.fl3(y)
        y = self.fl4(y)
        y = self.fl5(y)

        return y
        
    def forward(self, x):
        n_imgs, n_ptchs_per_img = x.shape[0:2]
        
        # Reshape
        x = x.view(-1, *x.shape[-3:])

        f = self.extract_feature(x)

        flatten = f.view(f.shape[0], -1)

        y = self.rl1(flatten)
        y = self.rl2(y)
        y = self.rl3(y)

        score = torch.mean(y)

        return score

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
                
