"""Zero-DCE++ Model Definition"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels):
        super(DSConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class DCENet_pp(nn.Module):
    """Zero-DCE++ Network for Low-Light Image Enhancement"""
    def __init__(self):
        super(DCENet_pp, self).__init__()
        self.e_conv1 = DSConv(3, 32)
        self.e_conv2 = DSConv(32, 32)
        self.e_conv3 = DSConv(32, 32)
        self.e_conv4 = DSConv(32, 32)
        self.e_conv5 = DSConv(64, 32)
        self.e_conv6 = DSConv(64, 32)
        self.e_conv7 = DSConv(64, 3)

    def forward(self, x):
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        for _ in range(8):
            x = x + x_r * (torch.pow(x, 2) - x)
        return x
