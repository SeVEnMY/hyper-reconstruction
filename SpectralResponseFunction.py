import torch
from torch import nn


class SpRes(nn.Module):
    def __init__(self,in_channels = 31):
        super(SpRes,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 31, out_channels = 3,bias=False,kernel_size = 1,stride= 1)
        # self.conv2 = nn.Conv2d(in_channels = 62, out_channels = 3,bias=False,kernel_size = 1,stride= 1)
        # self.LR = nn.LeakyReLU(negative_slope= 1e-2)
    def forward(self,x):
        x = self.conv1(x)
        x = nn.Tanh()(x)
        return x