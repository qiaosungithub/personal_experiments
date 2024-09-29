import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_c, out_c, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, c, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(in_c, c, stride)
        # self.bn1 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(c, c)
        # self.bn2 = nn.BatchNorm2d(c)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # here we may not use batch norm

        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


