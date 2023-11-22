import torch
from torch import nn
import math


class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接
        """self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False)"""
        self.conv1 = nn.Conv2d(in_planes*2, in_planes // ratio, (1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        cat_out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv2(self.sigmoid(self.conv1(cat_out))))
        return out


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel=7):
        super(SpatialAttention, self).__init__()
        assert kernel in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel == 7 else 1
        self.conv = nn.Conv2d(2, 1, (kernel, kernel), padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 最大池化
        x = torch.cat([avg_out, max_out], dim=1)
        # 堆叠: (2, 1, 26, 26) & (2, 1, 26, 26)->(2, 2, 26, 26)
        conv = self.conv(x)
        return self.sigmoid(conv)


class can_block(nn.Module):
    """
    CAN: Complementary Attention Block Module
    paper: https://ieeexplore.ieee.org.
    """

    def __init__(self, channel, ratio=16, kernel=7):
        super(can_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio)
        self.spatialattention = SpatialAttention(kernel)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


if __name__ == '__main__':
    model = can_block(512)
    # print(model)
    inputs = torch.ones([2, 512, 26, 26])
    outputs = model(inputs)
    print(outputs)