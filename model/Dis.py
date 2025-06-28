import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class AllOnesConv(nn.Module):
    def __init__(self):
        super(AllOnesConv, self).__init__()
        self.gap_day = 5
        self.kernel_size = int(360 / self.gap_day)
        # 初始化标准卷积层
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, stride=1, dilation=5, bias=False)

        # 将卷积核的权重全部设置为 1，并禁用参数更新
        with torch.no_grad():
            self.conv.weight.fill_(1.0/self.kernel_size)  # 全一初始化权重
        self.conv.weight.requires_grad = False  # 禁止权重更新

    def forward(self, x):
        x = self.conv(x)
        return x


class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        return x


class Model_D(nn.Module):
    def __init__(self, input_channels=1):
        super(Model_D, self).__init__()
        self.feat = 64

        self.mov_window = nn.Conv1d(input_channels, self.feat, 11, 1, 5)

        self.mlp = OrderedDict()
        self.mlp['in'] = nn.Linear(self.feat, self.feat * 4)
        self.mlp['act'] = nn.ReLU(True)
        self.mlp['out'] = nn.Linear(self.feat * 4, 1)
        self.mlp = nn.Sequential(self.mlp)

    def forward(self, x):
        x = self.mov_window(x)
        x = x.permute(0, 2, 1)
        x = self.mlp(x)
        x = x.permute(0, 2, 1)
        return x


class Model_D_dilate(nn.Module):
    def __init__(self, input_channels=1):
        super(Model_D_dilate, self).__init__()
        self.feat = 64
        self.gap_day = 5
        self.kernel_size = int(360 / self.gap_day)
        # self.mov_window = nn.Conv1d(input_channels, self.feat, 11, 1, 5)
        self.dilate_mov_window = nn.Conv1d(input_channels, out_channels=self.feat,
                                           kernel_size=self.kernel_size, dilation=self.gap_day)
        self.mlp = OrderedDict()
        self.mlp['in'] = nn.Linear(self.feat, self.feat * 4)
        self.mlp['act'] = nn.ReLU(True)
        self.mlp['out'] = nn.Linear(self.feat * 4, 1)
        self.mlp = nn.Sequential(self.mlp)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x = self.mov_window(x)
        x = self.dilate_mov_window(x)
        x = x.permute(0, 2, 1)
        x = self.mlp(x)
        x = x.permute(0, 2, 1)
        x = self.sig(x)
        return x
