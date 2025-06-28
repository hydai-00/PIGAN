import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from model.utils import generate_interp_functions, savgol_filter_pytorch


def interpolate_tensor(tensor):
    D, H, W = tensor.shape
    filled_tensor = tensor.clone()

    # 获取输入 tensor 的设备
    device = tensor.device

    for i in range(D):
        for j in range(H):
            missing = filled_tensor[i, j] == 0
            if missing.any():
                # 获取当前通道的有效值和坐标
                valid_indices = filled_tensor[i, j] != 0
                valid_values = filled_tensor[i, j][valid_indices]
                valid_coords = torch.arange(W, device=device)[valid_indices]

                # 插值过程
                for k in torch.where(missing)[0]:
                    # 寻找左边和右边最近的有效点
                    left_idx = (valid_coords[valid_coords < k].max() if (valid_coords < k).any() else None)
                    right_idx = (valid_coords[valid_coords > k].min() if (valid_coords > k).any() else None)

                    if left_idx is not None and right_idx is not None:
                        left_value = valid_values[valid_coords == left_idx]
                        right_value = valid_values[valid_coords == right_idx]
                        # 线性插值
                        filled_tensor[i, j, k] = (
                                left_value + (right_value - left_value) * ((k - left_idx) / (right_idx - left_idx))
                        )
                    elif left_idx is not None:
                        filled_tensor[i, j, k] = valid_values[valid_coords == left_idx]
                    elif right_idx is not None:
                        filled_tensor[i, j, k] = valid_values[valid_coords == right_idx]

    return filled_tensor.to(device)  # 确保输出在同一设备上


# all_eqs12

class block_model(nn.Module):
    def __init__(self, input_channels, input_len, out_len, individual, all):
        super(block_model, self).__init__()
        self.channels = input_channels
        self.in_channels = 2
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual
        self.all = all

        if self.all:
            AC = OrderedDict()
            AC['conv1'] = nn.Conv1d(self.channels, self.in_channels, 5, 1, 2)
            AC['act'] = nn.ReLU(inplace=True)
            AC['conv2'] = nn.Conv1d(self.in_channels, self.channels, 5, 1, 2)
            self.AC = nn.Sequential(AC)
        if self.individual:
            self.Linear_channel = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_channel.append(nn.Linear(self.input_len, self.out_len))
        else:
            self.Linear_channel = nn.Linear(self.input_len, self.out_len)
        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.all:
            x = self.AC(x)
        if self.individual:
            output = torch.zeros([x.size(0), x.size(1), self.out_len], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, i, :] = self.Linear_channel[i](x[:, i, :])
        else:
            output = self.Linear_channel(x)

        return output  # [Batch, Channel, Output length]


class Model(nn.Module):
    def __init__(self, s1_len=20, s2_len=48, s1_channels=2, s2_channels=12, act=True):
        super(Model, self).__init__()

        # s1:2,s2:12,ind:10

        self.s1_len = s1_len
        self.s2_len = s2_len
        self.s1_channels = s1_channels
        self.s2_channels = s2_channels

        self.mid_channels = 4
        self.individual = True
        self.act = act

        n1 = 1
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        down_in_s1 = [int(np.ceil(self.s1_len / filters[0])), int(np.ceil(self.s1_len / filters[1])),
                      int(np.ceil(self.s1_len / filters[2])),
                      int(np.ceil(self.s1_len / filters[3]))]
        down_in_s2 = [int(np.ceil(self.s2_len / filters[0])), int(np.ceil(self.s2_len / filters[1])),
                      int(np.ceil(self.s2_len / filters[2])),
                      int(np.ceil(self.s2_len / filters[3]))]
        down_out = [int(np.ceil(self.s2_len / filters[0])), int(np.ceil(self.s2_len / filters[1])),
                    int(np.ceil(self.s2_len / filters[2])),
                    int(np.ceil(self.s2_len / filters[3]))]

        S1B = OrderedDict()
        S1B['conv1'] = nn.Conv1d(self.s1_channels, self.mid_channels, 1, 1)
        S1B['act'] = nn.ReLU(inplace=True)
        self.S1B = nn.Sequential(S1B)

        S2B = OrderedDict()
        S2B['conv1'] = nn.Conv1d(self.s2_channels, self.mid_channels, 1, 1)
        S2B['act'] = nn.ReLU(inplace=True)
        self.S2B = nn.Sequential(S2B)

        # 最大池化层
        self.Maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.Maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.Maxpool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.down_block1_s1 = block_model(self.mid_channels, down_in_s1[0], down_out[0], self.individual, True)
        self.down_block2_s1 = block_model(self.mid_channels, down_in_s1[1], down_out[1], self.individual, True)
        self.down_block3_s1 = block_model(self.mid_channels, down_in_s1[2], down_out[2], self.individual, True)
        self.down_block4_s1 = block_model(self.mid_channels, down_in_s1[3], down_out[3], self.individual, True)

        self.down_block1_s2 = block_model(self.mid_channels, down_in_s2[0], down_out[0], self.individual, True)
        self.down_block2_s2 = block_model(self.mid_channels, down_in_s2[1], down_out[1], self.individual, True)
        self.down_block3_s2 = block_model(self.mid_channels, down_in_s2[2], down_out[2], self.individual, True)
        self.down_block4_s2 = block_model(self.mid_channels, down_in_s2[3], down_out[3], self.individual, True)

        # 右边特征融合层

        self.up_block3 = block_model(self.mid_channels * 2, down_out[2] + down_out[3], down_out[2],
                                     self.individual, True)

        self.up_block2 = block_model(self.mid_channels * 2, down_out[1] + down_out[2], down_out[1],
                                     self.individual, True)

        self.up_block1 = block_model(self.mid_channels * 2, down_out[0] + down_out[1], down_out[0],
                                     self.individual, True)
        self.tanh = nn.Tanh()

        self.out = nn.Conv1d(self.mid_channels * 2, 1, 1, 1)

        # self.linear_out = nn.Linear(self.out_len * 2, self.out_len)

    def forward(self, s1, s2):

        x1_1 = self.S1B(s1)
        e1_1 = self.down_block1_s1(x1_1)

        x2_1 = self.Maxpool1(x1_1)  # 48
        e2_1 = self.down_block2_s1(x2_1)

        x3_1 = self.Maxpool2(x2_1)  # 24
        e3_1 = self.down_block3_s1(x3_1)

        x4_1 = self.Maxpool3(x3_1)  # 12
        e4_1 = self.down_block4_s1(x4_1)

        # s2 process
        x1_2 = self.S2B(s2)
        e1_2 = self.down_block1_s2(x1_2)

        x2_2 = self.Maxpool1(x1_2)  # 48
        e2_2 = self.down_block2_s2(x2_2)

        x3_2 = self.Maxpool2(x2_2)  # 24
        e3_2 = self.down_block3_s2(x3_2)

        x4_2 = self.Maxpool3(x3_2)  # 12
        e4_2 = self.down_block4_s2(x4_2)

        e4 = torch.cat((e4_1, e4_2), dim=1)
        e3 = torch.cat((e3_1, e3_2), dim=1)
        e2 = torch.cat((e2_1, e2_2), dim=1)
        e1 = torch.cat((e1_1, e1_2), dim=1)

        d4 = torch.cat((e3, e4), dim=2)
        d4 = self.up_block3(d4)  # 24
        d3 = torch.cat((e2, d4), dim=2)
        d3 = self.up_block2(d3)  # 48
        d2 = torch.cat((e1, d3), dim=2)
        out = self.up_block1(d2)

        out = self.out(out)

        if self.act:
            out = self.tanh(out)
        ori = out
        out = s2[:, 0, :].unsqueeze(1) * s2[:, -1, :].unsqueeze(1) + ori * (1 - s2[:, -1, :].unsqueeze(1))

        return out, ori
