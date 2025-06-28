import torch
import torch.nn as nn
import numpy as np
from scipy.signal import savgol_coeffs


def is_boolean(var):
    return var is True or var is False


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.MSELoss()
        self.real_label = torch.tensor(1.0)
        self.fake_label = torch.tensor(0.0)

    def get_target_tensor(self, prediction, target_is_real):
        device = prediction.device
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).to(device)

    def __call__(self, prediction, target_is_real):
        if is_boolean(target_is_real):
            target_is_real = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_is_real)
        return loss


class imask_loss_2(nn.Module):
    def __init__(self):
        super(imask_loss_2, self).__init__()
        self.l1_loss = nn.MSELoss()
        self.TV_loss = TV_loss()
        self.lam1 = 5
        self.lam2 = 0.002

    def forward(self, predict, predict_ori, target, mask_fake, mask_real):
        loss = self.lam1 * self.l1_loss(predict * (1.0 - mask_fake.unsqueeze(1)),
                                        target * (1.0 - mask_fake.unsqueeze(1))) / (
                   torch.mean(1.0 - mask_fake.unsqueeze(1))) \
               + self.l1_loss(predict_ori * mask_real.unsqueeze(1), target * mask_real.unsqueeze(1)) / (
                   torch.mean(mask_real.unsqueeze(1))) + self.lam2 * self.TV_loss(predict)
        return loss


class imask_loss(nn.Module):
    def __init__(self):
        super(imask_loss, self).__init__()
        self.l1_loss = nn.MSELoss()
        self.TV_loss = TV_loss()
        self.lam1 = 5
        self.lam2 = 0.002

    def forward(self, predict, target, mask_fake, mask_real):
        loss = self.lam1 * self.l1_loss(predict * (1.0 - mask_fake.unsqueeze(1)),
                                        target * (1.0 - mask_fake.unsqueeze(1))) / (
                   torch.mean(1.0 - mask_fake.unsqueeze(1))) \
               + self.l1_loss(predict * mask_real.unsqueeze(1), target * mask_real.unsqueeze(1)) / (
                   torch.mean(mask_real.unsqueeze(1))) + self.lam2 * self.TV_loss(predict)
        return loss


class mask_loss(nn.Module):
    def __init__(self):
        super(mask_loss, self).__init__()
        self.l1_loss = nn.MSELoss()
        self.TV_loss = TV_loss()
        self.lam1 = 5
        self.lam2 = 0.005

    def forward(self, predict, target, mask_fake, mask_real):
        loss = self.lam1 * self.l1_loss(predict * mask_fake.unsqueeze(1),
                                        target * mask_fake.unsqueeze(1)) / (
                   torch.mean(mask_fake.unsqueeze(1))) \
               + self.l1_loss(predict * (1.0 - mask_real.unsqueeze(1)), target * (1.0 - mask_real.unsqueeze(1))) / (
                   torch.mean(1.0 - mask_real.unsqueeze(1))) + self.lam2 * self.TV_loss(predict)
        return loss


class TV_loss(nn.Module):
    def __init__(self):
        super(TV_loss, self).__init__()

    def forward(self, x):
        diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        loss = diff.sum()

        return loss


import torch


def huber_loss_cuda(y_true, y_pred, delta=0.05):

    # 计算误差
    error = y_pred - y_true

    # 计算均方误差和线性损失
    squared_loss = 0.5 * error ** 2  # 对误差小的部分，均方误差
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)  # 对误差大的部分，线性损失

    # 创建掩码：当误差的绝对值小于delta时，使用均方误差；否则使用线性损失
    is_small_error = torch.abs(error) <= delta

    # 选择使用均方误差或线性损失
    loss = torch.where(is_small_error, squared_loss, linear_loss)

    # 返回损失的平均值
    return torch.mean(loss)

def savgol_filter_pytorch(data: torch.Tensor, window_length: int, polyorder: int) -> torch.Tensor:
    """
    Apply Savitzky-Golay filter along the time dimension (last dimension) of a 3D tensor using PyTorch.

    Args:
        data (torch.Tensor): The input tensor of shape (4, 1, 360).
        window_length (int): The length of the window used for filtering (must be odd).
        polyorder (int): The order of the polynomial used for filtering.

    Returns:
        torch.Tensor: The filtered tensor of the same shape as input.
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")

    kernel = savgol_coeffs(window_length, polyorder)
    kernel = torch.tensor(kernel, dtype=torch.float32, device=data.device)
    half_window = (window_length - 1) // 2

    # Apply the filter using convolution (1D convolution along the time dimension)
    # Padding to handle borders: padding size is half_window
    padded_data = torch.nn.functional.pad(data.squeeze(1), (half_window, half_window), mode='constant', value=0)

    # Perform convolution along the last dimension (time dimension)
    filtered_data = torch.nn.functional.conv1d(padded_data.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding=0)

    # The output shape is (4, 1, 360) as input, but we have to ensure the result has the correct dimensions
    return filtered_data


def interp1d_pytorch(x, y, new_x):
    """
    使用线性插值计算 `new_x` 对应的 `y` 值。

    Args:
    - x: 原始数据的自变量 (PyTorch tensor)，形状为 [n]。
    - y: 原始数据的因变量 (PyTorch tensor)，形状为 [n]。
    - new_x: 需要插值的自变量值 (PyTorch tensor)，形状为 [m]。

    Returns:
    - 插值后的结果 (PyTorch tensor)，形状为 [m]。
    """
    # 通过torch的函数对原始数据点进行排序
    sorted_x, indices = torch.sort(x)
    sorted_y = y[indices]

    # 计算插值
    diffs = sorted_x[1:] - sorted_x[:-1]  # 相邻x值的差
    slopes = (sorted_y[1:] - sorted_y[:-1]) / diffs  # 计算斜率

    # 对 new_x 进行插值
    new_x = new_x.unsqueeze(-1)  # 确保 new_x 是列向量
    left = torch.searchsorted(sorted_x, new_x) - 1  # 找到 new_x 对应的区间位置
    left = torch.clamp(left, 0, len(sorted_x) - 2)  # 确保不越界
    right = left + 1  # 右边的索引

    # 线性插值公式：y = y1 + slope * (x - x1)
    interp_y = sorted_y[left] + slopes[left] * (new_x - sorted_x[left])
    return interp_y.squeeze()


def generate_interp_functions(data, mask, x):
    """
    对 4 * 10 * 360 的数据进行逐通道插值，每个通道生成一个插值函数。

    Args:
    - data: 4 * 10 * 360 的张量 (PyTorch tensor)，表示原始 NDVI 数据。
    - mask: 4 * 10 * 360 的布尔掩码张量，用于标记有效数据点。
    - x: 原始数据的自变量，形状为 [360]，用于插值。

    Returns:
    - 插值后的数据，形状为 [4, 10, 360]。
    """
    interp_data = torch.zeros_like(data)

    # 对每个通道进行插值
    for c in range(data.shape[0]):  # 4 个通道
        for i in range(data.shape[1]):  # 10 个样本
            # 获取有效的 x 和 y 数据点（根据掩码）
            valid_x = x[mask[c, :].bool()]  # 获取有效的 x 值
            valid_y = data[c, i, mask[c, :].bool()]  # 获取有效的 y 值

            # 使用线性插值
            interp_func = interp1d_pytorch(valid_x, valid_y, x)
            interp_data[c, i, :] = interp_func

    return interp_data
