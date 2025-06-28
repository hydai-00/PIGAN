import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


class metrics_generate():
    def __init__(self, name='No_one'):
        self.real_data = []
        self.fake_data = []
        self.cropland = []
        self.ori_real = []
        self.ori_fake = []
        self.ori_mask = []
        self.ori_point = []
        self.i = 0
        self.name = name

    def __call__(self, real, fake, mask, cropland, point=None):
        self.ori_real.append(real)
        self.ori_fake.append(fake)
        self.ori_mask.append(mask.astype(bool))
        self.real_data.append(real[mask.astype(bool)])
        self.fake_data.append(fake[mask.astype(bool)])
        self.cropland.append(cropland)
        if point is not None:
            self.ori_point.append(point)
        self.i = self.i + 1

    def calculate_metrics(self):
        # self.cropland = [arr[0] for arr in self.cropland]
        unique = set(self.cropland)
        mae_dict = {key: 0 for key in unique}
        rmse_dict = {key: 0 for key in unique}
        r2_dict = {key: 0 for key in unique}
        self.real = np.concatenate([arr.ravel() for arr in self.real_data])
        self.fake = np.concatenate([arr.ravel() for arr in self.fake_data])
        for name in unique:
            mask = [arr == name for arr in self.cropland]
            real = [self.real_data[i] for i in range(len(self.cropland)) if mask[i]]
            fake = [self.fake_data[i] for i in range(len(self.cropland)) if mask[i]]
            real = np.concatenate([arr.ravel() for arr in real])
            fake = np.concatenate([arr.ravel() for arr in fake])
            mae = mean_absolute_error(real, fake)
            rmse = np.sqrt(mean_squared_error(real, fake))
            r2 = r2_score(real, fake)
            mae_dict[name] = mae
            rmse_dict[name] = rmse
            r2_dict[name] = r2
        print('MAE:', mae_dict)
        print('RMSE:', rmse_dict)
        print('R2:', r2_dict)

        mae_dict = {key: 0 for key in unique}
        rmse_dict = {key: 0 for key in unique}
        r2_dict = {key: 0 for key in unique}


        mae = mean_absolute_error(self.real, self.fake)
        rmse = np.sqrt(mean_squared_error(self.real, self.fake))
        r2 = r2_score(self.real, self.fake)
        self.mae =mae
        self.rmse = rmse
        self.r2 = r2
        print('MAE:{} RMSE:{} R2:{}'.format(mae, rmse, r2))
        self.save_numpy()
        self.draw_r2()

    def save_numpy(self):
        with open(self.name + '.pkl', 'wb') as f:
            pickle.dump(
                {'real': self.ori_real, 'fake': self.ori_fake, 'cropland': self.cropland, 'mask': self.ori_mask}, f)

    def draw_r2(self):
        valid_mask = (self.real >= 0) & (self.real <= 1) & (self.fake >= 0) & (self.fake <= 1)
        self.real = self.real[valid_mask]
        self.fake = self.fake[valid_mask]
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 16, }
        x = self.real
        y = self.fake
        # 计算密度
        xy = np.vstack([x, y])  # 堆叠 x 和 y
        z = gaussian_kde(xy)(xy)  # 计算点密度

        # 根据密度排序
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        # 创建图形
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

        # 绘制密度散点图
        scatter = ax.scatter(
            x, y, marker='o', c=z, edgecolors='none', s=15, cmap='Spectral_r', label='LST'
        )
        plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.5, label="$y=x$")

        # 线性拟合
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(0, 1, 100)
        fit_line = slope * x_fit + intercept

        # 绘制拟合直线
        ax.plot(x_fit, fit_line, color="red", linestyle="-", linewidth=1.5, label="Fit line")
        text_str = f"y={slope:.3f}x+{intercept:.4f}"
        ax.text(
            0.05, 0.9, text_str,
            transform=ax.transAxes,
            fontsize=font['size'],
            fontweight='bold',
            fontname='Times New Roman',
            color="black",
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4', alpha=0.8)
        )

        metrics_text = (
            f"{' MAE'.ljust(5)}= {self.mae:.4f}\n"
            f"{'RMSE'.ljust(5)}= {self.rmse:.3f}\n"
            f"{'  R²'.ljust(5)}= {self.r2:.3f}"
        )
        ax.text(
            0.95, 0.05, metrics_text,
            transform=ax.transAxes,
            fontsize=font['size'],
            fontweight='bold',
            fontname='Times New Roman',
            color="black",
            ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4', alpha=0.8),
            parse_math=False  # 禁用数学解析
        )

        ax.tick_params(axis='x', labelsize=font['size'] - 4, labelcolor='black', direction='in', length=6)
        ax.tick_params(axis='y', labelsize=font['size'] - 4, labelcolor='black', direction='in', length=6)

        # 添加颜色条
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # 颜色条位置和大小
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label('Frequency', fontsize=font['size'], fontdict=font)
        cbar.ax.tick_params(labelsize=font['size'], labelcolor='black')

        # 添加图例和标签
        ax.set_xlabel("True Data", fontsize=font['size'], fontdict=font)
        ax.set_ylabel("Predicted Data", fontsize=font['size'], fontdict=font)
        # 设置坐标范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        output_path = "test.png"  # 指定保存路径
        plt.savefig(output_path, bbox_inches='tight', dpi=600)

        plt.show()
