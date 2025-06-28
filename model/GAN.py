import os
import time

import numpy as np
import torch
from matplotlib import pyplot
from torch import nn

from model.Dis import Model_D_dilate, AllOnesConv
from model.utils import GANLoss, imask_loss_2, huber_loss_cuda
from utils_py import AverageMeter, initialize_logger, save_checkpoint
from model.Gen import Model


class TSFGAN(nn.Module):
    def __init__(self, s1_len, s2_len, isTrain=True, name='init', message=None,s1_name=[],s2_name=[],device='cuda:0'):
        super(TSFGAN, self).__init__()
        self.s1_len = s1_len
        self.s2_len = s2_len
        self.sar_channels = len(s1_name)
        self.optical_channels = len(s2_name)
        self.isTrain = isTrain
        self.test_target = True

        self.netG = Model(self.s1_len, self.s2_len, s1_channels=self.sar_channels,
                          s2_channels=self.optical_channels + 1)
        self.lr = 1e-3
        # self.init_model(self.netG)
        self.loss_name = ['G_GAN', 'G_L2', 'D_real', 'D_fake']
        self.device = device
        self.epoch = -1
        self.model_path = os.path.join('./models', name)
        self.mask_hint_ratio = 0.8
        self.pro_conv = AllOnesConv()
        self.cl_loss = huber_loss_cuda
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.message = message

        self.test_loss_record = AverageMeter()
        self.test_loss_record_2 = AverageMeter()
        self.G_GAN_record = AverageMeter()
        self.G_L2_record = AverageMeter()
        self.D_real_record = AverageMeter()
        self.D_fake_record = AverageMeter()
        self.mix_loss_record = AverageMeter()
        self.D_all_record = AverageMeter()
        self.consist_loss_record = AverageMeter()

        self.loss_G_GAN = 0
        self.loss_G_L2 = 0
        self.loss_D_real = 0
        self.loss_D_fake = 0
        self.loss_mix = 0
        self.loss_D = 0
        self.consist_loss = 0
        self.best_test = 10000

        if self.isTrain:
            self.criterion_train = imask_loss_2()
            self.L2_loss = nn.MSELoss()
            self.criterionGAN = GANLoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
            log_dir = os.path.join(self.model_path, 'train.log')
            self.logger = initialize_logger(log_dir)
        self.netD = Model_D_dilate(input_channels=1)
        # self.init_model(self.netD)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def set_input(self, input):
        self.input = input
        self.real_s1 = input['s1'].to(self.device)
        self.real_s1_syn = input['s1_fake'].to(self.device)
        self.mask_s1 = (1.0 - input['s1_mask']).to(self.device)
        self.mask_s1_fake = (1.0 - input['s1_mask_fake']).to(self.device)
        self.mask_s1_real = (1.0 - input['s1_mask_real']).to(self.device)

        self.real_s2 = input['s2'].to(self.device)
        self.real_s2_syn = input['s2_fake'].to(self.device)
        self.mask_s2 = (1.0 - input['s2_mask']).to(self.device)
        # self.mask_s2 有数据缺失就是0，输入网络包含的数据点为1
        self.mask_s2_fake = (1.0 - input['s2_mask_fake']).to(self.device)
        # self.mask_s2_fake 生成的伪数据是0，其他数据为1
        self.mask_s2_real = (1.0 - input['s2_mask_real']).to(self.device)
        # self.mask_s2_real 本来就缺失的数据是0，其他数据为1

        self.ind = input['ind'].to(self.device)
        self.ind_syn = input['ind_fake'].to(self.device)
        self.ndvi = input['NDVI'].to(self.device)
        self.ndvi_syn = input['NDVI_fake'].to(self.device)

    def set_new_epoch(self, epoch):
        if epoch == 1 and self.message:
            self.logger.info(self.message)
        if hasattr(self, 'start_time'):
            self.end_time = time.time()
            self.record()
        self.epoch = epoch
        self.start_time = time.time()

    def test(self, iter=-1):
        self.fake_B, self.fake_B_ori = self.netG(self.real_s1,
                                                 torch.concat([self.ind_syn, self.mask_s2.unsqueeze(1)], dim=1))
        self.loss = self.L2_loss(self.fake_B_ori * self.mask_s2_real.unsqueeze(1),
                                 self.ndvi * self.mask_s2_real.unsqueeze(1)) / torch.mean(self.mask_s2_real)
        self.loss_2 = self.L2_loss(self.fake_B * (1 - self.mask_s2_fake.unsqueeze(1)),
                                   self.ndvi * (1 - self.mask_s2_fake.unsqueeze(1))) / torch.mean(
            (1 - self.mask_s2_fake))

        self.test_loss_record.update(self.loss.item())
        self.test_loss_record_2.update(self.loss_2.item())
        self.visual(iter)

    def forward(self):
        self.fake_B, self.fake_B_ori = self.netG(self.real_s1,
                                                torch.concat([self.ind_syn, self.mask_s2.unsqueeze(1)], dim=1))

    def backward_D(self):
        D_in = self.fake_B
        pred_fake = self.netD(D_in.detach())
        self.loss_D = self.criterionGAN(pred_fake, self.pro_mask.detach())
        self.D_all_record.update(self.loss_D.item())
        self.loss_D.backward()

    def backward_G(self):
        self.loss_G_L2 = self.criterion_train(self.fake_B, self.fake_B_ori, self.ndvi, self.mask_s2_fake,
                                              self.mask_s2_real)
        self.G_L2_record.update(self.loss_G_L2.item())
        self.full_B, self.full_B_ori = self.netG(self.real_s1,
                                                torch.concat([self.ind, self.mask_s2_real.unsqueeze(1)], dim=1))
        self.consist_loss = self.cl_loss(self.fake_B_ori, self.full_B_ori.detach())
        self.consist_loss_record.update(self.consist_loss.item())
        mean_value = self.pro_mask.mean()
        self.pro_gt = torch.where(self.pro_mask < mean_value, mean_value, self.pro_mask)
        D_in = self.fake_B
        self.pred_fake = self.netD(D_in)
        self.loss_G_GAN = self.criterionGAN(self.pred_fake, self.pro_gt.detach())
        self.G_GAN_record.update(self.loss_G_GAN.item())
        self.loss_G = 0.01 * self.loss_G_GAN + self.loss_G_L2 + self.consist_loss
        self.loss_G.backward()

    def optimize_paramters(self, i):
        self.forward()
        self.pro_mask = self.pro_conv(self.mask_s2.unsqueeze(1))
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        if i % 100 == 0:
            print(
                'Epoch[{}]|i:{}|G_GAN:{:.4f}|G_L2:{:.4f}|C_L: {:.4f}|D_real:{:.4f}|D_fake:{:.4f}|D_mix:{:.4f}|D_all:{:.4f}'.format(
                    self.epoch, i, self.loss_G_GAN, self.loss_G_L2, self.consist_loss, self.loss_D_real,
                    self.loss_D_fake, self.loss_mix,
                    self.loss_D))
        if i % 300 == 0:
            self.draw_D(i)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_model(self, name=None):
        if name is None:
            name = 'TSFGAN'
        save_checkpoint(self.model_path, self.epoch, self.netD, self.optimizer_D, name=name + '_D')
        save_checkpoint(self.model_path, self.epoch, self.netG, self.optimizer_G, name=name + '_G')

    def load_model(self, name):
        path_G = os.path.join(self.model_path, name + '_G.pkl')
        checkpoint = torch.load(path_G)
        self.netG.load_state_dict(checkpoint['state_dict'])
        if self.isTrain:
            self.optimizer_G.load_state_dict(checkpoint['optimizer'])
            path_D = os.path.join(self.model_path, name + '_D.pkl')
            checkpoint = torch.load(path_D)
            self.netD.load_state_dict(checkpoint['state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer'])

    def record(self):
        time = self.end_time - self.start_time
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']
        save_str = 'Epoch [{}]|Time:{:.4f}|learning rate : {:.9f}|G_GAN: {:.4f}|G_L2: {:.4f}|C_L: {:.4f}|D_all: {:.4f}|test_loss: {:.6f}|test_loss_fake: {:.6f}|'.format(
            self.epoch, time, lr, self.G_GAN_record.avg, self.G_L2_record.avg, self.consist_loss_record.avg,
            self.D_all_record.avg,
            self.test_loss_record.avg, self.test_loss_record_2.avg)
        print(save_str)
        self.logger.info(save_str)
        if self.test_loss_record_2.avg + self.test_loss_record.avg < self.best_test:
            self.best_test = self.test_loss_record_2.avg + self.test_loss_record.avg
            self.save_model('best')
            save_str = '________________save best model________________________'
            print(save_str)
            self.logger.info(save_str)

        self.G_GAN_record.reset()
        self.G_L2_record.reset()
        self.D_real_record.reset()
        self.D_fake_record.reset()
        self.mix_loss_record.reset()
        self.D_all_record.reset()
        self.test_loss_record.reset()
        self.test_loss_record_2.reset()
        self.consist_loss_record.reset()

    def visual(self, iter):
        if iter % 10 == 0:
            path = os.path.join(self.model_path, 'vis')
            path = os.path.join(path, 'epoch{}'.format(self.epoch))
            if not os.path.exists(path):
                os.makedirs(path)
            pyplot.figure()
            pyplot.plot(self.fake_B[0, 0, :].cpu().detach().numpy(), label='predict', color='blue', alpha=0.2)
            pyplot.plot(self.fake_B_ori[0, 0, :].cpu().detach().numpy(), label='predict_ori', color='cyan', alpha=0.6)
            ndvi_data = self.ndvi[0, 0, :].cpu().detach().numpy()
            mask_s2 = (1.0 - self.mask_s2[0, :]).cpu().detach().numpy()
            mask_s2_fake = (1.0 - self.mask_s2_fake[0, :]).cpu().detach().numpy()
            real_mask = np.nonzero(1.0 - mask_s2)  # 获取非零值的索引
            pyplot.scatter(real_mask[0], ndvi_data[real_mask[0]], label='give', color='green', s=5)
            fake_mask = np.nonzero(mask_s2_fake)
            pyplot.scatter(fake_mask[0], ndvi_data[fake_mask[0]], label='unsupport', color='yellow', s=5)
            pyplot.grid(True, which='both')
            pyplot.legend()
            pyplot.axhline(y=0, color='k')
            pyplot.savefig(os.path.join(path, "TSFGAN_{}.png".format(iter)))
            pyplot.close()

    def draw_D(self, iter):
        path = os.path.join(self.model_path, 'D_vis')
        path = os.path.join(path, 'epoch{}'.format(self.epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        pyplot.figure()
        pyplot.plot(self.fake_B[0, 0, :].cpu().detach().numpy(), label='predict', color='blue', alpha=0.2)
        pyplot.plot(self.full_B[0, 0, :].cpu().detach().numpy(), label='full', color='purple', alpha=0.2)
        pyplot.plot(self.fake_B_ori[0, 0, :].cpu().detach().numpy(), label='predict_ori', color='cyan', alpha=0.6)
        pyplot.plot(self.full_B_ori[0, 0, :].cpu().detach().numpy(), label='full_ori', color='violet', alpha=0.6)
        ndvi_data = self.ndvi[0, 0, :].cpu().detach().numpy()
        mask_s2 = (1.0 - self.mask_s2[0, :]).cpu().detach().numpy()
        mask_s2_fake = (1.0 - self.mask_s2_fake[0, :]).cpu().detach().numpy()
        real_mask = np.nonzero(1.0 - mask_s2)  # 获取非零值的索引
        pyplot.scatter(real_mask[0], ndvi_data[real_mask[0]], label='give', color='green', s=5)
        fake_mask = np.nonzero(mask_s2_fake)
        pyplot.scatter(fake_mask[0], ndvi_data[fake_mask[0]], label='unsupport', color='yellow', s=5)
        pyplot.grid(True, which='both')
        pyplot.legend()
        pyplot.axhline(y=0, color='k')
        pyplot.savefig(os.path.join(path, "{}_predict.png".format(iter)))
        pyplot.close()
        pyplot.figure()
        y = np.arange(1, self.pred_fake[0].size()[-1] + 1)
        pyplot.scatter(y, self.pred_fake[0].cpu().detach().numpy(), label='predict', color='blue',
                        s=10)
        # pyplot.scatter(y, self.pro_hm[0].cpu().detach().numpy(), label='HM', color='yellow', s=10)
        pyplot.scatter(y, self.pro_gt[0].cpu().detach().numpy(), label='learn', color='red', s=10)
        pyplot.scatter(y, self.pro_mask[0].cpu().detach().numpy(), label='mask', color='green', s=10)
        pyplot.grid(True, which='both')
        pyplot.legend()
        pyplot.axhline(y=0, color='k')
        pyplot.savefig(os.path.join(path, "{}_dis.png".format(iter)))
        pyplot.close()

    def init_model(self, m):
        class_name = m.__class__.__name__
        if class_name.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()  # 偏置初始化为0

        elif class_name.find('Conv1d') != -1:
            # Xavier 初始化（均匀分布）
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()  # 偏置初始化为0

