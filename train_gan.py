from tqdm import tqdm
from model.GAN import TSFGAN
from dataset import Dataset
import matplotlib
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

matplotlib.use("Agg")
train_data = Dataset(is_train=True)
batch_size = 4
start_epoch = 0
end_epoch = 101
s2_name = train_data.indicator_S2
s1_name = train_data.indicator_S1


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TSFGAN(30, 360, name='PIGAN', s1_name=s1_name, s2_name=s2_name, device=device)
    model = model.to(device=device, dtype=torch.float32)
    train_data_loader = DataLoader(dataset=train_data, num_workers=2, batch_size=batch_size, shuffle=True,
                                   pin_memory=True, drop_last=True)
    for epoch in range(start_epoch + 1, end_epoch):
        model.set_new_epoch(epoch)
        torch.cuda.empty_cache()
        train(model, train_data_loader)
        val_data = Dataset(is_train=False)
        val_data_loader = DataLoader(dataset=val_data, num_workers=2, batch_size=batch_size, shuffle=False,
                                     pin_memory=True, drop_last=True)
        val(model, val_data_loader)
        if epoch % 10 == 0:
            model.save_model()

    model.set_new_epoch(epoch)


def train(model, train_data_loader):
    model.train()
    i = 0
    for input in tqdm(train_data_loader):
        model.set_input(input)
        model.optimize_paramters(i)
        i = i + 1


def val(model, val_data_loader):
    model.eval()
    i = 0
    for input in tqdm(val_data_loader):
        model.set_input(input)
        model.test(i)
        i = i + 1


if __name__ == '__main__':
    main()
