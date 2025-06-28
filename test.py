from tqdm import tqdm
from model.GAN import TSFGAN
from dataset import Dataset
import matplotlib
import torch
from torch.utils.data import DataLoader
import os
from utils_test import metrics_generate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

matplotlib.use("Agg")
ind = [['CI', 'RVI', 'NGRDI', 'B1', 'CVI', 'B3'], ['VV', 'VH']]
train_data = Dataset(is_train=True, ind=ind)
batch_size = 1
start_epoch = 0
end_epoch = 2
s2_name = train_data.indicator_S2
s1_name = train_data.indicator_S1

message = 'test\n' + 's1:' + str(s1_name) + '  s2:' + str(s2_name)


def main():
    model = TSFGAN(30, 360, name='PIGAN', s1_name=s1_name, s2_name=s2_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mg = metrics_generate('PIGAN')
    model = model.to(device=device, dtype=torch.float)
    model.load_model('best')
    for epoch in range(start_epoch + 1, end_epoch):
        model.set_new_epoch(epoch)
        torch.cuda.empty_cache()
        val_data = Dataset(is_train=False, is_test=True, ind=ind)
        val_data_loader = DataLoader(dataset=val_data, num_workers=2, batch_size=batch_size, shuffle=False,
                                     pin_memory=True, drop_last=True)
        val(model, val_data_loader, mg)

    model.set_new_epoch(epoch)
    mg.calculate_metrics()


def val(model, val_data_loader, mg):
    model.eval()
    i = 0
    for input in tqdm(val_data_loader):
        model.set_input(input)
        model.test(i)
        mg(model.ndvi.cpu().detach().numpy(), model.fake_B_ori.cpu().detach().numpy(),
           (1 - model.mask_s2_fake).unsqueeze(1).cpu().detach().numpy(), input['cropland'].numpy()[0])
        i = i + 1


if __name__ == '__main__':
    main()
