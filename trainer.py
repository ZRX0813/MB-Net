import os.path
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch as t
from torch import optim
from MB_Net import MB_Net as Ours


class trainer(object):
    def __init__(self, arg, device, train_data, val_data):
        self.device = device
        self.eopch = arg.epochs
        self.batch_size = arg.batch_size
        self.train_loader = train_data
        self.val_loader = val_data
        self.save_path = arg.save_path
        self.model_name = arg.model

    def train(self):
        if not os.path.exists(os.path.join(self.save_path, self.model_name)):
            os.mkdir(os.path.join(self.save_path, self.model_name))
            os.mkdir(os.path.join(self.save_path, self.model_name, 'or_weights'))
        Net = Ours(3, 2).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(Net.parameters(), lr=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.eopch)
        train_loss_list = []
        val_loss_list = []
        for epoch in range(self.eopch):
            net = Net.train()
            train_loss = 0.0
            tbar_train = tqdm(self.train_loader)
            for i, sample in enumerate(tbar_train):
                channel_1_data = Variable(sample['channel_1'].to(self.device))
                channel_2_data = Variable(sample['channel_2'].to(self.device))
                channel_3_data = Variable(sample['channel_3'].to(self.device))
                img_label = Variable(sample['label'].to(self.device))
                out = net(channel_1_data, channel_2_data, channel_3_data).float()
                img_label = F.one_hot(img_label.squeeze(1), 2).permute(0, 3, 1, 2).float()
                loss = criterion(out, img_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                tbar_train.set_description(f'Train Epoch [{epoch + 1}/{self.eopch}]')
            train_loss_list.append(train_loss / len(self.train_loader))
            net = Net.eval()
            val_loss = 0.0
            tbar_val = tqdm(self.val_loader)
            for i, sample in enumerate(tbar_val):
                channel_1_data = Variable(sample['channel_1'].to(self.device))
                channel_2_data = Variable(sample['channel_2'].to(self.device))
                channel_3_data = Variable(sample['channel_3'].to(self.device))
                img_label = Variable(sample['label'].to(self.device))
                with torch.no_grad():
                    out = net(channel_1_data, channel_2_data, channel_3_data).float()
                out = F.log_softmax(out, dim=1)
                img_label = F.one_hot(img_label.squeeze_(1), 2).permute(0, 3, 1, 2).float()
                loss = criterion(out, img_label)
                val_loss += loss.item()
                tbar_val.set_description(f'VAL Epoch [{epoch + 1}/{self.eopch}]')
            val_loss_avg = val_loss / len(self.val_loader)
            scheduler.step()
            if epoch == 0:
                val_loss_list.append(val_loss_avg)
                t.save(net.state_dict(),
                       os.path.join(os.path.join(self.save_path, self.model_name), 'or_weights/best.pth'))
            if epoch != 0:
                if val_loss_avg < min(val_loss_list):
                    t.save(net.state_dict(),
                           os.path.join(os.path.join(self.save_path, self.model_name), 'or_weights/best.pth'))
                val_loss_list.append(val_loss_avg)

