import os
from config import path_cfg
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import torch as t


class LabelProcessor:

    def __init__(self, file_path):
        self.colormap = self.read_color_map(file_path)
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class LoadDataset(Dataset):
    def __init__(self, channels_1_root, channels_2_root, channels_3_root, label_root):
        self.channels_1_root = channels_1_root
        self.channels_2_root = channels_2_root
        self.channels_3_root = channels_3_root
        self.label_root = label_root
        self.channels_1 = self.read_file(self.channels_1_root)
        self.channels_2 = self.read_file(self.channels_2_root)
        self.channels_3 = self.read_file(self.channels_3_root)
        self.labels = self.read_file(self.label_root)

    def __getitem__(self, index):
        channel_1 = self.channels_1[index]
        channel_2 = self.channels_2[index]
        channel_3 = self.channels_3[index]
        label = self.labels[index]
        file_name = channel_1.split('/')[-1].split('.')[0]
        channel_1 = Image.open(channel_1)
        channel_2 = Image.open(channel_2)
        channel_3 = Image.open(channel_3)

        # Normalization (optional)
        # channel_1 = np.array(channel_1).astype(np.float32) / 255.0
        # channel_2 = np.array(channel_2).astype(np.float32) / 255.0
        # channel_3 = np.array(channel_3).astype(np.float32) / 255.0

        label = Image.open(label).convert('RGB')
        channel_1, channel_2, channel_3, label = self.img_transform(channel_1, channel_2, channel_3, label)
        sample = {'channel_1': channel_1, 'channel_2': channel_2, 'channel_3': channel_3, 'label': label,
                  'fname': file_name}
        return sample

    def __len__(self):
        return len(self.channels_1)

    def read_file(self, root):
        files_list = os.listdir(root)
        file_path_list = [os.path.join(root, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def img_transform(self, channel_1, channel_2, channel_3, label):
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        channel_1 = transform_img(channel_1)
        channel_2 = transform_img(channel_2)
        channel_3 = transform_img(channel_3)
        label_processor = LabelProcessor(path_cfg.class_dict_path)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)
        return channel_1, channel_2, channel_3, label
