import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from config import path_cfg
import pandas as pd
import torch as t
import torch.nn.functional as F
from PIL import Image
import os
import torch
from tqdm import tqdm
from utils.evaluate import get_value, evaluation
import numpy as np
from MB_Net import MB_Net as Ours


def test(args, device, test_loader):
    weights_path = os.path.join(args.save_path, args.model, 'or_weights')
    weights = os.listdir(weights_path)
    result_path = os.path.join(args.save_path, args.model, 'test_results')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    pd_label_color = pd.read_csv(path_cfg.class_dict_path, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)
    cm = np.array(colormap).astype('uint8')
    for weight in weights:
        Net = Ours(3, 2).to(device)
        print("current processing weight {}".format(weight))
        weight_path = os.path.join(weights_path, weight)
        Net.load_state_dict(t.load(weight_path))
        Net.eval()
        tbar_test = tqdm(test_loader)
        with torch.no_grad():
            for i, sample in enumerate(tbar_test):
                channel_1_data = sample['channel_1'].to(device)
                channel_2_data = sample['channel_2'].to(device)
                channel_3_data = sample['channel_3'].to(device)
                out = Net(channel_1_data, channel_2_data, channel_3_data)
                out = F.log_softmax(out, dim=1)
                pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
                pre = Image.fromarray(cm[pre_label])
                name = str(sample['fname'][0])
                name = eval(repr(name).replace('\\', '@'))
                name = str(name).split('@@')[4]
                pre.save(os.path.join(result_path, name) + '.png')
        TP, TN, FP, FN = get_value(result_path, path_cfg.TEST_LABEL)
        evo, iou = evaluation(TP, TN, FP, FN)
        print(evo)
