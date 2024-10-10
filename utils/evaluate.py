import os
import math
import numpy as np
from PIL import Image


def get_value(predict_folders_path, label_folders_path, value=255):
    predict_folders = os.listdir(predict_folders_path)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for folder in predict_folders:
        predict_folder_path = os.path.join(predict_folders_path, folder)
        label_folder_path = os.path.join(label_folders_path, folder)
        predict = np.array(Image.open(predict_folder_path).convert('L'))
        label = np.array(Image.open(label_folder_path).convert('L'))
        mask_predict = predict == value
        mask_label = label == value
        TP += np.sum(mask_predict & mask_label)
        FP += np.sum(mask_predict & ~mask_label)
        FN += np.sum(~mask_predict & mask_label)
        TN += np.sum(~mask_predict & ~mask_label)
    return float(TP), float(TN), float(FP), float(FN)


def evaluation(TP, TN, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * ((precision * recall) / (precision + recall))
    iou = TP / (TP + FP + FN)
    under = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    mcc = (TP * TN - FP * FN) / math.sqrt(under)
    evo = 'mcc: {:.4f}  iou: {:.4f}  f1: {:.4f}'.format(mcc, iou, f1)
    return evo, iou
