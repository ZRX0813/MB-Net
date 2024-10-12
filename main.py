import argparse
import tester
from trainer import *
from utils.data_load_transform import *
import os
from torch.utils.data import DataLoader
from config import path_cfg

parser = argparse.ArgumentParser(description="Ruixuan Zhang")
parser.add_argument('--save_path', type=str, default='./output',
                    help='output path')
parser.add_argument('--model', type=str, default='MB_Net', help='the name of model you select')
parser.add_argument('--workers', type=int, default=4,
                    metavar='N', help='dataloader threads')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training(default: 4)')
parser.add_argument("--train", action='store_true',
                    help="Whether on a training mission")
parser.add_argument("--test", action='store_true',
                    help="Whether on a testing mission")

if __name__ == '__main__':
    args = parser.parse_args()
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # train
    if args.train:
        train_loader = LoadDataset(path_cfg.CHANNELS_1_TRAIN_ROOT, path_cfg.CHANNELS_2_TRAIN_ROOT,
                                   path_cfg.CHANNELS_3_TRAIN_ROOT, path_cfg.TRAIN_LABEL)
        train_data = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = LoadDataset(path_cfg.CHANNELS_1_VAL_ROOT, path_cfg.CHANNELS_2_VAL_ROOT,
                                 path_cfg.CHANNELS_3_VAL_ROOT, path_cfg.VAL_LABEL)
        val_data = DataLoader(val_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        trainer = trainer(args, device, train_data, val_data)
        trainer.train()
    # test
    if args.test:
        test_loader = LoadDataset(path_cfg.CHANNELS_1_TEST_ROOT, path_cfg.CHANNELS_2_TEST_ROOT,
                                  path_cfg.CHANNELS_3_TEST_ROOT, path_cfg.TEST_LABEL)
        test_data = DataLoader(test_loader, batch_size=1, shuffle=True, num_workers=args.workers)
        tester = tester.test(args, device, test_data)
