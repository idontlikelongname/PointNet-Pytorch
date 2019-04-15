""" Test """

import os.path
import sys
import argparse
import datetime

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

from datasets.squeeseSeg import SqueezeSegDataset
from nets import pointnet
from utils.squeezeSeg_util import *

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PointNet')

parser.add_argument('--network', default='PointNet', type=str)

# path
parser.add_argument('--csv_path', default='./data/csv/', type=str, help='path to where csv file')
parser.add_argument('--data_path', default='./data/lidar_2d/', type=str, help='path to where data')
parser.add_argument('--model_path', default='./models', type=str, help='path to where model')

parser.add_argument('--epoch', default=0, type=int)

parser.add_argument('--cls', default=['unkwon', 'car', 'pedestrian', 'cyclist'], nargs='+', type=str)

# Device Option
parser.add_argument('--gpu_ids', dest='gpu_ids', default=[0,1], nargs="+", type=int, help='which gpu you use')
parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, test_loader):
    model.eval()

    total_tp = np.zeros(len(args.cls))
    total_fp = np.zeros(len(args.cls))
    total_fn = np.zeros(len(args.cls))

    with torch.no_grad():
        for batch_idx, datas in enumerate(test_loader, 1):
            inputs, targets = datas
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, trans, trans_feat = model(inputs[:,:3,:])

            _, predicted = torch.max(outputs.data, 2)

            if batch_idx == len(test_loader):
                input_img = img_normalize(inputs[0][3].view(64, 512))
                Image.fromarray(np.uint8(input_img)).save('input.png')

                output_img = visualize_seg(predicted[0].view(64, 512))
                Image.fromarray(np.uint8(output_img)).save('output.png')

                target_img = visualize_seg(targets[0].view(64,512))
                Image.fromarray(np.uint8(target_img)).save('target.png')

                print('save image')

            tp, fp, fn = evaluate(targets, predicted, len(args.cls))
            total_tp += tp
            total_fp += fp
            total_fn += fn

        iou = total_tp / (total_tp+total_fn+total_fp+1e-12)
        precision = total_tp / (total_tp+total_fp+1e-12)
        recall = total_tp / (total_tp+total_fn+1e-12)

        print()
        print_evaluate('IoU', iou, args.cls)
        print_evaluate('Precision', precision, args.cls)
        print_evaluate('Recall', recall, args.cls)
        print()

if __name__ == '__main__':

    # test data 読み込み
    test_datasets = SqueezeSegDataset(
        csv_file = args.csv_path + 'test.csv',
        root_dir = args.data_path,
        data_augmentation = False,
        random_flipping = False,
        transform = transforms.Compose([transforms.ToTensor()])
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4
    )

    if args.network == 'PointNet':
        model = pointnet.PointNetSegmentation(k=len(args.cls)).to(device)

    load_checkpoint(
        args.model_path,
        args.network,
        args.epoch,
        model
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    test(model, test_dataloader)