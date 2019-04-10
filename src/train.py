""" Train """

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

from datasets.dataset import KittiDataset
from nets import resnet, resnetwithUnet
from utils.util import *

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PointNet')

parser.add_argument('--network', default='PointNet', type=str)

# path
parser.add_argument('--csv_path', default='./data/csv/', type=str, help='path to where csv file')
parser.add_argument('--data_path', default='./data/lidar_2d/', type=str, help='path to where data')
parser.add_argument('--model_path', default='./models', type=str, help='path to where model')
parser.add_argument('--save_path', default='./data/')

# Hyper Params
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optim')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay for optim')
parser.add_argument('--lr_step', default=1000, type=int, help='number of lr step')
parser.add_argument('--lr_gamma', default=0.1, type=float, help='gamma for lr scheduler')

parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='number of epoch to start learning')
parser.add_argument('--pretrain', default=False, type=bool, help='Whether or not to pretrain')
parser.add_argument('--resume', default=False, type=bool, help='Whether or not to resume')

parser.add_argument('--classes', default=['unkwon', 'car', 'pedestrian', 'cyclist'], nargs='+', type=str)
parser.add_argument('--cls_loss_coef', default=15.0, type=float)

# Device Option
parser.add_argument('--gpu_ids', dest='gpu_ids', default=[0,1], nargs="+", type=int, help='which gpu you use')
parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size')

args = parser.parse_args()

# To use TensorboardX
writer = SummaryWriter()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()

    total_loss = 0.0
    total_size = 0.0

    for batch_idx, datas in enumerate(train_loader, 1):
        inputs, mask, targets, weights = datas
        inputs, mask, targets, weights = \
            inputs.to(device), mask.to(device), targets.to(device), weights.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        loss = criterion(outputs, targets, mask, weights)

        writer.add_scalar('data/loss', loss/args.batch_size, epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        total_size += inputs.size(0)

        if batch_idx % 10 == 0:
            now = datetime.datetime.now()

            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)] \t Average Loss: {:.6f}'.format(
                now,
                epoch,
                batch_idx * len(inputs),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss / total_size)
            )

def test(model, val_loader, epoch, best):
    model.eval()

    total_tp = np.zeros(len(args.classes))
    total_fp = np.zeros(len(args.classes))
    total_fn = np.zeros(len(args.classes))

    with torch.no_grad():
        for batch_idx, datas in enumerate(val_loader, 1):
            inputs, mask, targets, weights = datas
            inputs, mask, targets, weights = \
                inputs.to(device), mask.to(device), targets.to(device), weights.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            tp, fp, fn = evaluate(targets, predicted, len(args.classes))
            total_tp += tp
            total_fp += fp
            total_fn += fn

        iou = total_tp / (total_tp+total_fn+total_fp+1e-12)
        precision = total_tp / (total_tp+total_fp+1e-12)
        recall = total_tp / (total_tp+total_fn+1e-12)

        save = False

        if iou[1] > best:
            best = iou[1]
            save = True

            input_img = img_normalize(inputs[0][3])
            Image.fromarray(np.uint8(input_img)).save(os.path.join(args.save_path, args.network, 'Inputs/') + '{}.png'.format(epoch))

            output_img = visualize_seg(predicted)
            Image.fromarray(np.uint8(output_img[0])).save(os.path.join(args.save_path, args.network, 'Outputs/') + '{}.png'.format(epoch))

            target_img = visualize_seg(targets)
            Image.fromarray(np.uint8(target_img[0])).save(os.path.join(args.save_path, args.network, 'Targets/') + '{}.png'.format(epoch))

            print('save image as {}.png'.format(epoch))

        print()
        print_evaluate('IoU', iou, args.classes)
        print_evaluate('Precision', precision, args.classes)
        print_evaluate('Recall', recall, args.classes)
        print()

    return best, save

if __name__ == '__main__':
    if os.path.exists(args.model_path) is False:
        os.mkdir(args.model_path)

    if os.path.exists(os.path.join(args.model_path, args.network)) is False:
        os.mkdir(os.path.join(args.model_path, args.network))

    if os.path.exists(os.path.join(args.save_path, args.network)) is False:
        os.mkdir(os.path.join(args.save_path, args.network))
        os.mkdir(os.path.join(args.save_path, args.network, 'Inputs'))
        os.mkdir(os.path.join(args.save_path, args.network, 'Outputs'))
        os.mkdir(os.path.join(args.save_path, args.network, 'Targets'))

    # train data 読み込み
    train_datasets = KittiDataset(
        csv_file = args.csv_path + 'train.csv',
        root_dir = args.data_path,
        transform = transforms.Compose([transforms.ToTensor()])
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4
    )

    # val data 読み込み
    val_datasets = KittiDataset(
        csv_file = args.csv_path + 'val.csv',
        root_dir = args.data_path,
        data_augmentation = False,
        random_flipping = False,
        transform = transforms.Compose([transforms.ToTensor()])
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_datasets,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4
    )

    if args.network == 'PointNet':
        model = resnet.ResNet101().to(device)

    if args.resume:
        load_checkpoint(
            os.path.join(args.model_path, args.network),
            args.start_epoch - 1,
            model
        )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = SqueezeSegLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr = args.lr,
        momentum = args.momentum,
        weight_decay = args.weight_decay
    )

    schduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = args.lr_step,
        gamma = args.lr_gamma
    )

    best = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        schduler.step()
        print('-------------------------------------------------------------------')
        train(model, train_dataloader, criterion, optimizer, epoch)
        best, save = test(model, val_dataloader, epoch, best)
        if save:
            save_checkpoint(os.path.join(args.model_path, args.network), epoch, model)
        print('-------------------------------------------------------------------')
        print()

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
