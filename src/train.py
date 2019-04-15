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
parser.add_argument('--save_path', default='./data/')

# Hyper Params
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--betas',default=[0.9, 0.999], nargs='+', type=float, help='betas for Adam optim')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD optim')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay for optim')
parser.add_argument('--lr_step', default=20, type=int, help='number of lr step')
parser.add_argument('--lr_gamma', default=0.5, type=float, help='gamma for lr scheduler')

parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='number of start epochs to run')

parser.add_argument('--resume', default=False, type=bool, help='Whether or not to resume')

parser.add_argument('--cls', default=['unkwon', 'car', 'pedestrian', 'cyclist'], nargs='+', type=str)
parser.add_argument('--cls_loss_weight', default=[1.0/15.0, 1.0, 10.0, 10.0], nargs='+', type=float)
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
        inputs, targets = datas
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, trans, trans_feat = model(inputs[:,:3,:])

        _, predicted = torch.max(outputs.data, 2)

        loss = criterion(outputs.view(-1, len(args.cls)), targets.view(-1,))

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
                100. * batch_idx * len(inputs) / len(train_loader.dataset),
                total_loss / total_size))

        if batch_idx == len(train_loader):
            input_img = img_normalize(inputs[0][3].view(64, 512))
            Image.fromarray(np.uint8(input_img)).save(os.path.join(args.save_path, args.network, 'Inputs/') + '{}.png'.format(epoch))

            output_img = visualize_seg(predicted[0].view(64, 512))
            Image.fromarray(np.uint8(output_img)).save(os.path.join(args.save_path, args.network, 'Outputs/') + '{}.png'.format(epoch))

            target_img = visualize_seg(targets[0].view(64,512))
            Image.fromarray(np.uint8(target_img)).save(os.path.join(args.save_path, args.network, 'Targets/') + '{}.png'.format(epoch))

            print('save image as {}.png'.format(epoch))

def test(model, val_loader, epoch):
    model.eval()

    total_loss = 0.0

    total_tp = np.zeros(len(args.cls))
    total_fp = np.zeros(len(args.cls))
    total_fn = np.zeros(len(args.cls))

    with torch.no_grad():
        for batch_idx, datas in enumerate(val_loader, 1):
            inputs, targets = datas
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, trans, trans_feat = model(inputs[:,:3,:])

            _, predicted = torch.max(outputs.data, 2)

            tp, fp, fn = evaluate(targets, predicted, len(args.cls))
            total_tp += tp
            total_fp += fp
            total_fn += fn

        iou = total_tp / (total_tp+total_fn+total_fp+1e-12)
        precision = total_tp / (total_tp+total_fp+1e-12)
        recall = total_tp / (total_tp+total_fn+1e-12)

        writer.add_scalars(
            'data/iou',
            {'car': iou[1], 'pedestrian': iou[2], 'cyclist': iou[3]},
            epoch)

        writer.add_scalars(
            'data/precision',
            {'car': precision[1], 'pedestrian': precision[2], 'cyclist': precision[3]},
            epoch)

        writer.add_scalars(
            'data/recall',
            {'car': recall[1], 'pedestrian': recall[2], 'cyclist': recall[3]},
            epoch)

        print()
        print_evaluate('IoU', iou, args.cls)
        print_evaluate('Precision', precision, args.cls)
        print_evaluate('Recall', recall, args.cls)
        print()

if __name__ == '__main__':
    os.makedirs(os.path.join(args.model_path, args.network), exist_ok=True)

    os.makedirs(os.path.join(args.save_path, args.network, 'Inputs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.network, 'Outputs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.network, 'Targets'), exist_ok=True)

    # train data 読み込み
    train_datasets = SqueezeSegDataset(
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
    val_datasets = SqueezeSegDataset(
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
        model = pointnet.PointNetSegmentation(k=len(args.cls)).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.NLLLoss(
        weight=torch.tensor(args.cls_loss_weight).to(device))

    optimizer = optim.Adam(
        model.parameters(),
        lr = args.lr,
        betas = args.betas,
    )

    schduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = args.lr_step,
        gamma = args.lr_gamma
    )

    if args.resume:
        load_checkpoint(
            os.path.join(args.model_path, args.network),
            args.network,
            args.start_epoch - 1,
            model
        )

    for epoch in range(args.start_epoch, args.epochs):
        schduler.step()
        print('-----------------------------------------------------------')
        train(model, train_dataloader, criterion, optimizer, epoch)
        test(model, val_dataloader, epoch)
        save_checkpoint(
            os.path.join(args.model_path, args.network),
            args.network,
            epoch,
            model
        )
        print('-----------------------------------------------------------')
        print()

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
