""" Kitti Dataset for SqueezeSeg """

import pandas as pd

import numpy as np

import torch
from torch.utils.data import Dataset

import os
import sys

# x, y, z, intensity, distance for Normalization
INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
INPUT_STD = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

# クラスの出現度に応じてlossに重さを設定
CLS_LOSS_WEIGHT = np.array([1/15.0, 1.0, 10.0, 10.0])

class KittiDataset(Dataset):
    def __init__(self, csv_file, root_dir, data_augmentation=True, random_flipping=True, transform=None):
        self.lidar_2d_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.random_flipping = random_flipping

    def __len__(self):
        return len(self.lidar_2d_csv)

    def __getitem__(self, idx):
        lidar_name = os.path.join(self.root_dir, self.lidar_2d_csv.iloc[idx, 0])
        lidar_data = np.load(lidar_name).astype(np.float32)

        if self.data_augmentation:
            if self.random_flipping:
                if np.random.rand() > 0.5:
                    # flip y
                    lidar_data = lidar_data[:, ::-1, :]
                    lidar_data[:, :, 1] *= -1

        lidar_inputs =  lidar_data[:, :, :5] # x, y, z, intensity, depth(range)

        lidar_mask = np.reshape(
            (lidar_inputs[:, :, 4] > 0) * 1,
            [64, 512, 1]
        )

        # Normalize Inputs
        lidar_inputs = (lidar_inputs - INPUT_MEAN) / INPUT_STD

        lidar_label = lidar_data[:, :, 5]

        weight = np.zeros(lidar_label.shape)
        for l in range(len(CLS_LOSS_WEIGHT)):
            weight[lidar_label == l] = CLS_LOSS_WEIGHT[int(l)]

        if self.transform:
            lidar_inputs = self.transform(lidar_inputs)
            lidar_mask = self.transform(lidar_mask)

        return (lidar_inputs.float(), lidar_mask.float(), torch.from_numpy(lidar_label.copy()).long(), torch.from_numpy(weight.copy()).float())
