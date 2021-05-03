import os
import torch
from torch import nn
from torch.utils.data import Dataset, random_split
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import pandas as pd
from torch.utils.data import Subset


class PRESSURE_Dataset(TorchvisionDataset):
    def __init__(self, data_path, normal_class=1, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = tuple([1 - normal_class])
        self.train_set = PRESSURE(data_path + '/train.csv')
        self.test_set = PRESSURE(data_path + '/test.csv')


class PRESSURE(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.data = pd.read_csv(data_path).values.tolist()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx][-1]
        feature = self.data[idx][:-1]
        feature = torch.tensor(feature)
        return feature, label, idx
