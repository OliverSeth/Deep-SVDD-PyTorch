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
        data = pd.read_csv(data_path)
        train_data = data.loc[data['target'] == normal_class].values.tolist()
        self.train_set, test_data = random_split(train_data, [int(len(data) * 0.7), len(data) - int(len(data) * 0.7)])
        self.test_set = data.loc[data['target'] != normal_class].values.tolist()
        self.test_set.extend(test_data)


class PRESSURE(Dataset):
    def __init__(self, data_list, normal_class=1, transform=None, target_transform=None):
        self.data = data_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx][-1]
        feature = self.data[idx][:-1]
        feature = torch.tensor(feature)
        return feature, label, idx
