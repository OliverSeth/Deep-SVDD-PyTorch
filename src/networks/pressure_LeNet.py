import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class PRESSURE_LeNET(BaseNet):
    def __init__(self):
        super().__init__()
        self.rep_dim = 32
        self.pool = nn.MaxPool1d(2)

        self.conv1 = nn.Conv1d(1, 10, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm1d(10, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(10, 20, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm1d(20, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(20 * 100, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
