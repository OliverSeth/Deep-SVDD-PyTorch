import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class PRESSURE_LeNET(BaseNet):
    def __init__(self):
        super().__init__()
        self.rep_dim = 32
        self.pool = nn.MaxPool1d(4)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm1d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 25, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
