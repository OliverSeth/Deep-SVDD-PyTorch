import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


# class PRESSURE_LeNET(BaseNet):
#     def __init__(self):
#         super().__init__()
#         self.rep_dim = 32
#         self.pool = nn.MaxPool1d(4)
#
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, bias=False, padding=2)
#         self.bn1 = nn.BatchNorm1d(16, eps=1e-04, affine=False)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, bias=False, padding=2)
#         self.bn2 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
#         self.fc1 = nn.Linear(32 * 25, self.rep_dim, bias=False)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool(F.leaky_relu(self.bn1(x)))
#         x = self.conv2(x)
#         x = self.pool(F.leaky_relu(self.bn2(x)))
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, 3, stride, 1, bias=False),  # 要采样的话在这里改变stride
            nn.BatchNorm1d(outchannel),  # 批处理正则化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv1d(outchannel, outchannel, 3, 1, 1, bias=False),  # 采样之后注意保持feature map的大小不变
            nn.BatchNorm1d(outchannel),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # 计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)


class PRESSURE_LeNET(nn.Module):
    def __init__(self):
        super(PRESSURE_LeNET, self).__init__()
        self.rep_dim = 1000
        self.pre = nn.Sequential(
            nn.Conv1d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )  # 开始的部分
        self.body = self.makelayers([3, 4, 6, 3])  # 具有重复模块的部分
        self.classifier = nn.Linear(512, 1000)  # 末尾的部分

    def makelayers(self, blocklist):  # 注意传入列表而不是解列表
        self.layers = []
        for index, blocknum in enumerate(blocklist):
            if index != 0:
                shortcut = nn.Sequential(
                    nn.Conv1d(64 * 2 ** (index - 1), 64 * 2 ** index, 1, 2, bias=False),
                    nn.BatchNorm1d(64 * 2 ** index)
                )  # 使得输入输出通道数调整为一致
                self.layers.append(ResidualBlock(64 * 2 ** (index - 1), 64 * 2 ** index, 2, shortcut))  # 每次变化通道数时进行下采样
            for i in range(0 if index == 0 else 1, blocknum):
                self.layers.append(ResidualBlock(64 * 2 ** index, 64 * 2 ** index, 1))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.body(x)
        x = nn.AvgPool1d(7)(x)  # kernel_size为7是因为经过多次下采样之后feature map的大小为7*7，即224->112->56->28->14->7
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
