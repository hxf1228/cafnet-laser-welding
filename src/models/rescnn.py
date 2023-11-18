"""
Created on: 2021-10-02 22:14:48 UTC+8
@File: rescnn.py
@Description: The model used in paper "Intelligent Mechanical Fault Diagnosis Using Multi-Sensor Fusion and Convolution Neural Network"
@Copy Right: Licensed under the MIT License.
"""

import torch
import torch.nn as nn


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ImprovedResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.lrelu(out)
        return out


class ImprovedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImprovedConvBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv = conv1x1(in_channels, out_channels, stride=2)

    def forward(self, x):
        residual = self.conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.lrelu(out)
        return out


class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        self.in_channels = 16
        #self.conv = conv3x3(1, 16, stride=1)
        self.conv = conv3x3(1, 16, stride=1)
        self.bn = nn.BatchNorm2d(16)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.iresblock1 = ImprovedResidualBlock(16, 16, stride=1)
        self.iresblock2 = self.iresblock1
        self.iresblock3 = ImprovedResidualBlock(32, 32, stride=1)
        self.iresblock4 = ImprovedResidualBlock(64, 64, stride=1)
        self.iconvblock1 = ImprovedConvBlock(16, 32)
        self.iconvblock2 = ImprovedConvBlock(32, 64)
        self.globalavgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, 3)  # 64
        self.last_feature = None

    def forward(self, x):
        batch_size, wide, high = x.size(0), x.size(1), x.size(2)
        # Conv x1
        x = x.view(batch_size, 1, wide, high)
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)
        # Improved Residual Block x 2
        out = self.iresblock1(out)
        out = self.iresblock2(out)
        out = self.iconvblock1(out)
        out = self.iresblock3(out)
        out = self.iconvblock2(out)
        out = self.iresblock4(out)
        self.last_feature = out.view(out.size(0), -1)
        out = self.globalavgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResCNNEncoder(nn.Module):
    def __init__(self):
        super(ResCNNEncoder, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16, stride=1)
        self.bn = nn.BatchNorm2d(16)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.iresblock1 = ImprovedResidualBlock(16, 16, stride=1)
        self.iresblock2 = self.iresblock1
        self.iresblock3 = ImprovedResidualBlock(32, 32, stride=1)
        self.iresblock4 = ImprovedResidualBlock(64, 64, stride=1)
        self.iconvblock1 = ImprovedConvBlock(16, 32)
        self.iconvblock2 = ImprovedConvBlock(32, 64)
        self.globalavgpool = nn.AvgPool2d(8, stride=1)

    def forward(self, x):
        # Conv x1
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)
        # Improved Residual Block x 2
        out = self.iresblock1(out)
        out = self.iresblock2(out)
        out = self.iconvblock1(out)
        out = self.iresblock3(out)
        out = self.iconvblock2(out)
        out = self.iresblock4(out)
        out = self.globalavgpool(out)
        out = out.view(out.size(0), -1)
        return out


class MultiResCNN(nn.Module):
    def __init__(self):
        super(MultiResCNN, self).__init__()
        self.acoustic_model = ResCNNEncoder()
        self.photodiode_model = ResCNNEncoder()
        self.classifier_1 = nn.Sequential(nn.Linear(128, 3), )
        self.last_feature = None

    def forward(self, x_acoustic, x_photodiode):
        acoustic = self.acoustic_model(x_acoustic)
        photodiode = self.photodiode_model(x_photodiode)

        acoustic_pooled = acoustic
        photodiode_pooled = photodiode
        x = torch.cat((acoustic_pooled, photodiode_pooled), dim=-1)
        self.last_feature = x
        x1 = self.classifier_1(x)
        return x1
