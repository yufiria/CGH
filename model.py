# -*- coding: utf-8 -*-
"""
标准U-Net神经网络模型定义模块
"""
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__(); self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class HologramGenerator(nn.Module):
    """基于U-Net的全息图生成器模型。"""
    def __init__(self, config):
        super(HologramGenerator, self).__init__()
        features = config.features
        self.inc = DoubleConv(config.in_channels, features[0])
        self.down1 = Down(features[0], features[1]); self.down2 = Down(features[1], features[2]); self.down3 = Down(features[2], features[3])
        self.up1 = Up(features[3], features[2]); self.up2 = Up(features[2], features[1]); self.up3 = Up(features[1], features[0])
        self.outc = OutConv(features[0], config.out_channels)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4, x3); x = self.up2(x, x2); x = self.up3(x, x1)
        logits = self.outc(x)
        phase = torch.tanh(logits[:, 0:1, :, :]) * torch.pi
        amplitude = torch.sigmoid(logits[:, 1:2, :, :])
        return phase, amplitude