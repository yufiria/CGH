# -*- coding: utf-8 -*-
"""
高级神经网络模型定义模块

该文件定义了一个增强版的U-Net模型，集成了残差连接和注意力门，
以提升模型在复杂物理任务（如全息图生成）上的性能。
"""
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """带有两个卷积层的残差块。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels); self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class AttentionGate(nn.Module):
    """注意力门 (Attention Gate, AG)"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g); x1 = self.W_x(x)
        psi = self.relu(g1 + x1); psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    """集成了残差块和注意力门的增强版U-Net模型。"""
    def __init__(self, config):
        super().__init__()
        features = config.features
        self.in_block = ResidualBlock(config.in_channels, features[0])
        self.pool = nn.MaxPool2d(2)
        self.down1 = ResidualBlock(features[0], features[1]); self.down2 = ResidualBlock(features[1], features[2]); self.down3 = ResidualBlock(features[2], features[3])
        self.up1 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2); self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2); self.up3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.attn1 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[1]); self.attn2 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[0]); self.attn3 = AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0]//2)
        self.up_conv1 = ResidualBlock(features[3], features[2]); self.up_conv2 = ResidualBlock(features[2], features[1]); self.up_conv3 = ResidualBlock(features[1], features[0])
        self.out_conv = nn.Conv2d(features[0], config.out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.in_block(x); x2 = self.down1(self.pool(x1)); x3 = self.down2(self.pool(x2)); x4 = self.down3(self.pool(x3))
        d1 = self.up1(x4); x3_att = self.attn1(g=d1, x=x3); d1 = torch.cat((x3_att, d1), dim=1); d1 = self.up_conv1(d1)
        d2 = self.up2(d1); x2_att = self.attn2(g=d2, x=x2); d2 = torch.cat((x2_att, d2), dim=1); d2 = self.up_conv2(d2)
        d3 = self.up3(d2); x1_att = self.attn3(g=d3, x=x1); d3 = torch.cat((x1_att, d3), dim=1); d3 = self.up_conv3(d3)
        logits = self.out_conv(d3)
        phase = torch.tanh(logits[:, 0:1, :, :]) * torch.pi
        amplitude = torch.sigmoid(logits[:, 1:2, :, :])
        return phase, amplitude