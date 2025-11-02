# -*- coding: utf-8 -*-
"""
损失函数模块

定义了用于训练全息生成网络的组合损失函数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchmetrics.image import StructuralSimilarityIndexMeasure

class CombinedLoss(nn.Module):
    """结合MSE, SSIM, Perceptual Loss, 和 Total Variation Loss的组合损失模块。"""
    def __init__(self, config):
        super().__init__()
        self.weights = config.loss_weights
        self.device = config.device
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].to(self.device).eval()
        for param in vgg.parameters(): param.requires_grad = False
        self.vgg = vgg
        print(f"损失函数已初始化，权重: {self.weights}")

    def tv_loss(self, img):
        wh = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        wv = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        return (wh + wv) / torch.numel(img)

    def perceptual_loss(self, recon_img, target_img):
        recon_norm = (recon_img + 1.0) / 2.0; target_norm = (target_img + 1.0) / 2.0
        recon_features = self.vgg(recon_norm); target_features = self.vgg(target_norm)
        return F.mse_loss(recon_features, target_features)

    def forward(self, recon_img, target_img):
        losses = {}
        losses['mse'] = self.mse_loss(recon_img, target_img)
        losses['ssim'] = 1.0 - self.ssim_loss(recon_img, target_img)
        losses['perceptual'] = self.perceptual_loss(recon_img, target_img)
        losses['tv'] = self.tv_loss(recon_img)
        total_loss = sum(self.weights[key] * losses[key] for key in self.weights)
        return total_loss, {k: v.item() for k, v in losses.items()}