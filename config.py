# -*- coding: utf-8 -*-
"""
项目配置参数文件

该文件集中管理项目的所有可调参数，包括数据集、模型、物理传播、训练和路径等。
方便用户在不修改核心代码的情况下，快速调整实验设置。
"""
import torch

class Config:
    """配置类，存储所有训练、模型和物理参数"""
    
    # 1. 数据集参数
    # -------------------------------------------------------------------------
    dataset_path = './synthetic_dataset/'
    img_size = 32
    batch_size = 32

    # 2. 模型参数
    # -------------------------------------------------------------------------
    # 模型选择: 'UNet' (原始模型) 或 'AttentionUNet' (高级模型)
    model_name = 'AttentionUNet' 
    in_channels = 3
    out_channels = 2
    features = [64, 128, 256, 512]

    # 3. 物理传播参数
    # -------------------------------------------------------------------------
    wavelengths = {
        'r': 638e-9,  # 红色波长 (m)
        'g': 520e-9,  # 绿色波长 (m)
        'b': 450e-9   # 蓝色波长 (m)
    }
    pixel_size = 8e-6
    slm_resolution = (img_size, img_size)
    propagation_distance = 0.1

    # 4. 训练参数
    # -------------------------------------------------------------------------
    epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 5. 损失函数权重
    # -------------------------------------------------------------------------
    loss_weights = {
        'mse': 1.0,
        'ssim': 1.0,
        'perceptual': 0.1,
        'tv': 1e-5,
    }

    # 6. 保存与加载路径
    # -------------------------------------------------------------------------
    model_save_path = './checkpoints/'
    train_results_path = './results/train/'
    inference_results_path = './results/inference/'