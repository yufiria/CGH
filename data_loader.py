# -*- coding: utf-8 -*-
"""
数据集加载和预处理模块

该文件定义了一个自定义的PyTorch Dataset类，用于读取 `generate_dataset.py`
生成的本地图像文件，并将其转换为适合神经网络训练的Tensor格式。
"""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SyntheticHologramDataset(Dataset):
    """用于加载合成数据集的自定义Dataset类。"""
    def __init__(self, root_dir, split='train', transform=None):
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.transform = transform
        self.image_files = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_data_loader(config):
    """创建并返回训练和验证数据加载器。"""
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = SyntheticHologramDataset(root_dir=config.dataset_path, split='train', transform=data_transform)
    val_dataset = SyntheticHologramDataset(root_dir=config.dataset_path, split='val', transform=data_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True if config.device == 'cuda' else False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True if config.device == 'cuda' else False)
    
    return train_loader, val_loader