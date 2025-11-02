# -*- coding: utf-8 -*-
"""
模型训练主脚本
"""
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from data_loader import get_data_loader
from propagation import ComplexWavePropagator
from loss import CombinedLoss

def denormalize(tensor):
    return (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0

def save_training_results(epoch, target, recon, phase, amp, losses, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    target_np = denormalize(target[0]).permute(1, 2, 0).cpu().numpy()
    recon_np = denormalize(recon[0]).permute(1, 2, 0).cpu().numpy()
    phase_np = phase[0, 0].cpu().numpy(); amp_np = amp[0, 0].cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(target_np); axes[0].set_title('目标'); axes[0].axis('off')
    axes[1].imshow(recon_np); axes[1].set_title('重建'); axes[1].axis('off')
    im2 = axes[2].imshow(phase_np, cmap='hsv'); axes[2].set_title('相位'); plt.colorbar(im2, ax=axes[2]); axes[2].axis('off')
    im3 = axes[3].imshow(amp_np, cmap='viridis'); axes[3].set_title('振幅'); plt.colorbar(im3, ax=axes[3]); axes[3].axis('off')
    loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
    fig.suptitle(f'Epoch {epoch}\nLosses: {loss_str}', fontsize=12)
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}.png')); plt.close()

def train(config):
    device = torch.device(config.device)
    if config.model_name == 'AttentionUNet':
        from advanced_model import AttentionUNet as Model
    else:
        from model import HologramGenerator as Model
    model = Model(config).to(device)
    print(f"使用模型: {config.model_name}, 设备: {device}")

    propagators = {c: ComplexWavePropagator(config.slm_resolution, config.pixel_size, config.propagation_distance, w).to(device) for c, w in config.wavelengths.items()}
    criterion = CombinedLoss(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader, val_loader = get_data_loader(config)
    
    for epoch in range(config.epochs):
        model.train()
        train_loss_agg = {k: 0.0 for k in config.loss_weights}
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for target_imgs in progress_bar:
            target_imgs = target_imgs.to(device)
            optimizer.zero_grad()
            phase, amp = model(target_imgs)
            complex_field = amp * torch.exp(1j * phase)
            
            recon_channels = [torch.abs(propagators[c](complex_field))**2 for c in ['r', 'g', 'b']]
            recon_imgs = torch.cat(recon_channels, dim=1)
            
            recon_imgs_norm = recon_imgs / recon_imgs.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
            recon_imgs_norm = (recon_imgs_norm * 2.0 - 1.0).clamp(-1.0, 1.0)
            
            total_loss, losses = criterion(recon_imgs_norm, target_imgs)
            total_loss.backward(); optimizer.step()
            for k, v in losses.items(): train_loss_agg[k] += v
            progress_bar.set_postfix({k: f"{v / (progress_bar.n + 1):.4f}" for k, v in train_loss_agg.items()})

        model.eval()
        val_loss_agg = {k: 0.0 for k in config.loss_weights}
        with torch.no_grad():
            for target_imgs_val in val_loader:
                target_imgs_val = target_imgs_val.to(device)
                phase_val, amp_val = model(target_imgs_val)
                complex_field_val = amp_val * torch.exp(1j * phase_val)
                recon_channels_val = [torch.abs(propagators[c](complex_field_val))**2 for c in ['r', 'g', 'b']]
                recon_imgs_val = torch.cat(recon_channels_val, dim=1)
                recon_imgs_val_norm = recon_imgs_val / recon_imgs_val.amax(dim=(1,2,3), keepdim=True).clamp(min=1e-6)
                recon_imgs_val_norm = (recon_imgs_val_norm * 2.0 - 1.0).clamp(-1.0, 1.0)
                _, losses_val = criterion(recon_imgs_val_norm, target_imgs_val)
                for k, v in losses_val.items(): val_loss_agg[k] += v
        
        train_avg_loss = {k: v / len(train_loader) for k, v in train_loss_agg.items()}
        val_avg_loss = {k: v / len(val_loader) for k, v in val_loss_agg.items()}
        print(f"\nEpoch {epoch+1} Summary: Train Loss: { {k:f'{v:.4f}' for k,v in train_avg_loss.items()} } | Val Loss: { {k:f'{v:.4f}' for k,v in val_avg_loss.items()} }\n")

        torch.save(model.state_dict(), os.path.join(config.model_save_path, f'model_epoch_{epoch+1}.pth'))
        save_training_results(epoch + 1, target_imgs_val, recon_imgs_val_norm, phase_val, amp_val, val_avg_loss, config.train_results_path)

if __name__ == '__main__':
    train(Config())