# -*- coding: utf-8 -*-
"""
推理及琼斯矩阵计算脚本
"""
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from config import Config
from holo_io import save_hologram_fields
from jones import compute_and_save_jones

def load_model(config, model_path):
    device = torch.device(config.device)
    if config.model_name == 'AttentionUNet': from advanced_model import AttentionUNet as Model
    else: from model import HologramGenerator as Model
    model = Model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型 ({config.model_name}) 已从 {model_path} 加载。")
    return model

def create_target_image(text, color, img_size):
    image = Image.new('RGB', (img_size, img_size), (0, 0, 0)); draw = ImageDraw.Draw(image)
    try: font = ImageFont.truetype("arial.ttf", size=int(img_size * 0.8))
    except IOError: font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    pos = ((img_size - (text_bbox[2]-text_bbox[0])) / 2, (img_size - (text_bbox[3]-text_bbox[1])) / 2 - 2)
    draw.text(pos, text, fill=color, font=font)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])(image).unsqueeze(0)

def denormalize(tensor):
    return (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0

def main():
    config = Config()
    device = torch.device(config.device)
    model_path = os.path.join(config.model_save_path, f'model_epoch_{config.epochs}.pth')
    if not os.path.exists(model_path): raise FileNotFoundError(f"未找到模型: {model_path}")
    model = load_model(config, model_path)

    targets = {"H_LL": ("L", (255, 255, 255)), "H_LR": ("R", (255, 0, 0)), "H_RL": ("G", (0, 255, 0)), "H_RR": ("B", (0, 0, 255))}
    scene_dir = os.path.join(config.inference_results_path, "jones_scene_1"); os.makedirs(scene_dir, exist_ok=True)
    print(f"\n开始为4个偏振通道生成全息图，结果将保存在: {scene_dir}")

    with torch.no_grad():
        for tag, (text, color) in targets.items():
            print(f"  处理通道: {tag}...")
            target_image = create_target_image(text, color, config.img_size).to(device)
            phase, amplitude = model(target_image)
            save_hologram_fields(save_dir=scene_dir, tag=tag, phase=phase, amplitude=amplitude, meta={"target_text": text, "target_color": color})
            
            plt.figure(figsize=(8, 4)); plt.subplot(1, 2, 1); plt.imshow(denormalize(target_image[0]).permute(1, 2, 0).cpu().numpy()); plt.title(f'目标: {tag}'); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(phase[0, 0].cpu().numpy(), cmap='hsv'); plt.title('相位'); plt.axis('off')
            plt.savefig(os.path.join(scene_dir, f'result_{tag}.png')); plt.close()

    print("\n所有通道的全息图已生成。开始计算琼斯矩阵...")
    compute_and_save_jones(scene_dir=scene_dir, out_prefix="J_matrix")

if __name__ == '__main__':
    main()