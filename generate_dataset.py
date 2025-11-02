# -*- coding: utf-8 -*-
"""
为彩色全息超表面项目生成合成数据集。

该脚本创建包含多个随机颜色和形状的图像，用于训练神经网络
从目标图像反演超表面所需的出射光场。
"""
import os
import json
import random
import math
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

class DatasetGenConfig:
    IMG_SIZE = (32, 32)
    IMG_MODE = 'RGB'
    DATASET_SPLITS = {"train": 10000, "val": 1000}
    OUTPUT_DIR = './synthetic_dataset'
    SHAPE_TYPES = ['circle', 'rectangle', 'triangle']
    SHAPES_PER_IMAGE = (1, 4)
    SHAPE_SIZE_RANGE = (8, 16)
    COLORS = {
        "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
        "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
        "white": (255, 255, 255),
    }

def draw_shape(draw, shape_type, position, size, color, rotation_angle=0):
    x, y = position
    half_size = size / 2
    if shape_type == 'circle':
        bbox = [x - half_size, y - half_size, x + half_size, y + half_size]
        draw.ellipse(bbox, fill=color)
    elif shape_type in ['rectangle', 'triangle']:
        if shape_type == 'rectangle':
            points = [(-half_size, -half_size), (half_size, -half_size), (half_size, half_size), (-half_size, half_size)]
        else: # triangle
            p1 = (0, -2 * half_size / (3**0.5)); p2 = (-half_size, half_size / (3**0.5)); p3 = (half_size, half_size / (3**0.5))
            points = [p1, p2, p3]
        
        angle_rad = math.radians(rotation_angle)
        rotated_points = []
        for px, py in points:
            new_px = x + px * math.cos(angle_rad) - py * math.sin(angle_rad)
            new_py = y + px * math.sin(angle_rad) + py * math.cos(angle_rad)
            rotated_points.append((new_px, new_py))
        draw.polygon(rotated_points, fill=color)

def generate_image(config):
    image = Image.new(config.IMG_MODE, config.IMG_SIZE, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    metadata = {"image_size": config.IMG_SIZE, "shapes": []}
    num_shapes = random.randint(*config.SHAPES_PER_IMAGE)
    for _ in range(num_shapes):
        shape_type = random.choice(config.SHAPE_TYPES)
        shape_size = random.randint(*config.SHAPE_SIZE_RANGE)
        margin = shape_size // 2 + 2
        pos_x = random.randint(margin, config.IMG_SIZE[0] - margin)
        pos_y = random.randint(margin, config.IMG_SIZE[1] - margin)
        color_name, color_rgb = random.choice(list(config.COLORS.items()))
        rotation = random.randint(0, 359)
        draw_shape(draw, shape_type, (pos_x, pos_y), shape_size, color_rgb, rotation)
        metadata["shapes"].append({ "type": shape_type, "position": (pos_x, pos_y), "size": shape_size, "color_name": color_name, "color_rgb": color_rgb, "rotation_degrees": rotation })
    return image, metadata

def main():
    config = DatasetGenConfig()
    print("开始生成合成数据集...")
    for split, num_images in config.DATASET_SPLITS.items():
        print(f"\n正在生成 '{split}' 数据集 ({num_images} 张图像)...")
        image_dir = os.path.join(config.OUTPUT_DIR, split, 'images')
        label_dir = os.path.join(config.OUTPUT_DIR, split, 'labels')
        os.makedirs(image_dir, exist_ok=True); os.makedirs(label_dir, exist_ok=True)
        for i in tqdm(range(num_images), desc=f"Generating {split}"):
            image, metadata = generate_image(config)
            filename_base = f"{i:05d}"
            image.save(os.path.join(image_dir, f"{filename_base}.png"))
            with open(os.path.join(label_dir, f"{filename_base}.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
    print(f"\n数据集生成完成！保存在: {os.path.abspath(config.OUTPUT_DIR)}")

if __name__ == '__main__':
    main()