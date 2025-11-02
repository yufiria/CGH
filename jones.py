# -*- coding: utf-8 -*-
"""
琼斯矩阵计算模块

该文件提供了从四个基本偏振通道的出射光场反解出每个像素的琼斯矩阵的函数。
"""
import os
import numpy as np

def load_field_npz(path_npz: str) -> np.ndarray:
    if not os.path.exists(path_npz): raise FileNotFoundError(f"文件不存在: {path_npz}")
    data = np.load(path_npz, allow_pickle=True)
    if "field" not in data: raise ValueError(f"{path_npz} 中未找到 'field' 键。")
    return data["field"].astype(np.complex64, copy=False)

def build_jones_circular(H_LL, H_RL, H_LR, H_RR) -> np.ndarray:
    H, W = H_LL.shape
    J = np.empty((H, W, 2, 2), dtype=np.complex64)
    J[..., 0, 0] = H_LL; J[..., 1, 0] = H_RL; J[..., 0, 1] = H_LR; J[..., 1, 1] = H_RR
    return J

def circular_to_linear(J_circ: np.ndarray) -> np.ndarray:
    """[关键] 将琼斯矩阵从圆偏振基转换为线偏振基。"""
    factor = np.float32(1.0 / np.sqrt(2.0))
    T = np.array([[1, -1j], [1,  1j]], dtype=np.complex64) * factor
    T_inv = np.array([[1,  1], [1j, -1j]], dtype=np.complex64) * factor
    tmp = np.einsum('ab,...bc->...ac', T_inv, J_circ)
    J_lin = np.einsum('...ab,bc->...ac', tmp, T)
    return J_lin.astype(np.complex64, copy=False)

def compute_and_save_jones(scene_dir: str, out_prefix: str = "J"):
    paths = {tag: os.path.join(scene_dir, f"{tag}.npz") for tag in ["H_LL", "H_RL", "H_LR", "H_RR"]}
    fields = {tag: load_field_npz(p) for tag, p in paths.items()}
    J_circ = build_jones_circular(H_LL=fields["H_LL"], H_RL=fields["H_RL"], H_LR=fields["H_LR"], H_RR=fields["H_RR"])
    J_lin = circular_to_linear(J_circ)
    circ_path = os.path.join(scene_dir, f"{out_prefix}_circ.npz")
    lin_path = os.path.join(scene_dir, f"{out_prefix}_lin.npz")
    np.savez(circ_path, J=J_circ); np.savez(lin_path, J=J_lin)
    print(f"琼斯矩阵已保存至: {circ_path} 和 {lin_path}")