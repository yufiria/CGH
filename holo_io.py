# -*- coding: utf-8 -*-
"""
全息图数据输入/输出模块
"""
import os
import json
import numpy as np
import torch

def _to_numpy(x):
    if x is None: return None
    if isinstance(x, torch.Tensor): x = x.detach().cpu()
    return np.array(x)

def save_hologram_fields(save_dir, tag, phase, amplitude, reconstructed=None, meta: dict=None):
    os.makedirs(save_dir, exist_ok=True)
    phase_np = _to_numpy(phase).squeeze().astype(np.float32)
    amp_np = _to_numpy(amplitude).squeeze().astype(np.float32)
    field_np = (amp_np * np.exp(1j * phase_np)).astype(np.complex64)
    pack = {"phase": phase_np, "amplitude": amp_np, "field": field_np}
    if reconstructed is not None: pack["reconstructed"] = _to_numpy(reconstructed).squeeze()
    if meta is not None: pack["meta_json"] = np.array(json.dumps(meta), dtype=object)
    
    out_path = os.path.join(save_dir, f"{tag}.npz")
    np.savez(out_path, **pack)
    if meta is not None:
        with open(os.path.join(save_dir, f"{tag}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)