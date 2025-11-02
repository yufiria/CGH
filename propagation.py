# -*- coding: utf-8 -*-
"""
光学传播物理模型模块

该文件实现了基于角谱法(Angular Spectrum Method, ASM)的光学传播模型。
"""
import torch
import torch.nn as nn
import numpy as np

class ComplexWavePropagator(nn.Module):
    """使用角谱法(ASM)实现光波传播的模块。"""
    def __init__(self, resolution, pixel_size, distance, wavelength):
        super().__init__()
        self.H, self.W = resolution
        self.pixel_size = pixel_size
        self.distance = distance
        self.wavelength = wavelength
        self.register_buffer('H_tf', self._precompute_transfer_function())

    def _precompute_transfer_function(self):
        fx = torch.fft.fftfreq(self.W, d=self.pixel_size); fy = torch.fft.fftfreq(self.H, d=self.pixel_size)
        fxx, fyy = torch.meshgrid(fy, fx, indexing='ij')
        k = 2 * np.pi / self.wavelength
        argument = (k**2 - (2 * np.pi * fxx)**2 - (2 * np.pi * fyy)**2).clamp(min=0)
        return torch.exp(1j * self.distance * torch.sqrt(argument))

    def forward(self, complex_field):
        fft_field = torch.fft.fft2(complex_field)
        fft_field_propagated = fft_field * self.H_tf
        return torch.fft.ifft2(fft_field_propagated)