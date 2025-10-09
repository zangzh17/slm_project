# optics_utils.py

"""
Utility functions for optical calculations.
"""

import torch
import numpy as np
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Tuple, Optional
import math

def create_grid(L: float, N: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create coordinate grid"""
    x = torch.linspace(-L / 2, L / 2, N, device=device)
    return torch.meshgrid(x, x, indexing='ij')


def generate_spherical_wave(
    F: float, N: int, L: float, wavelength: float, device: torch.device
) -> torch.Tensor:
    """
    Generate spherical wave.

    Args:
        F (float): Radius of curvature of the spherical wave (focal length). F > 0 for diverging wave, F < 0 for converging wave.
        N (int): Number of grid points.
        L (float): Physical size of the grid.
        wavelength (float): Wavelength.
        device (torch.device): Computing device.

    Returns:
        torch.Tensor: Complex-valued spherical wave field.
    """
    X, Y = create_grid(L, N, device)
    k = 2 * np.pi / wavelength
    
    # Use the sign of F to determine diverging or converging wave
    r_sq = X**2 + Y**2 + F**2
    r = torch.sqrt(r_sq)
    
    phase_sign = 1.0 if F >= 0 else -1.0
    complex_phase = torch.exp(1j * phase_sign * k * r)
    
    # Avoid division by zero
    spherical_wave = complex_phase / torch.clamp(r, min=1e-12)
    return spherical_wave.to(torch.complex64)


def create_template_with_centers(
    N: int, centers: torch.Tensor, sigma_px: float, device: torch.device
) -> torch.Tensor:
    """Create Gaussian template based on given centers and sigma."""
    Y, X = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')
    gaussian_sum = torch.zeros((N, N), device=device)
    for center_x, center_y in centers:
        dist_sq = (X - center_x)**2 + (Y - center_y)**2
        gaussian_sum += torch.exp(-0.5 * dist_sq / (sigma_px**2))
    
    return gaussian_sum / gaussian_sum.sum() * N * N

def disparity_shift(
    radius_of_curvature: float, centers_pixel: torch.Tensor, 
    focal_plane_dist: float, pixel_size: float, N: int
) -> torch.Tensor:
    """Calculate target spot displacement caused by spherical wave incidence."""
    center_idx = (N - 1) / 2.0
    offsets_pix = centers_pixel - center_idx
    offsets_m = offsets_pix * pixel_size
    magnification = focal_plane_dist / radius_of_curvature
    shift_m = magnification * offsets_m
    shift_pix = shift_m / pixel_size
    return centers_pixel + shift_pix

def calculate_linear_phase(shape, angle_x_mrad, angle_y_mrad):
    """Calculate linear phase gradient for beam steering."""
    height, width = shape
    angle_x_rad = angle_x_mrad * 1e-3
    angle_y_rad = angle_y_mrad * 1e-3
    
    phase_gradient_x = (2 * np.pi / config.WAVELENGTH) * np.sin(angle_x_rad)
    phase_gradient_y = (2 * np.pi / config.WAVELENGTH) * np.sin(angle_y_rad)
    
    y_coords, x_coords = np.indices(shape)
    x_meters = (x_coords - width / 2) * config.PIXEL_SIZE
    y_meters = (y_coords - height / 2) * config.PIXEL_SIZE
    
    return phase_gradient_x * x_meters + phase_gradient_y * y_meters

def calculate_airy_disk(focal_length_m, aperture_width_m, wavelength):
    """Calculate diffraction-limited spot size (Airy disk diameter)."""
    f_number = focal_length_m / aperture_width_m
    airy_disk_diameter_m = 2.44 * wavelength * f_number
    return airy_disk_diameter_m

def create_checkerboard(shape):
    """Create checkerboard background pattern (0 and π)."""
    y, x = np.indices(shape)
    return np.pi * ((x + y) % 2)

def process_parameters(raw_params):
    """
    Takes a dictionary of raw parameters and returns a processed dictionary
    ready for simulation functions. This function is UI-independent.
    """
    # 复制一份以避免修改原始字典
    params = raw_params.copy()

    # --- 执行所有计算和逻辑处理 ---
    # 结合粗调和细调焦距
    params['focal_length'] = (params.get('focal_length_coarse', 0) + params.get('focal_length_fine', 0)) * 1e-3
    
    # 获取SLM尺寸，这里假设它在raw_params中
    slm_shape = params.get('shape', (1080, 1920)) # 提供一个默认值以防万一
    slm_height, slm_width = slm_shape

    # 计算ROI边界
    roi_size = params.get('N', 512) # 使用N作为ROI大小的来源
    roi_center_x = params.get('roi_center_x', slm_width // 2)
    roi_center_y = params.get('roi_center_y', slm_height // 2)

    roi_left = max(0, int(roi_center_x - roi_size // 2))
    roi_right = min(slm_width, int(roi_center_x + roi_size // 2))
    roi_top = max(0, int(roi_center_y - roi_size // 2))
    roi_bottom = min(slm_height, int(roi_center_y + roi_size // 2))

    # 确保ROI是正方形
    actual_roi_width = roi_right - roi_left
    actual_roi_height = roi_bottom - roi_top
    params['N'] = min(actual_roi_width, actual_roi_height)
    params['roi_rect'] = (roi_left, roi_top, params['N'], params['N'])

    return params