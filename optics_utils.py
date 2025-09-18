# optics_utils.py

"""
光学计算相关的工具函数。
"""

import torch
import numpy as np
import config

def create_grid(L: float, N: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """创建坐标网格"""
    x = torch.linspace(-L / 2, L / 2, N, device=device)
    return torch.meshgrid(x, x, indexing='ij')

def generate_spherical_wave(
    F: float, N: int, L: float, wavelength: float, device: torch.device
) -> torch.Tensor:
    """
    生成球面波。

    Args:
        F (float): 球面波的曲率半径 (焦距)。F > 0 为发散波, F < 0 为汇聚波。
        N (int): 网格点数。
        L (float): 网格物理尺寸。
        wavelength (float): 波长。
        device (torch.device): 计算设备。

    Returns:
        torch.Tensor: 复数形式的球面波场。
    """
    X, Y = create_grid(L, N, device)
    k = 2 * np.pi / wavelength
    
    # 使用 F 的符号来决定是发散波还是汇聚波
    r_sq = X**2 + Y**2 + F**2
    r = torch.sqrt(r_sq)
    
    phase_sign = 1.0 if F >= 0 else -1.0
    complex_phase = torch.exp(1j * phase_sign * k * r)
    
    # 避免除以零
    spherical_wave = complex_phase / torch.clamp(r, min=1e-12)
    return spherical_wave.to(torch.complex64)

def create_gaussian_template(
    N: int, L: float, focal_length: float, wavelength: float, M: int, 
    overlap_ratio: float, size_factor: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """创建目标高斯光斑阵列模板。"""
    pixel_size = L / N
    sub_aperture_L = L / M
    r_airy = 1.22 * wavelength * focal_length * size_factor / sub_aperture_L
    sigma = 0.42 * r_airy
    sigma_px = sigma / pixel_size

    print(f"衍射极限参数: Airy半径 = {r_airy/pixel_size:.2f} px, 高斯宽度 σ = {sigma_px:.2f} px")

    Y, X = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')

    region_size_norm = 1.0 / (M - (M - 1) * overlap_ratio)
    stride_norm = region_size_norm * (1 - overlap_ratio)
    
    centers_pixel = []
    gaussian_sum = torch.zeros((N, N), device=device)

    for i in range(M):
        for j in range(M):
            cx_norm = (i * stride_norm) + region_size_norm / 2
            cy_norm = (j * stride_norm) + region_size_norm / 2
            cx = cx_norm * (N - 1)
            cy = cy_norm * (N - 1)
            centers_pixel.append((cx, cy))
            dist_sq = (X - cx)**2 + (Y - cy)**2
            gaussian_sum += torch.exp(-0.5 * dist_sq / (sigma_px**2))
    
    # 归一化模板
    normalized_gaussian = gaussian_sum / gaussian_sum.sum() * N * N
    return normalized_gaussian, torch.tensor(centers_pixel, device=device), sigma_px

def create_template_with_centers(
    N: int, centers: torch.Tensor, sigma_px: float, device: torch.device
) -> torch.Tensor:
    """根据给定的中心点和sigma创建高斯模板。"""
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
    """计算球面波入射引起的目标光斑位移。"""
    center_idx = (N - 1) / 2.0
    offsets_pix = centers_pixel - center_idx
    offsets_m = offsets_pix * pixel_size
    magnification = focal_plane_dist / radius_of_curvature
    shift_m = magnification * offsets_m
    shift_pix = shift_m / pixel_size
    return centers_pixel + shift_pix

def calculate_linear_phase(shape, angle_x_mrad, angle_y_mrad):
    """计算用于光束偏转的线性相位梯度。"""
    height, width = shape
    angle_x_rad = angle_x_mrad * 1e-3
    angle_y_rad = angle_y_mrad * 1e-3
    
    phase_gradient_x = (2 * np.pi / config.WAVELENGTH) * np.sin(angle_x_rad)
    phase_gradient_y = (2 * np.pi / config.WAVELENGTH) * np.sin(angle_y_rad)
    
    y_coords, x_coords = np.indices(shape)
    x_meters = (x_coords - width / 2) * config.PIXEL_SIZE
    y_meters = (y_coords - height / 2) * config.PIXEL_SIZE
    
    return phase_gradient_x * x_meters + phase_gradient_y * y_meters

def calculate_airy_disk(focal_length_m, aperture_width_m):
    """计算衍射极限光斑尺寸（艾里斑直径）。"""
    f_number = focal_length_m / aperture_width_m
    airy_disk_diameter_m = 2.44 * config.WAVELENGTH * f_number
    return airy_disk_diameter_m * 1e6  # 返回微米

def create_checkerboard(shape):
    """创建棋盘格背景图案 (0 和 π)。"""
    y, x = np.indices(shape)
    return np.pi * ((x + y) % 2)