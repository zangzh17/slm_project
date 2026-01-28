# optics_utils.py

"""
Utility functions for optical calculations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Tuple, Optional
import math
import json
import os
from datetime import datetime
from scipy.io import savemat

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

def calculate_airy_disk(focal_length_m, aperture_width_m, wavelength):
    """Calculate diffraction-limited spot size (Airy disk diameter)."""
    f_number = focal_length_m / aperture_width_m
    airy_disk_diameter_m = 2.44 * wavelength * f_number
    return airy_disk_diameter_m

def create_checkerboard(shape):
    """Create checkerboard background pattern (0 and π)."""
    y, x = np.indices(shape)
    return np.pi * ((x + y) % 2)


def compute_psf_centers(
    M: int,
    overlap_ratio: float,
    center_blend: float,
    z_ratio: float = 1.0,
    N: int = 512,
    output_size = None,
    device: torch.device = torch.device('cpu'),
    randomness: float = 0.0,
    random_seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    计算不同传播距离下的PSF中心位置。

    Parameters
    ----------
    M : int
        透镜阵列维度 (M x M)
    overlap_ratio : float
        重叠比例 [0, 1)
    center_blend : float
        中心混合参数 [0, 1]，控制无重叠和重叠几何的混合
    z_ratio : float
        传播距离比例 [0, 1]
        - 0.0: 透镜平面（子透镜的几何中心）
        - 1.0: 焦平面（考虑重叠后的PSF中心）
    N : int
        输入像素数量
    output_size: int
        输出像素数目，不输入则和输入像素一样
    device : torch.device
        计算设备
    randomness : float
        随机化因子 [0, 1]，控制PSF位置的随机偏移程度
        - 0.0: 无随机偏移（默认，保持均匀周期分布）
        - >0: PSF位置施加高斯随机偏移，标准差 = randomness × pitch
    random_seed : int, optional
        随机种子，用于确保可复现性

    Returns
    -------
    Dict[str, torch.Tensor]
        - 'centers_pixel': PSF中心的像素坐标 [M*M, 2]
        - 'centers_geom_pixel': 几何中心的像素坐标 [M*M, 2]
        - 'scale': 输出像素坐标的缩放系数
        - 'ratio_norm': 输入归一化坐标到输出归一化坐标的缩放系数
        - 'shift_norm': 像素坐标到输出坐标的归一化平移量
        - 'region_size_norm': 归一化的区域大小
        - 'stride_norm': 归一化的步长
        - 'random_offsets': 随机偏移量 [M*M, 2]（仅当randomness > 0时）
    """
    # 归一化几何参数（透镜坐标）
    region_size_norm = 1.0 / (M - (M - 1) * overlap_ratio)
    stride_norm = region_size_norm * (1.0 - overlap_ratio)
    # 注意输出是output坐标
    if output_size is None:
        output_size = N

    scale = output_size - 1
    shift_norm = float(output_size - N) / output_size /2
    ratio_norm = float(N-1)/float(output_size-1)
    
    # 创建归一化索引
    i_idx = torch.arange(M, device=device, dtype=torch.float32)
    j_idx = torch.arange(M, device=device, dtype=torch.float32)
    
    # 无重叠时的归一化均匀网格中心 （输出坐标）
    cx_uniform_norm = (i_idx + 0.5) / M
    cy_uniform_norm = (j_idx + 0.5) / M
    CX_uniform, CY_uniform = torch.meshgrid(cx_uniform_norm, cy_uniform_norm, indexing='ij')
    
    # 归一化重叠透镜几何的中心 （透镜坐标）
    cx_overlap_norm = i_idx * stride_norm + region_size_norm / 2.0
    cy_overlap_norm = j_idx * stride_norm + region_size_norm / 2.0
    CX_overlap, CY_overlap = torch.meshgrid(cx_overlap_norm, cy_overlap_norm, indexing='ij')
    # 如输出坐标和输入不相等，则需要转换到输出归一化坐标
    if output_size != N:
        CX_overlap = CX_overlap * ratio_norm + shift_norm
        CY_overlap = CY_overlap * ratio_norm + shift_norm
    
    # 根据center_blend插值（控制无重叠和重叠几何的混合，决定PSF中心）（输出坐标）
    t_blend = float(max(0.0, min(1.0, center_blend)))
    CX_blended = (1.0 - t_blend) * CX_uniform + t_blend * CX_overlap
    CY_blended = (1.0 - t_blend) * CY_uniform + t_blend * CY_overlap

    # 根据z_ratio进一步插值（控制传播距离）
    # z_ratio = 0，透镜中心
    # z_ratio = 1，焦平面
    z_ratio = float(max(0.0, min(1.0, z_ratio)))
    CX_final = (1.0 - z_ratio) * CX_overlap + z_ratio * CX_blended
    CY_final = (1.0 - z_ratio) * CY_overlap + z_ratio * CY_blended
    CX_final = CX_final.clamp(0.0, 1.0)
    CY_final = CY_final.clamp(0.0, 1.0)
    
    # 归一化坐标转换为像素坐标，到达输出面
    centers_pixel = torch.stack([
        (CX_final * scale).reshape(-1),
        (CY_final * scale).reshape(-1)
    ], dim=-1)

    # 几何中心（始终使用重叠几何，用于透镜覆盖计算）
    centers_geom_pixel = torch.stack([
        (CX_overlap * scale).reshape(-1),
        (CY_overlap * scale).reshape(-1)
    ], dim=-1)

    # 应用随机偏移
    random_offsets = None
    if randomness > 0:
        # 计算pitch（相邻PSF间距，像素单位）
        pitch = output_size / M
        # 标准差 = randomness × pitch
        sigma = randomness * pitch

        # 设置随机种子（如果提供）
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # 生成高斯随机偏移 [M*M, 2]
        random_offsets = torch.randn(M * M, 2, device=device, dtype=torch.float32) * sigma

        # 应用偏移到中心坐标
        centers_pixel = centers_pixel + random_offsets

        # 确保中心坐标在有效范围内
        centers_pixel = centers_pixel.clamp(0.0, scale)

    result = {
        'centers_pixel': centers_pixel, # 输出坐标
        'centers_geom_pixel': centers_geom_pixel, # 输出坐标
        'scale': scale, # 输出坐标
        'shift_norm': shift_norm, #输出归一化坐标
        'ratio_norm': ratio_norm,   # 归一化输入->归一化输出转换
        'region_size_norm': region_size_norm, # 透镜坐标
        'stride_norm': stride_norm # 透镜坐标
    }

    if random_offsets is not None:
        result['random_offsets'] = random_offsets

    return result


def assign_tile_group(
    kx: int, 
    ky: int, 
    interleaving: str,
    mask_count: int, 
    coarse_grid_size: int
) -> int:
    """
    根据interleaving策略为tile分配组号。
    
    Parameters
    ----------
    kx, ky : int
        Tile在网格中的位置
    interleaving : str
        交错策略（"checkerboard", "coarse1", "coarse2", "coarse3"）
    mask_count : int
        组的总数
    coarse_grid_size : int
        粗网格大小
        
    Returns
    -------
    int
        组号（0 到 mask_count-1）
    """
    if interleaving == "checkerboard":
        return (kx + ky) % mask_count
        
    elif interleaving.startswith("coarse"):
        super_tile_x = kx // coarse_grid_size
        super_tile_y = ky // coarse_grid_size
        local_x = kx % coarse_grid_size
        local_y = ky % coarse_grid_size
        tiles_per_super = coarse_grid_size * coarse_grid_size
        
        if interleaving == "coarse3":
            local_index = local_y * coarse_grid_size + local_x
            if mask_count <= tiles_per_super:
                group = local_index % mask_count
            else:
                group = (local_index + (super_tile_x + super_tile_y) * tiles_per_super) % mask_count
            
            # 奇数super tiles反转组号
            if (super_tile_x + super_tile_y) % 2 == 1:
                group = (mask_count - 1 - group) % mask_count
            return group
            
        elif interleaving == "coarse2":
            local_index = local_y * coarse_grid_size + local_x
            if mask_count <= tiles_per_super:
                group = local_index % mask_count
            else:
                group = (local_index + (super_tile_x + super_tile_y) * tiles_per_super) % mask_count
            return group
            
        elif interleaving == "coarse1":
            local_index = local_y + local_x
            return local_index % mask_count
    
    # 默认fallback
    return (kx + ky) % mask_count


def generate_tile_masks(
    M: int,
    L: float,
    overlap_ratio: float,
    center_blend: float,
    mask_count: int,
    interleaving: str,
    N: int = 512,
    output_size: int = 512,
    coarse_grid_size: int = 2,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    生成tile布局和对应的像素级mask。
    
    Parameters
    ----------
    M : int
        透镜阵列维度
    L : float
        透镜物理尺寸
    overlap_ratio : float
        重叠比例
    center_blend : float
        中心混合参数
    mask_count : int
        mask数量
    interleaving : str
        交错策略
    N : int
        透镜像素数量
    output_size: int
        输出PSF面像素数量
    coarse_grid_size : int
        粗网格大小
    device : torch.device
        计算设备
        
    Returns
    -------
    Dict[str, Any]
        - 'masks': 像素级mask张量 [mask_count, N, N]
        - 'tiles': tile信息列表
        - 'a_lens_mask': 每个透镜在每个mask中的面积占比 [M*M, mask_count]
        - 'lens_mask': 每个透镜的覆盖区域 [M*M, N, N]
    """
    # 验证参数
    assert mask_count >= 2
    if mask_count > coarse_grid_size**2:
        print(f'coarse_grid_size={coarse_grid_size} is too small for current mask_count.')
        coarse_grid_size = int(math.ceil(math.sqrt(mask_count)))
        print(f'Use {coarse_grid_size} instead.')
    
    # 获取几何参数(已为输出坐标)
    center_info = compute_psf_centers(M, overlap_ratio, 
                                      center_blend, 
                                      N=N,
                                      output_size=output_size,
                                      device=device)
    region_size_norm = center_info['region_size_norm']
    stride_norm = center_info['stride_norm']
    scale = center_info['scale']
    shift_norm = center_info['shift_norm']
    ratio_norm = center_info['ratio_norm']
    centers_geom_pixel = center_info['centers_geom_pixel']
    
    # ========== 生成tile边界：归一化透镜坐标 ==========
    lens_boundaries_x = set()
    lens_boundaries_y = set()
    
    for i in range(M):
        x_start = i * stride_norm
        x_end = min(x_start + region_size_norm, 1.0)
        lens_boundaries_x.add(x_start)
        lens_boundaries_x.add(x_end)
    
    for j in range(M):
        y_start = j * stride_norm
        y_end = min(y_start + region_size_norm, 1.0)
        lens_boundaries_y.add(y_start)
        lens_boundaries_y.add(y_end)
    
    lens_boundaries_x.update([0.0, 1.0])
    lens_boundaries_y.update([0.0, 1.0])
    
    norm_edges_x = sorted(lens_boundaries_x)
    norm_edges_y = sorted(lens_boundaries_y)
    
    # ========== 生成tiles并分配到组：归一化透镜坐标 （可用于画图） ==========
    tiles = []
    for kx in range(len(norm_edges_x) - 1):
        x_start_norm = norm_edges_x[kx]
        x_end_norm = norm_edges_x[kx + 1]
        
        if x_end_norm - x_start_norm < 1e-6:
            continue
            
        for ky in range(len(norm_edges_y) - 1):
            y_start_norm = norm_edges_y[ky]
            y_end_norm = norm_edges_y[ky + 1]
            
            if y_end_norm - y_start_norm < 1e-6:
                continue
            
            tile_width = (x_end_norm - x_start_norm) * L
            tile_height = (y_end_norm - y_start_norm) * L
            tile_area = tile_width * tile_height
            
            # 找出贡献的透镜
            contributing_lenses = []
            for ii in range(M):
                for jj in range(M):
                    lens_x_start = ii * stride_norm
                    lens_x_end = min(lens_x_start + region_size_norm, 1.0)
                    lens_y_start = jj * stride_norm
                    lens_y_end = min(lens_y_start + region_size_norm, 1.0)
                    
                    if (x_start_norm < lens_x_end and x_end_norm > lens_x_start and
                        y_start_norm < lens_y_end and y_end_norm > lens_y_start):
                        contributing_lenses.append((ii, jj))
            
            # 分配到mask组
            group = assign_tile_group(kx, ky, interleaving, mask_count, coarse_grid_size)
            
            tiles.append({
                'x_start_norm': x_start_norm,
                'x_end_norm': x_end_norm,
                'y_start_norm': y_start_norm,
                'y_end_norm': y_end_norm,
                'x_start_px': x_start_norm * scale,
                'x_end_px': x_end_norm * scale,
                'y_start_px': y_start_norm * scale,
                'y_end_px': y_end_norm * scale,
                'area': tile_area,
                'lenses': contributing_lenses,
                'grid_kx': kx,
                'grid_ky': ky,
                'group': group,
                'num_lenses': len(contributing_lenses)
            })
    
    # ========== 创建output_size像素级masks ==========
    # 创建输出坐标网格
    Y, X = torch.meshgrid(
        torch.arange(output_size, device=device, dtype=torch.float32),
        torch.arange(output_size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    # 从透镜归一化坐标转换到输出归一化坐标 (t * ratio_norm + shift_norm)
    # 再通过scale转换到输出坐标output_size
    x_edges_px = torch.tensor([(t * ratio_norm + shift_norm) * scale for t in norm_edges_x], device=device, dtype=torch.float32)
    y_edges_px = torch.tensor([(t * ratio_norm + shift_norm) * scale for t in norm_edges_y], device=device, dtype=torch.float32)
    
    x_bins = torch.searchsorted(x_edges_px[1:], X.reshape(-1), right=False).reshape(output_size, output_size)
    y_bins = torch.searchsorted(y_edges_px[1:], Y.reshape(-1), right=False).reshape(output_size, output_size)
    
    masks = torch.zeros((mask_count, output_size, output_size), device=device, dtype=torch.bool)

    for tile in tiles:
        kx = tile['grid_kx']
        ky = tile['grid_ky']
        group = tile['group']
        
        tile_mask = (x_bins == kx) & (y_bins == ky)
        masks[group] |= tile_mask
    
    # ========== 计算透镜覆盖区域和面积占比 ==========
    num_lenses = M * M
    # 从归一化透镜坐标换算为归一化输出坐标，再换算为实际输出坐标，再计算半径
    half_w = region_size_norm * ratio_norm * scale / 2.0
    
    X2 = X.unsqueeze(0)
    Y2 = Y.unsqueeze(0)
    cxg = centers_geom_pixel[:, 0].view(num_lenses, 1, 1)
    cyg = centers_geom_pixel[:, 1].view(num_lenses, 1, 1)
    
    lens_mask = (X2 - cxg).abs() <= half_w
    lens_mask &= (Y2 - cyg).abs() <= half_w
    
    inter_counts = (lens_mask[:, None] & masks[None, :]).sum(dim=(2, 3)).to(torch.float32)
    lens_counts = lens_mask.view(num_lenses, -1).sum(dim=1).clamp(min=1).to(torch.float32)
    a_lens_mask = inter_counts / lens_counts[:, None]
    
    return {
        'masks': masks, # 输出坐标网络
        'tiles': tiles, # 归一化坐标
        'a_lens_mask': a_lens_mask, # 归一化比例
        'lens_mask': lens_mask # 输出坐标网络
    }


def generate_gaussian_psf(
    centers_pixel: torch.Tensor,
    N: int,
    L: float,
    M: int,
    overlap_ratio: float,
    focal_length: float,
    wavelength: float,
    airy_correction: float,
    masked_airy_correction: float,
    mask_psf_type: str = 'gaussian',
    masks: Optional[torch.Tensor] = None,
    a_lens_mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """
    生成高斯PSF。
    
    Parameters
    ----------
    centers_pixel : torch.Tensor
        PSF中心的像素坐标 [M*M, 2]
    N : int
        像素数量
    L : float
        物理尺寸
    M : int
        透镜阵列维度
    overlap_ratio : float
        重叠比例
    focal_length : float
        焦距
    wavelength : float
        波长
    airy_correction : float
        Airy校正因子
    mask_psf_type : str
        PSF类型，'gaussian' 或 'disk'
    masks : torch.Tensor, optional
        像素级mask [mask_count, N, N]
    a_lens_mask : torch.Tensor, optional
        透镜在各mask中的面积占比 [M*M, mask_count]
    normalize : bool
        是否归一化
    device : torch.device
        计算设备
        
    Returns
    -------
    Dict[str, torch.Tensor]
        - 'total_psf': 总PSF [N, N]
        - 'mask_psfs': 各mask的PSF [mask_count, N, N] 如果提供了masks
    """
    num_lenses = centers_pixel.shape[0]
    assert num_lenses == M * M
    assert mask_psf_type in ['gaussian', 'disk'], f"mask_psf_type must be 'gaussian' or 'disk', got {psf_type}"
    
    
    # 创建坐标网格
    Y, X = torch.meshgrid(
        torch.arange(N, device=device, dtype=torch.float32),
        torch.arange(N, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # 计算高斯参数
    pixel_size = L / N
    region_size_norm = 1.0 / (M - (M - 1) * overlap_ratio)
    D_eff = L * region_size_norm
    r_airy = 1.22 * wavelength * focal_length * airy_correction / D_eff
    r_airy_px = float(r_airy / pixel_size)
    # 高斯PSF
    sigma = 0.42 * r_airy
    sigma_px = float(sigma / pixel_size)
    inv_two_sigma2 = 0.5 / (sigma_px ** 2 + 1e-20)
    # Gaussian PSF
    psf_sum_total = torch.zeros((N, N), device=device, dtype=torch.float32)
    for l in range(num_lenses):
        cx, cy = centers_pixel[l, 0], centers_pixel[l, 1]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        g = torch.exp(-dist_sq * inv_two_sigma2)
        psf_sum_total += g
    # mask PSF
    if masks is not None:
        r_airy = r_airy * masked_airy_correction
        r_airy_px = float(r_airy / pixel_size)
        inv_two_sigma2 = 0.5 / (sigma_px ** 2 + 1e-20)
        mask_count = masks.shape[0]
        psf_sum_masks = torch.zeros((mask_count, N, N), device=device, dtype=torch.float32)

        if mask_psf_type == 'gaussian':
            for l in range(num_lenses):
                cx, cy = centers_pixel[l, 0], centers_pixel[l, 1]
                dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
                g = torch.exp(-dist_sq * inv_two_sigma2)
                if a_lens_mask is not None:
                    weights = a_lens_mask[l].view(mask_count, 1, 1)
                    psf_sum_masks += weights * g
        elif mask_psf_type == 'disk':
            # 圆盘PSF
            for l in range(num_lenses):
                cx, cy = centers_pixel[l, 0], centers_pixel[l, 1]
                dist = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                disk = (dist <= r_airy_px).float()
                if a_lens_mask is not None:
                    weights = a_lens_mask[l].view(mask_count, 1, 1)
                    psf_sum_masks += weights * disk
    # 归一化
    result = {}
    if normalize:
        denom = psf_sum_total.sum().clamp_min(1e-12)
        result['total_psf'] = psf_sum_total / denom * (N * N)
        if masks is not None:
            result['mask_psfs'] = psf_sum_masks / denom * (N * N)
    else:
        result['total_psf'] = psf_sum_total
        if masks is not None:
            result['mask_psfs'] = psf_sum_masks
    
    return result



def generate_lens_circular_masks(
    centers_pixel: torch.Tensor,
    radii_pixels: torch.Tensor,
    N: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    为每个深度和每个透镜中心生成圆形mask。
    
    Parameters
    ----------
    centers_pixel : torch.Tensor
        中心坐标 [num_depths, num_lenses, 2]
    radii_pixels : torch.Tensor
        每个深度对应的半径 [num_depths]
    N : int
        图像尺寸
    device : torch.device
        计算设备
        
    Returns
    -------
    torch.Tensor
        圆形mask [num_depths, num_lenses, N, N]
        每个mask是以对应中心为圆心、对应半径的圆形区域
    """
    num_depths, num_lenses, _ = centers_pixel.shape
    
    # 创建坐标网格
    Y, X = torch.meshgrid(
        torch.arange(N, device=device, dtype=torch.float32),
        torch.arange(N, device=device, dtype=torch.float32),
        indexing='ij'
    )
    # [1, 1, N, N]
    X = X.unsqueeze(0).unsqueeze(0)
    Y = Y.unsqueeze(0).unsqueeze(0)
    
    # [num_depths, num_lenses, 1, 1]
    cx = centers_pixel[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    cy = centers_pixel[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    
    # [num_depths, 1, 1, 1]
    radii = radii_pixels.view(num_depths, 1, 1, 1)
    
    # 计算距离并生成圆形mask
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    circular_masks = dist_sq <= (radii ** 2)
    
    return circular_masks


def save_dict_as_json(data: dict, filename: str = None):
    # Create filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"data_{timestamp}.json"
    if not filename.lower().endswith('.json'):
        filename += '.json' 
    # Write the dictionary to the JSON file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"JSON file saved to: {filename}")

def load_dict_from_json(filename: str) -> Dict:
    if not filename.lower().endswith('.json'):
        filename += '.json' 
    with open(filename, 'r') as json_file:
        params = json.load(json_file)
    return params

def save_array(array, filename=None):
    """
    Save a numpy array to disk.
    
    Args:
        array: numpy array to save
        filename: filename (default: timestamp)
    
    Returns:
        str: full path to saved file
    """
    
    # Generate filename from timestamp if not provided
    if filename is None:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + '.npy'
    
    if not filename.lower().endswith('.npy'):
        filename += '.npy'    
    
    # Save array
    np.save(filename, array)
    print(f"Array saved to: {filename}")
    return filename

def load_array(filepath):
    """
    Load a numpy array from disk.
    
    Args:
        filepath: full path to the .npy file
    
    Returns:
        numpy array
    """
    if not filepath.lower().endswith('.npy'):
        filepath += '.npy'
    
    array = np.load(filepath)
    print(f"Array loaded from: {filepath}")
    print(f"Shape: {array.shape}, dtype: {array.dtype}")
    
    return array


def save_pupil_to_mat(pupil_array, filename, pupil_extent=None, pupil_pixel_size=None):
    """
    将pupil function保存为MATLAB .mat文件
    
    参数:
        pupil_array: [M*M, H, W] 的numpy数组
        filename: 输出文件名（.mat）
        pupil_extent: pupil的空间范围（可选，用于保存元数据）
        pupil_pixel_size: pupil的采样间距（可选，用于保存元数据）
    """
    # 转换维度从 [M*M, H, W] 到 [H, W, M*M] (MATLAB格式)
    pupil_matlab = np.transpose(pupil_array, (1, 2, 0))
    
    # 准备保存的数据
    save_dict = {
        'pupil_function': pupil_matlab,
        'num_lenslets': pupil_array.shape[0],
        'pupil_size': pupil_array.shape[1:],
    }
    
    if pupil_extent is not None:
        save_dict['pupil_extent'] = pupil_extent
    if pupil_pixel_size is not None:
        save_dict['pupil_pixel_size'] = pupil_pixel_size
    
    savemat(filename, save_dict)
    print(f"Pupil function saved to {filename}")
    print(f"  Shape (MATLAB format): {pupil_matlab.shape} [H, W, M^2]")
    print(f"  Number of lenslets: {pupil_array.shape[0]}")