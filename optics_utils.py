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


def compute_psf_centers(
    M: int,
    overlap_ratio: float,
    center_blend: float,
    z_ratio: float = 1.0,
    N: int = 512,
    device: torch.device = torch.device('cpu')
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
        像素数量
    device : torch.device
        计算设备
        
    Returns
    -------
    Dict[str, torch.Tensor]
        - 'centers_pixel': PSF中心的像素坐标 [M*M, 2]
        - 'centers_geom_pixel': 几何中心的像素坐标 [M*M, 2]
        - 'scale': 像素坐标的缩放系数
        - 'region_size_norm': 归一化的区域大小
        - 'stride_norm': 归一化的步长
    """
    # 几何参数
    region_size_norm = 1.0 / (M - (M - 1) * overlap_ratio)
    stride_norm = region_size_norm * (1.0 - overlap_ratio)
    scale = N - 1
    
    # 创建索引
    i_idx = torch.arange(M, device=device, dtype=torch.float32)
    j_idx = torch.arange(M, device=device, dtype=torch.float32)
    
    # 无重叠时的均匀网格中心（z=0时的位置）
    cx_uniform_norm = (i_idx + 0.5) / M
    cy_uniform_norm = (j_idx + 0.5) / M
    CX_uniform, CY_uniform = torch.meshgrid(cx_uniform_norm, cy_uniform_norm, indexing='ij')
    
    # 重叠几何的中心（z=focal_length时的位置）
    cx_overlap_norm = i_idx * stride_norm + region_size_norm / 2.0
    cy_overlap_norm = j_idx * stride_norm + region_size_norm / 2.0
    CX_overlap, CY_overlap = torch.meshgrid(cx_overlap_norm, cy_overlap_norm, indexing='ij')
    
    # 根据center_blend插值（控制无重叠和重叠几何的混合）
    t_blend = float(max(0.0, min(1.0, center_blend)))
    CX_blended = (1.0 - t_blend) * CX_uniform + t_blend * CX_overlap
    CY_blended = (1.0 - t_blend) * CY_uniform + t_blend * CY_overlap
    
    # 根据z_ratio进一步插值（控制传播距离）
    # z_ratio = 0，考虑重叠的透镜中心
    # z_ratio = 1，考虑混合的焦平面分布
    z_ratio = float(max(0.0, min(1.0, z_ratio)))
    CX_final = (1.0 - z_ratio) * CX_overlap + z_ratio * CX_blended
    CY_final = (1.0 - z_ratio) * CY_overlap + z_ratio * CY_blended
    CX_final = CX_final.clamp(0.0, 1.0)
    CY_final = CY_final.clamp(0.0, 1.0)
    
    # 转换为像素坐标
    centers_pixel = torch.stack([
        (CX_final * scale).reshape(-1), 
        (CY_final * scale).reshape(-1)
    ], dim=-1)
    
    # 几何中心（始终使用重叠几何，用于透镜覆盖计算）
    centers_geom_pixel = torch.stack([
        (CX_overlap * scale).reshape(-1), 
        (CY_overlap * scale).reshape(-1)
    ], dim=-1)
    
    return {
        'centers_pixel': centers_pixel,
        'centers_geom_pixel': centers_geom_pixel,
        'scale': scale,
        'region_size_norm': region_size_norm,
        'stride_norm': stride_norm
    }


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
        物理尺寸
    overlap_ratio : float
        重叠比例
    center_blend : float
        中心混合参数
    mask_count : int
        mask数量
    interleaving : str
        交错策略
    N : int
        像素数量
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
    
    # 获取几何参数
    center_info = compute_psf_centers(M, overlap_ratio, center_blend, N=N, device=device)
    region_size_norm = center_info['region_size_norm']
    stride_norm = center_info['stride_norm']
    scale = center_info['scale']
    centers_geom_pixel = center_info['centers_geom_pixel']
    
    # 创建坐标网格
    Y, X = torch.meshgrid(
        torch.arange(N, device=device, dtype=torch.float32),
        torch.arange(N, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # ========== 生成tile边界 ==========
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
    
    # ========== 生成tiles并分配到组 ==========
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
    
    # ========== 创建像素级masks ==========
    x_edges_px = torch.tensor([t * scale for t in norm_edges_x], device=device, dtype=torch.float32)
    y_edges_px = torch.tensor([t * scale for t in norm_edges_y], device=device, dtype=torch.float32)
    
    x_bins = torch.searchsorted(x_edges_px[1:], X.reshape(-1), right=False).reshape(N, N)
    y_bins = torch.searchsorted(y_edges_px[1:], Y.reshape(-1), right=False).reshape(N, N)
    
    masks = torch.zeros((mask_count, N, N), device=device, dtype=torch.bool)
    for tile in tiles:
        kx = tile['grid_kx']
        ky = tile['grid_ky']
        group = tile['group']
        
        tile_mask = (x_bins == kx) & (y_bins == ky)
        masks[group] |= tile_mask
    
    # ========== 计算透镜覆盖区域和面积占比 ==========
    num_lenses = M * M
    half_w = region_size_norm * scale / 2.0
    
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
        'masks': masks,
        'tiles': tiles,
        'a_lens_mask': a_lens_mask,
        'lens_mask': lens_mask
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
        - 'mask_psfs': 各mask的PSF [mask_count, N, N]（如果提供了masks）
    """
    num_lenses = centers_pixel.shape[0]
    assert num_lenses == M * M
    
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
    sigma = 0.42 * r_airy
    sigma_px = float(sigma / pixel_size)
    inv_two_sigma2 = 0.5 / (sigma_px ** 2 + 1e-20)
    
    # 初始化累加器
    gaussian_sum_total = torch.zeros((N, N), device=device, dtype=torch.float32)
    
    if masks is not None:
        mask_count = masks.shape[0]
        gaussian_sum_masks = torch.zeros((mask_count, N, N), device=device, dtype=torch.float32)
    
    # 累加所有透镜的高斯PSF
    for l in range(num_lenses):
        cx, cy = centers_pixel[l, 0], centers_pixel[l, 1]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        g = torch.exp(-dist_sq * inv_two_sigma2)
        gaussian_sum_total += g
        
        if masks is not None and a_lens_mask is not None:
            weights = a_lens_mask[l].view(mask_count, 1, 1)
            gaussian_sum_masks += weights * g
    
    # 归一化
    result = {}
    if normalize:
        denom = gaussian_sum_total.sum().clamp_min(1e-12)
        result['total_psf'] = gaussian_sum_total / denom * (N * N)
        if masks is not None:
            result['mask_psfs'] = gaussian_sum_masks / denom * (N * N)
    else:
        result['total_psf'] = gaussian_sum_total
        if masks is not None:
            result['mask_psfs'] = gaussian_sum_masks
    
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
