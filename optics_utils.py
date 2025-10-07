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


def visualize_lenses_and_tiles(
    tiles: List[Dict],
    M: int,
    stride_norm: float,
    region_size_norm: float,
    mask_count: int,
    display_lens_idx: Tuple[int, int] = (0, 0),
    figsize: Tuple[float, float] = (8, 12)
):
    """
    可视化透镜和tiles的布局
    
    Parameters:
    -----------
    tiles : List[Dict]
        Tiles列表
    M : int
        透镜阵列维度
    stride_norm : float
        归一化步长
    region_size_norm : float
        归一化区域大小
    mask_count : int
        掩膜数量
    display_lens_idx : Tuple[int, int]
        要高亮显示的透镜索引
    figsize : Tuple[float, float]
        图形大小
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 左图：显示透镜和tiles
    ax1.set_title(f'Lenses and Tiles Layout (M={M}, overlap_ratio={(1-stride_norm/region_size_norm):.2f})')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 绘制所有透镜（半透明）
    for i in range(M):
        for j in range(M):
            x_start = i * stride_norm
            y_start = j * stride_norm
            
            if (i, j) == display_lens_idx:
                # 高亮显示选定的透镜
                rect = patches.Rectangle(
                    (x_start, y_start), region_size_norm, region_size_norm,
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.3
                )
                ax1.add_patch(rect)
                ax1.text(x_start + region_size_norm/2, y_start + region_size_norm/2,
                        f'L[{i},{j}]', ha='center', va='center', fontsize=13, fontweight='bold')
            else:
                # 其他透镜用淡蓝色
                rect = patches.Rectangle(
                    (x_start, y_start), region_size_norm, region_size_norm,
                    linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.1
                )
                ax1.add_patch(rect)
    
    # 绘制tiles边界
    for tile in tiles:
        rect = patches.Rectangle(
            (tile['x_start_norm'], tile['y_start_norm']),
            tile['x_end_norm'] - tile['x_start_norm'],
            tile['y_end_norm'] - tile['y_start_norm'],
            linewidth=1, edgecolor='black', facecolor='none'
        )
        ax1.add_patch(rect)
        
        # 在tile中心显示贡献透镜的数量
        cx = (tile['x_start_norm'] + tile['x_end_norm']) / 2
        cy = (tile['y_start_norm'] + tile['y_end_norm']) / 2
        ax1.text(cx, cy, str(tile['num_lenses']), 
                ha='center', va='center', fontsize=11, color='green')
    
    ax1.set_xlabel('Normalized X')
    ax1.set_ylabel('Normalized Y')
    
    # 右图：显示mask分组
    ax2.set_title(f'Tile Mask Groups ({mask_count} groups)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 为每个mask组分配颜色
    colors = plt.cm.get_cmap('tab10', mask_count)
    
    for tile in tiles:
        rect = patches.Rectangle(
            (tile['x_start_norm'], tile['y_start_norm']),
            tile['x_end_norm'] - tile['x_start_norm'],
            tile['y_end_norm'] - tile['y_start_norm'],
            linewidth=1, edgecolor='black',
            facecolor=colors(tile['group']), alpha=0.5
        )
        ax2.add_patch(rect)
        
        # 显示组号
        cx = (tile['x_start_norm'] + tile['x_end_norm']) / 2
        cy = (tile['y_start_norm'] + tile['y_end_norm']) / 2
        ax2.text(cx, cy, str(tile['group']), 
                ha='center', va='center', fontsize=11)
    
    ax2.set_xlabel('Normalized X')
    ax2.set_ylabel('Normalized Y')
    
    # 添加图例
    legend_elements = [patches.Patch(facecolor=colors(i), alpha=0.5, label=f'Group {i}') 
                       for i in range(mask_count)]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\nTile Statistics:")
    print(f"Total tiles: {len(tiles)}")
    print(f"Grid dimensions: {max(t['grid_kx'] for t in tiles)+1} x {max(t['grid_ky'] for t in tiles)+1}")
    
    # 统计每个组的tiles数量
    group_counts = {i: 0 for i in range(mask_count)}
    for tile in tiles:
        group_counts[tile['group']] += 1
    print(f"Tiles per group: {group_counts}")
    
    # 统计透镜贡献分布
    lens_contributions = {}
    for tile in tiles:
        n = tile['num_lenses']
        lens_contributions[n] = lens_contributions.get(n, 0) + 1
    print(f"Lens contribution distribution: {lens_contributions}")


def create_gaussian_template(
    N: int,
    L: float,
    focal_length: float,
    wavelength: float,
    M: int,
    overlap_ratio: float,
    device: torch.device,
    *,
    # PSF center interpolation: 0=grid centers without overlap, 1=geometric centers with overlap
    center_blend: float = 0.0,
    # Airy disk correction factor
    airy_correction: float = 1.0,
    # Number of interleaved masks (>=2)
    mask_count: int = 2,
    # Interleaving strategy
    interleaving: str = "checkerboard",
    # Coarse grid size for coarse interleaving strategies (e.g., 2 means 2x2 subregions)
    coarse_grid_size: int = 2,
    # Visualization options
    visualize: bool = False,
    display_lens_idx: Tuple[int, int] = (0, 0)
) -> Dict[str, Any]:
    """
    Create Gaussian PSF template with overlap geometry and interleaved tile masks.
    
    改进版本：
    1. 修正了tile划分逻辑，确保覆盖所有像素
    2. 添加了可视化功能
    3. 实现了基于粗网格的交错策略
    
    Parameters:
    -----------
    N : int
        Image size in pixels (N x N)
    L : float
        Physical aperture size
    focal_length : float
        Focal length of the system
    wavelength : float
        Wavelength of light
    M : int
        Number of lenslets in each dimension (M x M array)
    overlap_ratio : float
        Overlap ratio between adjacent lenslets (0 <= overlap_ratio < 1)
    device : torch.device
        Device for tensor operations
    center_blend : float
        Interpolation factor for PSF centers (0=no-overlap grid, 1=overlap geometry)
    airy_correction : float
        Multiplicative correction for Airy disk radius
    mask_count : int
        Number of interleaved masks to generate
    interleaving : str
        Strategy for tile interleaving
    coarse_grid_size : int
        Size of coarse grid for coarse_* interleaving strategies
        (defines super tile size as coarse_grid_size × coarse_grid_size)
    visualize : bool
        Whether to visualize the lens and tile layout
    display_lens_idx : Tuple[int, int]
        Index of lens to highlight in visualization
    
    Returns:
    --------
    Dict containing all PSF-related data and tile information
    """
    
    assert N > 0 and M > 0
    assert 0.0 <= overlap_ratio < 1.0
    assert mask_count >= 2
    if mask_count>coarse_grid_size**2:
        print(f'coarse_grid_size={coarse_grid_size} is too small for current mask_count.')
        coarse_grid_size = int(math.ceil(math.sqrt(mask_count)))
        print(f'Use {coarse_grid_size} instead.')
        
    # Basic grid
    pixel_size = L / N
    Y, X = torch.meshgrid(
        torch.arange(N, device=device, dtype=torch.float32),
        torch.arange(N, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Geometry with overlap
    region_size_norm = 1.0 / (M - (M - 1) * overlap_ratio)
    stride_norm = region_size_norm * (1.0 - overlap_ratio)
    
    # Effective sub-aperture size
    D_eff = L * region_size_norm
    
    # Airy disk and Gaussian parameters
    r_airy = 1.22 * wavelength * focal_length * airy_correction / D_eff
    sigma = 0.42 * r_airy
    sigma_px = float(sigma / pixel_size)
    
    # PSF centers: interpolate between no-overlap and overlap positions
    i_idx = torch.arange(M, device=device, dtype=torch.float32)
    j_idx = torch.arange(M, device=device, dtype=torch.float32)
    
    # No-overlap centers (uniform grid)
    cx_no_norm = (i_idx + 0.5) / M
    cy_no_norm = (j_idx + 0.5) / M
    
    # Overlap geometry centers
    cx_ov_norm = i_idx * stride_norm + region_size_norm / 2.0
    cy_ov_norm = j_idx * stride_norm + region_size_norm / 2.0
    
    # Create meshgrids
    CX_no, CY_no = torch.meshgrid(cx_no_norm, cy_no_norm, indexing='ij')
    CX_ov, CY_ov = torch.meshgrid(cx_ov_norm, cy_ov_norm, indexing='ij')
    
    # Interpolate centers
    t = float(max(0.0, min(1.0, center_blend)))
    CX = (1.0 - t) * CX_no + t * CX_ov
    CY = (1.0 - t) * CY_no + t * CY_ov
    CX = CX.clamp(0.0, 1.0)
    CY = CY.clamp(0.0, 1.0)
    
    # Convert to pixel coordinates
    scale = N - 1
    centers_pixel = torch.stack([(CX * scale).reshape(-1), (CY * scale).reshape(-1)], dim=-1)
    
    # Geometric centers for lens coverage (always use overlap geometry)
    centers_geom_pixel = torch.stack([(CX_ov * scale).reshape(-1), (CY_ov * scale).reshape(-1)], dim=-1)
    
    # Lens dimensions in pixels
    w_px = region_size_norm * scale
    half_w = w_px / 2.0
    
    # 改进的tile边界生成 - 确保覆盖所有区域
    # 创建包含所有透镜边界和图像边界的完整边界集合
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
    
    # 添加图像边界
    lens_boundaries_x.add(0.0)
    lens_boundaries_x.add(1.0)
    lens_boundaries_y.add(0.0)
    lens_boundaries_y.add(1.0)
    
    # 转换为排序列表
    norm_edges_x = sorted(lens_boundaries_x)
    norm_edges_y = sorted(lens_boundaries_y)
    
    # 生成tiles
    tiles = []
    for kx in range(len(norm_edges_x) - 1):
        x_start_norm = norm_edges_x[kx]
        x_end_norm = norm_edges_x[kx + 1]
        
        # 跳过零宽度的tiles
        if x_end_norm - x_start_norm < 1e-6:
            continue
            
        for ky in range(len(norm_edges_y) - 1):
            y_start_norm = norm_edges_y[ky]
            y_end_norm = norm_edges_y[ky + 1]
            
            # 跳过零高度的tiles
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
                    
                    # 检查是否有重叠
                    if (x_start_norm < lens_x_end and x_end_norm > lens_x_start and
                        y_start_norm < lens_y_end and y_end_norm > lens_y_start):
                        contributing_lenses.append((ii, jj))
            
            # 分配到mask组
            if interleaving == "checkerboard":
                # 简单棋盘模式
                group = (kx + ky) % mask_count
                
            elif interleaving.startswith("coarse"):
                # 基于粗网格的分组策略
                # 确定当前tile属于哪个super tile
                super_tile_x = kx // coarse_grid_size
                super_tile_y = ky // coarse_grid_size
                
                # 在super tile内部的位置
                local_x = kx % coarse_grid_size
                local_y = ky % coarse_grid_size
                
                # Super tile内部的tiles总数
                tiles_per_super = coarse_grid_size * coarse_grid_size
                
                if interleaving == "coarse3":
                    # 顺序分配：Super tile内的tiles按行优先顺序分配
                    local_index = local_y * coarse_grid_size + local_x
                    
                    if mask_count <= tiles_per_super:
                        # 如果mask数量不超过super tile内的tiles数
                        group = local_index % mask_count
                    else:
                        # 如果mask数量超过super tile内的tiles数，使用扩展模式
                        group = (local_index + (super_tile_x + super_tile_y) * tiles_per_super) % mask_count
                    
                    # 对奇数super tiles进行组号反转，增加多样性
                    if (super_tile_x + super_tile_y) % 2 == 1:
                        group = (mask_count - 1 - group) % mask_count
                        
                elif interleaving == "coarse2":
                    # 顺序分配：Super tile内的tiles按行优先顺序分配
                    local_index = local_y * coarse_grid_size + local_x
                    
                    if mask_count <= tiles_per_super:
                        # 如果mask数量不超过super tile内的tiles数
                        group = local_index % mask_count
                    else:
                        # 如果mask数量超过super tile内的tiles数，使用扩展模式
                        group = (local_index + (super_tile_x + super_tile_y) * tiles_per_super) % mask_count
                elif interleaving == "coarse1":
                    local_index = local_y  + local_x 
                    group = local_index % mask_count
                    
            else:
                # 默认fallback到checkerboard
                group = (kx + ky) % mask_count
            
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
    
    # 将tiles转换为像素级的mask
    # 使用更精确的边界映射
    x_edges_px = torch.tensor([t * scale for t in norm_edges_x], device=device, dtype=torch.float32)
    y_edges_px = torch.tensor([t * scale for t in norm_edges_y], device=device, dtype=torch.float32)
    
    x_bins = torch.searchsorted(x_edges_px[1:], X.reshape(-1), right=False).reshape(N, N)
    y_bins = torch.searchsorted(y_edges_px[1:], Y.reshape(-1), right=False).reshape(N, N)
    
    # 创建masks
    masks = torch.zeros((mask_count, N, N), device=device, dtype=torch.bool)
    for tile in tiles:
        kx = tile['grid_kx']
        ky = tile['grid_ky']
        group = tile['group']
        
        # 找到属于这个tile的像素
        tile_mask = (x_bins == kx) & (y_bins == ky)
        masks[group] |= tile_mask
    
    # Lens coverage masks
    num_lenses = M * M
    X2 = X.unsqueeze(0)
    Y2 = Y.unsqueeze(0)
    
    cxg = centers_geom_pixel[:, 0].view(num_lenses, 1, 1)
    cyg = centers_geom_pixel[:, 1].view(num_lenses, 1, 1)
    
    lens_mask = (X2 - cxg).abs() <= half_w
    lens_mask &= (Y2 - cyg).abs() <= half_w
    
    # Calculate area fractions for each lens in each mask
    inter_counts = (lens_mask[:, None] & masks[None, :]).sum(dim=(2, 3)).to(torch.float32)
    lens_counts = lens_mask.view(num_lenses, -1).sum(dim=1).clamp(min=1).to(torch.float32)
    a_lens_mask = inter_counts / lens_counts[:, None]
    
    # Accumulate PSFs
    gaussian_sum_total = torch.zeros((N, N), device=device, dtype=torch.float32)
    gaussian_sum_masks = torch.zeros((mask_count, N, N), device=device, dtype=torch.float32)
    
    inv_two_sigma2 = 0.5 / (sigma_px ** 2 + 1e-20)
    
    for l in range(num_lenses):
        cx, cy = centers_pixel[l, 0], centers_pixel[l, 1]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        g = torch.exp(-dist_sq * inv_two_sigma2)
        gaussian_sum_total += g
        
        # Weight by area fraction for each mask
        weights = a_lens_mask[l].view(mask_count, 1, 1)
        gaussian_sum_masks += weights * g
    
    # Normalize
    denom = gaussian_sum_total.sum().clamp_min(1e-12)
    normalized_gaussian = gaussian_sum_total / denom * (N * N)
    mask_psfs = gaussian_sum_masks / denom * (N * N)
    
    # Group tiles by mask
    masks_tiles = [[] for _ in range(mask_count)]
    for tile in tiles:
        masks_tiles[tile['group']].append(tile)
    
    # Information summary
    info = {
        'pixel_size': pixel_size,
        'region_size_norm': float(region_size_norm),
        'stride_norm': float(stride_norm),
        'w_px': float(w_px),
        's_px': float(stride_norm * scale),
        'D_eff': float(D_eff),
        'r_airy': float(r_airy),
        'sigma': float(sigma),
        'sigma_px': float(sigma_px),
        'center_blend': float(t),
        'airy_correction': float(airy_correction),
        'overlap_ratio': float(overlap_ratio),
        'M': int(M),
        'mask_count': int(mask_count),
        'interleaving': interleaving,
        'num_tiles': len(tiles),
        'airy_radius_px': float(r_airy / pixel_size / airy_correction),
        'tile_grid_size': (len(norm_edges_x)-1, len(norm_edges_y)-1)
    }
    
    # 可视化
    if visualize:
        visualize_lenses_and_tiles(
            tiles, M, stride_norm, region_size_norm,
            mask_count, display_lens_idx
        )
    
    return {
        'total_psfs': normalized_gaussian,
        'centers_pixel': centers_pixel,
        'sigma_px': sigma_px,
        'masks': masks,
        'mask_psfs': mask_psfs,
        'tiles': tiles,
        'masks_tiles': masks_tiles,
        'tile_indices': (x_bins.to(torch.long), y_bins.to(torch.long)),
        'info': info
    }


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