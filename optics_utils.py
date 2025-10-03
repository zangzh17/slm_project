# optics_utils.py

"""
Utility functions for optical calculations.
"""

import torch
import numpy as np
import config
from collections import defaultdict
from typing import Dict, Any, Tuple
import math

def create_grid(L: float, N: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create coordinate grid"""
    x = torch.linspace(-L / 2, L / 2, N, device=device)
    return torch.meshgrid(x, x, indexing='ij')

def create_gaussian_template_v2(
    N: int,
    L: float,
    focal_length: float,
    wavelength: float,
    M: int,
    overlap_ratio: float,
    size_factor: float,
    device: torch.device,
    *,
    # 1) 解耦重叠率与PSF中心：0=按“无重叠栅格(M等分)”布PSF中心；1=按“重叠几何中心”布PSF中心；中间值为线性插值
    center_blend: float = 0.0,
    # 2) 艾里斑修正：在“考虑重叠后的有效子孔径尺寸”基础上的乘性修正
    airy_correction: float = 1.0,
    # 3) 生成交错mask的数量（>=2 时产生互补/远离的分组；2=棋盘格）
    mask_count: int = 2,
    # 4) 每个mask的PSF权重模式：目前实现为基于“每个透镜的面积分配(per_lens)”以保证所有mask之和等于总PSF
    weighting: str = "per_lens",
) -> Dict[str, Any]:
    """
    构造高斯PSF模板（考虑重叠几何），并生成交错tile mask与对应的分组PSF。

    关键改变：
      - 用 center_blend 控制 PSF 中心从“无重叠等分中心”(0) 到 “重叠几何中心”(1) 的连续插值；默认 0。
      - 艾里斑半径 r_airy 使用“考虑重叠后的有效子孔径宽度 D_eff=L*region_size_norm”，并乘 airy_correction（默认1）。
      - 把总孔径划分为由 stride 决定的最小重复单元(tile)，基于 (i,j) 的广义棋盘着色生成 mask_count 个互补mask，使同组 tile 尽量远离。
      - 对每个 mask 生成相应的 PSF sum：仅累加那些对该组 tile 有贡献的透镜之PSF，且各透镜按其在该组内的“覆盖面积 / 该透镜总面积”加权。
        统一用同一个全局归一化因子，保证所有分组PSF之和严格等于总 normalized_gaussian。

    返回：
      {
        'normalized_gaussian': Tensor[N,N],                # 总PSF（归一化后）
        'centers_pixel': Tensor[num_lenses, 2],            # PSF中心（像素，已按 center_blend）
        'sigma_px': float,                                  # 高斯σ（像素）
        'masks': BoolTensor[mask_count,N,N],               # 每个分组的像素级mask
        'normalized_gaussians_per_mask': Tensor[mask_count,N,N],  # 每组的PSF（已与总PSF共用归一化因子）
        'tile_xy_index': Tuple[LongTensor[N,N], LongTensor[N,N]], # 每个像素的tile索引 (ix, iy)
        'info': dict                                       # 各种几何/物理派生量
      }
    """

    assert N > 0 and M > 0
    assert 0.0 <= overlap_ratio < 1.0
    assert mask_count >= 2
    weighting = weighting.lower()
    if weighting != "per_lens":
        raise NotImplementedError("Only weighting='per_lens' is implemented to preserve additivity across masks.")

    # --- 基本网格 ---
    pixel_size = L / N
    Y, X = torch.meshgrid(
        torch.arange(N, device=device, dtype=torch.float32),
        torch.arange(N, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # --- 几何：重叠方形透镜的归一化宽度与步进（见用户原始代码公式）---
    region_size_norm = 1.0 / (M - (M - 1) * overlap_ratio)  # 归一化的方形透镜边长
    stride_norm = region_size_norm * (1.0 - overlap_ratio)  # 邻近几何中心之间的步进

    # 有效子孔径（考虑重叠后的单个透镜“宽度”）
    D_eff = L * region_size_norm  # 用于艾里斑估算

    # --- 2) 艾里斑与高斯σ ---
    # r_airy = 1.22 * λ * f / D × size_factor × airy_correction
    r_airy = 1.22 * wavelength * focal_length * size_factor * airy_correction / D_eff
    sigma = 0.42 * r_airy
    sigma_px = float(sigma / pixel_size)

    # --- 1) PSF中心：无重叠等分 vs 重叠几何中心，并做线性插值 ---
    #   无重叠等分中心（把总孔径L分为M份）：(i+0.5)/M
    #   重叠几何中心：i*stride + region_size/2
    i_idx = torch.arange(M, device=device, dtype=torch.float32)
    j_idx = torch.arange(M, device=device, dtype=torch.float32)

    cx_no_norm = (i_idx + 0.5) / M
    cy_no_norm = (j_idx + 0.5) / M

    cx_ov_norm = i_idx * stride_norm + region_size_norm / 2.0
    cy_ov_norm = j_idx * stride_norm + region_size_norm / 2.0

    # 网格展开
    CX_no, CY_no = torch.meshgrid(cx_no_norm, cy_no_norm, indexing='ij')
    CX_ov, CY_ov = torch.meshgrid(cx_ov_norm, cy_ov_norm, indexing='ij')

    # 插值（clamp到[0,1]以防数值误差越界）
    t = float(max(0.0, min(1.0, center_blend)))
    CX = (1.0 - t) * CX_no + t * CX_ov
    CY = (1.0 - t) * CY_no + t * CY_ov
    CX = CX.clamp(0.0, 1.0)
    CY = CY.clamp(0.0, 1.0)

    # 像素坐标（与原代码一致使用 (N-1) 缩放）
    scale = (N - 1)
    centers_pixel = torch.stack([(CX * scale).reshape(-1), (CY * scale).reshape(-1)], dim=-1)

    # --- 物理“几何透镜”的中心用于画透镜覆盖（总是采用重叠几何中心，不受 center_blend 的影响）---
    CX_geom = CX_ov  # 真实几何布置
    CY_geom = CY_ov
    centers_geom_pixel = torch.stack([(CX_geom * scale).reshape(-1), (CY_geom * scale).reshape(-1)], dim=-1)

    # 透镜方形的像素宽度（与 (N-1) 对齐，便于与中心相容）
    w_px = region_size_norm * scale
    half_w = w_px / 2.0

    # --- 3) 生成 tile 边界与 tile 索引（大多数tile为 stride×stride，边界处可能是剩余条带）---
    # 边界：0, s, 2s, ..., M*s, 1
    s_px = stride_norm * scale
    # 用 torch.linspace 可能引入舍入误差；直接构造
    edge_main = torch.arange(0, M + 1, device=device, dtype=torch.float32) * s_px
    x_edges = torch.cat([edge_main, torch.tensor([float(scale)], device=device)])  # [0, s, ..., M*s, (N-1)]
    y_edges = x_edges.clone()

    # 像素坐标 -> tile 索引（区间 [edge[k], edge[k+1])，最后一格闭区间）
    # torch.bucketize: 返回落入哪个边界区间的右端索引
    x_bins = torch.bucketize(X, x_edges, right=False) - 1
    y_bins = torch.bucketize(Y, y_edges, right=False) - 1
    x_bins = x_bins.clamp(0, x_edges.numel() - 2)
    y_bins = y_bins.clamp(0, y_edges.numel() - 2)

    # --- 交错分组（广义棋盘）：选择 (a,b)=(1,2) 对大多数 K 提供较好的分散；若 K 与2不互素则退化到(1,1)
    def choose_ab(K: int) -> Tuple[int, int]:
        a, b = 1, 2
        if math.gcd(K, b) != 1:
            b = 1
        return a, b

    a, b = choose_ab(mask_count)
    group_id = (a * x_bins + b * y_bins) % mask_count

    masks = torch.stack([group_id == k for k in range(mask_count)], dim=0)

    # --- 每个透镜的覆盖mask（像素级，基于几何中心 + 方形边长 w_px）---
    num_lenses = M * M
    X2 = X.unsqueeze(0)  # [1,N,N]
    Y2 = Y.unsqueeze(0)

    cxg = centers_geom_pixel[:, 0].view(num_lenses, 1, 1)
    cyg = centers_geom_pixel[:, 1].view(num_lenses, 1, 1)

    lens_mask = (X2 - cxg).abs() <= half_w
    lens_mask &= (Y2 - cyg).abs() <= half_w
    # lens_mask: [L,N,N]  (L=num_lenses)

    # --- 4) 计算每个 mask 对每个透镜的面积占比（per_lens）：a[l, k] ∈ [0,1] 且 sum_k a[l,k] = 1 ---
    # 交集像素数
    # (lens_mask[:,None] & masks[None,:]) -> [L,K,N,N]
    inter_counts = (lens_mask[:, None] & masks[None, :]).sum(dim=(2, 3)).to(torch.float32)  # [L,K]
    lens_counts = lens_mask.view(num_lenses, -1).sum(dim=1).clamp(min=1).to(torch.float32)  # [L]
    a_lens_mask = inter_counts / lens_counts[:, None]  # [L,K]

    # --- 累加 PSF：总和 + 每组（按 a_lens_mask 加权）---
    denom_eps = 1e-12
    gaussian_sum_total = torch.zeros((N, N), device=device, dtype=torch.float32)
    gaussian_sum_masks = torch.zeros((mask_count, N, N), device=device, dtype=torch.float32)

    inv_two_sigma2 = 0.5 / (sigma_px ** 2 + 1e-20)

    for l in range(num_lenses):
        cx, cy = centers_pixel[l, 0], centers_pixel[l, 1]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        g = torch.exp(-dist_sq * inv_two_sigma2)
        gaussian_sum_total += g
        # 各mask按该透镜的面积比例加权
        weights = a_lens_mask[l].view(mask_count, 1, 1)
        gaussian_sum_masks += weights * g

    # --- 归一化：统一用总和的归一化，从而保证 Σmask == total ---
    denom = gaussian_sum_total.sum().clamp_min(denom_eps)
    normalized_gaussian = gaussian_sum_total / denom * (N * N)
    normalized_gaussians_per_mask = gaussian_sum_masks / denom * (N * N)

    # --- 信息汇总 ---
    info: Dict[str, Any] = dict(
        pixel_size=pixel_size,
        region_size_norm=float(region_size_norm),
        stride_norm=float(stride_norm),
        w_px=float(w_px),
        s_px=float(s_px),
        D_eff=float(D_eff),
        r_airy=float(r_airy),
        sigma=float(sigma),
        sigma_px=float(sigma_px),
        center_blend=float(t),
        airy_correction=float(airy_correction),
        overlap_ratio=float(overlap_ratio),
        M=int(M),
        mask_count=int(mask_count),
    )

    return dict(
        normalized_gaussian=normalized_gaussian,
        centers_pixel=centers_pixel,
        sigma_px=sigma_px,
        masks=masks,
        normalized_gaussians_per_mask=normalized_gaussians_per_mask,
        tile_xy_index=(x_bins.to(torch.long), y_bins.to(torch.long)),
        info=info,
    )

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


def create_gaussian_template(
    N: int, L: float, focal_length: float, wavelength: float, M: int,
    overlap_ratio: float, size_factor: float, device: torch.device,
    center_blend: float = 0.0, correction_factor: float = 1.0, num_masks: int = 2
) -> tuple[torch.Tensor, torch.Tensor, float, list[torch.Tensor], list[dict]]:
    """Create target Gaussian spot array template."""

    pixel_size = L / N
    region_size_norm = 1.0 / (M - (M - 1) * overlap_ratio)
    stride_norm = region_size_norm * (1 - overlap_ratio)
    sub_aperture_L = L * region_size_norm  # Consider overlap
    r_airy = 1.22 * wavelength * focal_length * size_factor * correction_factor / sub_aperture_L
    sigma = 0.42 * r_airy
    sigma_px = sigma / pixel_size

    print(f"Diffraction limit parameters: Airy radius = {r_airy / size_factor / pixel_size / correction_factor:.2f} px, Gaussian width σ = {sigma_px:.2f} px")

    Y, X = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')

    centers_pixel = []
    gaussian_sum = torch.zeros((N, N), device=device)

    for i in range(M):
        for j in range(M):
            cx_norm_overlap = (i * stride_norm) + region_size_norm / 2
            cy_norm_overlap = (j * stride_norm) + region_size_norm / 2
            cx_norm_no = (i + 0.5) / M
            cy_norm_no = (j + 0.5) / M
            cx_norm = (1 - center_blend) * cx_norm_no + center_blend * cx_norm_overlap
            cy_norm = (1 - center_blend) * cy_norm_no + center_blend * cy_norm_overlap
            cx = cx_norm * (N - 1)
            cy = cy_norm * (N - 1)
            centers_pixel.append((cx, cy))
            dist_sq = (X - cx)**2 + (Y - cy)**2
            gaussian_sum += torch.exp(-0.5 * dist_sq / (sigma_px**2))

    # Normalize template
    normalized_gaussian = gaussian_sum / gaussian_sum.sum() * N * N

    # For masks and tile-based PSF
    norm_edges = sorted(set([float(k) * stride_norm for k in range(M)] + [float(k) * stride_norm + region_size_norm for k in range(M)]))
    tiles = []
    area_lens = (region_size_norm * L) ** 2

    for kx in range(len(norm_edges) - 1):
        x_start_norm = norm_edges[kx]
        x_end_norm = norm_edges[kx + 1]
        tile_width = (x_end_norm - x_start_norm) * L
        for ky in range(len(norm_edges) - 1):
            y_start_norm = norm_edges[ky]
            y_end_norm = norm_edges[ky + 1]
            tile_height = (y_end_norm - y_start_norm) * L
            tile_area = tile_width * tile_height
            contributing_lenses = []
            for ii in range(M):
                for jj in range(M):
                    lens_x_start = float(ii) * stride_norm
                    lens_x_end = lens_x_start + region_size_norm
                    lens_y_start = float(jj) * stride_norm
                    lens_y_end = lens_y_start + region_size_norm
                    if (x_start_norm < lens_x_end and x_end_norm > lens_x_start and
                        y_start_norm < lens_y_end and y_end_norm > lens_y_start):
                        contributing_lenses.append((ii, jj))
            tiles.append({
                'x_start_norm': x_start_norm,
                'x_end_norm': x_end_norm,
                'y_start_norm': y_start_norm,
                'y_end_norm': y_end_norm,
                'area': tile_area,
                'lenses': contributing_lenses,
                'grid_kx': kx,
                'grid_ky': ky
            })

    # Assign tiles to masks
    masks_tiles = [[] for _ in range(num_masks)]
    for tile in tiles:
        group = (tile['grid_kx'] + tile['grid_ky']) % num_masks
        masks_tiles[group].append(tile)

    # Compute PSF for each mask
    mask_psfs = []
    centers_tensor = torch.tensor(centers_pixel, device=device)
    for mask_id in range(num_masks):
        lens_contrib = defaultdict(float)
        for tile in masks_tiles[mask_id]:
            for lens_ij in tile['lenses']:
                lens_contrib[lens_ij] += tile['area']
        gaussian_sum_mask = torch.zeros((N, N), device=device)
        for lens_ij, contrib_area in lens_contrib.items():
            scale = contrib_area / area_lens
            idx = lens_ij[0] * M + lens_ij[1]
            cx, cy = centers_tensor[idx]
            dist_sq = (X - cx)**2 + (Y - cy)**2
            gaussian_sum_mask += scale * torch.exp(-0.5 * dist_sq / (sigma_px**2))
        if gaussian_sum_mask.sum() > 0:
            normalized_mask = gaussian_sum_mask / gaussian_sum_mask.sum() * N * N
        else:
            normalized_mask = gaussian_sum_mask
        mask_psfs.append(normalized_mask)

    return normalized_gaussian, centers_tensor, sigma_px, mask_psfs, masks_tiles

# def create_gaussian_template(
#     N: int, L: float, focal_length: float, wavelength: float, M: int, 
#     overlap_ratio: float, size_factor: float, device: torch.device
# ) -> tuple[torch.Tensor, torch.Tensor, float]:
#     """Create target Gaussian spot array template."""
#     pixel_size = L / N
#     sub_aperture_L = L / M
#     r_airy = 1.22 * wavelength * focal_length * size_factor / sub_aperture_L
#     sigma = 0.42 * r_airy
#     sigma_px = sigma / pixel_size

#     print(f"Diffraction limit parameters: Airy radius = {r_airy/size_factor/pixel_size:.2f} px, Gaussian width σ = {sigma_px:.2f} px")

#     Y, X = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')

#     region_size_norm = 1.0 / (M - (M - 1) * overlap_ratio)
#     stride_norm = region_size_norm * (1 - overlap_ratio)
    
#     centers_pixel = []
#     gaussian_sum = torch.zeros((N, N), device=device)

#     for i in range(M):
#         for j in range(M):
#             cx_norm = (i * stride_norm) + region_size_norm / 2
#             cy_norm = (j * stride_norm) + region_size_norm / 2
#             cx = cx_norm * (N - 1)
#             cy = cy_norm * (N - 1)
#             centers_pixel.append((cx, cy))
#             dist_sq = (X - cx)**2 + (Y - cy)**2
#             gaussian_sum += torch.exp(-0.5 * dist_sq / (sigma_px**2))
    
#     # Normalize template
#     normalized_gaussian = gaussian_sum / gaussian_sum.sum() * N * N
#     return normalized_gaussian, torch.tensor(centers_pixel, device=device), sigma_px



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

def calculate_airy_disk(focal_length_m, aperture_width_m):
    """Calculate diffraction-limited spot size (Airy disk diameter)."""
    f_number = focal_length_m / aperture_width_m
    airy_disk_diameter_m = 2.44 * config.WAVELENGTH * f_number
    return airy_disk_diameter_m * 1e6  # Return in micrometers

def create_checkerboard(shape):
    """Create checkerboard background pattern (0 and π)."""
    y, x = np.indices(shape)
    return np.pi * ((x + y) % 2)