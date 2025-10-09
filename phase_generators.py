"""
提供相位图生成类，支持菲涅尔透镜和优化算法。
所有优化逻辑已合并入类中。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import config
from visualization import visualize_lenses_and_tiles
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from optics_utils import create_checkerboard
# Assuming wave_propagation is an available external library
import wave_propagation as wp
import math

class PhaseGenerator:
    def __init__(self, params, device=torch.device('cuda')):
        self.shape = params['shape']
        self.N = params['N']  # ROI size (square)
        self.M = params['M']   # Array size (square)
        self.focal_length = params['focal_length'] 
        self.wavelength = config.WAVELENGTH
        self.pixel_size = config.PIXEL_SIZE
        self.roi_rect = params['roi_rect']
        self.two_pi_value = params['two_pi_value']
        self.psf_energy_level = params['psf_energy_level']
        self.dof_correction = params['dof_correction']
        self.airy_correction = params['airy_correction']
        self.overlap_ratio = params['overlap_ratio']
        self.mask_count = int(params['mask_count'])
        self.center_blend = params['center_blend']
        self.interleaving = params['interleaving']
        self.ni = params['ni']
        self.lr = params['lr']
        self.device = device
        print(f"Using device: {self.device}")
        
        # simple calculations for other parameters
        self.L = None
        self.lens_width = None
        self.f_number = None
        self.depth_of_focus = None
        self.airy_radius = None
        self._update_parameters()
        
        # Results (set after generate)
        self.phase = None
        self.phase_8bit = None
        self.history = None  # Only for optimized
        self.total_psfs = None
        self.centers_pixel = None
        
        # Optimization attributes (set in _optimize_phase)
        self.phase_param = None # raw phase in nn.parameters
        self.U_masked = None
        self.total_psfs_up = None
        self.mask_psfs_up = None
        self.centers_pixel_up = None

    def generate_fresnel_phase(self,N=None) -> np.ndarray:
        """
        生成菲涅尔相位。
        """
        if N is None:
            N = self.N
        print(N, self.M)
        y, x = np.indices((N, N))
        phase = np.zeros((N, N))
        for r in range(self.M):
            for c in range(self.M):
                y_start = (r * N) // self.M
                y_end = ((r + 1) * N) // self.M
                x_start = (c * N) // self.M
                x_end = ((c + 1) * N) // self.M
                center_y = (y_start + y_end) / 2
                center_x = (x_start + x_end) / 2
                region = (slice(y_start, y_end), slice(x_start, x_end))
                x_dist = (x[region] - center_x) * self.pixel_size 
                y_dist = (y[region] - center_y) * self.pixel_size 
                r_squared = x_dist**2 + y_dist**2
                f_squared = self.focal_length**2
                phase_calc = (2 * np.pi / self.wavelength) * (self.focal_length - np.sqrt(f_squared + r_squared))
                phase[region] = phase_calc
        return phase % (2 * np.pi)

    def forward(self, U_in=None, z=None, upsampling=1.0) -> torch.Tensor:
        """
        Single forward propagation process. 
        """
        if self.phase_param is None:
            raise ValueError("Phase parameter not set. Run generate('optimized') first.")
        if z is None:
            z = self.focal_length
        if upsampling != 1.0:
            N = int(self.N * upsampling)
        else:
            N = self.N
        if U_in is None:
            U_in = torch.ones((N, N), device=self.device, dtype=torch.complex64)
        
        if upsampling != 1.0:
            # Assume phase_param is a real tensor of shape (N, N) 
            phase = self.phase_param.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            phase_up = F.interpolate(phase, scale_factor=upsampling, mode='nearest').squeeze(0).squeeze(0)
            U_phase = U_in * torch.exp(1j * phase_up)
        else:
            U_phase = U_in * torch.exp(1j * self.phase_param)
        
        U_focal = wp.propagate_ASM(U_phase, z, self.L, self.wavelength, self.device)
        return torch.abs(U_focal)**2
    
    def compute_loss(self, upsampling=1.0) -> tuple[torch.Tensor, dict]:
        """Compute losses"""
        if upsampling != 1.0:
            N = int(self.N * upsampling)
            pixel_size = self.L/N
        else:
            N = self.N
            pixel_size = self.pixel_size
            
        loss_fn = nn.MSELoss()
        
        # MSE loss for focal plane
        I_focal_full = self.forward(z=self.focal_length, upsampling=upsampling)
        mse = loss_fn(I_focal_full, self.total_psfs_up)
        
        # Average focusing efficiency loss for focal plane
        x_grid, y_grid = torch.meshgrid(torch.arange(N, device=self.device),
                                        torch.arange(N, device=self.device),
                                        indexing='ij')
        efficiencies = []
        theoretical_efficiency = (N ** 2) / (self.M * self.M)
        for center in self.centers_pixel_up: 
            distances = torch.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)
            mask = distances <= self.airy_radius * self.airy_correction / pixel_size
            encircled_energy = I_focal_full[mask].sum()
            efficiencies.append(encircled_energy / theoretical_efficiency)
        efficiencies = torch.stack(efficiencies)
        efficiency_mean = efficiencies.mean()
        efficiency_std = efficiencies.std()
        
        # Masked loss
        I_focal_masked = self.forward(U_in=self.U_masked, z=self.focal_length, upsampling=upsampling)
        mse_masked = loss_fn(I_focal_masked, self.mask_psfs_up)
        
        total_loss = torch.sum(torch.stack((mse, 
                                            -10*efficiency_mean, 
                                            10*efficiency_std, 
                                            mse_masked)))
        loss_components = {
            'focal_mse': mse.item(),
            'eff_mean': efficiency_mean.item(),
            'eff_std': efficiency_std.item(),
            'masked': mse_masked.item(),
            'total_loss': total_loss.item()
        }
        return total_loss, loss_components
    
    def _update_parameters(self,mode='fresnel'):
        # simple calculations
        self.L = self.N * self.pixel_size
        # Geometry with or w/o overlap
        if mode == 'fresnel':
            self.lens_width = self.L / self.M
            self.f_number = self.focal_length / self.lens_width
            self.depth_of_focus = 2*self.wavelength*self.f_number**2 
            self.airy_radius = 1.22*self.wavelength*self.f_number
        elif mode == 'optimized':
            region_size_norm = 1.0 / (self.M - (self.M - 1) * self.overlap_ratio)
            self.lens_width = self.L * region_size_norm
            self.f_number = self.focal_length / self.lens_width 
            self.depth_of_focus = 2*self.wavelength*self.f_number**2
            self.airy_radius = 1.22*self.wavelength*self.f_number
    
    def _prepare_template(self,upsampling=1.0,visualize=True):
         # Target patterns for optimization
        results = self._create_gaussian_template(
            upsampling = upsampling, 
            visualize=visualize        )
        self.mask_psfs_up = results['mask_psfs'] * self.psf_energy_level
        self.total_psfs_up = results['total_psfs'] * self.psf_energy_level
        masks = results['masks']
        self.centers_pixel_up = results['centers_pixel']
        
        # target patterns for general use
        results = self._create_gaussian_template(visualize=False)
        self.total_psfs = results['total_psfs']
        self.centers_pixel = results['centers_pixel']
        
        # Incident modulation for optimization
        self.U_masked = masks.to(device=self.device, dtype=torch.complex64)
        
    def _optimize_phase(self, 
                        init_phase: torch.Tensor = None, 
                        update_callback=None,
                        upsampling = 1.0):
        """
        执行优化，设置incident wave, self.phase_param和self.history。
        """
        # Initialize phase
        if init_phase is not None:
            self.phase_param = nn.Parameter(init_phase.to(device=self.device, dtype=torch.float32))
        else:
            self.phase_param = nn.Parameter(torch.rand((self.N, self.N), device=self.device) * 2 * np.pi)
        
        optimizer = torch.optim.Adam([self.phase_param], lr=self.lr)
        self.history = defaultdict(list) 
        start_time = time.time()
        loss_str_history = [] 
        print(f"Starting optimization with {self.ni} iterations...")
        for i in range(self.ni):
            optimizer.zero_grad()
            total_loss, loss_components = self.compute_loss(upsampling=upsampling)
            total_loss.backward()
            optimizer.step()
            # Record history
            for key, value in loss_components.items():
                self.history[key].append(value)
            # Callback
            if (i % 50 == 0 or i == self.ni - 1):
                if update_callback:
                    update_callback(i, self.ni, total_loss.item(), self.phase_param)
                else:
                    loss_str = " ".join([f"{k.capitalize()}: {v:.4f}" for k, v in loss_components.items() if k != 'total_loss'])
                    loss_str += f" Total: {loss_components['total_loss']:.4e}" if 'total_loss' in loss_components else ""
                    print(f"Iter: {i+1}/{self.ni} {loss_str}", flush=True)
        
        elapsed_time = time.time() - start_time
        print() 
        print(f"Optimization completed. Time elapsed: {elapsed_time:.2f} seconds")
    
    def _post_process_phase(self):
        """
        共享的后处理逻辑：嵌入SLM图案、结合背景、转换为8位，并设置实例属性。
        """
        y, x = np.indices(self.shape)
        roi_left = self.roi_rect[0]
        roi_top = self.roi_rect[1]
        roi_mask = (x >= roi_left) & (x < roi_left+self.N) & (y >= roi_top) & (y < roi_top+self.N)

        final_phase = np.zeros(self.shape)
        final_phase[roi_mask] = self.phase.flatten()

        checkerboard = create_checkerboard(self.shape)
        combined_phase = np.where(roi_mask, final_phase, checkerboard)
        self.phase_8bit = np.uint8(combined_phase / (2 * np.pi) * self.two_pi_value)   
        
    def update_phase_8bit(self, two_pi_value) -> np.array:
        self.two_pi_value = two_pi_value
        self._post_process_phase()
        return self.phase_8bit
    
    def generate(self, mode: str = 'optimized', 
                 init_mode: str = 'random', 
                 upsampling = 1.0,
                 vis_callback=None):
        """
        统一生成相位图，支持'fresnel'或'optimized'模式。生成后，直接从实例属性访问结果。
        对于'optimized'，init_mode可为'random'（默认）或'fresnel'（用Fresnel作为初始相位）。
        """
        
        if mode not in ['fresnel', 'optimized']:
            raise ValueError("Mode must be 'fresnel' or 'optimized'.")
        if init_mode not in ['random', 'fresnel']:
            raise ValueError("init_mode must be 'random' or 'fresnel'.")

        
        init_phase = None
        if mode == 'fresnel':
            self.phase = self.generate_fresnel_phase()
            self.phase_param = torch.tensor(self.phase, device=self.device, dtype=torch.float32)
            self._prepare_template(visualize=False)
            self._update_parameters(mode=mode)
            
        elif mode == 'optimized':
            
            if init_mode == 'fresnel':
                fresnel_phase_np = self.generate_fresnel_phase()
                init_phase = torch.tensor(fresnel_phase_np, dtype=torch.float32)
            self._prepare_template(upsampling=upsampling)
            self._optimize_phase(init_phase=init_phase, update_callback=vis_callback, upsampling=upsampling)
            self.phase = torch.remainder(self.phase_param, 2 * np.pi).detach().cpu().numpy()
            self._update_parameters(mode=mode)
            
        self._post_process_phase()
    

    def _create_gaussian_template(self, upsampling: float = 1.0, 
                                  coarse_grid_size: int = 2,
                                  visualize: bool = True, 
                                  display_lens_idx: Tuple[int, int] = (0, 0)) -> Dict[str, Any]:
        """
        Create Gaussian PSF template using internal parameters of PhaseOptimizer.
        
        Parameters
        ----------
        upsampling : float
            Optional upsampling factor for higher-resolution template generation.
        visualize : bool
            Whether to visualize the lens and tile layout.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing PSF templates, masks, and pixel coordinates.
        """
        # Use internal attributes
        N = int(self.N * upsampling)
        L = self.L
        focal_length = self.focal_length
        wavelength = self.wavelength
        M = self.M
        overlap_ratio = self.overlap_ratio
        device = self.device

        # Optional parameters if defined
        center_blend = self.center_blend
        airy_correction = self.airy_correction
        mask_count = self.mask_count
        interleaving = self.interleaving
        
        assert N > 0 and M > 0
        assert 0.0 <= overlap_ratio < 1.0
        assert mask_count >= 2
        if mask_count > coarse_grid_size**2:
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
        
        # Group tiles by mask (removed as unused)
        # masks_tiles = [[] for _ in range(mask_count)]
        # for tile in tiles:
        #     masks_tiles[tile['group']].append(tile)
        
        # Compute airy_radius_px for focusing efficiency
        airy_radius_px = float(r_airy / pixel_size / airy_correction)
        
        # 可视化
        if visualize:
            visualize_lenses_and_tiles(
                tiles, M, stride_norm, region_size_norm,
                mask_count, display_lens_idx
            )
        
        return {
            'total_psfs': normalized_gaussian,
            'centers_pixel': centers_pixel,
            'masks': masks,
            'mask_psfs': mask_psfs
        }

            