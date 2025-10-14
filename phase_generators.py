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
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from optics_utils import create_checkerboard
from optics_utils import compute_psf_centers, generate_tile_masks, generate_gaussian_psf
from optics_utils import generate_lens_circular_masks
        
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
        self.depth_in_focus = params['depth_in_focus']
        self.depth_out_focus = params['depth_out_focus']
        
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
        self.depth_psfs_up = None
        self.centers_pixel_out_focus_up = None
        self.out_focus_masks_up = None

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

        # Depth MES Loss for in-focus planes

        # Centroid loss for out-focus planes 
        # (Regional centroid deviation from geo. centers pixel)
        
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
    
    def update_phase_8bit(self, two_pi_value) -> np.array:
        self.two_pi_value = two_pi_value
        self._post_process_phase()
        return self.phase_8bit
    
    def generate(self, mode: str = 'optimized', 
                 init_mode: str = 'random', 
                 upsampling = 1.0,
                 visualize = True,
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
            self._prepare_template(upsampling=upsampling, visualize=visualize)
            self._optimize_phase(init_phase=init_phase, update_callback=vis_callback, upsampling=upsampling)
            self.phase = torch.remainder(self.phase_param, 2 * np.pi).detach().cpu().numpy()
            self._update_parameters(mode=mode)
            
        self._post_process_phase()
    
    
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
        
        if self.depth_in_focus is None:
            self.depth_in_focus = self.focal_length
        else:
            self.depth_in_focus = [self.focal_length + d*self.depth_of_focus*self.dof_correction
                                    for d in self.depth_in_focus]
            
        if self.depth_out_focus is not None:
            self.depth_out_focus = [self.focal_length + d*self.depth_of_focus*self.dof_correction
                                    for d in self.depth_out_focus]
    
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
    
    def _prepare_template(self, upsampling=1.0, visualize=True):
        """
        准备优化模板
        使用类参数：
        - self.depth_in_focus: z_ratio列表，用于生成depth_psfs
        - self.depth_out_focus: z_ratio列表，用于生成离焦中心坐标
        """
        N_up = int(self.N * upsampling)
        
        # 上采样版本，生成in-focus的多平面PSF（包含depth_psfs）
        results = self._create_gaussian_template(
            upsampling=upsampling,
            visualize=visualize,
            z_ratios=self.depth_in_focus
        )
        self.mask_psfs_up = results['mask_psfs'] * self.psf_energy_level
        self.total_psfs_up = results['total_psfs'] * self.psf_energy_level
        self.depth_psfs_up = results['depth_psfs'] * self.psf_energy_level  # [num_depths, N, N]
        self.centers_pixel_up = results['centers_pixel']
        masks = results['masks']
        
        # 上采样版本，out-of-focus中心和mask
        if self.depth_out_focus is not None:
            # 生成out-of-focus的中心坐标（仅坐标，不需要PSF）
            centers_out_focus_list = []
            for z_ratio in self.depth_out_focus:
                center_info = compute_psf_centers(
                    M=self.M,
                    overlap_ratio=self.overlap_ratio,
                    center_blend=self.center_blend,
                    z_ratio=z_ratio,
                    N=N_up,
                    device=self.device
                )
                centers_out_focus_list.append(center_info['centers_pixel'])
            # [num_out_focus, M*M, 2]
            self.centers_pixel_out_focus_up = torch.stack(centers_out_focus_list, dim=0)

            # 上采样版本，生成out-of-focus的mask
            region_size_norm = 1.0 / (self.M - (self.M - 1) * self.overlap_ratio)
            lens_width_px_up = region_size_norm * (N_up - 1)
            radii_up = torch.tensor(
                [lens_width_px_up * z / 2.0 for z in self.depth_out_focus],
                device=self.device,
                dtype=torch.float32
            )
            self.out_focus_masks_up = generate_lens_circular_masks(
                centers_pixel=self.centers_pixel_out_focus_up,
                radii_pixels=radii_up,
                N=N_up,
                device=self.device
            )

        # 非上采样版本，生成in-focus的PSF（包含depth_psfs）
        results = self._create_gaussian_template(
            upsampling=1.0,
            visualize=False,
            z_ratios=self.depth_in_focus
        )
        self.total_psfs = results['total_psfs']
        self.centers_pixel = results['centers_pixel']

        # Incident modulation for optimization
        self.U_masked = masks.to(device=self.device, dtype=torch.complex64)

    def _create_gaussian_template(
        self,
        upsampling: float = 1.0,
        coarse_grid_size: int = 2,
        visualize: bool = True,
        display_lens_idx: Tuple[int, int] = (0, 0),
        z_ratios: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        创建高斯PSF模板。
        
        Parameters
        ----------
        upsampling : float
            上采样因子
        coarse_grid_size : int
            粗网格大小
        visualize : bool
            是否可视化
        display_lens_idx : Tuple[int, int]
            要高亮显示的透镜索引
        z_ratios : List[float], optional
            传播距离比例列表。如果为None，默认为[1.0]
            
        Returns
        -------
        Dict[str, Any]
            - 'total_psfs': z_ratio=0时的PSF [N, N]
            - 'mask_psfs': z_ratio=0时的mask PSF [mask_count, N, N]
            - 'depth_psfs': 不同z_ratio的PSF [len(z_ratios), N, N]
            - 'centers_pixel': z_ratio=0时的中心坐标 [M*M, 2]
            - 'masks': 像素级mask [mask_count, N, N]
        """
        N = int(self.N * upsampling)
        
        if z_ratios is None:
            z_ratios = [1.0]
        z_ratios_list = list(z_ratios)
        
        # 1. 生成masks（只需要一次）
        mask_info = generate_tile_masks(
            M=self.M,
            L=self.L,
            overlap_ratio=self.overlap_ratio,
            center_blend=self.center_blend,
            mask_count=self.mask_count,
            interleaving=self.interleaving,
            N=N,
            coarse_grid_size=coarse_grid_size,
            device=self.device
        )
        masks = mask_info['masks']
        tiles = mask_info['tiles']
        a_lens_mask = mask_info['a_lens_mask']
        
        # 2. 计算z_ratio=0时的PSF中心
        center_info_z0 = compute_psf_centers(
            M=self.M,
            overlap_ratio=self.overlap_ratio,
            center_blend=self.center_blend,
            z_ratio=0.0,
            N=N,
            device=self.device
        )
        centers_pixel_z0 = center_info_z0['centers_pixel']
        
        #  计算z_ratio=0时的PSF
        psf_result_z0 = generate_gaussian_psf(
            centers_pixel=centers_pixel_z0,
            N=N,
            L=self.L,
            M=self.M,
            overlap_ratio=self.overlap_ratio,
            focal_length=self.focal_length,
            wavelength=self.wavelength,
            airy_correction=self.airy_correction,
            masks=masks,
            a_lens_mask=a_lens_mask,
            normalize=True,
            device=self.device
        )
        
        # 3. 计算不同z_ratio的PSF
        depth_psfs_list = []
        for z_ratio in z_ratios_list:
            center_info = compute_psf_centers(
                M=self.M,
                overlap_ratio=self.overlap_ratio,
                center_blend=self.center_blend,
                z_ratio=z_ratio,
                N=N,
                device=self.device
            )
            
            psf_result = generate_gaussian_psf(
                centers_pixel=center_info['centers_pixel'],
                N=N,
                L=self.L,
                M=self.M,
                overlap_ratio=self.overlap_ratio,
                focal_length=self.focal_length,
                wavelength=self.wavelength,
                airy_correction=self.airy_correction,
                masks=None,  # depth_psfs不需要mask版本
                normalize=True,
                device=self.device
            )
            depth_psfs_list.append(psf_result['total_psf'])
        
        # 堆叠成3D张量
        depth_psfs = torch.stack(depth_psfs_list, dim=0)  # [len(z_ratios_list), N, N]
        
        # 4. 可视化
        if visualize:
            from visualization import visualize_lenses_and_tiles
            visualize_lenses_and_tiles(
                tiles=tiles,
                M=self.M,
                stride_norm=center_info_z0['stride_norm'],
                region_size_norm=center_info_z0['region_size_norm'],
                mask_count=self.mask_count,
                display_lens_idx=display_lens_idx
            )
        
        return {
            'total_psfs': psf_result_z0['total_psf'],
            'mask_psfs': psf_result_z0['mask_psfs'],
            'depth_psfs': depth_psfs,
            'centers_pixel': centers_pixel_z0,
            'masks': masks
        }