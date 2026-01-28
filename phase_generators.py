"""
提供相位图生成类，支持菲涅尔透镜和优化算法。
所有优化逻辑已合并入类中。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import textwrap
import os
import time
import config
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from optics_utils import create_checkerboard
from optics_utils import compute_psf_centers, generate_tile_masks, generate_gaussian_psf
from optics_utils import generate_lens_circular_masks,compute_psf_centers
from visualization import visualize_depth_psfs_and_masks, plot_psf_row
        
import wave_propagation as wp
import math

class PhaseGenerator:
    def __init__(self, params, device=torch.device('cuda'), mode='fresnel'):
        self.shape = params['shape']
        self.N = params['N']  # ROI size (square)
        self.output_size = params['output_size'] # output fov (square)
        self.roi_center_x = params['roi_center_x']
        self.roi_center_y = params['roi_center_y']
        self.M = params['M']   # Array size (square)
        self.focal_length = params['focal_length'] 
        self.wavelength = config.WAVELENGTH
        self.pixel_size = config.PIXEL_SIZE
        self.two_pi_value = params['two_pi_value']
        self.psf_energy_level = params['psf_energy_level']
        self.dof_correction = params['dof_correction']
        self.airy_correction = params['airy_correction']
        self.masked_airy_correction = params['masked_airy_correction']
        self.focusing_eff_correction = params['focusing_eff_correction']
        self.overlap_ratio = params['overlap_ratio']
        self.mask_count = int(params['mask_count'])
        self.center_blend = params['center_blend']
        self.interleaving = params['interleaving']
        self.ni = params['ni']
        self.lr = params['lr']
        self.show_iters = params['show_iters']
        self.weights = params['weights']
        self.randomness = params.get('randomness', 0.0)
        self.random_seed = params.get('random_seed', None)
        self.device = device
        print(f"Using device: {self.device}")
        if self.randomness > 0:
            print(f"PSF randomization enabled: randomness={self.randomness}, seed={self.random_seed}")

        self.depth_in_focus_dof = params['depth_in_focus']
        self.depth_out_focus_ratio = params['depth_out_focus']
        
        # simple calculations for other parameters
        self.roi_rect = None
        self.L = None
        self.lens_width = None
        self.f_number = None
        self.depth_of_focus = None
        self.airy_radius = None
        self.depth_in_focus_ratio = None
        self.depth_in_focus = None
        self._update_parameters(verbose=True, mode=mode)
        
        # Results (set after generate)
        self.phase = None
        self.phase_8bit = None
        self.history = None  # Only for optimized
        self.total_psfs = None
        self.centers_pixel = None
        
        # Optimization attributes (set in _optimize_phase)
        self.upsampling = 1.0
        self.phase_param = None # raw phase in nn.parameters
        self.U_masked = None
        self.total_psfs_up = None
        self.mask_psfs_up = None
        self.centers_pixel_up = None
        self.depth_psfs_up = None
        self.centers_pixel_out_focus_up = None
        self.out_focus_masks_up = None

    def generate_fresnel_phase(self,N=None,output_size=None) -> np.ndarray:
        """
        生成菲涅尔相位。
        输出像素为output_size
        但各菲涅尔透镜的对称中心由N输入像素的等分决定

        Usage 1: generate_fresnel_phase()
        Usage 2: generate_fresnel_phase(N=N)      
        Usage 3: generate_fresnel_phase(N=N,output_size=output_size)        
        """

        if N is None:
            N = self.N
            if output_size is None:
                # Use generate_fresnel_phase()  
                output_size = self.output_size
        else:
            # Use generate_fresnel_phase(N=N)  
            if output_size is None:
                output_size = N


        print(f"Fresnel Lens: {N}x{N} px; {self.M}x{self.M} lenses")
        if output_size!=N:
            print(f"Different Output size: {output_size}x{output_size}. Will generate de-centered Fresnel Lens.")
        
        y_grid, x_grid = np.indices((N, N))
        phase = np.zeros((N, N))
        # 计算透镜阵列的物理步长 (每个子透镜占据的像素数)
        sub_lens_pitch = N / self.M  # 注意这里是浮点数，切片时需要取整
        # 计算光学中心的步长 (根据 output_size 缩放)
        optical_pitch = output_size / self.M
        # 计算整体偏移量，确保 output_size 区域居中于 N
        global_offset = (N - output_size) / 2.0
        # 预计算焦距平方，避免循环中重复计算
        f_squared = self.focal_length**2
        k = 2 * np.pi / self.wavelength

        for r in range(self.M):
            for c in range(self.M):
                # --- 1. 计算物理切片范围 (解决 N%M != 0 的问题) ---
                y_start = int(round(r * sub_lens_pitch))
                y_end = int(round((r + 1) * sub_lens_pitch))
                x_start = int(round(c * sub_lens_pitch))
                x_end = int(round((c + 1) * sub_lens_pitch))
                
                # 强制最后一个透镜填满边缘 (防止浮点误差导致的 1px 缝隙)
                if r == self.M - 1: y_end = N
                if c == self.M - 1: x_end = N

                # --- 2. 计算该子透镜的光学中心 (Optical Center) ---
                # 逻辑：起始偏移 + (当前索引 + 0.5) * 光学步长
                center_y_px = global_offset + (r + 0.5) * optical_pitch
                center_x_px = global_offset + (c + 0.5) * optical_pitch

                # --- 3. 计算相位 ---
                region = (slice(y_start, y_end), slice(x_start, x_end))
                
                # 获取当前区域的全局像素坐标
                # 注意：这里直接利用网格值减去中心值
                y_coords = y_grid[region]
                x_coords = x_grid[region]

                # 转换为物理距离 (m)
                y_dist = (y_coords - center_y_px) * self.pixel_size
                x_dist = (x_coords - center_x_px) * self.pixel_size
                
                r_squared_val = x_dist**2 + y_dist**2
                
                # 球面波相位公式: 2pi/lambda * (f - sqrt(f^2 + r^2))
                # 注意：如果是聚焦透镜，相位通常是负的 (中心最厚/相位最大，边缘相位滞后)
                # 下面这个公式是标准的球面透镜相位分布
                phase_calc = k * (self.focal_length - np.sqrt(f_squared + r_squared_val))

                phase[region] = phase_calc
        return phase % (2 * np.pi)

    def forward(self, U_in=None, z=None, upsampling=1.0) -> torch.Tensor:
        """
        Single forward propagation process. 
        z may be a list or scalar
        """
        U_focal = self.propagate(U_in=U_in, z=z, upsampling=upsampling)
        return torch.abs(U_focal)**2

    def propagate(self, U_in=None, z=None, upsampling=1.0, apply_phase=True, extent=None) -> torch.Tensor:
        """
        Single forward propagation process. 
        z may be a list or scalar
        """
        if self.phase_param is None:
            raise ValueError("Phase parameter not set. Run generate('optimized') first.")
        if extent is None:
            extent = (self.output_size -1)*self.pixel_size

        if z is None:
            z = self.focal_length

        N = int(self.N * upsampling)

        if U_in is not None:
            output_size = U_in.shape[-1]
        else:
            output_size = int(self.output_size * upsampling)
            U_in = torch.ones((output_size, output_size), device=self.device, dtype=torch.complex64)
        
        if apply_phase:
            # load phase with [1, 1, N, N]
            phase = self.phase_param.unsqueeze(0).unsqueeze(0)  

            if self.output_size != self.N:
                # phase need zero padding first to self.output_size
                padding_left = (self.output_size - self.N) // 2
                padding_right = self.output_size - self.N - padding_left
                padding_top = (self.output_size - self.N) // 2
                padding_bottom = self.output_size - self.N - padding_top
                phase = F.pad(phase, 
                              (padding_left, padding_right, padding_top, padding_bottom), 
                              mode='constant', value=0)
                # if phase was padded, need to set outside region amplitude to zero
                # so we need a mask to apply after phase modulation
                mask = torch.ones((1, 1, self.N, self.N), 
                                    device=self.device)
                mask = F.pad(mask.float(), 
                              (padding_left, padding_right, padding_top, padding_bottom), 
                              mode='constant', value=0)
            if self.output_size != output_size:
                # phase need upsampling to output_size 
                # use 'nearest' to avoid smoothing
                phase = F.interpolate(phase, size=(output_size,output_size), mode='nearest')
                
                if self.output_size != self.N:
                    # also need to update existing mask
                    mask = F.interpolate(mask, 
                                         size=(output_size,output_size), 
                                         mode='nearest')
            
            phase = phase.squeeze()  # [output_size, output_size]
            # apply phase
            U_phase = U_in * torch.exp(1j * phase)
            # apply mask if needed
            if self.output_size != self.N:
                U_phase = U_phase * mask.squeeze()
                
        else:
            U_phase = U_in
        
        U_focal = wp.propagate_ASM(U_phase, z, extent, self.wavelength, self.device)
        return U_focal
    
    def compute_loss(self) -> tuple[torch.Tensor, dict]:
        """
        Compute losses including focal plane, depth planes, and out-of-focus centroid losses.
        Using upsampling factor for high-resolution evaluation
            
        Returns
        -------
        total_loss : torch.Tensor
            Combined weighted loss
        loss_components : dict
            Individual loss component values for monitoring
        """
        if self.upsampling != 1.0:
            N = int(self.N * self.upsampling)
            output_size = int(self.output_size * self.upsampling)
            pixel_size = self.pixel_size / self.upsampling
        else:
            N = self.N
            output_size = self.output_size
            pixel_size = self.pixel_size
            
        loss_fn = nn.MSELoss()
        loss_components = {}
        loss_terms = []
        
        # ========== 1. MSE loss for focal plane ==========
        I_focal_full = self.forward(z=self.focal_length, upsampling=self.upsampling)
        mse = loss_fn(I_focal_full, self.total_psfs_up)
        loss_terms.append(mse * self.weights['mse'])
        loss_components['focal_mse'] = loss_terms[-1].item()

        # ========== 2. Depth MSE Loss for in-focus planes ==========
        if hasattr(self, 'depth_in_focus_ratio') and self.depth_in_focus_ratio is not None:
            # Use depth_in_focus_ratio if available
            z_list = [self.focal_length * ratio for ratio in self.depth_in_focus_ratio]
        else:
            z_list = None

        if z_list is not None:
            # Forward propagate to multiple depths 
            # (returns [num_depths, output_size, output_size])
            I_depth_planes = self.forward(z=z_list, upsampling=self.upsampling)
            
            # Compute MSE with target depth PSFs
            depth_mse = loss_fn(I_depth_planes, self.depth_psfs_up)
            loss_terms.append(depth_mse * self.weights['depth_in_focus'])
            loss_components['depth_mse'] = loss_terms[-1].item()
        
        # ========== 3. Centroid loss for out-of-focus planes ==========
        if hasattr(self, 'depth_out_focus_ratio') and self.depth_out_focus_ratio is not None:
            # Propagate to out-of-focus depths
            z_out_list = [self.focal_length * ratio for ratio in self.depth_out_focus_ratio]
            # [num_out, output_size, output_size]
            I_out_focus_planes = self.forward(z=z_out_list, upsampling=self.upsampling)  
            
            num_out_focus = len(z_out_list)
            num_lenses = self.M * self.M
            
            # Create coordinate grids
            y_grid, x_grid = torch.meshgrid(
                torch.arange(output_size, device=self.device, dtype=torch.float32),
                torch.arange(output_size, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            
            centroid_distances = []
            
            for i in range(num_out_focus):
                I_plane = I_out_focus_planes[i]  # [output_size, output_size]
                
                for j in range(num_lenses):
                    # Get mask for this depth and lens
                    mask = self.out_focus_masks_up[i, j]  # [output_size, output_size]
                    
                    # Extract intensity values within mask
                    I_masked = I_plane * mask  # [output_size, output_size]
                    total_intensity = I_masked.sum()
                    
                    # Compute centroid (weighted average position)
                    centroid_x = (I_masked * x_grid).sum() / total_intensity
                    centroid_y = (I_masked * y_grid).sum() / total_intensity
                    
                    # Target center position
                    target_x = self.centers_pixel_out_focus_up[i, j, 0]
                    target_y = self.centers_pixel_out_focus_up[i, j, 1]
                    
                    # Euclidean distance
                    distance = torch.sqrt((centroid_x - target_x) ** 2 + 
                                        (centroid_y - target_y) ** 2)
                    centroid_distances.append(distance)
            
            # Average centroid deviation
            centroid_loss = torch.stack(centroid_distances).mean()
            if self.weights['depth_out_focus']>0:
                loss_terms.append(centroid_loss*self.weights['depth_out_focus'])
                loss_components['centroid_loss'] = loss_terms[-1].item()
        
        # ========== 4. Average focusing efficiency loss for focal plane ==========
        x_grid_eff, y_grid_eff = torch.meshgrid(
            torch.arange(output_size, device=self.device),
            torch.arange(output_size, device=self.device),
            indexing='ij'
        )
        efficiencies = []
        theoretical_efficiency = (N ** 2) / (self.M * self.M)
        
        for center in self.centers_pixel_up: 
            distances = torch.sqrt((x_grid_eff - center[0]) ** 2 + 
                                (y_grid_eff - center[1]) ** 2)
            mask = distances <= self.airy_radius * self.airy_correction * self.focusing_eff_correction / pixel_size
            encircled_energy = I_focal_full[mask].sum()
            efficiencies.append(encircled_energy / theoretical_efficiency)
        
        efficiencies = torch.stack(efficiencies)
        efficiency_mean = efficiencies.mean()
        efficiency_std = efficiencies.std()
        
        loss_terms.append(-efficiency_mean * self.weights['eff_mean'])
        loss_components['eff_mean'] = loss_terms[-1].item()        
        loss_terms.append(efficiency_std * self.weights['eff_std'])
        loss_components['eff_std'] = loss_terms[-1].item()

        loss_components['eff_mean (unweighted)'] = efficiency_mean.item()
        loss_components['eff_std (unweighted)'] = efficiency_std.item()
        
        
        # ========== 5. Masked loss ==========
        if self.mask_count > 0:
            I_focal_masked = self.forward(
                U_in=self.U_masked, 
                z=self.focal_length, 
                upsampling=self.upsampling
            )
            mse_masked = loss_fn(I_focal_masked, self.mask_psfs_up)
            # mse_masked = -torch.mean(I_focal_masked * self.mask_psfs_up)
            loss_terms.append(mse_masked * self.weights['masked'])
            loss_components['masked'] = loss_terms[-1].item()
        
         # ========== Combine all losses ==========
        total_loss = torch.sum(torch.stack(loss_terms))
        loss_components['total_loss'] = total_loss.item()

        return total_loss, loss_components
    
    def update_phase_8bit(self, two_pi_value=None) -> np.array:
        if two_pi_value is not None:
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

            self._prepare_template(upsampling=upsampling, 
                                   visualize=visualize)
            self._optimize_phase(init_phase=init_phase, 
                                 update_callback=vis_callback)
            
            self.phase = torch.remainder(self.phase_param, 2 * np.pi).detach().cpu().numpy()
            self._update_parameters(mode=mode)
            
        self._post_process_phase()
    
    
    def _update_parameters(self,mode='fresnel',verbose=False):
        # simple calculations
        self.L = (self.N-1) * self.pixel_size
        # calc ROI params
        # 获取SLM尺寸
        slm_height, slm_width = self.shape
        # 计算ROI边界
        roi_size = self.N # 使用N作为ROI大小的来源
        roi_center_x = self.roi_center_x
        roi_center_y = self.roi_center_y
        roi_left = max(0, int(roi_center_x - roi_size // 2))
        roi_right = min(slm_width, int(roi_center_x + roi_size // 2))
        roi_top = max(0, int(roi_center_y - roi_size // 2))
        roi_bottom = min(slm_height, int(roi_center_y + roi_size // 2))
        # 确保ROI是正方形
        actual_roi_width = roi_right - roi_left
        actual_roi_height = roi_bottom - roi_top
        self.N = min(actual_roi_width, actual_roi_height)
        self.roi_rect = (roi_left, roi_top, self.N, self.N)

        # Geometry with or w/o overlap
        if mode == 'fresnel':
            self.lens_width = self.L / self.M
            self.f_number = self.focal_length / self.lens_width
            self.depth_of_focus = 2*self.wavelength*self.f_number**2 
            self.airy_radius = 1.22*self.wavelength*self.f_number
            if verbose:
                print(f"F/{self.f_number:.2f}, {self.focal_length*1e3} mm")
                print(f"Lens width: {self.lens_width*1.0e3:.3f}mm")
                print(f"Airy radius: {self.airy_radius*1.0e6:.1f}um")
                print(f"Depth of focus: {self.depth_of_focus*1.0e3:.2f}mm")
        elif mode == 'optimized':
            region_size_norm = 1.0 / (self.M - (self.M - 1) * self.overlap_ratio)
            self.lens_width = self.L * region_size_norm
            self.f_number = self.focal_length / self.lens_width 
            self.depth_of_focus = 2*self.wavelength*self.f_number**2
            self.airy_radius = 1.22*self.wavelength*self.f_number
            if verbose:
                print(f"F/{self.f_number:.2f}, {self.focal_length*1e3} mm")
                print(f"Max. Lens width: {self.lens_width*1.0e3:.3f}mm")
                print(f"Lens width (Fresnel): {self.L/self.M*1.0e3:.3f}mm")
                print(f"Max. Overlap: {(self.lens_width/(self.L/self.M) - 1) *100:.1f} %")
                print('')
                lens_width_raw = self.L / self.M
                f_number_raw = self.focal_length / lens_width_raw
                airy_radius_raw = 1.22*self.wavelength*f_number_raw
                depth_of_focus_raw = 2*self.wavelength*f_number_raw**2
                print(f"Airy radius: {self.airy_radius*1.0e6:.1f}um")
                print(f"Airy radius (fresnel): {airy_radius_raw*1.0e6:.1f}um")
                print(f"Airy correction for Fresnel: {airy_radius_raw/self.airy_radius:.3f}")

                print(f"Depth of focus: {self.depth_of_focus*1.0e3:.2f}mm")
                print(f"Depth of focus (fresnel): {depth_of_focus_raw*1.0e3:.2f}mm")
                print(f"DOF correction for Fresnel: {depth_of_focus_raw/self.depth_of_focus:.2f}")
                print('')
        if self.depth_in_focus_dof is None:
            self.depth_in_focus_ratio = None
            self.depth_in_focus = None
            if verbose:
                print(f"Single depth plane is used at F={self.focal_length*1.0e3:.1f} mm")
        else:
            # self.depth_in_focus_ratio: 
            # convert to ratio (divided by focal length)
            self.depth_in_focus = [self.focal_length + d*self.depth_of_focus*self.dof_correction
                                    for d in self.depth_in_focus_dof]
            
            self.depth_in_focus_ratio = [d/self.focal_length
                                   for d in self.depth_in_focus]
            if verbose:
                print(f"Multi-depth planes are used at F={', '.join(f'{x*1.0e3:.1f} ({y:.1f}DOF)' for x,y in zip(self.depth_in_focus,self.depth_in_focus_dof))} mm")
        
        if (self.depth_out_focus_ratio is not None) and verbose:
            # already been ratio (divided by focal length)
            depth_out_focus_m = [self.focal_length * d
                                    for d in self.depth_out_focus_ratio]
            print(f"Out-of-focus planes are used at Z={', '.join(f'{x*1.0e3:.1f} ({y:.2f}x)' for x,y in zip(depth_out_focus_m, self.depth_out_focus_ratio))} mm")
        
    
    def _optimize_phase(self, 
                        init_phase: torch.Tensor = None, 
                        update_callback=None):
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
            total_loss, loss_components = self.compute_loss()
            total_loss.backward()
            optimizer.step()
            # Record history
            for key, value in loss_components.items():
                self.history[key].append(value)
            # Callback
            if (i % self.show_iters == 0 or i == self.ni - 1):
                if update_callback:
                    update_callback(i, self.ni, total_loss.item(), self.phase_param)
                else:
                    header = f"Iter: {i+1}/{self.ni}"
                    max_width = config.TEXT_WIDTH_WRAP  # 终端宽度
                    loss_str = " ".join([f"{k.capitalize()}: {v:.4f}" for k, v in loss_components.items() if k != 'total_loss'])
                    loss_str += f" Total: {loss_components['total_loss']:.4e}" if 'total_loss' in loss_components else ""
                    if len(header) + len(loss_str) + 1 > max_width:
                        print(header)
                        wrapped_lines = textwrap.wrap(loss_str, width=max_width - 2)
                        for line in wrapped_lines:
                            print(f"  {line}")
                    else:
                        print(f"{header} {loss_str}")
        
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
        - self.depth_in_focus_ratio: z_ratio列表，用于生成depth_psfs
        - self.depth_out_focus_ratio: z_ratio列表，用于生成离焦中心坐标
        """
        # 设置上采样因子
        self.upsampling = upsampling

         # 计算上采样后的参数
        N_up = int(self.N * self.upsampling)
        output_size_up = int(self.output_size * self.upsampling)
        pixel_size_up = self.pixel_size / self.upsampling
        
        # 0. 非上采样版本，生成in-focus的多平面PSF（包含depth_psfs）
        results = self._create_gaussian_template(
            upsampling=1.0,
            visualize=False,
            z_ratios=self.depth_in_focus_ratio
        )
        self.total_psfs = results['total_psfs']
        self.centers_pixel = results['centers_pixel']

        # 1. 上采样版本，生成in-focus的多平面PSF（包含depth_psfs）
        results = self._create_gaussian_template(
            upsampling=upsampling,
            visualize=visualize,
            z_ratios=self.depth_in_focus_ratio
        )
        if self.mask_count>0:
            if visualize:
                plot_psf_row(results['mask_psfs'])
            self.mask_psfs_up = results['mask_psfs'] * self.psf_energy_level
            masks = results['masks']
            # Incident modulation for optimization
            self.U_masked = masks.to(device=self.device, dtype=torch.complex64)

        self.total_psfs_up = results['total_psfs'] * self.psf_energy_level
        self.depth_psfs_up = results['depth_psfs'] * self.psf_energy_level  # [num_depths, N, N]
        self.centers_pixel_up = results['centers_pixel']
        
        
        # 2. 上采样版本，out-of-focus中心和mask
        if self.depth_out_focus_ratio is not None:
            # 生成out-of-focus的中心坐标（仅坐标，不需要PSF）
            # 使用相同的randomness和seed以保持一致性
            centers_out_focus_list = []
            for z_ratio in self.depth_out_focus_ratio:
                center_info = compute_psf_centers(
                    M=self.M,
                    overlap_ratio=self.overlap_ratio,
                    center_blend=self.center_blend,
                    z_ratio=z_ratio,
                    N=N_up,
                    output_size=output_size_up,
                    device=self.device,
                    randomness=self.randomness,
                    random_seed=self.random_seed
                )
                centers_out_focus_list.append(center_info['centers_pixel'])
            # [num_out_focus, M*M, 2]
            self.centers_pixel_out_focus_up = torch.stack(centers_out_focus_list, dim=0)

            # 上采样版本，生成out-of-focus的mask
            region_size_norm = 1.0 / (self.M - (self.M - 1) * self.overlap_ratio)
            lens_width_px_up = region_size_norm * (N_up - 1)
            # 考虑衍射airy blur半径
            radii_list = []
            for z in self.depth_out_focus_ratio:
                half_width = lens_width_px_up * (1.0 - z) / 2.0
                radius = math.sqrt(half_width ** 2 + (self.airy_radius/pixel_size_up) ** 2)
                radii_list.append(radius)
            radii_up = torch.tensor(radii_list, device=self.device, dtype=torch.float32)
            # [num_out_focus, M*M, output_size_up, output_size_up]
            self.out_focus_masks_up = generate_lens_circular_masks(
                centers_pixel=self.centers_pixel_out_focus_up,
                radii_pixels=radii_up,
                N=output_size_up,
                device=self.device
            )
            
            if visualize:
                out_focus_masks = self.out_focus_masks_up if self.depth_out_focus_ratio is not None else None
                depth_out_focus_ratio = self.depth_out_focus_ratio if self.depth_out_focus_ratio is not None else None

                visualize_depth_psfs_and_masks(
                    depth_psfs=self.depth_psfs_up,
                    depth_in_focus_ratio=self.depth_in_focus_ratio,
                    out_focus_masks=out_focus_masks,
                    depth_out_focus_ratio=depth_out_focus_ratio,
                    figsize=(12, 4)
                )
        

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
        # 输出大小
        output_size = int(self.output_size * upsampling)
        # 输入透镜尺寸
        N = int(self.N * upsampling)

        if z_ratios is None:
            z_ratios = [1.0]
        z_ratios_list = list(z_ratios)
        
        if self.mask_count > 0:
            # 1. 生成masks（只需要一次）
            mask_info = generate_tile_masks(
                M=self.M,
                L=self.L,
                overlap_ratio=self.overlap_ratio,
                center_blend=self.center_blend,
                mask_count=self.mask_count,
                interleaving=self.interleaving,
                N=N,
                output_size=output_size,
                coarse_grid_size=coarse_grid_size,
                device=self.device
            )
            masks = mask_info['masks']  # 输出坐标
            tiles = mask_info['tiles']  # 归一化透镜坐标
            a_lens_mask = mask_info['a_lens_mask'] # 归一化输出比例
        else:
            masks = None
            a_lens_mask = None
        
        # 2. 计算z_ratio=1.0 时的PSF中心(输出坐标)
        center_info_z0 = compute_psf_centers(
            M=self.M,
            overlap_ratio=self.overlap_ratio,
            center_blend=self.center_blend,
            z_ratio=1.0,
            N=N,
            output_size=output_size,
            device=self.device,
            randomness=self.randomness,
            random_seed=self.random_seed
        )
        centers_pixel_z0 = center_info_z0['centers_pixel']
        
        #  计算z_ratio=1时的PSF
        # 注意是输出坐标，因此N用output_size
        psf_result_z0 = generate_gaussian_psf(
            centers_pixel=centers_pixel_z0,
            N=output_size,
            L=(self.output_size -1)*self.pixel_size,
            M=self.M,
            overlap_ratio=self.overlap_ratio,
            focal_length=self.focal_length,
            wavelength=self.wavelength,
            airy_correction=self.airy_correction,
            masked_airy_correction=self.masked_airy_correction,
            masks=masks,
            a_lens_mask=a_lens_mask,
            normalize=True,
            device=self.device
        )
        if self.mask_count == 0:
            psf_result_z0['mask_psfs'] = None
        
        # 3. 计算不同z_ratio的PSF
        depth_psfs_list = []
        for z_ratio in z_ratios_list:
            # 也用输出坐标
            center_info = compute_psf_centers(
                M=self.M,
                overlap_ratio=self.overlap_ratio,
                center_blend=self.center_blend,
                z_ratio=z_ratio,
                N=N,
                output_size=output_size,
                device=self.device,
                randomness=self.randomness,
                random_seed=self.random_seed
            )
            # 注意是输出坐标，因此N用output_size
            psf_result = generate_gaussian_psf(
                centers_pixel=center_info['centers_pixel'],
                N=output_size,
                L=(self.output_size -1)*self.pixel_size,
                M=self.M,
                overlap_ratio=self.overlap_ratio,
                focal_length=self.focal_length,
                wavelength=self.wavelength,
                airy_correction=self.airy_correction,
                masked_airy_correction=self.masked_airy_correction,
                masks=None,  # depth_psfs不需要mask版本
                normalize=True,
                device=self.device
            )
            depth_psfs_list.append(psf_result['total_psf'])
        
        # 堆叠成3D张量
        depth_psfs = torch.stack(depth_psfs_list, dim=0)  # [len(z_ratios_list), output_size, output_size]
        
        # 4. 可视化
        if visualize and self.mask_count>0:
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
    
    
    def compute_pupil_function(self, pupil_N=None, pupil_pixel_size=None, 
                               upsampling=3.0, 
                               z=None, 
                               verbose=True) -> np.ndarray:
        """
        计算阵列的pupil function
        
        基于给定的pupil空间范围和采样间距，通过以下步骤计算:
        1. 根据pupil extent确定传播时的CSF采样间距
        2. 通过上采样参数size实现该采样间距
        3. 对每个微透镜单元独立计算CSF->ATF->Pupil的转换
        4. 将各微透镜的pupil function组合输出
        
        参数:
            pupil_N: pupil function的空间采样点数
            pupil_pixel_size: pupil function的采样间距 (单位: 米)
            z: 传播距离 (单位: 米)，默认为焦距
            upsampling: 传播时的上采样因子
            verbose: 是否打印计算信息
        
        返回:
            pupil_array: [M*M, H, W] 的numpy数组，复数类型
                        每个[i, :, :]对应第i个微透镜的pupil function
        """
        if pupil_N is None:
            pupil_N = self.N
        if pupil_pixel_size is None:
            pupil_pixel_size = self.pixel_size
        pupil_extent = pupil_N * pupil_pixel_size
        
        if z is None:
            z = self.focal_length
        
        # ===== 步骤1: 以固定上采样传播得到CSF =====
        csf_full = self.propagate(z=z, upsampling=upsampling).detach()
        
        # ===== 步骤2: 获取上采样后的csf焦斑中心坐标和pupil中心坐标 =====
        results = compute_psf_centers(
            M=self.M,
            overlap_ratio=self.overlap_ratio,
            center_blend=self.center_blend,
            z_ratio=1.0,
            N=csf_full.shape[-1],
            device=self.device
        )
        centers_pixel_csf = torch.floor(results['centers_pixel']).to(torch.int64)  # [M*M, 2]

        # ===== 步骤3: 计算采样要求 =====
        current_pixel_size = self.pixel_size / upsampling
        current_pupil_extent = csf_full.shape[-1] * current_pixel_size
        if verbose:
            print(f"Computing pupil extent: {pupil_extent*1e3:.2f} mm")
            print(f"CSF extent before padding: {current_pupil_extent*1e3:.2f} mm")
        if pupil_pixel_size < current_pixel_size:
            # 警告：传播时的CSF采样间距大于需要的采样间距，可能导致欠采样
            print(f"Minimum CSF pixel size required ({pupil_pixel_size*1e6:.2f} um).")
            print(f"Current CSF pixel size ({current_pixel_size * 1e6:.2f} um).")
            print(f"Recommended upsampling >= {self.pixel_size / pupil_pixel_size:.2f} during propagation to avoid undersampling.")

        # ===== 步骤4: 为每个微透镜计算pupil function ====
        patch_size_csf = csf_full.shape[-1] // self.M
        patch_size_csf_half = patch_size_csf//2 - 1
        num_lens = self.M * self.M
        pupil_array = np.zeros((num_lens, pupil_N, pupil_N), dtype=np.complex64)
        atf_array = np.zeros((num_lens, pupil_N, pupil_N), dtype=np.complex64)
        for i in range(num_lens):
            # 获取当前微透镜的焦点位置
            cy, cx = centers_pixel_csf[i]
            # 提取以焦点为中心的CSF区域（大小为patch_size_csf）
            csf_local = csf_full[cy - patch_size_csf_half : cy + patch_size_csf_half + 1,
                                 cx - patch_size_csf_half : cx + patch_size_csf_half + 1]
            # 把CSF patch放回全局大小，其他位置为0
            csf_local_full = torch.zeros_like(csf_full)
            csf_local_full[cy - patch_size_csf_half : cy + patch_size_csf_half + 1,
                           cx - patch_size_csf_half : cx + patch_size_csf_half + 1] = csf_local
            # 把CSF进一步补零到与pupil大小相同
            if pupil_extent > current_pupil_extent:
                pad_size = int((pupil_extent/current_pixel_size - csf_local_full.shape[-1]) / 2)
                csf_local_full = F.pad(csf_local_full, (pad_size, pad_size, pad_size, pad_size))
                padded_size = csf_local_full.shape[-1]
                padded_extent = padded_size * current_pixel_size
            else:
                padded_extent = current_pupil_extent
                padded_size = csf_local_full.shape[-1]

            # 反向传播到透镜面，再去除透镜相位，得到Pupil function
            pupil_local = self.propagate(U_in=csf_local_full, z=-z, 
                                         upsampling=1.0, 
                                         apply_phase=False, 
                                         extent=pupil_extent)
            # 定义坐标
            y = (torch.arange(padded_size, device=self.device) - padded_size // 2) * current_pixel_size
            x = (torch.arange(padded_size, device=self.device) - padded_size // 2) * current_pixel_size
            X, Y = torch.meshgrid(x, y, indexing='xy')
            # 生成全局透镜相位
            r_squared = X**2 + Y**2
            f_squared = self.focal_length**2
            lens_phase = (2 * np.pi / self.wavelength) * (self.focal_length - torch.sqrt(f_squared + r_squared))
            lens_phase_complex = torch.exp(1j * lens_phase)
            # 去除透镜相位
            pupil_local = pupil_local * torch.conj(lens_phase_complex)

            # 进行精确插值，满足预定的pupil采样间距和大小
            y_coords = torch.linspace(-pupil_extent/padded_extent,
                                    pupil_extent/padded_extent,
                                    pupil_N)
            x_coords = torch.linspace(-pupil_extent/padded_extent,
                                    pupil_extent/padded_extent,
                                    pupil_N)
            # y_coords = torch.arange(-pupil_extent/padded_extent,
            #                         pupil_extent/padded_extent,
            #                         step=2*pupil_pixel_size/padded_extent)
            # x_coords = torch.arange(-pupil_extent/padded_extent,
            #                         pupil_extent/padded_extent,
            #                         step=2*pupil_pixel_size/padded_extent)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            # 执行复数插值，满足预定的pupil采样间距和大小
            pupil_local_real = torch.real(pupil_local).unsqueeze(0).unsqueeze(0)
            pupil_local_imag = torch.imag(pupil_local).unsqueeze(0).unsqueeze(0)
            pupil_local_real = F.grid_sample(pupil_local_real, grid, mode='bilinear', align_corners=True)
            pupil_local_imag = F.grid_sample(pupil_local_imag, grid, mode='bilinear', align_corners=True)
            pupil_local = (pupil_local_real + 1j * pupil_local_imag).squeeze(0).squeeze(0).cpu().numpy()           

            # 保存到输出数组
            pupil_array[i, :, :] = pupil_local

            if verbose:
                print(f"    Processed lenslet {i+1}/{num_lens}")

        if verbose:
            print(f"  Done! Output shape: {pupil_array.shape}")
        return pupil_array
    
    def compute_transfer_function(self, N=None, pixel_size=None, 
                               upsampling=3.0, 
                               z=None, 
                               verbose=True) -> np.ndarray:
        """
        计算阵列的amplitude transfer function
        
        基于给定的ATF空域范围和空域间距计算
        
        参数:
            N: pupil function的空间采样点数
            pixel_size: pupil function的采样间距 (单位: 米)
            z: 传播距离 (单位: 米)，默认为焦距
            upsampling: 传播时的上采样因子
            verbose: 是否打印计算信息
        
        返回:
            atf_array: [M*M, H, W] 的numpy数组，复数类型
                        每个[i, :, :]对应第i个微透镜的ATF
        """
        if N is None:
            N = self.N
        if pixel_size is None:
            pixel_size = self.pixel_size
        spatial_extent = N * pixel_size
        target_atf_extent = 1/pixel_size
        target_atf_pixel_size = 1/spatial_extent

        if z is None:
            z = self.focal_length
        
        # ===== 步骤1: 以固定上采样传播得到CSF =====
        csf_full = self.propagate(z=z, upsampling=upsampling).detach()
        
        # ===== 步骤2: 获取上采样后的csf焦斑中心坐标和pupil中心坐标 =====
        results = compute_psf_centers(
            M=self.M,
            overlap_ratio=self.overlap_ratio,
            center_blend=self.center_blend,
            z_ratio=1.0,
            N=csf_full.shape[-1],
            device=self.device
        )
        centers_pixel_csf = torch.floor(results['centers_pixel']).to(torch.int64)  # [M*M, 2]

        # ===== 步骤3: 计算采样要求 =====
        current_pupil_pixel_size = self.pixel_size / upsampling
        current_pupil_extent = csf_full.shape[-1] * current_pupil_pixel_size
        current_atf_extent = current_pupil_extent / (self.wavelength * z)
        current_atf_pixel_size = current_pupil_pixel_size / (self.wavelength * z)
        target_pupil_pixel_size = target_atf_pixel_size * (self.wavelength * z)
        target_pupil_extent = target_atf_extent * (self.wavelength * z)
        if current_atf_pixel_size > target_atf_pixel_size:
            # 警告：传播时的CSF采样间距大于需要的采样间距，可能导致欠采样
            print(f"Minimum CSF pixel size required ({target_pupil_pixel_size*1e6:.2f} um).")
            print(f"Current CSF pixel size ({current_pupil_pixel_size * 1e6:.2f} um).")
            print(f"Recommended upsampling >= {self.pixel_size / target_pupil_pixel_size:.2f} during propagation to avoid undersampling.")

        # ===== 步骤4: 为每个微透镜计算pupil function ====
        patch_size_csf = csf_full.shape[-1] // self.M
        patch_size_csf_half = patch_size_csf//2 - 1
        num_lens = self.M * self.M
        atf_array = np.zeros((num_lens, N, N), dtype=np.complex64)
        for i in range(num_lens):
            # 获取当前微透镜的焦点位置
            cy, cx = centers_pixel_csf[i]
            # 提取以焦点为中心的CSF区域（大小为patch_size_csf）
            csf_local = csf_full[cy - patch_size_csf_half : cy + patch_size_csf_half + 1,
                                 cx - patch_size_csf_half : cx + patch_size_csf_half + 1]
            # 把CSF patch放回全局大小，其他位置为0
            csf_local_full = torch.zeros_like(csf_full)
            csf_local_full[cy - patch_size_csf_half : cy + patch_size_csf_half + 1,
                           cx - patch_size_csf_half : cx + patch_size_csf_half + 1] = csf_local
            
            # 反向传播到透镜面，再去除透镜相位，得到Pupil function
            pupil_local = self.propagate(U_in=csf_local_full, z=-z, 
                                         upsampling=1.0, 
                                         apply_phase=False)
            # 定义坐标
            y = (torch.arange(csf_local_full.shape[0], device=self.device) - csf_local_full.shape[0] // 2) * current_pupil_pixel_size
            x = (torch.arange(csf_local_full.shape[1], device=self.device) - csf_local_full.shape[1] // 2) * current_pupil_pixel_size
            X, Y = torch.meshgrid(x, y, indexing='xy')
            # 生成全局透镜相位
            r_squared = X**2 + Y**2
            f_squared = self.focal_length**2
            lens_phase = (2 * np.pi / self.wavelength) * (self.focal_length - torch.sqrt(f_squared + r_squared))
            lens_phase_complex = torch.exp(1j * lens_phase)
            # 去除透镜相位
            pupil_local = pupil_local * torch.conj(lens_phase_complex)

            # 进行精确插值，满足预定的pupil采样间距和大小
            y_coords = torch.linspace(-target_pupil_extent/current_pupil_extent,
                                    target_pupil_extent/current_pupil_extent,
                                    N)
            x_coords = torch.linspace(-target_pupil_extent/current_pupil_extent,
                                    target_pupil_extent/current_pupil_extent,
                                    N)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            # 执行复数插值，满足预定的pupil采样间距和大小
            pupil_local_real = torch.real(pupil_local).unsqueeze(0).unsqueeze(0)
            pupil_local_imag = torch.imag(pupil_local).unsqueeze(0).unsqueeze(0)
            pupil_local_real = F.grid_sample(pupil_local_real, grid, 
                                             mode='bilinear', align_corners=True,
                                             padding_mode="zeros")
            pupil_local_imag = F.grid_sample(pupil_local_imag, grid, 
                                             mode='bilinear', align_corners=True,
                                             padding_mode="zeros")
            pupil_local = (pupil_local_real + 1j * pupil_local_imag).squeeze(0).squeeze(0).cpu().numpy()           

            # 反转pupil，得到atf
            atf_array[i, :, :] = pupil_local[-1::-1,-1::-1]

            if verbose:
                print(f"    Processed lenslet {i+1}/{num_lens}")

        if verbose:
            print(f"  Done! Output shape: {atf_array.shape}")
        return atf_array
    def compute_local_transfer_function(self, N=None, pixel_size=None, 
                               upsampling=3.0, 
                               z=None, 
                               verbose=True) -> np.ndarray:
        """
        计算阵列的amplitude transfer function
        
        基于给定的ATF空域范围和空域间距计算，以每个子图像中心作为坐标中心
        
        参数:
            N: pupil function的空间采样点数
            pixel_size: pupil function的采样间距 (单位: 米)
            z: 传播距离 (单位: 米)，默认为焦距
            upsampling: 传播时的上采样因子
            verbose: 是否打印计算信息
        
        返回:
            atf_array: [M*M, H, W] 的numpy数组，复数类型
                        每个[i, :, :]对应第i个微透镜的ATF
        """
        if N is None:
            N = self.N
        if pixel_size is None:
            pixel_size = self.pixel_size
        spatial_extent = N * pixel_size
        target_atf_extent = 1/pixel_size
        target_atf_pixel_size = 1/spatial_extent

        if z is None:
            z = self.focal_length
        
        # ===== 步骤1: 以固定上采样传播得到CSF =====
        csf_full = self.propagate(z=z, upsampling=upsampling).detach()
        
        # ===== 步骤2: 获取上采样后的csf焦斑中心坐标和pupil中心坐标 =====
        results = compute_psf_centers(
            M=self.M,
            overlap_ratio=self.overlap_ratio,
            center_blend=self.center_blend,
            z_ratio=1.0,
            N=csf_full.shape[-1],
            device=self.device
        )
        centers_pixel_csf = torch.floor(results['centers_pixel']).to(torch.int64)  # [M*M, 2]

        # ===== 步骤3: 计算采样要求 =====
        csf_pixel_size = self.pixel_size / upsampling
        csf_extent = csf_full.shape[-1] * csf_pixel_size
        current_atf_extent = 1/csf_pixel_size
        current_atf_pixel_size = 1/csf_extent

        if current_atf_extent < target_atf_extent:
            # 警告：传播时的CSF采样间距大于需要的采样间距，可能导致欠采样
            print(f"Minimum CSF pixel size required ({1/target_atf_extent*1e6:.2f} um).")
            print(f"Current CSF pixel size ({csf_pixel_size * 1e6:.2f} um).")
            print(f"Recommended upsampling >= {self.pixel_size * target_atf_extent:.2f} during propagation to avoid undersampling.")

        # ===== 步骤4: 为每个微透镜计算pupil function ====
        patch_size_csf = csf_full.shape[-1] // self.M
        patch_size_csf_half = patch_size_csf//2 - 1
        num_lens = self.M * self.M
        atf_array = np.zeros((num_lens, N, N), dtype=np.complex64)
        for i in range(num_lens):
            # 获取当前微透镜的焦点位置
            cy, cx = centers_pixel_csf[i]
            # 提取以焦点为中心的CSF区域（大小为patch_size_csf）
            csf_local = csf_full[cy - patch_size_csf_half : cy + patch_size_csf_half + 1,
                                 cx - patch_size_csf_half : cx + patch_size_csf_half + 1]
            # 把CSF patch放回全局大小，其他位置为0
            csf_local_full = torch.zeros_like(csf_full)
            csf_local_full[cy - patch_size_csf_half : cy + patch_size_csf_half + 1,
                           cx - patch_size_csf_half : cx + patch_size_csf_half + 1] = csf_local
            
            # 不用反向传播
            # 直接对CSF进行变换，得到local ATF
            atf_local = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(csf_local_full)))

            # 进行精确插值，满足预定的pupil采样间距和大小
            y_coords = torch.linspace(-target_atf_extent/current_atf_extent,
                                    target_atf_extent/current_atf_extent,
                                    N)
            x_coords = torch.linspace(-target_atf_extent/current_atf_extent,
                                    target_atf_extent/current_atf_extent,
                                    N)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            # 执行复数插值，满足预定的pupil采样间距和大小
            atf_local_real = torch.real(atf_local).unsqueeze(0).unsqueeze(0)
            atf_local_imag = torch.imag(atf_local).unsqueeze(0).unsqueeze(0)
            atf_local_real = F.grid_sample(atf_local_real, grid, 
                                             mode='bilinear', align_corners=True,
                                             padding_mode="zeros")
            atf_local_imag = F.grid_sample(atf_local_imag, grid, 
                                             mode='bilinear', align_corners=True,
                                             padding_mode="zeros")
            atf_local = (atf_local_real + 1j * atf_local_imag).squeeze(0).squeeze(0).cpu().numpy()           

            # 得到atf
            atf_array[i, :, :] = atf_local

            if verbose:
                print(f"    Processed lenslet {i+1}/{num_lens}")

        if verbose:
            print(f"  Done! Output shape: {atf_array.shape}")
        return atf_array