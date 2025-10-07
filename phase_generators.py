"""
提供相位图生成类，支持菲涅尔透镜和优化算法。
所有优化逻辑已合并入类中。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import config
from collections import defaultdict
from optics_utils import create_checkerboard, create_gaussian_template
# Assuming wave_propagation is an available external library
import wave_propagation as wp

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
        self._update_parameters()
        
        # Results (set after generate)
        self.phase = None
        self.phase_8bit = None
        self.history = None  # Only for optimized
        self.total_psfs = None
        
        # Optimization attributes (set in _optimize_phase)
        self.phase_param = None # raw phase in nn.parameters
        self.U_masked = None
        self.total_psfs_up = None
        self.mask_psfs_up = None

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
        loss_fn = nn.MSELoss()
        
        I_focal_full = self.forward(z=self.focal_length, upsampling=upsampling)
        loss1 = loss_fn(I_focal_full, self.total_psfs_up)
        
        I_focal_masked = self.forward(U_in=self.U_masked, z=self.focal_length, upsampling=upsampling)
        loss2 = loss_fn(I_focal_masked, self.mask_psfs_up)
        
        total_loss = torch.sum(torch.stack((loss1, loss2)))
        loss_components = {
            'focal_loss': loss1.item(),
            'mask_loss': loss2.item(),
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
        elif mode == 'optimized':
            region_size_norm = 1.0 / (self.M - (self.M - 1) * self.overlap_ratio)
            self.lens_width = self.L * region_size_norm
            self.f_number = self.focal_length / self.lens_width 
            self.depth_of_focus = 2*self.wavelength*self.f_number**2
    
    def _prepare_template(self,upsampling=1.0):
        if upsampling != 1.0:
            N = int(self.N * upsampling)
        else:
            N = self.N
        # Target patterns for optimization
        results = create_gaussian_template(
            N, self.L, self.focal_length, 
            self.wavelength, self.M, 
            self.overlap_ratio, self.device,
            airy_correction=self.airy_correction, 
            center_blend=self.center_blend, mask_count=self.mask_count,
            visualize=True, interleaving=self.interleaving
        )
        self.mask_psfs_up = results['mask_psfs'] * self.psf_energy_level
        self.total_psfs_up = results['total_psfs'] * self.psf_energy_level
        masks = results['masks']
        
        # target patterns for general use
        results = create_gaussian_template(
            self.N, self.L, self.focal_length, 
            self.wavelength, self.M, 
            self.overlap_ratio, self.device,
            airy_correction=self.airy_correction, 
            center_blend=self.center_blend, mask_count=self.mask_count,
            visualize=True, interleaving=self.interleaving
        )
        self.total_psfs = results['total_psfs']
        
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
                    print(f"Iter: {i+1}/{self.ni} Loss: {total_loss.item():.4e}", end='\r', flush=True)
                    # print(f"Iter: {i}/{self.ni} Loss: {total_loss.item():.4e}")
        
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
            self._prepare_template()
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
        