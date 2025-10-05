# optimizer.py

import torch
import torch.nn as nn
import time
import numpy as np

# Assuming wave_propagation is an available external library
import wave_propagation as wp
from optics_utils import generate_spherical_wave, create_gaussian_template
import config

class PhaseOptimizer:
    def __init__(self, N: int, pixel_size: float, wavelength: float, focal_length: float,
                 psf_energy_level: float, dof_correction: float, airy_correction: float,
                 M: int, aperture_overlap_ratio: float, 
                 mask_count=2, center_blend=0.0, interleaving='coarse2', device=None):
        
        self.N = N
        self.pixel_size = pixel_size
        self.L = N * pixel_size
        self.wavelength = wavelength
        self.focal_length = focal_length
        self.M = M
        self.airy_correction = airy_correction
        self.aperture_overlap_ratio = aperture_overlap_ratio
        self.mask_count = mask_count
        self.center_blend = center_blend
        self.interleaving = interleaving
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize phase parameters
        self.phase_param = nn.Parameter(torch.rand((N, N), device=self.device) * 2 * np.pi)
        
        # # 为每个损失分量创建一个可学习的对数方差
        # self.log_vars = nn.Parameter(torch.zeros(2, device=self.device))

        # Set physical parameters
        self._setup_parameters(dof_correction, psf_energy_level)
        
        # Create target patterns
        self._setup_target_patterns()
        
        # Create incident wave fields
        self._setup_incident_waves()
        
        # Initialize history records
        self.history = {}

    def _setup_parameters(self, dof_correction, psf_energy_level):
        """Calculate depth of field, spherical wave curvature radius and other parameters."""        
        # obj and DOF
        sub_aperture_NA = config.NA_OBJECTIVE / self.M
        self.depth_of_focus = dof_correction * self.wavelength / (sub_aperture_NA**2)
        
        # Spherical wave curvature radius for simulating defocus
        self.radius_of_spherical_wave = config.FOCAL_OBJECTIVE**2 / self.depth_of_focus

        # New focal point position of spherical wave after passing through lens
        self.spherical_focal_plane_dist_div = 1 / (1/self.focal_length - 1/self.radius_of_spherical_wave)
        self.psf_energy_level = psf_energy_level
        
    def _setup_target_patterns(self):
        """Create target Gaussian templates"""
        results = create_gaussian_template(
            self.N, self.L, self.focal_length, 
            self.wavelength, self.M, 
            self.aperture_overlap_ratio, self.device,
            airy_correction = self.airy_correction, 
            center_blend = self.center_blend, 
            mask_count=self.mask_count, 
            visualize = True,
            interleaving=self.interleaving
        )
        self.sigma_px = results['sigma_px']
        self.mask_psfs = results['mask_psfs'] * self.psf_energy_level
        self.total_psfs = results['total_psfs'] * self.psf_energy_level
        self.centers_pixel = results['centers_pixel']
        self.masks = results['masks']
        
    def _setup_incident_waves(self):
        """Create plane wave, masked wave and spherical incident wave fields."""
        self.U_masked = self.masks.to(device=self.device, dtype=torch.complex64)
        self.U_in_plane = torch.ones((self.N, self.N), device=self.device, dtype=torch.complex64)
        self.U_in_spherical_divergent = generate_spherical_wave(
            self.radius_of_spherical_wave, self.N, self.L, self.wavelength, self.device)
        
    def forward(self, U_in: torch.Tensor, z: float) -> torch.Tensor:
        """Single forward propagation process."""
        U_phase = U_in * torch.exp(1j * self.phase_param)
        U_focal = wp.propagate_ASM(U_phase, z, self.L, self.wavelength, self.device)
        return torch.abs(U_focal)**2

    def compute_loss(self) -> tuple[torch.Tensor, ...]:
        """Compute MSE losses for three components and total loss."""
        loss_fn = nn.MSELoss()
        
        
        # Component 1: focal plane loss for full field
        I_focal = self.forward(self.U_in_plane, self.focal_length)
        loss1 = loss_fn(I_focal, self.total_psfs)
        
        # Component 2: focal plane loss for masked field
        I_focal = self.forward(self.U_masked, self.focal_length)
        loss2 = loss_fn(I_focal, self.mask_psfs)

        # # 基于不确定性的加权总损失 = Σ (exp(-log_var_i) * loss_i + log_var_i)
        # precision = torch.exp(-self.log_vars)
        # weights = (precision / precision.sum()).detach()
        # losses_tensor = torch.stack((loss1,loss2))
        # total_loss = torch.sum(precision * losses_tensor + self.log_vars)
        
        weights = torch.tensor(config.OPTIMIZER['weights'], device=self.device)
        total_loss = torch.sum(weights * torch.stack((loss1, loss2)))
        
        loss_components = {
            'focal_loss': loss1.item(),
            'mask_loss': loss2.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components

    def optimize(self, num_iterations: int, learning_rate: float, update_callback=None):
        """Execute optimization loop."""
        # optimizer = torch.optim.Adam([self.phase_param, self.log_vars], lr=learning_rate)
        optimizer = torch.optim.Adam([self.phase_param], lr=learning_rate)
        start_time = time.time()
        print(f"Starting optimization with {num_iterations} iterations...")

        for i in range(num_iterations):
            optimizer.zero_grad()
            total_loss, loss_components = self.compute_loss()
            total_loss.backward()
            optimizer.step()
            
            # Record history data
            if not self.history:
                # 使用 loss_components 的键来初始化 self.history
                self.history = {key: [] for key in loss_components.keys()}
            for key, value in loss_components.items():
                self.history[key].append(value)
            
            # visualize or other callback
            if update_callback and (i % 50 == 0 or i == num_iterations - 1):
                update_callback(i, num_iterations, total_loss.item(), self)
        
        elapsed_time = time.time() - start_time
        print(f"Optimization completed. Time elapsed: {elapsed_time:.2f} seconds")
        return self.phase_param.detach().clone()