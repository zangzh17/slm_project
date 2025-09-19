# optimizer.py

import torch
import torch.nn as nn
import time
import numpy as np

# Assuming wave_propagation is an available external library
import wave_propagation as wp
from loss import AdaptiveOpticalLoss
from optics_utils import generate_spherical_wave, create_gaussian_template, create_template_with_centers, disparity_shift
import config

class PhaseOptimizer:
    def __init__(self, N: int, pixel_size: float, wavelength: float, focal_length: float,
                 psf_energy_level: float, dof_tol_factor: float, size_factor: float,
                 M: int, aperture_overlap_ratio: float, device=None):
        
        self.N = N
        self.pixel_size = pixel_size
        self.L = N * pixel_size
        self.wavelength = wavelength
        self.focal_length = focal_length
        self.M = M
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize phase parameters
        self.phase_param = nn.Parameter(torch.rand((N, N), device=self.device) * 2 * np.pi)
        
        # Initialize loss function
        self.adaptive_loss = AdaptiveOpticalLoss(num_losses=3).to(self.device)

        # Set physical parameters
        self._setup_parameters(dof_tol_factor, psf_energy_level)

        # Create incident wave fields
        self._setup_incident_waves()

        # Create target patterns
        self._setup_target_patterns(size_factor, aperture_overlap_ratio)

        # Initialize history records
        self.history = {'loss': [], 'mse1': [], 'mse2': [], 'mse3': []}

    def _setup_parameters(self, dof_tol_factor, psf_energy_level):
        """Calculate depth of field, spherical wave curvature radius and other parameters."""
        sub_aperture_NA = config.NA_OBJECTIVE / self.M
        self.depth_of_focus = dof_tol_factor * self.wavelength / (sub_aperture_NA**2)
        
        # Spherical wave curvature radius for simulating defocus
        self.radius_of_spherical_wave = config.FOCAL_OBJECTIVE**2 / self.depth_of_focus
        self.f_divergent = self.radius_of_spherical_wave
        self.f_convergent = -self.f_divergent

        # New focal point position of spherical wave after passing through lens
        self.spherical_focal_plane_dist_div = 1 / (1/self.focal_length - 1/self.f_divergent)
        self.spherical_focal_plane_dist_conv = 1 / (1/self.focal_length - 1/self.f_convergent)
        self.psf_energy_level = psf_energy_level

    def _setup_incident_waves(self):
        """Create plane wave and spherical incident wave fields."""
        self.U_in_plane = torch.ones((self.N, self.N), device=self.device, dtype=torch.complex64)
        self.U_in_spherical_divergent = generate_spherical_wave(
            self.f_divergent, self.N, self.L, self.wavelength, self.device)
        self.U_in_spherical_convergent = generate_spherical_wave(
            self.f_convergent, self.N, self.L, self.wavelength, self.device)

    def _setup_target_patterns(self, size_factor, aperture_overlap_ratio):
        """Create target Gaussian templates for plane wave and spherical waves."""
        # 1. Basic target: plane wave incidence
        self.target_plane, self.centers_pixel, self.sigma_px = create_gaussian_template(
            self.N, self.L, self.focal_length, self.wavelength, self.M, 
            aperture_overlap_ratio, size_factor, self.device
        )
        self.target_plane *= self.psf_energy_level
        
        # 2. Divergent wave target: calculate center point displacement
        shifted_centers_div = disparity_shift(
            self.f_divergent, self.centers_pixel, self.spherical_focal_plane_dist_div,
            self.pixel_size, self.N)
        self.target_divergent = create_template_with_centers(
            self.N, shifted_centers_div, self.sigma_px, self.device) * self.psf_energy_level
        
        # 3. Convergent wave target: calculate center point displacement
        shifted_centers_conv = disparity_shift(
            self.f_convergent, self.centers_pixel, self.spherical_focal_plane_dist_conv,
            self.pixel_size, self.N)
        self.target_convergent = create_template_with_centers(
            self.N, shifted_centers_conv, self.sigma_px, self.device) * self.psf_energy_level

    def forward(self, U_in: torch.Tensor, z: float) -> torch.Tensor:
        """Single forward propagation process."""
        U_phase = U_in * torch.exp(1j * self.phase_param)
        U_focal = wp.propagate_ASM(U_phase, z, self.L, self.wavelength, self.device)
        return torch.abs(U_focal)**2

    def compute_loss(self) -> tuple[torch.Tensor, ...]:
        """Compute MSE losses for three components and total loss."""
        # Component 1: Depth of field loss (on both sides of focal plane)
        I_focal_pos_dof = self.forward(self.U_in_plane, self.focal_length + self.depth_of_focus / 2)
        I_focal_neg_dof = self.forward(self.U_in_plane, self.focal_length - self.depth_of_focus / 2)
        mse1 = torch.mean((I_focal_pos_dof - self.target_plane)**2) + \
               torch.mean((I_focal_neg_dof - self.target_plane)**2)
        
        # Component 2: Divergent wave loss
        I_focal_div = self.forward(self.U_in_spherical_divergent, self.spherical_focal_plane_dist_div)
        mse2 = torch.mean((I_focal_div - self.target_divergent)**2)

        # Component 3: Convergent wave loss
        I_focal_conv = self.forward(self.U_in_spherical_convergent, self.spherical_focal_plane_dist_conv)
        mse3 = torch.mean((I_focal_conv - self.target_convergent)**2)
        
        total_loss, weights = self.adaptive_loss(mse1, mse2, mse3)
        return total_loss, mse1, mse2, mse3, weights

    def optimize(self, num_iterations: int, learning_rate: float, update_callback=None):
        """Execute optimization loop."""
        optimizer = torch.optim.Adam([self.phase_param, *self.adaptive_loss.parameters()], lr=learning_rate)
        start_time = time.time()
        print(f"Starting optimization with {num_iterations} iterations...")

        for i in range(num_iterations):
            optimizer.zero_grad()
            loss, mse1, mse2, mse3, weights = self.compute_loss()
            loss.backward()
            optimizer.step()
            
            # Record history data
            self.history['loss'].append(loss.item())
            self.history['mse1'].append(mse1.item() * weights[0].item())
            self.history['mse2'].append(mse2.item() * weights[1].item())
            self.history['mse3'].append(mse3.item() * weights[2].item())

            if update_callback and (i % 50 == 0 or i == num_iterations - 1):
                update_callback(i, num_iterations, loss.item(), self)

        elapsed_time = time.time() - start_time
        print(f"Optimization completed. Time elapsed: {elapsed_time:.2f} seconds")
        return self.phase_param.detach().clone()