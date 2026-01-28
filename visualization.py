# visualization.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from scipy import ndimage
import torch
from scipy.ndimage import zoom
from typing import Dict, List, Tuple, Optional
import matplotlib.patches as patches
from IPython.display import display, clear_output
import ipywidgets as widgets
from optics_utils import calculate_airy_disk
from optics_utils import compute_psf_centers

def plot_live_update(iteration: int, total_iterations: int, loss: float, optimizer):
    """Display and update images in real-time during optimization."""
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # 1. Loss curve
    ax = axes[0]
    # only total loss
    # ax.semilogy(optimizer.history['total_loss']) 
    # show every loss item
    for loss_name, loss_values in optimizer.history.items():
        if loss_name == 'total_loss':
            linewidth = 2
            zorder = 10 
        else:
            linewidth = 1
            zorder = 5
        ax.semilogy(loss_values, label=loss_name, linewidth=linewidth, zorder=zorder)
    ax.legend()
    
    ax.set_title(f'Iteration {iteration+1}/{total_iterations}')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss (log scale)') 
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Current phase
    ax = axes[1]
    phase_wrapped = torch.remainder(optimizer.phase_param.detach(), 2 * np.pi).cpu().numpy()
    im = ax.imshow(phase_wrapped, cmap='hsv', vmin=0, vmax=2 * np.pi)
    ax.set_title('Current Optimized Phase')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Phase (rad)')


    
    # 4. Current focal plane intensity
    ax = axes[2]
    with torch.no_grad():
        intensity = optimizer.forward().cpu().numpy()
    im = ax.imshow(intensity, cmap='hot', norm=mcolors.LogNorm())
    ax.set_title(f'Focal Plane Intensity (Loss: {loss:.4e})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Intensity (log scale)')
    
    plt.tight_layout()
    display(fig)
    plt.close(fig)

    # 3. Cross sectional plot for phase
    # Create separate figure for horizontal cross-section
    fig_cross = plt.figure(figsize=(12, 4))
    ax_cross = fig_cross.add_subplot(111)

    center_row = phase_wrapped.shape[0] // 2
    cross_section = phase_wrapped[center_row, :]
    ax_cross.plot(cross_section, linewidth=2)
    ax_cross.set_title(f'Horizontal Cross-Section Phase (Row {center_row})')
    ax_cross.set_xlabel('Pixel Position')
    ax_cross.set_ylabel('Phase (rad)')
    ax_cross.set_ylim(0, 2 * np.pi)
    ax_cross.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_phase(optimizer):
    """Display phase"""
    # Visualize the generated Fresnel microlens array phase pattern.
    plt.figure(figsize=(12, 7))
    plt.imshow(optimizer.phase_8bit, cmap='gray', vmin=0, vmax=255)
    plt.colorbar(label=f"Gray Level (0-{optimizer.two_pi_value})")
    # Draw ROI boundary
    rect = optimizer.roi_rect
    plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                      edgecolor='r', facecolor='none', lw=2, label='ROI'))
    # Calculate and display information
    f_m = abs(optimizer.focal_length)
    lens_w_m = optimizer.lens_width
    airy_disk_m = calculate_airy_disk(f_m, lens_w_m, optimizer.wavelength)
    title = (
        f"Diff.Lim. Spot: {airy_disk_m*1e6:.2f} µm / {airy_disk_m/optimizer.pixel_size:.1f} pix"
    )
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_2d_comparisons(optimizer, slice_y=None):
    """Side-by-side comparison of optimized intensity patterns with propagation analysis.

    Parameters
    ----------
    optimizer : PhaseGenerator
        The optimizer object
    slice_y : int, optional
        Y position for the slice. If None, uses center (N//2).
    """
    if slice_y is None:
        slice_y = optimizer.N // 2

    def compute_propagation_map(U_in, z_values, optimizer, slice_pos, max_norm=True):
        """Compute intensity map for multiple propagation distances."""
        n_z = len(z_values)
        # Get the shape from a test propagation
        test_field = optimizer.forward(U_in=U_in, z=z_values[0])
        n_y = test_field.shape[0]

        # Initialize the intensity map
        intensity_map = torch.zeros((n_z, n_y))

        for i, z in enumerate(z_values):
            # Propagate to distance z
            intensity = optimizer.forward(U_in=U_in,z=z_values[i])
            # Sum along x-axis (axis=1) to get 1D intensity profile
            intensity_1d = intensity[slice_pos,:]
            if max_norm:
                intensity_1d = intensity_1d/intensity_1d.max()
            intensity_map[i, :] = intensity_1d

        return intensity_map.cpu().numpy()
    
    with torch.no_grad():
        # Original intensity calculations
        I_opt_plane = optimizer.forward().cpu().numpy()
        
        # Define number of samples for propagation analysis
        n_samples = 50
        
        # Compute propagation ranges for each wave type
        wave_configs = {
            "Plane Wave": {
                "U_in": None,
                "focal_dist": optimizer.focal_length,
                "optimized": I_opt_plane
            }
        }
        
        for name, config in wave_configs.items():
            # Create figure with 2 rows: top row with 4 plots, bottom row with 1 elongated plot
            fig = plt.figure(figsize=(18, 6))
            gs = fig.add_gridspec(2, 4, height_ratios=[4, 2], hspace=0.3)
            
            # Top row: 4 plots
            axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
            # Bottom row: 1 elongated plot spanning all columns
            ax_zoom = fig.add_subplot(gs[1, :])
            
            fig.suptitle(f'Intensity Analysis: {name}', fontsize=14, y=0.98)
            
            # Plot 1: Optimized Intensity
            im1 = axes[0].imshow(config["optimized"], cmap='hot', norm=mcolors.LogNorm())
            axes[0].set_title('Optimized Intensity', fontsize=10)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.02)
            
            # Plot 2: Propagation from 0 to 1.2x focal length, linear scale
            z_range_1 = torch.linspace(0, 1.2 * config["focal_dist"], n_samples)
            intensity_map_1 = compute_propagation_map(config["U_in"], z_range_1, optimizer, slice_y)
            intensity_map_2 = compute_propagation_map(config["U_in"], z_range_1, optimizer, slice_y, max_norm=False)
            
            im2 = axes[1].imshow(intensity_map_1, cmap='hot',
                                aspect='auto', extent=[0, intensity_map_1.shape[1]*optimizer.pixel_size*1e3, 
                                                      z_range_1[-1].item()*1e3, 
                                                      z_range_1[0].item()*1e3])
            axes[1].set_title('Propagation (0 - 1.2× focal)', fontsize=10)
            axes[1].set_xlabel('Y (mm)', fontsize=9)
            axes[1].set_ylabel('Z (mm)', fontsize=9)
            axes[1].tick_params(axis='both', labelsize=8)
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.02)
            cbar2.set_label('I', fontsize=9)
            cbar2.ax.tick_params(labelsize=8)
            
            # Plot 3: Propagation from 0 to 1.2x focal length, log scale
            intensity_map_log = intensity_map_2/intensity_map_2.max()
            intensity_map_log = np.log10(intensity_map_log + 1e-10)
            intensity_map_log[intensity_map_log<-4] = -4
            im3 = axes[2].imshow(intensity_map_log, cmap='hot', 
                                aspect='auto', extent=[0, intensity_map_log.shape[1]*optimizer.pixel_size*1e3, 
                                                      z_range_1[-1].item()*1e3, 
                                                      z_range_1[0].item()*1e3])
            axes[2].set_title('Propagation (0 - 1.2× focal)', fontsize=10)
            axes[2].set_xlabel('X (mm)', fontsize=9)
            axes[2].set_ylabel('Z (mm)', fontsize=9)
            axes[2].tick_params(axis='both', labelsize=8)
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.7, pad=0.02)
            cbar3.set_label('log₁₀(I)', fontsize=9)
            cbar3.ax.tick_params(labelsize=8)
            
            # Plot 4: Propagation around focal length ± 4*depth_of_focus
            z_min = config["focal_dist"] - 4 * optimizer.depth_of_focus
            z_max = config["focal_dist"] + 4 * optimizer.depth_of_focus

            z_range_2 = torch.linspace(z_min, z_max, n_samples)
            intensity_map_3 = compute_propagation_map(config["U_in"], z_range_2, optimizer, slice_y)
            
            # Convert to relative distance in micrometers
            z_range_2_relative_mm = (z_range_2 - config["focal_dist"]).numpy() * 1e3  # Convert to mm
            
            im4 = axes[3].imshow(intensity_map_3, cmap='hot', 
                                aspect='auto', extent=[0, intensity_map_3.shape[1]*optimizer.pixel_size*1e3, 
                                                      z_range_2_relative_mm[-1], 
                                                      z_range_2_relative_mm[0]])
            
            axes[3].set_title('Around Focal (±4×DOF)', fontsize=10)
            axes[3].set_xlabel('Y (mm)', fontsize=9)
            axes[3].set_ylabel('Δz (mm)', fontsize=9)
            axes[3].tick_params(axis='both', labelsize=8)
            cbar4 = plt.colorbar(im4, ax=axes[3], shrink=0.7, pad=0.02)
            cbar4.set_label('log₁₀(I)', fontsize=9)
            cbar4.ax.tick_params(labelsize=8)
            # Add horizontal lines at focal plane and DOF markers
            axes[3].axhline(y=0, color='white', linestyle='-', linewidth=0.5)
            axes[3].axhline(y=optimizer.depth_of_focus/2 *1e3, color='white', linestyle='--', linewidth=1, alpha=0.5)
            axes[3].axhline(y=-optimizer.depth_of_focus/2 *1e3, color='white', linestyle='--', linewidth=1, alpha=0.5)
            axes[3].axhline(y=optimizer.depth_of_focus *1e3, color='white', linestyle='--', linewidth=1, alpha=0.5)
            axes[3].axhline(y=-optimizer.depth_of_focus *1e3, color='white', linestyle='--', linewidth=1, alpha=0.5)
            
            # Bottom elongated plot: Zoomed version of Plot 4 (±3 DOF, with 0.5 DOF grid lines)
            dof_mm = optimizer.depth_of_focus * 1e3  # DOF in mm
            
            im_zoom = ax_zoom.imshow(intensity_map_3, cmap='hot', 
                                    aspect='auto', extent=[0, intensity_map_3.shape[1]*optimizer.pixel_size*1e3, 
                                                          z_range_2_relative_mm[-1], 
                                                          z_range_2_relative_mm[0]])
            
            ax_zoom.set_ylim(-2.5 * dof_mm, 2.5 * dof_mm)
            ax_zoom.set_title('Zoomed: Around Focal (±2.5×DOF)', fontsize=10)
            ax_zoom.set_xlabel('Y (mm)', fontsize=9)
            ax_zoom.set_ylabel('Δz (mm)', fontsize=9)
            ax_zoom.tick_params(axis='both', labelsize=8)
            
            # Add horizontal grid lines at every 0.5 DOF
            for i in np.arange(-2.5, 3, 0.5):
                linestyle = '-' if i == 0 else '--'
                linewidth = 0.5 if i == 0 else 1.0
                ax_zoom.axhline(y=i * dof_mm, color='white', linestyle=linestyle, 
                               linewidth=linewidth, alpha=0.8)
            
            cbar_zoom = plt.colorbar(im_zoom, ax=ax_zoom, shrink=0.5, pad=0.02)
            cbar_zoom.set_label('log₁₀(I)', fontsize=9)
            cbar_zoom.ax.tick_params(labelsize=8)
            
            plt.tight_layout()
            plt.show()

def plot_2d_comparisons_interactive(optimizer, upsampling=1.0):
    """Interactive side-by-side comparison with adjustable slice position.

    Provides coarse/fine slice position control to navigate through PSF positions,
    especially useful when randomness is applied to PSF centers.

    Parameters
    ----------
    optimizer : PhaseGenerator
        The optimizer object
    upsampling : float
        Upsampling factor for visualization (default 1.0)
    """
    import matplotlib.patches as mpatches

    N = optimizer.N
    M = optimizer.M
    N_up = int(N * upsampling)
    pixel_size_up = optimizer.pixel_size / upsampling

    # Get PSF center positions (accounting for randomness)
    randomness = getattr(optimizer, 'randomness', 0.0)
    random_seed = getattr(optimizer, 'random_seed', None)

    center_info = compute_psf_centers(
        M=M, overlap_ratio=optimizer.overlap_ratio, center_blend=optimizer.center_blend,
        z_ratio=1.0, N=N_up, output_size=N_up, device=optimizer.device,
        randomness=randomness, random_seed=random_seed
    )
    centers = center_info['centers_pixel'].cpu().numpy()  # [M*M, 2] format: [x, y]

    # Calculate efficiency circle radius (same as plot_energy_distribution)
    radius_meters = optimizer.airy_radius
    radius_pixels = radius_meters / pixel_size_up

    # Get unique y positions (rows) - centers[:, 1] is y coordinate
    y_centers = sorted(set(int(c[1]) for c in centers))

    # Create widgets - use HTML labels to avoid truncation
    style = {'description_width': '90px'}
    layout = widgets.Layout(width='350px')

    w_coarse = widgets.IntSlider(
        value=len(y_centers) // 2, min=0, max=len(y_centers) - 1, step=1,
        description='PSF Row:', style=style, layout=layout,
        continuous_update=False
    )
    w_fine = widgets.IntSlider(
        value=0, min=-50, max=50, step=1,
        description='Fine Adj:', style=style, layout=layout,
        continuous_update=False
    )
    w_log_scale = widgets.Checkbox(value=True, indent=False)
    w_log_range = widgets.FloatSlider(
        value=4, min=1, max=6, step=0.5,
        layout=widgets.Layout(width='150px'),
        continuous_update=False, readout_format='.1f'
    )
    w_show_circles = widgets.Checkbox(value=False, indent=False)
    w_position_label = widgets.HTML(value='')
    plot_output = widgets.Output()

    def update_position_label():
        base_y = y_centers[w_coarse.value]
        actual_y = base_y + w_fine.value
        w_position_label.value = (
            f'<b>Slice Y:</b> {actual_y} px (PSF row {w_coarse.value + 1}/{len(y_centers)}, '
            f'base={base_y}, fine={w_fine.value:+d})'
        )

    def plot_all(change=None):
        with plot_output:
            clear_output(wait=True)
            plt.close('all')

            base_y = y_centers[w_coarse.value]
            slice_y = base_y + w_fine.value
            slice_y = max(0, min(N_up - 1, slice_y))  # Clamp to valid range
            use_log = w_log_scale.value
            log_range = w_log_range.value
            show_circles = w_show_circles.value

            update_position_label()

            # Compute propagation map with specified slice position
            def compute_propagation_map(U_in, z_values, max_norm=True):
                n_z = len(z_values)
                intensity_map = torch.zeros((n_z, N_up))
                with torch.no_grad():
                    for i, z in enumerate(z_values):
                        intensity = optimizer.forward(U_in=U_in, z=z, upsampling=upsampling)
                        intensity_1d = intensity[slice_y, :]
                        if max_norm and intensity_1d.max() > 0:
                            intensity_1d = intensity_1d / intensity_1d.max()
                        intensity_map[i, :] = intensity_1d
                return intensity_map.cpu().numpy()

            n_samples = 50
            focal_dist = optimizer.focal_length

            # Create figure
            fig = plt.figure(figsize=(18, 6))
            gs = fig.add_gridspec(2, 4, height_ratios=[4, 2], hspace=0.3)
            axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
            ax_zoom = fig.add_subplot(gs[1, :])

            fig.suptitle(f'Intensity Analysis - Slice Y={slice_y}', fontsize=14, y=0.98)

            # Plot 1: Optimized Intensity with slice line (toggle log/linear)
            with torch.no_grad():
                I_opt = optimizer.forward(upsampling=upsampling).cpu().numpy()

            if use_log:
                vmin = I_opt.max() * (10 ** (-log_range))
                im1 = axes[0].imshow(I_opt, cmap='hot',
                                    norm=mcolors.LogNorm(vmin=vmin, vmax=I_opt.max()))
                title1 = f'Optimized Intensity (log, {log_range:.0f} decades)'
            else:
                im1 = axes[0].imshow(I_opt, cmap='hot')
                title1 = 'Optimized Intensity (linear)'

            axes[0].axhline(y=slice_y, color='cyan', linestyle='--', linewidth=1, alpha=0.8)

            # Draw efficiency circles if enabled
            if show_circles:
                for i in range(len(centers)):
                    cx, cy = centers[i, 0], centers[i, 1]
                    circle = mpatches.Circle((cx, cy), radius_pixels,
                                            fill=False, edgecolor='lime',
                                            linestyle='--', linewidth=0.8, alpha=0.7)
                    axes[0].add_patch(circle)

            axes[0].set_title(title1, fontsize=10)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.02)

            # Plot 2: Propagation from 0 to 1.2x focal length
            z_range_1 = torch.linspace(0, 1.2 * focal_dist, n_samples)
            intensity_map_1 = compute_propagation_map(None, z_range_1)
            intensity_map_2 = compute_propagation_map(None, z_range_1, max_norm=False)

            im2 = axes[1].imshow(intensity_map_1, cmap='hot', aspect='auto',
                                extent=[0, N_up * pixel_size_up * 1e3,
                                        z_range_1[-1].item() * 1e3, z_range_1[0].item() * 1e3])
            axes[1].set_title('Propagation (0 - 1.2× focal)', fontsize=10)
            axes[1].set_xlabel('X (mm)', fontsize=9)
            axes[1].set_ylabel('Z (mm)', fontsize=9)
            plt.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.02)

            # Plot 3: Propagation log scale
            intensity_map_log = intensity_map_2 / (intensity_map_2.max() + 1e-10)
            intensity_map_log = np.log10(intensity_map_log + 1e-10)
            intensity_map_log = np.clip(intensity_map_log, -4, 0)

            im3 = axes[2].imshow(intensity_map_log, cmap='hot', aspect='auto',
                                extent=[0, N_up * pixel_size_up * 1e3,
                                        z_range_1[-1].item() * 1e3, z_range_1[0].item() * 1e3])
            axes[2].set_title('Propagation (Log scale)', fontsize=10)
            axes[2].set_xlabel('X (mm)', fontsize=9)
            axes[2].set_ylabel('Z (mm)', fontsize=9)
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.7, pad=0.02)
            cbar3.set_label('log₁₀(I)', fontsize=9)

            # Plot 4: Around focal
            z_min = focal_dist - 4 * optimizer.depth_of_focus
            z_max = focal_dist + 4 * optimizer.depth_of_focus
            z_range_2 = torch.linspace(z_min, z_max, n_samples)
            intensity_map_3 = compute_propagation_map(None, z_range_2)
            z_range_2_relative_mm = (z_range_2 - focal_dist).numpy() * 1e3

            im4 = axes[3].imshow(intensity_map_3, cmap='hot', aspect='auto',
                                extent=[0, N_up * pixel_size_up * 1e3,
                                        z_range_2_relative_mm[-1], z_range_2_relative_mm[0]])
            axes[3].set_title('Around Focal (±4×DOF)', fontsize=10)
            axes[3].set_xlabel('X (mm)', fontsize=9)
            axes[3].set_ylabel('Δz (mm)', fontsize=9)
            axes[3].axhline(y=0, color='white', linestyle='-', linewidth=0.5)
            dof_mm = optimizer.depth_of_focus * 1e3
            for mult in [0.5, 1.0]:
                axes[3].axhline(y=mult * dof_mm, color='white', linestyle='--', linewidth=1, alpha=0.5)
                axes[3].axhline(y=-mult * dof_mm, color='white', linestyle='--', linewidth=1, alpha=0.5)
            plt.colorbar(im4, ax=axes[3], shrink=0.7, pad=0.02)

            # Bottom zoom plot
            im_zoom = ax_zoom.imshow(intensity_map_3, cmap='hot', aspect='auto',
                                    extent=[0, N_up * pixel_size_up * 1e3,
                                            z_range_2_relative_mm[-1], z_range_2_relative_mm[0]])
            ax_zoom.set_ylim(-2.5 * dof_mm, 2.5 * dof_mm)
            ax_zoom.set_title('Zoomed: Around Focal (±2.5×DOF)', fontsize=10)
            ax_zoom.set_xlabel('X (mm)', fontsize=9)
            ax_zoom.set_ylabel('Δz (mm)', fontsize=9)

            for i in np.arange(-2.5, 3, 0.5):
                linestyle = '-' if i == 0 else '--'
                linewidth = 0.5 if i == 0 else 1.0
                ax_zoom.axhline(y=i * dof_mm, color='white', linestyle=linestyle,
                               linewidth=linewidth, alpha=0.8)
            plt.colorbar(im_zoom, ax=ax_zoom, shrink=0.5, pad=0.02)

            plt.tight_layout()
            plt.show()

    # Setup callbacks
    w_coarse.observe(plot_all, names='value')
    w_fine.observe(plot_all, names='value')
    w_log_scale.observe(plot_all, names='value')
    w_log_range.observe(plot_all, names='value')
    w_show_circles.observe(plot_all, names='value')

    # Navigation buttons
    btn_layout = widgets.Layout(width='80px')
    prev_btn = widgets.Button(description='◀ Prev', layout=btn_layout)
    next_btn = widgets.Button(description='Next ▶', layout=btn_layout)
    reset_btn = widgets.Button(description='Reset', layout=btn_layout)

    def on_prev(b):
        if w_coarse.value > 0:
            w_coarse.value -= 1

    def on_next(b):
        if w_coarse.value < len(y_centers) - 1:
            w_coarse.value += 1

    def on_reset(b):
        w_coarse.value = len(y_centers) // 2
        w_fine.value = 0

    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    reset_btn.on_click(on_reset)

    # Display with HTML labels to avoid truncation
    controls = widgets.VBox([
        widgets.HTML('<h4>Slice Position Control</h4>'),
        widgets.HBox([w_coarse, prev_btn, next_btn]),
        widgets.HBox([w_fine, reset_btn]),
        widgets.HBox([
            widgets.HTML('<span style="margin-right:5px">Log:</span>'), w_log_scale,
            widgets.HTML('<span style="margin:0 10px">Range:</span>'), w_log_range,
            widgets.HTML('<span style="margin:0 10px">Circles:</span>'), w_show_circles,
        ]),
        w_position_label
    ])
    display(controls)
    display(plot_output)

    # Initial plot
    plot_all()


def plot_cross_sections(optimizer,upsampling=1.0):
    """
    Plot intensity comparison of central cross-sections (Wider Figure).
    Uses scipy.signal.find_peaks to mark all peak positions with vertical dashed lines.
    """
    N_up = optimizer.N * upsampling
    tgt_psfs = optimizer.total_psfs.cpu().numpy()
    tgt_psfs_up = zoom(tgt_psfs, zoom=upsampling, order=0)
    
    with torch.no_grad():
        I_opt_plane = optimizer.forward(upsampling=upsampling).cpu().numpy()
        
    y_slice = int(N_up // 2)
    plane_slice = I_opt_plane[y_slice, :]
    tgt_slice = tgt_psfs_up[y_slice, :]
    
    plane_slice = plane_slice/tgt_slice.max()
    tgt_slice = tgt_slice/tgt_slice.max()
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 8), sharex=True)

    # --- Subplot 1: log Scale ---
    ax1.plot(tgt_slice, 'r-', label='Optimized', lw=1)
    ax1.plot(plane_slice, 'b--', label='Target', lw=1)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-6, 1.5) 
    ax1.set_ylabel('Intensity (log scale)')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    # --- Subplot 2: Linear Scale ---
    ax2.plot(tgt_slice, 'r-', label='Optimized', lw=1)
    ax2.plot(plane_slice, 'b--', label='Target', lw=1)
    # y-scale is linear by default
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Intensity (linear scale)')
    ax2.legend()
    ax2.grid(True, ls="--", alpha=0.6)
    # Add a main title for the entire figure
    fig.suptitle(f'Central Cross-section Intensity Comparison (y={y_slice})', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect adjusts for suptitle
    plt.show()


def plot_energy_distribution(optimizer, upsampling=1.0, fig=None):
    """
    可视化衍射透镜阵列中所有焦斑的聚焦效率（相对比例）。

    使用精确的几何中心位置计算每个焦斑的圈入能量效率。

    Parameters
    ----------
    optimizer : PhaseOptimizer
        优化器对象，包含所有必要的参数和方法
    upsampling : float
        上采样因子，用于更精确的能量计算
    fig : matplotlib.figure.Figure, optional
        外部提供的Figure对象。如果为None，则创建新Figure。

    Notes
    -----
    该函数执行以下步骤：
    1. 使用compute_psf_centers获取精确的焦斑中心位置
    2. 根据Airy半径计算圈入能量的半径
    3. 在优化后的光强图中为每个焦斑计算圈入能量
    4. 将圈入能量除以理论分配能量，得到效率比例
    5. 绘制柱状图展示效率分布，并标注统计数据
    """
    output_size_up = int(optimizer.output_size * upsampling)
    N_up = int(optimizer.N * upsampling)
    pixel_size_up = optimizer.pixel_size / upsampling
    
    with torch.no_grad():
        # 获取优化后的光强分布
        I_opt = optimizer.forward(z=optimizer.focal_length, upsampling=upsampling).cpu().numpy()
    
    # ========== 步骤 1: 使用精确的几何中心 ==========
    # 获取 randomness 和 random_seed 参数（如果存在）
    randomness = getattr(optimizer, 'randomness', 0.0)
    random_seed = getattr(optimizer, 'random_seed', None)

    center_info = compute_psf_centers(
        M=optimizer.M,
        overlap_ratio=optimizer.overlap_ratio,
        center_blend=optimizer.center_blend,
        z_ratio=1.0,  # 焦平面
        N=N_up,
        output_size=output_size_up,
        device=optimizer.device,
        randomness=randomness,
        random_seed=random_seed
    )
    centers_pixel = center_info['centers_pixel'].cpu().numpy()  # [M*M, 2]
    
    num_spots = centers_pixel.shape[0]
    print(f"\n{'='*60}")
    print(f"Focusing Efficiency Analysis")
    print(f"{'='*60}")
    print(f"Number of focal spots: {num_spots} (M={optimizer.M}x{optimizer.M})")
    if randomness > 0:
        print(f"PSF Randomness: {randomness} (seed={random_seed})")
    
    # ========== 步骤 2: 计算半径 ==========
    # 使用airy半径和校正因子
    radius_meters = optimizer.airy_radius
    radius_pixels = radius_meters / pixel_size_up
    
    print(f"\nEncircled Energy Calculation:")
    print(f"  Radius: {radius_pixels:.2f} pixels = {radius_meters*1e6:.1f} μm")
    
    # ========== 步骤 3: 计算圈入能量 ==========
    encircled_energies = []
    Y, X = np.ogrid[:output_size_up, :output_size_up]
    
    for i in range(num_spots):
        center_x = centers_pixel[i, 0]
        center_y = centers_pixel[i, 1]
        
        # 计算到中心的距离
        dist_sq = (X - center_x)**2 + (Y - center_y)**2
        mask = dist_sq <= radius_pixels**2
        
        # 计算圈内能量
        energy = I_opt[mask].sum()
        encircled_energies.append(energy)
    
    abs_energies = np.array(encircled_energies)
    
    # ========== 步骤 4: 计算效率比例 ==========
    # 理论上分配给单个焦点的总能量
    ideal_energy_per_spot = (N_up * N_up) / (optimizer.M * optimizer.M)
    
    # 计算效率（每个焦斑的圈入能量 / 理论分配的总能量）
    efficiencies = abs_energies / ideal_energy_per_spot
    efficiencies_percent = efficiencies * 100
    
    # 统计信息
    mean_eff = efficiencies_percent.mean()
    max_eff = efficiencies_percent.max()
    min_eff = efficiencies_percent.min()
    std_eff = efficiencies_percent.std()
    median_eff = np.median(efficiencies_percent)
    
    # 找出最高和最低效率的焦斑
    max_idx = np.argmax(efficiencies_percent)
    min_idx = np.argmin(efficiencies_percent)
    
    # print(f"\nEfficiency Statistics:")
    # print(f"  Mean:   {mean_eff:.2f}%")
    # print(f"  Median: {median_eff:.2f}%")
    # print(f"  Std:    {std_eff:.2f}%")
    # print(f"  Range:  [{min_eff:.2f}%, {max_eff:.2f}%]")
    # print(f"  Max efficiency at spot #{max_idx}")
    # print(f"  Min efficiency at spot #{min_idx}")
    # print(f"{'='*60}\n")
    
    # ========== 步骤 5: 可视化 ==========
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    else:
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
    
    # --- 左图：柱状图 ---
    spot_indices = np.arange(len(efficiencies_percent))
    bars = ax1.bar(spot_indices, efficiencies_percent, color='darkcyan', 
                   label='Focusing Efficiency', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 高亮最高和最低效率的焦斑
    bars[max_idx].set_color('green')
    bars[max_idx].set_alpha(1.0)
    bars[min_idx].set_color('red')
    bars[min_idx].set_alpha(1.0)
    
    # 添加平均线和标准差区域
    ax1.axhline(y=mean_eff, color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_eff:.2f}%', zorder=3)
    ax1.axhspan(mean_eff - std_eff, mean_eff + std_eff, 
                alpha=0.2, color='blue', label=f'±1 Std: {std_eff:.2f}%')
    
    # 统计信息文本
    stats_text = (f'Total Spots: {len(efficiencies_percent)}\n'
                  f'Mean: {mean_eff:.2f}%\n'
                  f'Median: {median_eff:.2f}%\n'
                  f'Std Dev: {std_eff:.2f}%\n'
                  f'Max: {max_eff:.2f}% (#{max_idx})\n'
                  f'Min: {min_eff:.2f}% (#{min_idx})\n'
                  f'Radius: {radius_pixels:.2f} px')
    if randomness > 0:
        stats_text += f'\nRandomness: {randomness}'
    
    ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='lightgoldenrodyellow', alpha=0.9))
    
    ax1.set_title('Focusing Efficiency Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Focal Spot Index', fontsize=12)
    ax1.set_ylabel('Focusing Efficiency (%)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.set_ylim(0, max(110, np.max(efficiencies_percent) * 1.2))
    
    # --- 右图：2D效率分布热力图 ---
    efficiencies_2d = efficiencies_percent.reshape(optimizer.M, optimizer.M)
    # 获取colormap的前 0.4 部分
    cmap = plt.get_cmap('gist_heat')
    colors = cmap(np.linspace(0, 0.4, 256))
    cmap = mcolors.ListedColormap(colors)

    im = ax2.imshow(efficiencies_2d, cmap=cmap, origin='lower', 
                    aspect='auto', interpolation='nearest')
    
    # 标记最高和最低
    max_i, max_j = max_idx // optimizer.M, max_idx % optimizer.M
    min_i, min_j = min_idx // optimizer.M, min_idx % optimizer.M
    # 添加数值标注
    for i in range(optimizer.M):
        for j in range(optimizer.M):
            if i==max_i and j==max_j:
                text = ax2.text(j, i, f'{efficiencies_2d[i, j]:.0f}',
                            ha="center", va="center", color="Blue", fontsize=13,
                            fontweight='bold')
            elif i==min_i and j==min_j:
                text = ax2.text(j, i, f'{efficiencies_2d[i, j]:.0f}',
                            ha="center", va="center", color="Green", fontsize=13,
                            fontweight='bold')
            else:
                text = ax2.text(j, i, f'{efficiencies_2d[i, j]:.0f}',
                            ha="center", va="center", color="White", fontsize=12)
    
    ax2.set_title('Efficiency Map (2D Layout)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lens Column', fontsize=12)
    ax2.set_ylabel('Lens Row', fontsize=12)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Efficiency (%)', fontsize=11)
    
    # 设置刻度
    ax2.set_xticks(range(optimizer.M))
    ax2.set_yticks(range(optimizer.M))
    
    plt.tight_layout()
    plt.show()

def plot_fresnel_pattern(optimizer):
    """
    Visualize the generated Fresnel microlens array phase pattern.
    """
    plt.figure(figsize=(12, 7))
    plt.imshow(optimizer.phase, cmap='gray', vmin=0, vmax=255)
    plt.colorbar(label=f"Gray Level (0-{optimizer.two_pi_value})")
    
    # Draw ROI boundary
    rect = optimizer.roi_rect
    plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                      edgecolor='r', facecolor='none', lw=2, label='ROI'))
    
    # Calculate and display information
    f_m = abs(optimizer.focal_length)
    lens_w_m = optimizer.lens_width * optimizer.pixel_size
    airy_disk_m = calculate_airy_disk(f_m, lens_w_m)
    
    title = (
        f"Fresnel Microlens Array: {optimizer.M}×{optimizer.M}, f={optimizer.focal_length:.1f} mm\n"
        f"Diff.Lim. Spot: {airy_disk_m*1.0e6:.2f} µm / {airy_disk_m/optimizer.pixel_size:.1f} pix"
    )
    plt.title(title)
    plt.legend()
    plt.show()




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

def plot_psf_row(psf_tensor, figsize=None, titles=None, show_colorbar=True):
    """
    将多个PSF绘制在一行子图中，使用对数scale
    
    参数:
        psf_tensor: torch.Tensor, 形状为 (mask_count, N, N)
        figsize: tuple, 图形大小，默认自动计算
        titles: list, 每个子图的标题，默认为 "PSF 1", "PSF 2", ...
        show_colorbar: bool, 是否显示colorbar
    """
    # 转换为numpy数组
    if isinstance(psf_tensor, torch.Tensor):
        psf_array = psf_tensor.detach().cpu().numpy()
    else:
        psf_array = np.array(psf_tensor)
    
    mask_count = psf_array.shape[0]
    
    # 自动计算图形大小
    if figsize is None:
        figsize = (4 * mask_count, 4)
    
    # 创建子图
    fig, axes = plt.subplots(1, mask_count, figsize=figsize)
    
    # 如果只有一个PSF，axes不是数组，需要转换
    if mask_count == 1:
        axes = [axes]
    
    # 绘制每个PSF
    for i, ax in enumerate(axes):
        im = ax.imshow(psf_array[i], cmap='hot')
        
        # 设置标题
        if titles is not None:
            ax.set_title(titles[i])
        else:
            ax.set_title(f'PSF {i+1}')
        
        # 添加colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensity', rotation=270, labelpad=15)
        
        ax.axis('off')
    
    plt.tight_layout()
    return fig, axes

def visualize_depth_psfs_and_masks(
    depth_psfs: torch.Tensor,
    depth_in_focus_ratio: List[float],
    out_focus_masks: Optional[torch.Tensor] = None,
    depth_out_focus_ratio: Optional[List[float]] = None,
    figsize: tuple = (12, 4)
):
    """
    可视化不同深度的PSF和离焦mask。
    
    Parameters
    ----------
    depth_psfs : torch.Tensor
        深度PSF张量 [num_depths, N, N]
    depth_in_focus_ratio : List[float]
        焦内z_ratio列表
    out_focus_masks : torch.Tensor, optional
        离焦mask张量 [num_out_focus, M*M, N, N]
    depth_out_focus_ratio : List[float], optional
        焦外z_ratio列表
    figsize : tuple
        每行的图形大小
    """
    # 转换为numpy并移到CPU
    depth_psfs_np = depth_psfs.cpu().numpy()
    
    # 确定绘图布局
    has_masks = out_focus_masks is not None and depth_out_focus_ratio is not None
    num_in_focus = len(depth_in_focus_ratio)
    
    if has_masks:
        # 处理mask：沿M*M维度求和
        out_focus_masks_np = out_focus_masks.sum(dim=1).cpu().numpy()  # [num_out_focus, N, N]
        num_out_focus = len(depth_out_focus_ratio)
        num_rows = max(num_in_focus, num_out_focus)
        num_cols = 2
    else:
        num_rows = num_in_focus
        num_cols = 1
    
    # 创建图形
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0], figsize[1] * num_rows))
    
    # 确保axes是2D数组
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 绘制depth_psfs（对数scale）
    for i in range(num_rows):
        ax_psf = axes[i, 0]
        
        if i < num_in_focus:
            # 绘制PSF（对数scale，避免log(0)）
            psf = depth_psfs_np[i]
            psf_log = np.log10(psf + 1e-10)  # 加小值避免log(0)
            
            im = ax_psf.imshow(psf_log, cmap='hot', origin='lower')
            ax_psf.set_title(f'In-Focus PSF (z_ratio={depth_in_focus_ratio[i]:.3f})\nLog Scale')
            ax_psf.set_xlabel('X (pixels)')
            ax_psf.set_ylabel('Y (pixels)')
            plt.colorbar(im, ax=ax_psf, label='log10(Intensity)')
        else:
            # 空白子图
            ax_psf.axis('off')
    
    # 绘制out_focus_masks（线性scale）
    if has_masks:
        for i in range(num_rows):
            ax_mask = axes[i, 1]
            
            if i < num_out_focus:
                # 绘制mask sum（线性scale）
                mask_sum = out_focus_masks_np[i]
                
                im = ax_mask.imshow(mask_sum, cmap='viridis', origin='lower')
                ax_mask.set_title(f'Out-Focus Mask Sum (z_ratio={depth_out_focus_ratio[i]:.3f})\nLinear Scale')
                ax_mask.set_xlabel('X (pixels)')
                ax_mask.set_ylabel('Y (pixels)')
                plt.colorbar(im, ax=ax_mask, label='Mask Count')
            else:
                # 空白子图
                ax_mask.axis('off')
    
    plt.tight_layout()
    plt.show()