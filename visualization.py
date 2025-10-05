# visualization.py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.signal import find_peaks
import torch
from IPython.display import display, clear_output
from optics_utils import calculate_airy_disk
import config

def plot_live_update(iteration: int, total_iterations: int, loss: float, optimizer):
    """Display and update images in real-time during optimization."""
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    
    # 3. Current focal plane intensity
    ax = axes[2]
    with torch.no_grad():
        intensity = optimizer.forward(optimizer.U_in_plane, optimizer.focal_length)
    im = ax.imshow(intensity.cpu().numpy(), cmap='hot', norm=colors.LogNorm())
    ax.set_title(f'Focal Plane Intensity (Loss: {loss:.4e})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Intensity (log scale)')
    
    plt.tight_layout()
    display(fig)
    plt.close(fig)

def plot_final_results(optimizer,info):
    """Display all final result plots after optimization."""

    # Visualize the generated Fresnel microlens array phase pattern.
    plt.figure(figsize=(12, 7))
    plt.imshow(info['phi'], cmap='gray', vmin=0, vmax=255)
    plt.colorbar(label=f"Gray Level (0-{info['two_pi_value']})")
    # Draw ROI boundary
    rect = info['roi_rect']
    plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                      edgecolor='r', facecolor='none', lw=2, label='ROI'))
    # Calculate and display information
    f_m = abs(info['focal_length'])
    lens_w_m = info['lens_width'] * config.PIXEL_SIZE
    airy_disk_um = calculate_airy_disk(f_m, lens_w_m)
    title = (
        f"Diff.Lim. Spot: {airy_disk_um:.2f} µm / {airy_disk_um/config.PIXEL_SIZE:.1f} pix"
    )
    plt.title(title)
    plt.legend()
    plt.show()

    print("\n--- Final Results Visualization ---")
    plot_2d_comparisons(optimizer)
    plot_cross_sections(optimizer)
    plot_zoomed_on_peaks(optimizer)
    
def plot_2d_comparisons(optimizer):
    """Side-by-side comparison of optimized intensity patterns with propagation analysis."""
    
    def compute_propagation_map(U_in, z_values, optimizer):
        """Compute intensity map for multiple propagation distances."""
        n_z = len(z_values)
        # Get the shape from a test propagation
        test_field = optimizer.forward(U_in, z_values[0])
        n_y = test_field.shape[0]
        
        # Initialize the intensity map
        intensity_map = torch.zeros((n_z, n_y))
        
        for i, z in enumerate(z_values):
            # Propagate to distance z
            field = optimizer.forward(U_in, z)
            intensity = torch.abs(field)**2
            # Sum along x-axis (axis=1) to get 1D intensity profile
            intensity_1d = intensity.sum(dim=1)
            intensity_map[i, :] = intensity_1d
        
        return intensity_map.cpu().numpy()
    
    with torch.no_grad():
        # Original intensity calculations
        I_opt_plane = optimizer.forward(optimizer.U_in_plane, optimizer.focal_length).cpu().numpy()
        
        # Define number of samples for propagation analysis
        n_samples = 50
        
        # Compute propagation ranges for each wave type
        wave_configs = {
            "Plane Wave": {
                "U_in": optimizer.U_in_plane,
                "focal_dist": optimizer.focal_length,
                "optimized": I_opt_plane
            }
        }
        
        for name, config in wave_configs.items():
            # Create figure with 1x3 subplots - more compact layout
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f'Intensity Analysis: {name}', fontsize=14, y=1.02)
            
            # Plot 1: Optimized Intensity
            im1 = axes[0].imshow(config["optimized"], cmap='hot', norm=colors.LogNorm())
            axes[0].set_title('Optimized Intensity', fontsize=10)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.02)
            
            # Plot 2: Propagation from 0 to 1.2x focal length, log scale
            z_range_1 = torch.linspace(0, 1.2 * config["focal_dist"], n_samples)
            intensity_map_1 = compute_propagation_map(config["U_in"], z_range_1, optimizer)
            
            # Apply logarithmic scale with some orders of magnitude
            intensity_map_1_log = np.log10(intensity_map_1 + 1e-10)  # Add small value to avoid log(0)
            vmin_1 = np.percentile(intensity_map_1_log[intensity_map_1_log > -10], 1)
            vmax_1 = vmin_1 + 4  # some orders of magnitude
            
            im2 = axes[1].imshow(intensity_map_1_log, cmap='hot', 
                                vmin=vmin_1, vmax=vmax_1,
                                aspect='auto', extent=[0, intensity_map_1.shape[1], 
                                                      z_range_1[-1].item(), 
                                                      z_range_1[0].item()])
            axes[1].set_title('Propagation (0 - 1.2× focal), log scale', fontsize=10)
            axes[1].set_xlabel('Y Position', fontsize=9)
            axes[1].set_ylabel('Distance', fontsize=9)
            axes[1].tick_params(axis='both', labelsize=8)
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.02)
            cbar2.set_label('log₁₀(I)', fontsize=9)
            cbar2.ax.tick_params(labelsize=8)
            
            # Plot 3: Propagation from 0 to 1.2x focal length, linear scale
            im3 = axes[2].imshow(intensity_map_1, cmap='hot', 
                                aspect='auto', extent=[0, intensity_map_1.shape[1], 
                                                      z_range_1[-1].item(), 
                                                      z_range_1[0].item()])
            axes[2].set_title('Propagation (0 - 1.2× focal)', fontsize=10)
            axes[2].set_xlabel('Y Position', fontsize=9)
            axes[2].set_ylabel('Distance', fontsize=9)
            axes[2].tick_params(axis='both', labelsize=8)
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.7, pad=0.02)
            cbar3.set_label('log₁₀(I)', fontsize=9)
            cbar3.ax.tick_params(labelsize=8)
            # Add a horizontal line at focal plane (z=0) for reference
            axes[3].axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
            
            
            # Plot 4: Propagation around focal length ± 5*depth_of_focus
            # Check if depth_of_focus exists, otherwise use a default range
            if hasattr(optimizer, 'depth_of_focus'):
                z_min = config["focal_dist"] - 5 * optimizer.depth_of_focus
                z_max = config["focal_dist"] + 5 * optimizer.depth_of_focus
            else:
                # Use 10% of focal length as default if depth_of_focus not available
                delta = 0.1 * config["focal_dist"]
                z_min = config["focal_dist"] - delta
                z_max = config["focal_dist"] + delta
            
            z_range_2 = torch.linspace(z_min, z_max, n_samples)
            intensity_map_2 = compute_propagation_map(config["U_in"], z_range_2, optimizer)
            
            # Convert to relative distance in micrometers
            focal_dist_um = config["focal_dist"] * 1e6  # Convert to micrometers
            z_range_2_relative_um = (z_range_2 - config["focal_dist"]).numpy() * 1e6  # Convert to μm
            
            # # Apply logarithmic scale with 4 orders of magnitude
            # intensity_map_2_log = np.log10(intensity_map_2 + 1e-10)
            # vmin_2 = np.percentile(intensity_map_2_log[intensity_map_2_log > -10], 1)
            # vmax_2 = vmin_2 + 4  # 4 orders of magnitude
            # im3 = axes[2].imshow(intensity_map_2_log, cmap='hot', 
            #                     vmin=vmin_2, vmax=vmax_2,
            #                     aspect='auto', extent=[0, intensity_map_2.shape[1], 
            #                                           z_range_2_relative_um[-1], 
            #                                           z_range_2_relative_um[0]])
            im4 = axes[3].imshow(intensity_map_2, cmap='hot', 
                                aspect='auto', extent=[0, intensity_map_2.shape[1], 
                                                      z_range_2_relative_um[-1], 
                                                      z_range_2_relative_um[0]])
            
            axes[3].set_title('Around Focal (±5×DOF)', fontsize=10)
            axes[3].set_xlabel('Y Position', fontsize=9)
            axes[3].set_ylabel('Δz (μm)', fontsize=9)
            axes[3].tick_params(axis='both', labelsize=8)
            cbar4 = plt.colorbar(im4, ax=axes[3], shrink=0.7, pad=0.02)
            cbar4.set_label('log₁₀(I)', fontsize=9)
            cbar4.ax.tick_params(labelsize=8)
            # Add a horizontal line at focal plane (z=0) for reference
            axes[3].axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
            
            plt.tight_layout()
            plt.show()

def plot_cross_sections(optimizer):
    """
    Plot intensity comparison of central cross-sections (Wider Figure).
    Uses scipy.signal.find_peaks to mark all peak positions with vertical dashed lines.
    """
    with torch.no_grad():
        I_opt_plane = optimizer.forward(optimizer.U_in_plane, optimizer.focal_length).cpu().numpy()
        tgt_plane = optimizer.target_plane.cpu().numpy()

    
    y_slice = optimizer.N // 2
    
    plane_slice = I_opt_plane[y_slice, :]
    
    plt.figure(figsize=(20, 6))
    
    # Plot optimized curves
    plt.plot(plane_slice, 'b-', label='Optimized (Plane Wave)', lw=1)
    
    # 使用 find_peaks 找到所有峰值的索引
    #    返回的第一个元素是包含所有峰值索引的numpy数组
    peaks_plane, _ = find_peaks(tgt_plane[y_slice, :],prominence=5)
    
    # 循环为每个找到的峰值绘制垂直线
    # Plane Wave Peaks (blue)
    for i, peak in enumerate(peaks_plane):
        # 只为第一个峰值添加标签，避免图例混乱
        label = 'Plane Peaks' if i == 0 else None
        plt.axvline(x=peak, color='b', linestyle='--', linewidth=1, label=label)

    plt.yscale('log')
    plt.title(f'Central Cross-section Intensity Comparison (y={y_slice})')
    plt.xlabel('X (pixels)')
    plt.ylabel('Intensity (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_zoomed_on_peaks(optimizer):
    """
    Plot zoomed-in views of peak regions from central cross-sections.
    Each peak region (based on plane wave peaks) is displayed in a separate subplot arranged in a single row.
    The zoom window width is approximately 1/10 of the average peak spacing.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    import torch
    
    with torch.no_grad():
        I_opt_plane = optimizer.forward(optimizer.U_in_plane, optimizer.focal_length).cpu().numpy()
        tgt_plane = optimizer.target_plane.cpu().numpy()
    
    y_slice = optimizer.N // 2
    plane_slice = I_opt_plane[y_slice, :]
    tgt_plane_slice = tgt_plane[y_slice, :]
    
    # Find peaks for all three target patterns
    peaks_plane, _ = find_peaks(tgt_plane[y_slice, :], prominence=5)
    
    # Use only plane wave peaks for creating subplots
    if len(peaks_plane) == 0:
        print("No peaks found in plane wave")
        return
    
    # Calculate average peak spacing to determine zoom window width
    if len(peaks_plane) > 1:
        peak_spacings = np.diff(peaks_plane)
        avg_spacing = np.mean(peak_spacings)
    else:
        avg_spacing = optimizer.N // 10
    
    # Set zoom window width as a fraction of average peak spacing
    zoom_fraction = 1/8  # Easy to modify: 1/10, 1/8, 1/5, etc.
    half_window = int(avg_spacing * zoom_fraction)
    half_window = max(half_window, 10)  # Ensure minimum window size
    
    # Create figure with subplots for each plane wave peak
    n_peaks = len(peaks_plane)
    fig, axes = plt.subplots(1, n_peaks, figsize=(5*n_peaks, 5))
    
    # Handle case where there's only one peak
    if n_peaks == 1:
        axes = [axes]
    
    # Plot zoomed view for each plane wave peak
    for idx, peak in enumerate(peaks_plane):
        ax = axes[idx]
        
        # Calculate window boundaries
        x_min = max(0, peak - half_window)
        x_max = min(len(plane_slice) - 1, peak + half_window)
        x_range = np.arange(x_min, x_max + 1)
        
        # Plot the zoomed sections
        ax.plot(x_range, plane_slice[x_min:x_max+1], 'b-', 
                label='Optimized (Plane)', linewidth=1.5)
        
        # Plot the zoomed sections
        ax.plot(x_range, tgt_plane_slice[x_min:x_max+1], 'b--', 
                label='Target (Plane)', linewidth=1)
        
        # Set log scale and limits
        ax.set_yscale('log')
        
        # Labels and formatting
        ax.set_xlabel('X (pixels)')
        if idx == 0:
            ax.set_ylabel('Intensity (log scale)')
        ax.set_title(f'Peak at x={peak}')
        ax.grid(True, which="both", ls="--", alpha=0.6)
        
        # Only show legend on first subplot to avoid clutter
        if idx == 0:
            ax.legend(fontsize=8, loc='best')
    
    plt.suptitle(f'Zoomed Views of Peak Regions (y={y_slice}, window=±{half_window} pixels)', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_fresnel_pattern(info: dict):
    """
    Visualize the generated Fresnel microlens array phase pattern.
    """
    plt.figure(figsize=(12, 7))
    plt.imshow(info['phi'], cmap='gray', vmin=0, vmax=255)
    plt.colorbar(label=f"Gray Level (0-{info['two_pi_value']})")
    
    # Draw ROI boundary
    rect = info['roi_rect']
    plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                      edgecolor='r', facecolor='none', lw=2, label='ROI'))
    
    # Calculate and display information
    f_m = abs(info['focal_length']) * 1e-3
    lens_w_m = info['lens_width'] * config.PIXEL_SIZE
    airy_disk_um = calculate_airy_disk(f_m, lens_w_m)
    
    title = (
        f"Fresnel Microlens Array: {info['rows']}×{info['cols']}, f={info['focal_length']:.1f} mm\n"
        f"Diff.Lim. Spot: {airy_disk_um:.2f} µm / {airy_disk_um/config.PIXEL_SIZE:.1f} pix"
    )
    plt.title(title)
    plt.legend()
    plt.show()