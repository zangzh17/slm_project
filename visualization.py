# visualization.py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from scipy import ndimage
import torch
from scipy.ndimage import zoom
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output
from optics_utils import calculate_airy_disk

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
    im = ax.imshow(intensity, cmap='hot', norm=colors.LogNorm())
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
    lens_w_m = optimizer.lens_width * optimizer.pixel_size
    airy_disk_m = calculate_airy_disk(f_m, lens_w_m, optimizer.wavelength)
    title = (
        f"Diff.Lim. Spot: {airy_disk_m*1e6:.2f} µm / {airy_disk_m/optimizer.pixel_size:.1f} pix"
    )
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_2d_comparisons(optimizer):
    """Side-by-side comparison of optimized intensity patterns with propagation analysis."""
    
    def compute_propagation_map(U_in, z_values, optimizer, max_norm=True):
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
            intensity_1d = intensity[optimizer.N//2,:]
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
            # Create figure with 1x3 subplots - more compact layout
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f'Intensity Analysis: {name}', fontsize=14, y=1.02)
            
            # Plot 1: Optimized Intensity
            im1 = axes[0].imshow(config["optimized"], cmap='hot', norm=colors.LogNorm())
            axes[0].set_title('Optimized Intensity', fontsize=10)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.02)
            
            # Plot 2: Propagation from 0 to 1.2x focal length, linear scale
            z_range_1 = torch.linspace(0, 1.2 * config["focal_dist"], n_samples)
            intensity_map_1 = compute_propagation_map(config["U_in"], z_range_1, optimizer)
            intensity_map_2 = compute_propagation_map(config["U_in"], z_range_1, optimizer, max_norm=False)
            
            im2 = axes[1].imshow(intensity_map_1, cmap='hot',
                                aspect='auto', extent=[0, intensity_map_1.shape[1], 
                                                      z_range_1[-1].item(), 
                                                      z_range_1[0].item()])
            axes[1].set_title('Propagation (0 - 1.2× focal)', fontsize=10)
            axes[1].set_xlabel('Y Position', fontsize=9)
            axes[1].set_ylabel('Distance', fontsize=9)
            axes[1].tick_params(axis='both', labelsize=8)
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.02)
            cbar2.set_label('I', fontsize=9)
            cbar2.ax.tick_params(labelsize=8)
            
            # Plot 3: Propagation from 0 to 1.2x focal length, log scale
            intensity_map_log = intensity_map_2/intensity_map_2.max()
            intensity_map_log = np.log10(intensity_map_log + 1e-10)
            intensity_map_log[intensity_map_log<-4] = -4
            im3 = axes[2].imshow(intensity_map_log, cmap='hot', 
                                aspect='auto', extent=[0, intensity_map_log.shape[1], 
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
            z_min = config["focal_dist"] - 4 * optimizer.depth_of_focus
            z_max = config["focal_dist"] + 4 * optimizer.depth_of_focus
            
            z_range_2 = torch.linspace(z_min, z_max, n_samples)
            intensity_map_3 = compute_propagation_map(config["U_in"], z_range_2, optimizer)
            
            # Convert to relative distance in micrometers
            z_range_2_relative_um = (z_range_2 - config["focal_dist"]).numpy() * 1e6  # Convert to μm
            

            im4 = axes[3].imshow(intensity_map_3, cmap='hot', 
                                aspect='auto', extent=[0, intensity_map_3.shape[1], 
                                                      z_range_2_relative_um[-1], 
                                                      z_range_2_relative_um[0]])
            
            axes[3].set_title('Around Focal (±4×DOF)', fontsize=10)
            axes[3].set_xlabel('Y Position', fontsize=9)
            axes[3].set_ylabel('Δz (μm)', fontsize=9)
            axes[3].tick_params(axis='both', labelsize=8)
            cbar4 = plt.colorbar(im4, ax=axes[3], shrink=0.7, pad=0.02)
            cbar4.set_label('log₁₀(I)', fontsize=9)
            cbar4.ax.tick_params(labelsize=8)
            # Add a horizontal line at focal plane (z=0) for reference
            axes[3].axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
            
            plt.tight_layout()
            plt.show()

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


def plot_energy_distribution(optimizer,upsampling=1.0):
    """
    可视化衍射透镜阵列中所有焦斑的聚焦效率（相对比例）。

    1. 从理想模板(total_psfs)中定位所有焦斑的中心。
    2. 计算圈入能量的计算半径。
    3. 在优化后的实际光强图中，为每个焦斑计算圈入能量。
    4. 将圈入能量除以理论分配能量，得到效率比例。
    5. 绘制柱状图展示效率分布，并标注统计数据。
    """
    N_up = optimizer.N * upsampling
    pixel_size_up = optimizer.pixel_size / upsampling
    
    with torch.no_grad():
        I_opt = optimizer.forward(upsampling=upsampling).cpu().numpy()
        tgt_psfs = optimizer.total_psfs.cpu().numpy()
        tgt_psfs_up = zoom(tgt_psfs, zoom=upsampling, order=0)


    # --- 步骤 1: 定位焦斑中心 (与之前相同) ---
    threshold = tgt_psfs_up.max() * 0.5
    binary_mask = tgt_psfs_up > threshold
    labeled_array, num_features = ndimage.label(binary_mask)
    centers = ndimage.center_of_mass(tgt_psfs_up, labeled_array, range(1, num_features + 1))
    
    expected_spots = optimizer.M * optimizer.M
    if num_features != expected_spots:
        print(f"Warning: Found {num_features} spots, but expected {expected_spots}.")

    # --- 步骤 2: 计算半径 (与之前相同) ---
    radius_meters = 1.22 * optimizer.wavelength * optimizer.f_number
    radius_pixels = radius_meters / pixel_size_up
    
    print(f"Encircled energy radius used for calculation: {radius_pixels:.2f} pixels")

    # --- 步骤 3: 计算圈入能量 (与之前相同) ---
    encircled_energies = []
    Y, X = np.ogrid[:N_up, :N_up]
    
    for center_y, center_x in centers:
        dist_sq = (X - center_x)**2 + (Y - center_y)**2
        mask = dist_sq <= radius_pixels**2
        energy = I_opt[mask].sum()
        encircled_energies.append(energy)
        
    abs_energies = np.array(encircled_energies)

    # --- 步骤 4: 计算效率比例并进行统计 ---
    # 计算理论上分配给单个焦点的总能量
    ideal_energy_per_spot = (N_up * N_up) / (optimizer.M * optimizer.M)
    
    # 防止除以零的错误
    if ideal_energy_per_spot == 0:
        print("Error: Ideal energy per spot is zero. Cannot calculate efficiency.")
        return
        
    # 计算效率（每个焦斑的圈入能量 / 理论分配的总能量）
    efficiencies = abs_energies / ideal_energy_per_spot
    
    # 将效率转换为百分比进行显示
    efficiencies_percent = efficiencies * 100
    
    mean_eff = efficiencies_percent.mean()
    max_eff = efficiencies_percent.max()
    min_eff = efficiencies_percent.min()
    std_eff = efficiencies_percent.std()
    
    # --- 步骤 5: 可视化 ---
    plt.figure(figsize=(16, 8))
    spot_indices = np.arange(len(efficiencies_percent))
    plt.bar(spot_indices, efficiencies_percent, color='darkcyan', label='Focusing Efficiency per Spot')
    
    # 更新统计信息文本
    stats_text = (f'Total Spots: {len(efficiencies_percent)}\n'
                  f'Mean Efficiency: {mean_eff:.2f}%\n'
                  f'Max Efficiency: {max_eff:.2f}%\n'
                  f'Min Efficiency: {min_eff:.2f}%\n'
                  f'Std Dev: {std_eff:.2f}%')
                  
    plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='lightgoldenrodyellow', alpha=0.6))
             
    plt.title('Focusing Efficiency Distribution of Focal Spots', fontsize=16)
    plt.xlabel('Focal Spot Index', fontsize=12)
    plt.ylabel('Focusing Efficiency (%)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    # 根据数据范围设置一个合适的Y轴，例如0到最大效率的1.2倍
    plt.ylim(0, max(110, np.max(efficiencies_percent) * 1.2)) 
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