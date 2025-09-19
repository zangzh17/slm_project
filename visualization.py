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
    ax.semilogy(optimizer.history['loss'])
    ax.set_title(f'Iteration {iteration+1}/{total_iterations}')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Total Loss (log scale)')
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

def plot_final_results(optimizer):
    """Display all final result plots after optimization."""
    print("\n--- Final Results Visualization ---")
    plot_loss_history(optimizer.history)
    plot_2d_comparisons(optimizer)
    plot_cross_sections(optimizer)
    plot_zoomed_on_peaks(optimizer)
    
def plot_loss_history(history):
    """Plot historical curves of loss components."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['mse1'], label='DOF Loss (MSE1 * w1)', marker='o', markersize=3)
    plt.plot(history['mse2'], label='Divergent Wave Loss (MSE2 * w2)', marker='s', markersize=3)
    plt.plot(history['mse3'], label='Convergent Wave Loss (MSE3 * w3)', marker='^', markersize=3)
    plt.yscale('log')
    plt.title('Weighted Loss Components History')
    plt.xlabel('Iterations')
    plt.ylabel('Weighted MSE (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.show()

    
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
        I_opt_div = optimizer.forward(optimizer.U_in_spherical_divergent, 
                                    optimizer.spherical_focal_plane_dist_div).cpu().numpy()
        I_opt_conv = optimizer.forward(optimizer.U_in_spherical_convergent, 
                                      optimizer.spherical_focal_plane_dist_conv).cpu().numpy()
        
        # Define number of samples for propagation analysis
        n_samples = 40
        
        # Compute propagation ranges for each wave type
        wave_configs = {
            "Plane Wave": {
                "U_in": optimizer.U_in_plane,
                "focal_dist": optimizer.focal_length,
                "optimized": I_opt_plane
            },
            "Divergent Wave": {
                "U_in": optimizer.U_in_spherical_divergent,
                "focal_dist": optimizer.spherical_focal_plane_dist_div,
                "optimized": I_opt_div
            },
            "Convergent Wave": {
                "U_in": optimizer.U_in_spherical_convergent,
                "focal_dist": optimizer.spherical_focal_plane_dist_conv,
                "optimized": I_opt_conv
            }
        }
        
        for name, config in wave_configs.items():
            # Create figure with 1x3 subplots - more compact layout
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.suptitle(f'Intensity Analysis: {name}', fontsize=14, y=1.02)
            
            # Plot 1: Optimized Intensity
            im1 = axes[0].imshow(config["optimized"], cmap='hot', norm=colors.LogNorm())
            axes[0].set_title('Optimized Intensity', fontsize=10)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.02)
            
            # Plot 2: Propagation from 0 to 1.2x focal length
            z_range_1 = torch.linspace(0, 1.2 * config["focal_dist"], n_samples)
            intensity_map_1 = compute_propagation_map(config["U_in"], z_range_1, optimizer)
            
            # Apply logarithmic scale with 4 orders of magnitude
            intensity_map_1_log = np.log10(intensity_map_1 + 1e-10)  # Add small value to avoid log(0)
            vmin_1 = np.percentile(intensity_map_1_log[intensity_map_1_log > -10], 1)
            vmax_1 = vmin_1 + 4  # 4 orders of magnitude
            
            im2 = axes[1].imshow(intensity_map_1_log, cmap='hot', 
                                vmin=vmin_1, vmax=vmax_1,
                                aspect='auto', extent=[0, intensity_map_1.shape[1], 
                                                      z_range_1[-1].item(), 
                                                      z_range_1[0].item()])
            axes[1].set_title('Propagation (0 - 1.2× focal)', fontsize=10)
            axes[1].set_xlabel('Y Position', fontsize=9)
            axes[1].set_ylabel('Distance', fontsize=9)
            axes[1].tick_params(axis='both', labelsize=8)
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.02)
            cbar2.set_label('log₁₀(I)', fontsize=9)
            cbar2.ax.tick_params(labelsize=8)
            
            # Plot 3: Propagation around focal length ± 5*depth_of_focus
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
            
            # Apply logarithmic scale with 4 orders of magnitude
            intensity_map_2_log = np.log10(intensity_map_2 + 1e-10)
            vmin_2 = np.percentile(intensity_map_2_log[intensity_map_2_log > -10], 1)
            vmax_2 = vmin_2 + 4  # 4 orders of magnitude
            
            # Convert to relative distance in micrometers
            focal_dist_um = config["focal_dist"] * 1e6  # Convert to micrometers
            z_range_2_relative_um = (z_range_2 - config["focal_dist"]).numpy() * 1e6  # Convert to μm
            
            im3 = axes[2].imshow(intensity_map_2_log, cmap='hot', 
                                vmin=vmin_2, vmax=vmax_2,
                                aspect='auto', extent=[0, intensity_map_2.shape[1], 
                                                      z_range_2_relative_um[-1], 
                                                      z_range_2_relative_um[0]])
            axes[2].set_title('Around Focal (±5×DOF)', fontsize=10)
            axes[2].set_xlabel('Y Position', fontsize=9)
            axes[2].set_ylabel('Δz (μm)', fontsize=9)
            axes[2].tick_params(axis='both', labelsize=8)
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.7, pad=0.02)
            cbar3.set_label('log₁₀(I)', fontsize=9)
            cbar3.ax.tick_params(labelsize=8)
            
            # Add a horizontal line at focal plane (z=0) for reference
            axes[2].axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
            
            plt.tight_layout()
            plt.show()

def plot_cross_sections(optimizer):
    """
    Plot intensity comparison of central cross-sections (Wider Figure).
    Uses scipy.signal.find_peaks to mark all peak positions with vertical dashed lines.
    """
    with torch.no_grad():
        I_opt_plane = optimizer.forward(optimizer.U_in_plane, optimizer.focal_length).cpu().numpy()
        I_opt_div = optimizer.forward(optimizer.U_in_spherical_divergent, optimizer.spherical_focal_plane_dist_div).cpu().numpy()
        I_opt_conv = optimizer.forward(optimizer.U_in_spherical_convergent, optimizer.spherical_focal_plane_dist_conv).cpu().numpy()
        tgt_plane = optimizer.target_plane.cpu().numpy()
        tgt_div = optimizer.target_divergent.cpu().numpy()
        tgt_conv = optimizer.target_convergent.cpu().numpy()
    
    y_slice = optimizer.N // 2
    
    plane_slice = I_opt_plane[y_slice, :]
    div_slice = I_opt_div[y_slice, :]
    conv_slice = I_opt_conv[y_slice, :]
    
    plt.figure(figsize=(20, 6))
    
    # Plot optimized curves
    plt.plot(plane_slice, 'b-', label='Optimized (Plane Wave)', lw=1)
    plt.plot(div_slice, 'g-', label='Optimized (Divergent Wave)', lw=1)
    plt.plot(conv_slice, 'r-', label='Optimized (Convergent Wave)', lw=1)
    
    # 使用 find_peaks 找到所有峰值的索引
    #    返回的第一个元素是包含所有峰值索引的numpy数组
    peaks_plane, _ = find_peaks(tgt_plane[y_slice, :],prominence=5)
    peaks_div, _ = find_peaks(tgt_div[y_slice, :],prominence=5)
    peaks_conv, _ = find_peaks(tgt_conv[y_slice, :],prominence=5)
    
    # 循环为每个找到的峰值绘制垂直线
    # Plane Wave Peaks (blue)
    for i, peak in enumerate(peaks_plane):
        # 只为第一个峰值添加标签，避免图例混乱
        label = 'Plane Peaks' if i == 0 else None
        plt.axvline(x=peak, color='b', linestyle='--', linewidth=1, label=label)

    # Divergent Wave Peaks (green)
    for i, peak in enumerate(peaks_div):
        label = 'Divergent Peaks' if i == 0 else None
        plt.axvline(x=peak, color='g', linestyle='--', linewidth=1, label=label)
        
    # Convergent Wave Peaks (red)
    for i, peak in enumerate(peaks_conv):
        label = 'Convergent Peaks' if i == 0 else None
        plt.axvline(x=peak, color='r', linestyle='--', linewidth=1, label=label)

    # --- 修改结束 ---

    all_data_max = np.max([
        plane_slice.max(), div_slice.max(), conv_slice.max()
    ])
    
    plt.ylim(all_data_max / 1e5, all_data_max * 1.1)
    
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
    在每个目标信号的中心横截面中查找主峰值，
    根据峰值宽度的10倍定义缩放区间，并创建一行子图来显示这些感兴趣的区域。
    
    Args:
        optimizer (object): 包含光学系统参数和前向传播方法的优化器对象。
    """
    # 1. 获取强度数据 (与原函数相同)
    with torch.no_grad():
        I_opt_plane = optimizer.forward(optimizer.U_in_plane, optimizer.focal_length).cpu().numpy()
        I_opt_div = optimizer.forward(optimizer.U_in_spherical_divergent, optimizer.spherical_focal_plane_dist_div).cpu().numpy()
        I_opt_conv = optimizer.forward(optimizer.U_in_spherical_convergent, optimizer.spherical_focal_plane_dist_conv).cpu().numpy()
    
    # 提取中心横截面
    y_slice = optimizer.N // 2
    
    # 将信号和相关信息打包到一个字典中，方便迭代处理
    signals = {
        'Plane Wave': {
            'data': I_opt_plane[y_slice, :],
            'color': 'b'
        },
        'Divergent Wave': {
            'data': I_opt_div[y_slice, :],
            'color': 'g'
        },
        'Convergent Wave': {
            'data': I_opt_conv[y_slice, :],
            'color': 'r'
        }
    }
    
    # 2. 创建一行子图
    num_signals = len(signals)
    # sharey=True 使所有子图共享Y轴，便于比较峰值高度
    fig, axes = plt.subplots(1, num_signals, figsize=(8 * num_signals, 6), sharey=True)
    
    # 如果只有一个信号，确保axes是一个可迭代的列表
    if num_signals == 1:
        axes = [axes]

    # 3. 遍历每个信号，查找峰值并绘图
    for ax, (title, props) in zip(axes, signals.items()):
        data = props['data']
        color = props['color']

        # 使用 find_peaks 查找所有峰值及其属性（如宽度）
        # prominence 参数可以帮助过滤掉噪声中的小峰值，这里设置为数据最大值的10%
        peaks, properties = find_peaks(data, width=1, prominence=(np.max(data) * 0.1, None))

        if len(peaks) == 0:
            # 如果没有找到峰值，则绘制完整信号并提示
            ax.plot(data, color=color)
            ax.set_title(f"{title}\n(No significant peak found)")
            ax.set_xlabel('X (pixels)')
            ax.grid(True, which="both", ls="--", alpha=0.6)
            continue # 处理下一个子图

        # 确定主峰值（强度最高的那个）
        main_peak_idx_in_peaks_array = np.argmax(data[peaks])
        peak_pos = peaks[main_peak_idx_in_peaks_array]
        peak_width = properties['widths'][main_peak_idx_in_peaks_array]

        # 4. 定义缩放区间（基于峰值宽度的10倍）
        # 缩放窗口的总宽度为 10 * peak_width，所以中心点左右各取 5 * peak_width
        zoom_half_width = int(np.ceil(5 * peak_width))
        x_start = max(0, peak_pos - zoom_half_width)
        x_end = min(len(data), peak_pos + zoom_half_width + 1) # Python切片不包含末尾，故+1

        # 5. 在子图上绘制缩放后的区域
        x_range = np.arange(x_start, x_end)
        ax.plot(x_range, data[x_start:x_end], color=color, lw=2)
        
        # 标记峰值位置
        ax.axvline(x=peak_pos, color='k', linestyle='--', linewidth=1.5, label=f'Peak @ {peak_pos}')
        
        # 可选：可视化find_peaks计算出的宽度（半高宽）
        # properties中包含了计算宽度时使用的左右边界点
        width_y_level = properties['width_heights'][main_peak_idx_in_peaks_array]
        width_x_min = properties['left_ips'][main_peak_idx_in_peaks_array]
        width_x_max = properties['right_ips'][main_peak_idx_in_peaks_array]
        ax.hlines(y=width_y_level, xmin=width_x_min, xmax=width_x_max,
                  color='black', linestyle=':', label=f'Width: {peak_width:.2f} px')

        # 设置子图的格式
        ax.set_title(f'Zoom on Peak: {title}')
        ax.set_xlabel('X (pixels)')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.6)
        
        # 严格限制X轴范围为缩放区间
        ax.set_xlim(x_start, x_end - 1)

    # 为共享的Y轴设置标签
    axes[0].set_ylabel('Intensity')
    
    # 为整个图像添加一个总标题
    fig.suptitle(f'Zoomed View of Main Peaks in Central Cross-section (y={y_slice})', fontsize=16)
    
    # 调整布局以防止标题重叠
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
        f"Diffraction Limited Spot: {airy_disk_um:.2f} µm"
    )
    plt.title(title)
    plt.legend()
    plt.show()