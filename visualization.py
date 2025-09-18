# visualization.py

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch
from IPython.display import display, clear_output

from optics_utils import calculate_airy_disk
import config

# 这一行设置了matplotlib在Jupyter中的后端，以支持实时更新
# %matplotlib inline

def plot_live_update(iteration: int, total_iterations: int, loss: float, optimizer):
    """在优化过程中实时更新和显示图像。"""
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 损失曲线
    ax = axes[0]
    ax.semilogy(optimizer.history['loss'])
    ax.set_title(f'迭代 {iteration+1}/{total_iterations}')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('总损失 (log scale)')
    ax.grid(True, linestyle='--', alpha=0.6)

    # 2. 当前相位
    ax = axes[1]
    phase_wrapped = torch.remainder(optimizer.phase_param.detach(), 2 * np.pi).cpu().numpy()
    im = ax.imshow(phase_wrapped, cmap='hsv', vmin=0, vmax=2 * np.pi)
    ax.set_title('当前优化相位')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='相位 (rad)')

    # 3. 当前焦平面强度
    ax = axes[2]
    with torch.no_grad():
        intensity = optimizer.forward(optimizer.U_in_plane, optimizer.focal_length)
    im = ax.imshow(intensity.cpu().numpy(), cmap='hot', norm=colors.LogNorm())
    ax.set_title(f'焦平面强度 (Loss: {loss:.4e})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='强度 (log scale)')

    plt.tight_layout()
    display(fig)
    plt.close(fig)

def plot_final_results(optimizer, optimized_phase):
    """优化结束后，显示所有最终结果图。"""
    print("\n--- 最终结果可视化 ---")
    _plot_loss_history(optimizer.history)
    _plot_2d_comparisons(optimizer, optimized_phase)
    _plot_cross_sections(optimizer, optimized_phase)
    
def _plot_loss_history(history):
    """绘制各个损失分量的历史曲线。"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['mse1'], label='景深损失 (MSE1 * w1)', marker='o', markersize=3)
    plt.plot(history['mse2'], label='发散波损失 (MSE2 * w2)', marker='s', markersize=3)
    plt.plot(history['mse3'], label='汇聚波损失 (MSE3 * w3)', marker='^', markersize=3)
    plt.yscale('log')
    plt.title('各加权损失分量历史')
    plt.xlabel('迭代次数')
    plt.ylabel('加权MSE (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.show()

def _plot_2d_comparisons(optimizer, optimized_phase):
    """并排比较目标强度和优化后的强度图。"""
    with torch.no_grad():
        I_opt_plane = optimizer.forward(optimizer.U_in_plane, optimizer.focal_length).cpu().numpy()
        I_opt_div = optimizer.forward(optimizer.U_in_spherical_divergent, optimizer.spherical_focal_plane_dist_div).cpu().numpy()
        I_opt_conv = optimizer.forward(optimizer.U_in_spherical_convergent, optimizer.spherical_focal_plane_dist_conv).cpu().numpy()

    targets = {
        "平面波": (optimizer.target_plane.cpu().numpy(), I_opt_plane),
        "发散波": (optimizer.target_divergent.cpu().numpy(), I_opt_div),
        "汇聚波": (optimizer.target_convergent.cpu().numpy(), I_opt_conv)
    }

    for name, (target, optimized) in targets.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'强度对比: {name}', fontsize=16)
        
        im1 = axes[0].imshow(target, cmap='hot', norm=colors.LogNorm())
        axes[0].set_title('目标强度')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        im2 = axes[1].imshow(optimized, cmap='hot', norm=colors.LogNorm())
        axes[1].set_title('优化后强度')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
def _plot_cross_sections(optimizer, optimized_phase):
    """绘制中心横截面的强度对比图。"""
    with torch.no_grad():
        I_opt_plane = optimizer.forward(optimizer.U_in_plane, optimizer.focal_length).cpu().numpy()
        I_opt_div = optimizer.forward(optimizer.U_in_spherical_divergent, optimizer.spherical_focal_plane_dist_div).cpu().numpy()
        I_opt_conv = optimizer.forward(optimizer.U_in_spherical_convergent, optimizer.spherical_focal_plane_dist_conv).cpu().numpy()
    
    tgt_plane = optimizer.target_plane.cpu().numpy()
    tgt_div = optimizer.target_divergent.cpu().numpy()
    tgt_conv = optimizer.target_convergent.cpu().numpy()
    
    y_slice = optimizer.N // 2

    plt.figure(figsize=(12, 7))
    plt.plot(tgt_plane[y_slice, :], 'b--', label='目标 (平面波)')
    plt.plot(I_opt_plane[y_slice, :], 'b-', label='优化后 (平面波)', lw=2)
    
    plt.plot(tgt_div[y_slice, :], 'g--', label='目标 (发散波)')
    plt.plot(I_opt_div[y_slice, :], 'g-', label='优化后 (发散波)', lw=2)

    plt.plot(tgt_conv[y_slice, :], 'r--', label='目标 (汇聚波)')
    plt.plot(I_opt_conv[y_slice, :], 'r-', label='优化后 (汇聚波)', lw=2)

    plt.yscale('log')
    plt.title(f'中心横截面强度对比 (y={y_slice})')
    plt.xlabel('X (像素)')
    plt.ylabel('强度 (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.show()

def plot_fresnel_pattern(info: dict):
    """
    可视化生成的菲涅尔微透镜阵列相位图。
    """
    plt.figure(figsize=(12, 7))
    plt.imshow(info['phi'], cmap='gray', vmin=0, vmax=255)
    plt.colorbar(label=f"灰度值 (0-{info['two_pi_value']})")

    # 绘制 ROI 边界
    rect = info['roi_rect']
    plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                      edgecolor='r', facecolor='none', lw=2, label='ROI'))

    # 计算并显示信息
    f_m = abs(info['focal_length']) * 1e-3
    lens_w_m = info['lens_width'] * config.PIXEL_SIZE
    airy_disk_um = calculate_airy_disk(f_m, lens_w_m)
    angle_x_deg = info['angle_x_mrad'] * 180 / np.pi / 1000
    angle_y_deg = info['angle_y_mrad'] * 180 / np.pi / 1000

    title = (
        f"菲涅尔微透镜阵列: {info['rows']}×{info['cols']}, f={info['focal_length']:.1f} mm\n"
        f"衍射极限光斑: {airy_disk_um:.2f} µm | "
        f"偏转: θx={angle_x_deg:.3f}°, θy={angle_y_deg:.3f}°"
    )
    plt.title(title)
    plt.legend()
    plt.show()