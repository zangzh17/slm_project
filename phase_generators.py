# phase_generators.py

"""
提供多种生成相位图的函数，包括菲涅尔透镜和优化算法。
"""
import numpy as np
import torch
import config
from optimizer import PhaseOptimizer
from optics_utils import calculate_linear_phase, create_checkerboard

def generate_fresnel_pattern(params: dict) -> tuple[np.ndarray, dict]:
    """
    计算菲涅尔微透镜阵列的相位图。

    Args:
        params (dict): 包含所有UI输入参数的字典。

    Returns:
        tuple[np.ndarray, dict]: (最终的8位相位图, 用于可视化的信息字典)
    """
    shape = params['shape']
    focal_length_m = params['focal_length'] * 1e-3
    
    # ROI 边界
    roi_left = max(0, int(params['roi_center_x'] - params['roi_width'] // 2))
    roi_right = min(shape[1], int(params['roi_center_x'] + params['roi_width'] // 2))
    roi_top = max(0, int(params['roi_center_y'] - params['roi_height'] // 2))
    roi_bottom = min(shape[0], int(params['roi_center_y'] + params['roi_height'] // 2))

    actual_roi_width = roi_right - roi_left
    actual_roi_height = roi_bottom - roi_top
    
    lens_height = actual_roi_height // params['rows']
    lens_width = actual_roi_width // params['cols']

    y, x = np.indices(shape)
    phase = np.zeros(shape)
    roi_mask = (x >= roi_left) & (x < roi_right) & (y >= roi_top) & (y < roi_bottom)
    
    linear_phase = calculate_linear_phase(shape, params['angle_x_mrad'], params['angle_y_mrad'])

    for r in range(params['rows']):
        for c in range(params['cols']):
            center_y = roi_top + r * lens_height + lens_height // 2
            center_x = roi_left + c * lens_width + lens_width // 2
            
            y_start, y_end = roi_top + r * lens_height, roi_top + (r + 1) * lens_height
            x_start, x_end = roi_left + c * lens_width, roi_left + (c + 1) * lens_width
            
            region = (slice(y_start, y_end), slice(x_start, x_end))
            
            x_dist = (x[region] - center_x) * config.PIXEL_SIZE
            y_dist = (y[region] - center_y) * config.PIXEL_SIZE
            r_squared = x_dist**2 + y_dist**2
            
            # 精确公式: φ = (2π/λ) * (f - sqrt(f² + r²))
            f_squared = focal_length_m**2
            phase_calc = (2 * np.pi / config.WAVELENGTH) * \
                         (focal_length_m - np.sqrt(f_squared + r_squared))
            
            phase[region] = phase_calc + linear_phase[region]
    
    phase = phase % (2 * np.pi)
    
    # 结合背景
    checkerboard = create_checkerboard(shape)
    combined_phase = np.where(roi_mask, phase, checkerboard)
    
    # 转换为 8-bit
    phi = np.uint8(combined_phase / (2 * np.pi) * params['two_pi_value'])
    
    info = {
        'phi': phi, 'focal_length': params['focal_length'], 'rows': params['rows'], 'cols': params['cols'],
        'roi_width': actual_roi_width, 'roi_height': actual_roi_height, 'lens_width': lens_width, 'lens_height': lens_height,
        'angle_x_mrad': params['angle_x_mrad'], 'angle_y_mrad': params['angle_y_mrad'],
        'two_pi_value': params['two_pi_value'], 'roi_rect': (roi_left, roi_top, actual_roi_width, actual_roi_height)
    }
    return phi, info

def generate_optimized_pattern(params: dict, vis_callback) -> tuple[np.ndarray, object]:
    """
    通过优化算法生成微透镜阵列的相位图。

    Args:
        params (dict): 包含所有UI输入参数的字典。
        vis_callback (function): 用于实时可视化的回调函数。

    Returns:
        tuple[np.ndarray, object]: (最终的8位相位图, 优化器对象本身用于后续分析)
    """
    focal_length = (params['focal_length_coarse'] + params['focal_length_fine']) * 1e-3
    if params['lens_type']: focal_length = -focal_length

    roi_mask = (np.zeros(params['shape']) == 0) # 简化：假设优化总是在方形区域
    N = int(np.sqrt(np.sum(roi_mask)))

    optimizer = PhaseOptimizer(
        N=N, pixel_size=config.PIXEL_SIZE, wavelength=config.WAVELENGTH, focal_length=focal_length,
        psf_energy_level=params['psf_energy_level'], dof_tol_factor=params['dof_factor'],
        size_factor=params['size_factor'], M=min(params['rows'], params['cols']),
        aperture_overlap_ratio=params['overlap_ratio']
    )
    
    optimized_phase = optimizer.optimize(
        num_iterations=params['ni'], learning_rate=params['lr'], update_callback=vis_callback
    )
    
    # 将优化结果嵌入到整个SLM图案中
    final_phase = np.zeros(params['shape'])
    phase_wrapped = torch.remainder(optimized_phase, 2 * np.pi).cpu().numpy()
    final_phase[roi_mask] = phase_wrapped.flatten() # 简化赋值
    
    checkerboard = create_checkerboard(params['shape'])
    combined_phase = np.where(roi_mask, final_phase, checkerboard)
    
    phi = np.uint8(combined_phase / (2 * np.pi) * params['phase_range'])
    
    return phi, optimizer