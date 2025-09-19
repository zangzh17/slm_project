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

    rows, cols = params['rows'], params['cols']
    for r in range(rows):
        for c in range(cols):
            # --- 这是修改后的核心逻辑 ---
            # 通过将乘法放在整除前，可以更精确地分配像素
            y_start = roi_top + (r * actual_roi_height) // rows
            y_end = roi_top + ((r + 1) * actual_roi_height) // rows
            
            x_start = roi_left + (c * actual_roi_width) // cols
            x_end = roi_left + ((c + 1) * actual_roi_width) // cols

            # 使用新的起止点计算中心（这也是更精确的方法）
            center_y = (y_start + y_end) / 2
            center_x = (x_start + x_end) / 2
            
            # 创建slice对象，这里的变量现在保证是整数
            region = (slice(y_start, y_end), slice(x_start, x_end))
            
            # 后续计算保持不变
            x_dist = (x[region] - center_x) * config.PIXEL_SIZE
            y_dist = (y[region] - center_y) * config.PIXEL_SIZE
            r_squared = x_dist**2 + y_dist**2
            
            f_squared = focal_length_m**2
            phase_calc = (2 * np.pi / config.WAVELENGTH) * \
                        (focal_length_m - np.sqrt(f_squared + r_squared))
            
            phase[region] = phase_calc

    
    phase = phase % (2 * np.pi)
    
    # 结合背景
    checkerboard = create_checkerboard(shape)
    combined_phase = np.where(roi_mask, phase, checkerboard)
    
    # 转换为 8-bit
    phi = np.uint8(combined_phase / (2 * np.pi) * params['two_pi_value'])
    
    info = {
        'phi': phi, 'focal_length': params['focal_length'], 'rows': params['rows'], 'cols': params['cols'],
        'roi_width': actual_roi_width, 'roi_height': actual_roi_height, 'lens_width': lens_width, 'lens_height': lens_height,
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
    shape = params['shape']
    focal_length = (params['focal_length_coarse'] + params['focal_length_fine']) * 1e-3
    if params['lens_type']: 
        focal_length = -focal_length

    # 使用与generate_fresnel_pattern相同的ROI逻辑
    roi_left = max(0, int(params['roi_center_x'] - params['roi_width'] // 2))
    roi_right = min(shape[1], int(params['roi_center_x'] + params['roi_width'] // 2))
    roi_top = max(0, int(params['roi_center_y'] - params['roi_height'] // 2))
    roi_bottom = min(shape[0], int(params['roi_center_y'] + params['roi_height'] // 2))

    actual_roi_width = roi_right - roi_left
    actual_roi_height = roi_bottom - roi_top
    
    # 创建正确的ROI mask
    y, x = np.indices(shape)
    roi_mask = (x >= roi_left) & (x < roi_right) & (y >= roi_top) & (y < roi_bottom)
    
    # 计算优化区域的尺寸 - 使用较小的维度作为正方形边长
    # 或者根据你的具体需求调整
    N = min(actual_roi_width, actual_roi_height)

    optimizer = PhaseOptimizer(
        N=N, 
        pixel_size=config.PIXEL_SIZE, 
        wavelength=config.WAVELENGTH, 
        focal_length=focal_length,
        psf_energy_level=params['psf_energy_level'], 
        dof_tol_factor=params['dof_factor'],
        size_factor=params['size_factor'], 
        M=min(params['rows'], params['cols']),
        aperture_overlap_ratio=params['overlap_ratio']
    )
    
    optimized_phase = optimizer.optimize(
        num_iterations=params['ni'], 
        learning_rate=params['lr'], 
        update_callback=vis_callback
    )
    
    # 将优化结果嵌入到整个SLM图案中
    final_phase = np.zeros(shape)
    phase_wrapped = torch.remainder(optimized_phase, 2 * np.pi).cpu().numpy()
    
    # 方法1：如果优化器返回的是N×N的正方形，需要调整尺寸
    if phase_wrapped.shape == (N, N):
        # 将N×N的结果放置到ROI的中心区域
        roi_center_y = (roi_top + roi_bottom) // 2
        roi_center_x = (roi_left + roi_right) // 2
        
        opt_top = roi_center_y - N // 2
        opt_bottom = opt_top + N
        opt_left = roi_center_x - N // 2
        opt_right = opt_left + N
        
        # 确保边界在有效范围内
        opt_top = max(opt_top, 0)
        opt_bottom = min(opt_bottom, shape[0])
        opt_left = max(opt_left, 0)
        opt_right = min(opt_right, shape[1])
        
        final_phase[opt_top:opt_bottom, opt_left:opt_right] = phase_wrapped[:opt_bottom-opt_top, :opt_right-opt_left]
        
        # 更新mask以反映实际的优化区域
        roi_mask = np.zeros_like(roi_mask)
        roi_mask[opt_top:opt_bottom, opt_left:opt_right] = True
    
    # 方法2：如果你的PhaseOptimizer支持矩形输出，直接赋值
    # elif phase_wrapped.size == np.sum(roi_mask):
    #     final_phase[roi_mask] = phase_wrapped.flatten()
    
    # 结合背景
    checkerboard = create_checkerboard(shape)
    combined_phase = np.where(roi_mask, final_phase, checkerboard)
    
    phi = np.uint8(combined_phase / (2 * np.pi) * params['two_pi_value'])
    
    return phi, optimizer