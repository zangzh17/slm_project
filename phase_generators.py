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
    focal_length = params['focal_length']
    N = params['N']
    M = params['M']
    print(N,M)
    y, x = np.indices((N,N))
    phase = np.zeros((N,N))
    for r in range(M):
        for c in range(M):
            # 通过将乘法放在整除前，可以更精确地分配像素
            y_start = (r * N) // M
            y_end = ((r + 1) * N) // M
            
            x_start = (c * N) // M 
            x_end = ((c + 1) * N) // M

            # 使用起止点计算中心
            center_y = (y_start + y_end) / 2
            center_x = (x_start + x_end) / 2
            
            # 创建slice对象，这里的变量现在保证是整数
            region = (slice(y_start, y_end), slice(x_start, x_end))
            
            # 后续计算保持不变
            x_dist = (x[region] - center_x) * config.PIXEL_SIZE
            y_dist = (y[region] - center_y) * config.PIXEL_SIZE
            r_squared = x_dist**2 + y_dist**2
            
            f_squared = focal_length**2
            phase_calc = (2 * np.pi / config.WAVELENGTH) * \
                        (focal_length - np.sqrt(f_squared + r_squared))
            
            phase[region] = phase_calc
    phase = phase % (2 * np.pi)
    
    # 将优化结果嵌入到整个SLM图案中
    final_phase = np.zeros(shape)
    final_phase[params['roi_mask']] = phase.flatten()
    
    # 结合背景
    checkerboard = create_checkerboard(shape)
    combined_phase = np.where(params['roi_mask'], final_phase, checkerboard)
    
    # 转换为 8-bit
    phi = np.uint8(combined_phase / (2 * np.pi) * params['two_pi_value'])
    
    info = {
        'phi': phi, 'focal_length': params['focal_length'], 'rows': M, 'cols': M,
        'lens_width': N//M, 'lens_height': N//M,
        'two_pi_value': params['two_pi_value'], 
        'roi_rect': params['roi_rect']
    }
    return phi, info


def generate_optimized_pattern(params: dict, vis_callback=None) -> tuple[np.ndarray, object, dict]:
    """
    通过优化算法生成微透镜阵列的相位图。

    Args:
        params (dict): 包含所有UI输入参数的字典。
        vis_callback (function): 用于实时可视化的回调函数。

    Returns:
        tuple[np.ndarray, object]: (最终的8位相位图, 优化器对象本身用于后续分析)
    """
    shape = params['shape']
    focal_length = params['focal_length']
    if params['lens_type']: 
        focal_length = -focal_length

    N = params['N']
    M = params['M']
    
    optimizer = PhaseOptimizer(
        N=N, 
        pixel_size=config.PIXEL_SIZE, 
        wavelength=config.WAVELENGTH, 
        focal_length=focal_length,
        psf_energy_level=params['psf_energy_level'], 
        dof_correction=params['dof_correction'],
        airy_correction=params['airy_correction'], 
        M=M,
        aperture_overlap_ratio=params['overlap_ratio'],
        mask_count=int(params['mask_count']),
        center_blend=params['center_blend'],
        interleaving=params['interleaving'],
    )
    
    optimized_phase = optimizer.optimize(
        num_iterations=params['ni'], 
        learning_rate=params['lr'], 
        update_callback=vis_callback
    )
    
    # 将优化结果嵌入到整个SLM图案中
    final_phase = np.zeros(shape)
    phase_wrapped = torch.remainder(optimized_phase, 2 * np.pi).cpu().numpy()
    final_phase[params['roi_mask']] = phase_wrapped.flatten()
    
    # 结合背景
    checkerboard = create_checkerboard(shape)
    combined_phase = np.where(params['roi_mask'], final_phase, checkerboard)
    
    phi = np.uint8(combined_phase / (2 * np.pi) * params['two_pi_value'])

    lens_height = N // M
    lens_width = N // M
    info = {
        'phi': phi, 'focal_length': focal_length, 'rows': M, 'cols': M,
        'roi_width': N, 'roi_height': N, 'lens_width': lens_width, 'lens_height': lens_height,
        'two_pi_value': params['two_pi_value'], 'roi_rect': params['roi_rect']
    }
    
    return phi, optimizer, info