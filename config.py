"""
配置文件：存放物理常量、模拟参数和UI默认值。
"""

# Physical constants
WAVELENGTH = 515e-9         # 波长 (515 nm)
PIXEL_SIZE = 9.2e-6         # SLM像素尺寸 (9.2 μm)
FOCAL_OBJECTIVE = 10e-3      # 物镜焦距
NA_OBJECTIVE = 0.7          # 物镜数值孔径

# SLM/ROI default params
SLM_SHAPE = (1152, 1920)     # SLM default resolution in sim mode
SLM_SDK_PATH = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus"
SLM_LUT_PATH = "C:\\Program Files\\Meadowlark Optics\\SDK\\slm5691_at635.LUT"

# System params
TEXT_WIDTH_WRAP = 100       # Auto-wrap number of words for loss value print
ROI_DEFAULT_SIZE = 1152      # Square ROI Size in pix

# Predefined params (shares for both Optimized and Fresnel modes)
COMMON_DEFAULTS = {
    'focal_length': 60.0e-3, 
    'N': 900,                 # For optimized ROI size
    'output_size': 900,      # For optimized output FOV size
    'roi_center_y': 576,     # Center of ROI
    'roi_center_x': 960,     # Center of ROI
    'M': 7,                    # Microlens array: M*M
    'two_pi_value': 220,          # Gray value for 2π phase shift
}

# Only applied for
# Optimized mode
OPTIMIZED_DEFAULTS = {
    'overlap_ratio': 0.3,
    'dof_correction': 1.0,
    'airy_correction': 1.0,
    'center_blend': 0.0,       # 0：焦点采用原生均匀分布；1：考虑重叠后的中心
    'interleaving' : 'coarse1',
    'mask_count': 2,
    'psf_energy_level': 1.0,
    'z_factor': 0.2,
    'lr': 0.05,                  # learning rate
    'ni': 500,                    # number of iterations
    
    'depth_in_focus': [-0.5, 0.5],
    'depth_out_focus': None,
    # 'depth_out_focus': None,
    'focusing_eff_correction': 1.0, # airy correction for focusing efficiency loss computation
    'masked_airy_correction': 1.0, # airy correction for only masked loss computation
    'show_iters': 50, # show iteration info every show_iters
    'weights': {
        'mse': 1.0,
        'depth_in_focus': 1.0,
        'depth_out_focus':0.0,
        'masked': 1.0,
        'eff_mean': 20.0,
        'eff_std': 50.0
    }
}
