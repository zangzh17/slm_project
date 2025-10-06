"""
配置文件：存放物理常量、模拟参数和UI默认值。
"""

# 物理常量
WAVELENGTH = 532e-9         # 波长 (532 nm)
PIXEL_SIZE = 9.2e-6         # SLM像素尺寸 (9.2 μm)
FOCAL_OBJECTIVE = 10e-3      # 物镜焦距
NA_OBJECTIVE = 0.7          # 物镜数值孔径

# SLM/ROI 默认参数
SLM_SHAPE = (1152, 1920)     # SLM 分辨率
SLM_SDK_PATH = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus"
SLM_LUT_PATH = "C:\\Program Files\\Meadowlark Optics\\SDK\\slm5691_at635.LUT"

# ROI默认设置 (方形)
ROI_DEFAULT_SIZE = 1152      # 默认方形ROI尺寸

# 通用UI默认值 (both Optimized and Fresnel modes共享)
COMMON_DEFAULTS = {
    'focal_length_coarse': 50,   # mm
    'focal_length_fine': 0,       # mm (fine adjustment)
    'M': 5,                    # 微透镜阵列行列数
    'two_pi_value': 220,          # Gray value for 2π phase shift
}

# Optimized mode特有默认值
OPTIMIZED_DEFAULTS = {
    'overlap_ratio': 0.3,
    'dof_correction': 1.0,
    'airy_correction': 1.0,
    'center_blend': 0.0,
    'interleaving' : 'coarse1',
    'mask_count': 2,
    'psf_energy_level': 1.0,
    'z_factor': 0.2,
    'lr': 0.05,                  # learning rate
    'ni': 500,                    # number of iterations
}

# 优化器内部参数 (non-UI)
OPTIMIZER = {
    'weights' : [1.0, 1.0]
}

# 预定义的ROI尺寸选项
ROI_SIZE_OPTIONS = {
    '100%': 1152,
    '95%': 1094,
    '90%': 1036,
    '80%': 922,
    '70%': 806,
    '50%': 576,
}