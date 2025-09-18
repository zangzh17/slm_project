# config.py

"""
配置文件：存放物理常量、模拟参数和UI默认值。
"""

# 物理常量
WAVELENGTH = 532e-9         # 波长 (532 nm)
PIXEL_SIZE = 9.2e-6         # SLM像素尺寸 (9.2 μm)
FOCAL_OBJECTIVE = 10e-3      # 物镜焦距 (9 mm)
NA_OBJECTIVE = 0.7          # 物镜数值孔径

# SLM/ROI 默认参数
SLM_SHAPE = (1152, 1920)     # SLM 分辨率
ROI_DEFAULT_WIDTH = 1024
ROI_DEFAULT_HEIGHT = 1024

# 优化器默认参数
DEFAULT_PARAMS = {
    'focal_length': 50e-3,
    'rows': 5,
    'cols': 5,
    'overlap_ratio': 0.1,
    'dof_factor': 2.0,
    'size_factor': 1.0,
    'psf_energy_level': 1.0,
    'z_factor': 0.2,
    'learning_rate': 0.05,
    'num_iterations': 500,
}

# UI 默认值
UI_DEFAULTS = {
    'focal_length_coarse': 50,
    'focal_length_fine': 0,
    'lens_type': False,
    'rows': 5,
    'cols': 5,
    'roi_width': 1152,
    'roi_height': 1152,
    'overlap_ratio': 0.1,
    'dof_factor': 2.0,
    'size_factor': 1.0,
    'psf_energy_level': 1.0,
    'phase_range': 255.0,
    'z_factor': 0.2,
    'lr': 0.05,
    'ni': 500,
    'mask_box': False
}

# SLM/ROI 默认参数
SLM_SHAPE = (1152, 1920)
SLM_SDK_PATH = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus"
SLM_LUT_PATH = "C:\\Program Files\\Meadowlark Optics\\SDK\\slm5691_at635.LUT"

# ... (原有优化器默认参数) ...

# 菲涅尔透镜 UI 默认值
FRESNEL_DEFAULTS = {
    'focal_length_coarse': 50,
    'focal_length_fine': 0,
    'lens_type': False,
    'rows': 3,
    'cols': 3,
    'roi_width': 1152,
    'roi_height': 1152,
    'angle_x_mrad': 0,
    'angle_y_mrad': 0,
    'two_pi_value': 220,
}