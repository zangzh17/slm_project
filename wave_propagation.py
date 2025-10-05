"""
光场传播算法模块
提供用于数值光场传播的各种函数，包括角谱法(ASM)和瑞利-索末菲卷积(RSC)。
"""
import torch
import numpy as np

def critical_range(pitch, wavelength, N):
    """
    计算ASM和RSC方法切换的临界距离
    
    参数:
        pitch (float): 空间采样间隔（米）
        wavelength (float): 波长（米）
        N (int): 采样点数
        
    返回:
        float: 临界距离（米）
    """
    zc = 2 * N * pitch**2 / wavelength * np.sqrt(1 - (wavelength / (2 * pitch))**2)
    return zc

def ASM_Kernel(z_list, L, wavelength, N, device=None):
    """
    生成ASM或RSC传播核函数
    
    参数:
        z_list (list or array): 传播距离列表（米）
        L (float): 计算区域尺寸（米）
        wavelength (float): 波长（米）
        N (int): 原始场的采样点数
        device: PyTorch设备
        
    返回:
        tuple: (传播核函数列表, 填充后的尺寸)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pitch = L / N  # 采样间隔
    k = 2 * np.pi / wavelength
    zc = critical_range(pitch, wavelength, N)
    
    # 填充
    pad_x = int(np.ceil(N / 2))
    N_pad = N + 2 * pad_x
    
    # 创建传播核函数列表
    H_list = []
    
    for z in z_list:
        if abs(z) > zc:
            # RSC方法（大传播距离）
            lbx = N_pad * pitch
            x = torch.linspace(-lbx/2, lbx/2-pitch/N_pad, N_pad, device=device)
            y = torch.linspace(-lbx/2, lbx/2-pitch/N_pad, N_pad, device=device)
            X, Y = torch.meshgrid(x, y, indexing='xy')
            
            R = torch.sqrt(X**2 + Y**2 + z**2)
            sign_z = 1 if z > 0 else -1
            
            h = (1/(2*np.pi)) * abs(z)/R * (1/R - sign_z * 1j * k) * torch.exp(sign_z * 1j * k * R) / R
            H = torch.fft.fft2(torch.fft.fftshift(h))
            H = torch.fft.fftshift(H / torch.abs(H[0, 0]))
        else:
            # ASM方法（小传播距离）
            fx = torch.fft.fftfreq(N_pad, d=pitch, device=device)
            fy = torch.fft.fftfreq(N_pad, d=pitch, device=device)
            
            FX, FY = torch.meshgrid(fx, fy, indexing='xy')
            
            # 计算传输函数
            kx = 2 * np.pi * FX
            ky = 2 * np.pi * FY
            k_squared = k**2
            kz_squared = k_squared - (kx**2 + ky**2)
            
            # 处理消逝波
            kz = torch.sqrt(torch.clamp(kz_squared, min=0.0))
            H = torch.exp(1j * kz * z)
            H = torch.fft.fftshift(H)
        
        H_list.append(H)
    
    return H_list, N_pad

def prop(field, H, pad_x, init_wavefront=None, pad_val=0):
    """
    使用ASM或RSC方法传播光场
    
    参数:
        field: 输入复场 [N,N] 或 [B,N,N]
        H: 传播核函数（从ASM_Kernel获得）
        pad_x: 每边填充的像素数
        init_wavefront: 初始波前（可选）
        pad_val: 填充值（默认为0）或"circular"表示循环填充
        
    返回:
        torch.Tensor: 传播后的场
    """
    # 检查输入维度并统一处理
    is_batch = field.dim() == 3
    if not is_batch:
        field = field.unsqueeze(0)  # 添加batch维度 [1, N, N]
    
    # 获取场的尺寸
    B, Ny, Nx = field.shape
    
    # 填充
    if pad_val == "circular":
        # pad只对最后两个维度进行填充
        field_pad = torch.nn.functional.pad(field, (pad_x, pad_x, pad_x, pad_x), mode='circular')
    else:
        field_pad = torch.nn.functional.pad(field, (pad_x, pad_x, pad_x, pad_x), mode='constant', value=pad_val)
    
    # 应用初始波前（如果提供）
    if init_wavefront is not None:
        # 如果init_wavefront是2D的，需要扩展到匹配batch维度
        if init_wavefront.dim() == 2:
            init_wavefront = init_wavefront.unsqueeze(0).expand(B, -1, -1)
        field_pad = field_pad * init_wavefront
    
    # 计算ASM/RSC
    # fft2和ifft2会自动处理batch维度（对最后两个维度进行操作）
    field_pad_FT = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field_pad, dim=(-2, -1))), dim=(-2, -1))
    # H需要广播到batch维度
    if H.dim() == 2:
        H = H.unsqueeze(0)  # [1, Ny_pad, Nx_pad]
    field_pro = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(field_pad_FT * H, dim=(-2, -1))), dim=(-2, -1))
    
    # 移除填充
    field_result = field_pro[:, pad_x:pad_x+Ny, pad_x:pad_x+Nx]
    
    # 如果输入不是batch，移除batch维度
    if not is_batch:
        field_result = field_result.squeeze(0)
    
    return field_result

def propagate_ASM(field, z, L, wavelength, device=None):
    """
    使用ASM或RSC方法（基于传播距离自动选择）传播光场
    
    参数:
        field: 输入复场 [N,N] 或 [B,N,N]
        z: 传播距离（米）
        L: 计算区域尺寸（米）
        wavelength: 波长（米）
        device: PyTorch设备
        
    返回:
        torch.Tensor: 传播后的场
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取场的尺寸
    if field.dim() == 2:
        N = field.shape[0]  # [N, N]
    else:
        N = field.shape[1]  # [B, N, N]
    
    # 生成传播核函数
    H_list, N_pad = ASM_Kernel([z], L, wavelength, N, device)
    H = H_list[0]  # 只使用一个z值的核函数
    
    # 计算填充大小
    pad_x = int((N_pad - N) / 2)
    
    # 执行传播
    return prop(field, H, pad_x)

def create_grid(L, N, device=None):
    """
    创建空间坐标网格
    
    参数:
        L: 计算区域尺寸（米）
        N: 网格点数
        device: PyTorch设备
        
    返回:
        tuple: (X, Y) - 网格坐标张量
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    x = torch.linspace(-L/2, L/2, N, device=device)
    y = torch.linspace(-L/2, L/2, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    return X, Y

def fresnel_lens_phase(X, Y, focal_length, wavelength):
    """
    计算菲涅尔透镜的相位分布: φ(x,y) = (2π/λ) * (f - sqrt(f² + x² + y²))
    
    参数:
        X, Y: 坐标网格
        focal_length: 焦距（米）
        wavelength: 波长（米）
        
    返回:
        torch.Tensor: 相位分布
    """
    return (2 * np.pi / wavelength) * (focal_length - torch.sqrt(focal_length**2 + X**2 + Y**2))