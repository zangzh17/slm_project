# SLM Phase Optimization Project Documentation

## 项目概述

本项目是一个用于**空间光调制器 (SLM)** 相位图生成与优化的系统，主要应用于超分辨率成像和光场调控。系统支持生成微透镜阵列的相位图案，并通过优化算法提升成像质量。

## 核心功能

1. **相位图生成**: 支持菲涅尔透镜阵列和优化后的相位图案
2. **硬件控制**: 远程控制 SLM、电动平移台、旋转台
3. **自动化采集**: 自动化 Z-scan PSF 采集流程
4. **批量优化**: 批量生成和管理多组优化任务

---

## 主代码文件说明

### 1. `optimize_multiple.ipynb` - 相位图优化与生成

这是相位图生成和优化的主要工作流程。

#### 工作流程

```
[加载配置] → [GUI 创建优化任务] → [批量执行优化] → [浏览/可视化结果]
```

#### Cell 说明

**Cell 1: Generate Job List (生成任务列表)**
```python
from optics_utils import load_dict_from_json
from phase_optimizer_gui import create_optimizer_gui

# 加载基础配置
params = load_dict_from_json(os.path.join(path_json, filename_base))
gui = create_optimizer_gui(default_params=params)
```

功能:
- 从 JSON 配置文件加载光学参数
- 创建交互式 GUI 界面用于配置优化任务
- GUI 支持设置: M (阵列大小)、焦距、重叠比例、Airy 校正、深度范围等

**Cell 2: Run optimization Jobs (运行优化任务)**
```python
from batch_processor import process_jobs
from hardware import RemoteSLMManager, SLMManager

# 选择运行模式
remote_mode = True
sim_mode = True

if sim_mode:
    slm_manager = SLMManager(sim_mode=True)
elif remote_mode:
    slm_manager = RemoteSLMManager()

results = process_jobs(gui, slm_manager=slm_manager, device=device)
```

功能:
- 批量执行 GUI 中创建的所有优化任务
- 支持仿真模式 (`sim_mode=True`) 和远程 SLM 模式
- 输出文件保存到 `./output/` 目录

输出文件:
- `{job_title}.npy`: 8-bit 相位图
- `{job_title}.json`: 优化参数
- `{job_title}_optimizer.pkl`: 完整优化器对象 (用于可视化)

**Cell 3: Check results (查看结果)**
```python
from batch_processor import browse_jobs
browser = browse_jobs(output_dir='./output')
```

功能:
- 浏览已保存的优化任务
- 可视化相位图、PSF 对比、能量分布等
- 可直接上传相位图到 SLM

---

### 2. `capture_PSF.ipynb` - PSF 采集与扫描

用于采集点扩散函数 (PSF) 的自动化工作流程。

#### 工作流程

```
[硬件连接] → [选择 .npy 相位图] → [Z-Scan 采集] → [整理数据]
```

#### Cell 说明

**Cell 1: Stage Connection and Manual Control (硬件连接)**
```python
from hardware import RemoteHardwareManager

hw = RemoteHardwareManager(host="127.0.0.1", port=18861)

# 获取位置
current_pos = hw.stage_get_position()
angle = hw.rotation_get_position()

# 上传测试相位图
from phase_generators import PhaseGenerator
Optimizer = PhaseGenerator(params)
Optimizer.generate(mode='fresnel')
hw.upload_slm(Optimizer.update_phase_8bit())

# 旋转台角度预设 (不同 M 值对应不同角度)
M_angle = {'M3': 282.75, 'M5': 268.0, 'M7': 261.6, 'M9': 258.5}
```

功能:
- 连接远程硬件服务 (SLM + 电动台 + 旋转台)
- 测试 SLM 连接和相位图上传
- 设置旋转台到正确角度

**Cell 2: NPY File Selection GUI (选择相位图文件)**
```python
from npy_file_selector import select_npy_files
file_selector_widget = select_npy_files(output_dir="./output")
```

功能:
- 扫描 `./output` 目录下的 `.npy` 相位图文件
- 提供多选 GUI 界面选择要扫描的文件
- 自动解析文件名中的 M 值 (如 `M5`, `M7`)

**Cell 3: Z-Scan Acquisition (Z 扫描采集)**
```python
# 关键参数
save_dir = r"Z:\\SLM_super_resolution\\data\\for_auto_scan\\"
z_focal_plane = 11.805  # 焦平面位置 (mm)
num_steps = 81          # 扫描步数
z_range = 0.4           # 扫描范围 (mm)

# 执行扫描
for npy_name in selected_npy_files:
    # 上传相位图到 SLM
    hw.upload_slm(pattern)

    # 切换旋转台角度
    hw.rotation_move_to(target_angle)

    # Z 扫描循环
    for z_pos in z_positions:
        hw.stage_move_to(z_pos)
        hw.click_at()  # 触发相机拍照
```

功能:
- 自动化 Z 轴扫描采集
- 对每个相位图案执行完整 Z-scan
- 自动切换旋转台角度匹配不同 M 值
- 通过 AutoHotkey 脚本触发相机采集

**Cell 4: Organize Results (整理结果)**
```python
user_prefix = "PSF_4um"

# 自动整理 TIFF 文件到对应文件夹
# 重命名格式: {user_prefix}_frame{n}_z{z_pos}mm.tiff
# 生成 scan_info.json 记录扫描参数
```

功能:
- 按相位图名称创建子文件夹
- 重命名 TIFF 文件包含 Z 位置信息
- 生成扫描信息 JSON 文件

---

## 核心模块说明

### `phase_generators.py` - 相位生成器

主类: `PhaseGenerator`

```python
class PhaseGenerator:
    def __init__(self, params, device=torch.device('cuda'), mode='fresnel'):
        # 光学参数: 焦距、波长、像素尺寸
        # 阵列参数: M (阵列大小)、N (ROI 尺寸)、重叠比例
        # 优化参数: 学习率、迭代次数、损失权重
```

关键方法:
- `generate(mode='fresnel'|'optimized')`: 生成相位图
- `generate_fresnel_phase()`: 生成菲涅尔透镜阵列相位
- `forward(z)`: 光传播前向模拟
- `compute_loss()`: 计算多目标损失函数
- `update_phase_8bit()`: 转换为 8-bit SLM 格式

损失函数组成:
- `mse`: 焦平面 PSF 匹配
- `depth_in_focus`: 多深度平面 PSF 匹配
- `depth_out_focus`: 离焦面质心约束
- `eff_mean/eff_std`: 聚焦效率均匀性

### `hardware.py` - 硬件控制

**SLMManager**: 本地 SLM 控制
```python
slm = SLMManager(sim_mode=False)
slm.upload(phase_pattern)
```

**RemoteHardwareManager**: 统一远程硬件控制
```python
hw = RemoteHardwareManager(host="127.0.0.1", port=18861)

# SLM
hw.upload_slm(phase_8bit)

# Z 轴平移台 (Z825B)
hw.stage_move_to(position_mm)
hw.stage_get_position()

# 旋转台 (PRM1-Z8)
hw.rotation_move_to(angle_deg)
hw.rotation_get_position()

# 相机触发 (AutoHotkey)
hw.capture_position()  # 捕获点击位置
hw.click_at()          # 触发相机
```

### `batch_processor.py` - 批量处理

```python
from batch_processor import process_jobs, browse_jobs

# 批量执行优化任务
results = process_jobs(gui, slm_manager, device)

# 浏览和可视化结果
browser = browse_jobs(output_dir='./output')
optimizer = browser.get_current_optimizer()
```

---

## 配置文件说明

### `config/base.json` - 基础光学参数

```json
{
    "focal_length": 0.0739,      // 焦距 (m)
    "N": 850,                    // ROI 尺寸 (像素)
    "output_size": 850,          // 输出尺寸
    "M": 5,                      // 微透镜阵列大小 (5x5)
    "two_pi_value": 210,         // SLM 2π 相位对应灰度值
    "overlap_ratio": 0.3,        // 重叠比例
    "dof_correction": 1.0,       // 景深校正因子
    "airy_correction": 1.0,      // Airy 斑校正因子
    "lr": 0.05,                  // 学习率
    "ni": 500,                   // 迭代次数
    "depth_in_focus": [-0.5, 0.5], // 深度范围 (DOF 单位)
    "weights": {
        "mse": 1.0,
        "depth_in_focus": 1.0,
        "eff_mean": 20.0,
        "eff_std": 50.0
    }
}
```

---

## 典型使用流程

### 1. 生成优化相位图

```
1. 打开 optimize_multiple.ipynb
2. 运行 Cell 1 启动 GUI
3. 在 GUI 中设置参数并添加任务到列表
4. 运行 Cell 2 执行批量优化
5. 运行 Cell 3 浏览和可视化结果
```

### 2. 采集 PSF 数据

```
1. 确保硬件服务已启动
2. 打开 capture_PSF.ipynb
3. 运行 Cell 1 连接硬件并测试
4. 运行 Cell 2 选择要采集的相位图
5. 运行 Cell 3 执行自动化 Z-scan
6. 运行 Cell 4 整理和保存数据
```

---

## 文件结构

```
slm_project/
├── optimize_multiple.ipynb   # 相位优化主流程
├── capture_PSF.ipynb         # PSF 采集主流程
├── phase_generators.py       # 相位生成器核心类
├── batch_processor.py        # 批量处理模块
├── hardware.py               # 硬件控制模块
├── phase_optimizer_gui.py    # 优化器 GUI
├── optics_utils.py           # 光学工具函数
├── visualization.py          # 可视化工具
├── wave_propagation.py       # 光传播算法
├── config/
│   └── base.json            # 基础配置文件
└── output/                   # 输出目录
    └── {job_title}/
        ├── {job_title}.npy
        ├── {job_title}.json
        └── {job_title}_optimizer.pkl
```

---

## 环境配置

项目使用 **uv** 进行依赖管理，虚拟环境位于 `.venv/` 目录。

### 安装依赖

```bash
# 安装 uv (如果尚未安装)
pip install uv

# 创建虚拟环境并安装依赖
uv sync
```

### 主要依赖

| 包名 | 版本 | 说明 |
|------|------|------|
| torch | >=2.8.0 | GPU 计算 (CUDA 12.8) |
| numpy | >=2.3.3 | 数值计算 |
| matplotlib | >=3.10.7 | 可视化 |
| jupyter | >=1.1.1 | Notebook 环境 |
| rpyc | >=6.0.2 | 远程硬件通信 |
| pythonnet | >=3.0.5 | .NET 接口 (SLM SDK) |
| slmsuite | >=0.3.0 | SLM 工具库 |

### 激活环境

```bash
# Windows
.venv\Scripts\activate

# 或直接使用 uv 运行
uv run jupyter notebook
```

---

## 注意事项

1. **CUDA 支持**: 优化过程需要 GPU 加速，PyTorch 已配置使用 CUDA 12.8
2. **硬件连接**: 采集前确保本地硬件服务 (`rpyc` 服务端口 18861) 已启动
3. **NAS 映射**: 数据保存到 NAS 时需先执行 `quick_map()` 映射网络驱动器
4. **旋转台角度**: 不同 M 值需要不同的旋转台角度，参考 `M_angle` 字典
5. **Python 版本**: 需要 Python >= 3.11
