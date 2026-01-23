# SLM 相位优化项目

## 项目概述

空间光调制器 (SLM) 相位优化代码，用于超分辨和超景深光场（多孔径）成像。

## 项目环境

项目采用uv创建的，位于当前目录的.venv python环境.

## 工作流

基本工作流在目录中的几个jupyter notebook中。

1. 用户通过optimize_multiple.ipynb 优化一系列recipe，是最关键的一步
2. 用户通过capture_PSF.ipynb 读取优化好的recipe，并进行硬件控制和PSF采集
3. （可选）用户通过test_slm.ipynb 做快速的测试

### 优化工作流

optimize_multiple.ipynb 中，其中包含一个基于GUI的优化job生成工具，由phase_optimizer_gui创建。

核心优化逻辑在 `phase_generators.py` 的 `PhaseGenerator` 类中。

#### 优化目标

为 SLM 生成相位图，使其等效于 M×M 的微透镜阵列 (Micro-lens Array)。优化目标：
- 焦平面产生均匀的 M×M PSF 阵列
- 各子孔径聚焦效率高且均匀
- 在指定深度范围内保持聚焦（扩展景深）
- 离焦面上各子图像不串扰

#### 两种生成模式

1. **Fresnel 模式** (`mode='fresnel'`)
   - 直接生成理想菲涅尔透镜阵列相位，无优化
   - 每个子透镜独立计算球面波相位：`φ = k(f - √(f² + r²))`

2. **Optimized 模式** (`mode='optimized'`)
   - 基于目标 PSF 模板进行梯度下降优化
   - 使用 PyTorch 自动微分 + Adam 优化器

#### 关键参数

| 参数 | 说明 |
|------|------|
| `M` | 微透镜阵列大小 (M×M)，如 M=5 表示 5×5=25 个子孔径 |
| `focal_length` | 焦距 (m) |
| `overlap_ratio` | 相邻子孔径重叠比例 (0~1)，增大可扩展等效口径 |
| `airy_correction` | Airy 斑大小修正因子 |
| `depth_in_focus` | 需保持聚焦的深度列表，以 DOF 为单位，如 `[-0.5, 0.5]` |
| `depth_out_focus` | 离焦约束面的 z/f 比例列表 |
| `weights` | 各损失项权重字典 |

#### 损失函数

优化模式使用多目标损失函数 (`compute_loss`)：

1. **`mse`**: 焦平面 PSF 与高斯目标的 MSE
2. **`depth_in_focus`**: 多深度平面 PSF 的 MSE（扩展景深约束）
3. **`depth_out_focus`**: 离焦面质心位置约束（防止子图像串扰）
4. **`eff_mean`**: 平均聚焦效率（取负，越高越好）
5. **`eff_std`**: 效率均匀性（标准差越小越好）
6. **`masked`**: 可选的 mask 入射光 PSF 损失

#### overlap_ratio 的作用

`overlap_ratio` 控制相邻微透镜的重叠程度：
- `overlap_ratio = 0`: 无重叠，等效于标准 Fresnel MLA
- `overlap_ratio > 0`: 子孔径扩展，等效口径增大

设 α = 扩展因子，则：
- 等效 Airy 斑半径缩小为 1/α
- 等效景深缩小为 1/α²

这是分辨率与景深的 trade-off

## 目录结构

```
slm_project/
├── 核心模块
│   ├── phase_generators.py    # 相位生成与优化核心
│   ├── hardware.py            # 硬件控制接口 (SLM、平移台、相机)
│   ├── batch_processor.py     # 批量任务处理
│   ├── visualization.py       # 可视化工具
│   ├── optics_utils.py        # 光学计算工具函数
│   ├── wave_propagation.py    # 光传播算法 (ASM/RSC)
│   ├── meadowlark.py          # Meadowlark SLM 驱动
│   ├── slm.py                 # SLM 基类
│   ├── phase_optimizer_gui.py # 优化器 GUI
│   ├── npy_file_selector.py   # 文件选择 GUI
│   ├── nas_mapper.py          # NAS 网络映射
│   └── config.py              # 全局配置
│
├── Notebook 工作流
│   ├── optimize_multiple.ipynb  # 相位优化主流程
│   ├── capture_PSF.ipynb        # PSF 采集主流程
│   └── test_slm.ipynb           # SLM 简短测试
│
├── 配置文件
│   ├── config/base.json         # 基础配置
│   └── pyproject.toml           # 项目依赖
│
└── 数据
    ├── output/                  # 优化结果输出
    ├── atf.mat                  # ATF 数据
    └── pupil.mat                # 光瞳函数数据
```

## 核心模块说明

### phase_generators.py
- `PhaseGenerator` 类：相位生成与优化的核心
- 支持 Fresnel 相位生成和多目标优化
- 损失函数：MSE、深度聚焦、效率均匀性

### hardware.py
- `RemoteHardwareManager`：统一硬件接口
- 支持远程 SLM 控制、Z 轴平移台、旋转台、相机触发

### batch_processor.py
- `process_jobs()`：批量执行优化任务
- `JobBrowserGUI`：结果浏览与可视化

### wave_propagation.py
- ASM (角谱法)：小传播距离
- RSC (瑞利-索末菲卷积)：大传播距离

## 主要工作流

### 优化流程 (optimize_multiple.ipynb)
1. GUI 参数配置
2. 批量优化执行
3. 结果浏览与 SLM 上传

### 采集流程 (capture_PSF.ipynb)
1. 硬件连接
2. 选择相位图文件
3. Z 轴自动扫描采集
4. 数据整理

## 技术栈

- **计算框架**: PyTorch (GPU 加速)
- **科学计算**: NumPy, SciPy
- **可视化**: Matplotlib
- **远程通信**: RPyC
- **硬件接口**: pythonnet (Meadowlark SDK)
- **环境**: Python >= 3.11, Jupyter Notebook

## 关键参数

- 波长: 515 nm
- SLM 像素: 9.2 μm
- SLM 分辨率: 1152 × 1920
- 焦距: 73.9 mm (可配置)
- 微透镜阵列: M3/M5/M7/M9

## 输出文件格式

- `.npy`: 8-bit 相位图
- `.json`: 优化参数
- `.pkl`: 优化器对象 (可恢复)
