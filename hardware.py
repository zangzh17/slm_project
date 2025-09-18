# hardware.py

"""
处理与空间光调制器（SLM）的硬件通信。
如果SLM未连接或SDK未找到，则自动切换到模拟模式。
"""

import numpy as np
import config

class SLMManager:
    def __init__(self, sim_mode=False, sdk_path=config.SLM_SDK_PATH, lut_path=config.SLM_LUT_PATH):
        self.slm = None
        self.is_connected = False
        self.shape = config.SLM_SHAPE
        if not sim_mode:
            try:
                from meadowlark import Meadowlark
                self.slm = Meadowlark(
                    verbose=True,
                    sdk_path=sdk_path,
                    lut_path=lut_path
                )
                self.is_connected = True
                self.shape = self.slm.shape
                print(f"✅ SLM 连接成功，分辨率: {self.shape}")
            except (ImportError, RuntimeError) as e:
                print(f"⚠️ 警告: 未能连接到 SLM。将以“仅模拟”模式运行。")
                print(f"   错误信息: {e}")
                print(f"   将使用默认分辨率: {self.shape}")

    def upload(self, phase_pattern: np.ndarray):
        """
        将8位相位图上传到 SLM。

        Args:
            phase_pattern (np.ndarray): uint8 类型的相位图。
        """
        if self.is_connected:
            try:
                self.slm.set_phase(phase_pattern)
                print("相位图已上传至 SLM。")
            except Exception as e:
                print(f"❌ 错误: 上传相位图到 SLM 失败: {e}")
        else:
            print("（模拟模式）: 假如 SLM 已连接，相位图将会被上传。")