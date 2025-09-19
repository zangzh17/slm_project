# hardware.py

"""
Handle hardware communication with Spatial Light Modulator (SLM).
Automatically switches to simulation mode if SLM is not connected or SDK is not found.
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
                print(f"✅ SLM connected successfully, resolution: {self.shape}")
            except (ImportError, RuntimeError) as e:
                print(f"⚠️ Warning: Failed to connect to SLM. Running in simulation mode only.")
                print(f"   Error message: {e}")
                print(f"   Using default resolution: {self.shape}")

    def upload(self, phase_pattern: np.ndarray):
        """
        Upload 8-bit phase pattern to SLM.

        Args:
            phase_pattern (np.ndarray): Phase pattern in uint8 format.
        """
        if self.is_connected:
            try:
                self.slm.set_phase(phase_pattern)
                print("Phase pattern uploaded to SLM.")
            except Exception as e:
                print(f"❌ Error: Failed to upload phase pattern to SLM: {e}")
        else:
            print("(Simulation mode): Phase pattern would be uploaded if SLM was connected.")