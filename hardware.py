# hardware.py

"""
Handle hardware communication with Spatial Light Modulator (SLM).
Automatically switches to simulation mode if SLM is not connected or SDK is not found.
"""

import numpy as np
import config

class SLMManager:
    def __init__(self, sim_mode=False, 
                 sdk_path=config.SLM_SDK_PATH, 
                 lut_path=config.SLM_LUT_PATH):
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
        else:
            print("🎬 Simulation Mode Enabled")

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

import rpyc

class RemoteSLMManager:
    def __init__(self, host="127.0.0.1", port=18861):
        print(f"Connecting to local hardware service at {host}:{port} ...")
        conn_config = {'allow_public_attrs': True}
        self.conn = rpyc.connect(host, port, config=conn_config)
        self.service = self.conn.root
        self.shape = config.SLM_SHAPE
        print("✅ Connected to local SLM")

    def upload(self, phase_pattern: np.ndarray):
        """
        This function works exactly the same as the local slm_manager.upload
        """
        # Optimization: don't directly pass numpy objects, RPyC is slow with large arrays.
        # Trick: convert to bytes for transmission, much faster.
        data_bytes = phase_pattern.tobytes()
        shape = phase_pattern.shape
        dtype_str = str(phase_pattern.dtype)
        
        # Call the local exposed_upload_frame function
        self.service.exposed_upload_frame(data_bytes, shape, dtype_str)
        print("🚀 Data sent to local hardware")

    def close(self):
        self.conn.close()


class RemoteStageManager:
    """Remote control for Thorlabs Stage via RPyC"""
    
    # Stage types
    ROTATION = 1  # PRM1-Z8
    Z_AXIS = 2    # Z825B
    
    # Stage type name mapping
    STAGE_NAMES = {
        1: "PRM1-Z8 (Rotation)",
        2: "Z825B (Z-Axis)",
    }
    
    def __init__(self, host="127.0.0.1", port=18861, stage_type=2):
        """
        Initialize remote stage connection.
        Stage is already connected on local server startup.
        
        Args:
            host: Local server hostname
            port: Local server port
            stage_type: 1 = PRM1-Z8 (Rotation), 2 = Z825B (Z-Axis)
        """
        print(f"Connecting to local hardware service at {host}:{port} ...")
        conn_config = {'allow_public_attrs': True}
        self.conn = rpyc.connect(host, port, config=conn_config)
        self.service = self.conn.root
        self.stage_type = stage_type
        stage_name = self.STAGE_NAMES.get(stage_type, f"Unknown({stage_type})")
        print(f"✅ Connected to Stage {stage_type}: {stage_name}")

    def home(self, timeout=60000):
        """Home the stage"""
        return self.service.exposed_stage_home(self.stage_type, timeout)

    def get_position(self):
        """Get current position"""
        return self.service.exposed_stage_get_position(self.stage_type)

    def move_to(self, position, timeout=60000):
        """Move to absolute position"""
        return self.service.exposed_stage_move_to(position, self.stage_type, timeout)

    def is_connected(self):
        """Check if stage is connected"""
        return self.service.exposed_stage_is_connected(self.stage_type)

    def close(self):
        """Close RPyC connection"""
        self.conn.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RemoteAHKManager:
    """Remote control for AutoHotkey scripts via RPyC"""
    
    def __init__(self, host="127.0.0.1", port=18861):
        """
        Initialize remote AHK connection.
        
        Args:
            host: Local server hostname
            port: Local server port
        """
        print(f"Connecting to local hardware service at {host}:{port} ...")
        conn_config = {'allow_public_attrs': True}
        self.conn = rpyc.connect(host, port, config=conn_config)
        self.service = self.conn.root
        self.click_position = None  # Cached click position (x, y)
        print("✅ Connected to local AHK service")

    def capture_position(self):
        """
        Run capture_position.ahk on local machine and retrieve the captured coordinates.
        
        Returns:
            tuple: (x, y) coordinates, or None if capture failed
        """
        result = self.service.exposed_ahk_capture_position()
        if result is not None:
            self.click_position = tuple(result)
            print(f"📍 Captured position: {self.click_position}")
        return self.click_position

    def click_at(self, x=None, y=None):
        """
        Execute click_at.ahk on local machine at specified coordinates.
        If coordinates not provided, uses the last captured position.
        
        Args:
            x: X coordinate (optional, uses cached if not provided)
            y: Y coordinate (optional, uses cached if not provided)
            
        Returns:
            bool: True if click was successful
        """
        if x is None or y is None:
            if self.click_position is None:
                print("❌ No position captured. Call capture_position() first or provide coordinates.")
                return False
            x, y = self.click_position
        
        success = self.service.exposed_ahk_click_at(int(x), int(y))
        if success:
            print(f"🖱️ Clicked at ({x}, {y})")
        return success

    def trigger_camera(self):
        """
        Convenience method: trigger camera using captured position.
        Equivalent to click_at() with cached coordinates.
        
        Returns:
            bool: True if trigger was successful
        """
        return self.click_at()

    def get_cached_position(self):
        """Get the currently cached click position"""
        return self.click_position

    def set_position(self, x, y):
        """Manually set the click position without capturing"""
        self.click_position = (int(x), int(y))
        print(f"📍 Position set to: {self.click_position}")

    def close(self):
        """Close RPyC connection"""
        self.conn.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RemoteHardwareManager:
    """
    Unified remote hardware manager that provides access to SLM, Stage, and AHK functionality
    through a single connection.
    """
    
    # Stage types (for convenience)
    STAGE_ROTATION = 1  # PRM1-Z8
    STAGE_Z_AXIS = 2    # Z825B
    
    # Stage type name mapping
    STAGE_NAMES = {
        1: "PRM1-Z8 (Rotation)",
        2: "Z825B (Z-Axis)",
    }
    
    def __init__(self, host="127.0.0.1", port=18861):
        """
        Initialize unified remote hardware connection.
        
        Args:
            host: Local server hostname
            port: Local server port
        """
        print(f"Connecting to local hardware service at {host}:{port} ...")
        conn_config = {'allow_public_attrs': True}
        self.conn = rpyc.connect(host, port, config=conn_config)
        self.service = self.conn.root
        self.shape = config.SLM_SHAPE
        self.click_position = None
        print("✅ Connected to unified hardware service")
        
        # Print connected stages status
        self._print_stages_status()

    def _print_stages_status(self):
        """Print connection status of all stages"""
        print("   Stage status:")
        for stage_type, stage_name in self.STAGE_NAMES.items():
            connected = self.stage_is_connected(stage_type)
            status = "✅ Connected" if connected else "❌ Not connected"
            print(f"     - Stage {stage_type} ({stage_name}): {status}")

    # ============== SLM Functions ==============
    def upload_slm(self, phase_pattern: np.ndarray):
        """Upload phase pattern to SLM"""
        data_bytes = phase_pattern.tobytes()
        shape = phase_pattern.shape
        dtype_str = str(phase_pattern.dtype)
        self.service.exposed_upload_frame(data_bytes, shape, dtype_str)
        print("🚀 Phase pattern sent to SLM")

    # ============== Stage Functions ==============
    def stage_home(self, stage_type=2, timeout=60000):
        """Home the stage"""
        return self.service.exposed_stage_home(stage_type, timeout)

    def stage_get_position(self, stage_type=2):
        """Get stage position"""
        return self.service.exposed_stage_get_position(stage_type)

    def stage_move_to(self, position, stage_type=2, timeout=60000):
        """Move stage to position"""
        return self.service.exposed_stage_move_to(position, stage_type, timeout)

    def stage_is_connected(self, stage_type=2):
        """Check if stage is connected"""
        return self.service.exposed_stage_is_connected(stage_type)

    # ============== Convenience Stage Methods ==============
    def rotation_home(self, timeout=60000):
        """Home the rotation stage (PRM1-Z8)"""
        return self.stage_home(self.STAGE_ROTATION, timeout)

    def rotation_get_position(self):
        """Get rotation stage position"""
        return self.stage_get_position(self.STAGE_ROTATION)

    def rotation_move_to(self, position, timeout=60000):
        """Move rotation stage to position (degrees)"""
        return self.stage_move_to(position, self.STAGE_ROTATION, timeout)

    def z_home(self, timeout=60000):
        """Home the Z-axis stage (Z825B)"""
        return self.stage_home(self.STAGE_Z_AXIS, timeout)

    def z_get_position(self):
        """Get Z-axis stage position"""
        return self.stage_get_position(self.STAGE_Z_AXIS)

    def z_move_to(self, position, timeout=60000):
        """Move Z-axis stage to position (mm)"""
        return self.stage_move_to(position, self.STAGE_Z_AXIS, timeout)

    # ============== AHK Functions ==============
    def capture_position(self):
        """Capture click position via AHK"""
        result = self.service.exposed_ahk_capture_position()
        if result is not None:
            self.click_position = tuple(result)
            print(f"📍 Captured position: {self.click_position}")
        return self.click_position

    def click_at(self, x=None, y=None):
        """Click at position via AHK"""
        if x is None or y is None:
            if self.click_position is None:
                print("❌ No position captured. Call capture_position() first.")
                return False
            x, y = self.click_position
        success = self.service.exposed_ahk_click_at(int(x), int(y))
        if success:
            print(f"🖱️ Clicked at ({x}, {y})")
        return success

    def trigger_camera(self):
        """Trigger camera using captured position"""
        return self.click_at()

    def close(self):
        """Close RPyC connection"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()