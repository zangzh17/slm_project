"""
PSF Capture GUI for Jupyter Notebook
Provides interactive widgets for hardware setup and automated PSF capture.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import os
import re
import time
import json
import glob
import shutil
from datetime import datetime


# ============================================================
# Default Rotation Stage Angle Presets
# ============================================================
DEFAULT_ROTATION_ANGLES = {
    'M3': 282.75,
    'M5': 268.0,
    'M7': 261.6,
    'M9': 258.5,
}

# Rotation angles for patterns with randomness (typically different due to PSF spread)
DEFAULT_ROTATION_ANGLES_RAND = {
    'M3': 282.75,
    'M5': 268.0,
    'M7': 261.6,
    'M9': 258.5,
}

# Default scan parameters for different M values
DEFAULT_SCAN_PARAMS = {
    'M3': {'num_steps': 81, 'z_range': 0.3},
    'M5': {'num_steps': 81, 'z_range': 0.4},
    'M7': {'num_steps': 81, 'z_range': 0.6},
    'M9': {'num_steps': 101, 'z_range': 1.0},
}


class CaptureSetupGUI:
    """
    GUI for PSF capture preparation: hardware connection, Fresnel test, file selection.
    """

    def __init__(self, hw_manager=None, output_dir='./output', config_path=r".\\config\\base.json",
                 rotation_angles=None, rotation_angles_rand=None):
        """
        Initialize the capture setup GUI.

        Args:
            hw_manager: RemoteHardwareManager instance (optional, can connect later)
            output_dir: Directory containing .npy phase pattern files
            config_path: Path to base config JSON file
            rotation_angles: Dict of M value to rotation angle, e.g., {'M5': 268.0}
            rotation_angles_rand: Dict of M value to rotation angle for patterns with randomness
        """
        self.hw = hw_manager
        self.output_dir = os.path.normpath(output_dir)
        self.config_path = config_path
        self.file_info_list = []  # List of dicts with file info
        self.initial_z_pos = None

        # Initialize rotation angles with defaults, then override with user values
        self.rotation_angles = DEFAULT_ROTATION_ANGLES.copy()
        if rotation_angles:
            self.rotation_angles.update(rotation_angles)

        # Initialize rotation angles for randomness patterns
        self.rotation_angles_rand = DEFAULT_ROTATION_ANGLES_RAND.copy()
        if rotation_angles_rand:
            self.rotation_angles_rand.update(rotation_angles_rand)

        self._create_widgets()
        self._setup_layout()
        self._setup_callbacks()

        # Auto-refresh hardware status if connected
        if self.hw is not None:
            self._refresh_hw_status(None)

    def _create_widgets(self):
        """Create all GUI widgets."""
        # === Hardware Status Section ===
        self.w_hw_status = widgets.HTML(
            value='<span style="color: gray;">Not connected</span>'
        )

        self.w_connect_btn = widgets.Button(
            description='Connect Hardware',
            button_style='primary',
            icon='plug',
            layout=widgets.Layout(width='150px')
        )

        self.w_refresh_btn = widgets.Button(
            description='Refresh',
            button_style='info',
            icon='refresh',
            layout=widgets.Layout(width='100px')
        )

        # Position displays
        self.w_z_pos = widgets.HTML(value='Z: -- mm')
        self.w_rot_pos = widgets.HTML(value='Rot: -- deg')

        # === Rotation Angle Settings ===
        self.w_rotation_angles = {}
        for m in ['M3', 'M5', 'M7', 'M9']:
            self.w_rotation_angles[m] = widgets.FloatText(
                value=self.rotation_angles.get(m, 260.0),
                description=f'{m}:',
                style={'description_width': '30px'},
                layout=widgets.Layout(width='100px')
            )

        # === Rotation Angle Settings for Randomness ===
        self.w_rotation_angles_rand = {}
        for m in ['M3', 'M5', 'M7', 'M9']:
            self.w_rotation_angles_rand[m] = widgets.FloatText(
                value=self.rotation_angles_rand.get(m, 260.0),
                description=f'{m}:',
                style={'description_width': '30px'},
                layout=widgets.Layout(width='100px')
            )

        # === Fresnel Test Section ===
        self.w_test_m = widgets.Dropdown(
            options=[3, 5, 7, 9],
            value=5,
            description='M:',
            style={'description_width': '30px'},
            layout=widgets.Layout(width='100px')
        )

        self.w_upload_fresnel_btn = widgets.Button(
            description='Upload Fresnel',
            button_style='success',
            icon='upload',
            layout=widgets.Layout(width='130px')
        )

        self.w_set_rotation_btn = widgets.Button(
            description='Set Rotation',
            button_style='warning',
            icon='cog',
            layout=widgets.Layout(width='120px')
        )

        self.w_test_status = widgets.HTML(value='')

        # === File Selection Section ===
        self.w_scan_files_btn = widgets.Button(
            description='Scan Files',
            button_style='info',
            icon='search',
            layout=widgets.Layout(width='120px')
        )

        self.w_file_select = widgets.SelectMultiple(
            options=[],
            description='Files:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='450px', height='200px')
        )

        self.w_load_selected_btn = widgets.Button(
            description='Load Selected',
            button_style='success',
            icon='download',
            tooltip='Load first selected pattern to SLM',
            layout=widgets.Layout(width='130px'),
            disabled=True
        )

        self.w_file_info = widgets.HTML(
            value='<i>Click "Scan Files" to find .npy files (double-click to load)</i>'
        )

        # === Scan Parameters Section ===
        self.w_z_focal = widgets.FloatText(
            value=12.0,
            description='Z Focal (mm):',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='180px')
        )

        self.w_use_current_z = widgets.Button(
            description='Use Current Z',
            button_style='info',
            layout=widgets.Layout(width='120px')
        )

        self.w_scan_summary = widgets.HTML(value='')

        # === Status Output ===
        self.w_status = widgets.Output(
            layout=widgets.Layout(height='100px', overflow='auto')
        )

    def _setup_layout(self):
        """Setup widget layout."""
        # Hardware section
        hw_section = widgets.VBox([
            widgets.HTML('<h3>Hardware Connection</h3>'),
            widgets.HBox([self.w_connect_btn, self.w_refresh_btn, self.w_hw_status]),
            widgets.HBox([self.w_z_pos, self.w_rot_pos])
        ])

        # Rotation angle settings
        rotation_section = widgets.VBox([
            widgets.HTML('<h3>Rotation Angles (deg)</h3>'),
            widgets.HTML('<b>Standard:</b>'),
            widgets.HBox([
                self.w_rotation_angles['M3'],
                self.w_rotation_angles['M5'],
                self.w_rotation_angles['M7'],
                self.w_rotation_angles['M9']
            ]),
            widgets.HTML('<b>With Randomness:</b>'),
            widgets.HBox([
                self.w_rotation_angles_rand['M3'],
                self.w_rotation_angles_rand['M5'],
                self.w_rotation_angles_rand['M7'],
                self.w_rotation_angles_rand['M9']
            ])
        ])

        # Fresnel test section
        fresnel_section = widgets.VBox([
            widgets.HTML('<h3>Fresnel Test</h3>'),
            widgets.HBox([self.w_test_m, self.w_upload_fresnel_btn, self.w_set_rotation_btn]),
            self.w_test_status
        ])

        # File selection section
        file_section = widgets.VBox([
            widgets.HTML('<h3>Select Phase Patterns</h3>'),
            widgets.HBox([self.w_scan_files_btn, self.w_load_selected_btn]),
            self.w_file_select,
            self.w_file_info
        ])

        # Scan parameters section
        scan_section = widgets.VBox([
            widgets.HTML('<h3>Scan Parameters</h3>'),
            widgets.HBox([self.w_z_focal, self.w_use_current_z]),
            self.w_scan_summary
        ])

        # Main layout
        self.main_layout = widgets.VBox([
            hw_section,
            rotation_section,
            fresnel_section,
            file_section,
            scan_section,
            self.w_status
        ])

    def _setup_callbacks(self):
        """Setup widget callbacks."""
        self.w_connect_btn.on_click(self._on_connect)
        self.w_refresh_btn.on_click(self._refresh_hw_status)
        self.w_upload_fresnel_btn.on_click(self._on_upload_fresnel)
        self.w_set_rotation_btn.on_click(self._on_set_rotation)
        self.w_scan_files_btn.on_click(self._on_scan_files)
        self.w_file_select.observe(self._on_file_select, names='value')
        self.w_load_selected_btn.on_click(self._on_load_selected)
        self.w_use_current_z.on_click(self._on_use_current_z)

        # Rotation angle change callbacks
        for m, w in self.w_rotation_angles.items():
            w.observe(lambda change, m=m: self._on_rotation_angle_change(m, change, False), names='value')
        for m, w in self.w_rotation_angles_rand.items():
            w.observe(lambda change, m=m: self._on_rotation_angle_change(m, change, True), names='value')

    def _on_rotation_angle_change(self, m_value, change, is_rand=False):
        """Handle rotation angle widget change."""
        if is_rand:
            self.rotation_angles_rand[m_value] = change['new']
        else:
            self.rotation_angles[m_value] = change['new']

    def _on_connect(self, btn):
        """Handle connect button click."""
        if self.hw is not None:
            self._refresh_hw_status(None)
            return

        with self.w_status:
            clear_output()
            print("Connecting to hardware service...")

        try:
            from hardware import RemoteHardwareManager
            self.hw = RemoteHardwareManager(host="127.0.0.1", port=18861)
            self._refresh_hw_status(None)
        except Exception as e:
            self.w_hw_status.value = f'<span style="color: red;">Connection failed: {e}</span>'

    def _refresh_hw_status(self, btn):
        """Refresh hardware status display."""
        if self.hw is None:
            self.w_hw_status.value = '<span style="color: gray;">Not connected</span>'
            self.w_z_pos.value = 'Z: -- mm'
            self.w_rot_pos.value = 'Rot: -- deg'
            return

        try:
            z_pos = self.hw.stage_get_position()
            rot_pos = self.hw.rotation_get_position()
            self.initial_z_pos = z_pos

            self.w_hw_status.value = '<span style="color: green;">Connected</span>'
            self.w_z_pos.value = f'Z: {z_pos:.4f} mm'
            self.w_rot_pos.value = f'Rot: {rot_pos:.2f} deg'

            # Auto-update Z focal to current position
            self.w_z_focal.value = z_pos

        except Exception as e:
            self.w_hw_status.value = f'<span style="color: red;">Error: {e}</span>'

    def _get_rotation_angle(self, m_value, has_randomness=False):
        """Get rotation angle for M value from current settings."""
        if isinstance(m_value, int):
            m_value = f'M{m_value}'
        if has_randomness:
            return self.rotation_angles_rand.get(m_value, 260.0)
        return self.rotation_angles.get(m_value, 260.0)

    def _on_upload_fresnel(self, btn):
        """Upload Fresnel test pattern to SLM."""
        if self.hw is None:
            self.w_test_status.value = '<span style="color: red;">Hardware not connected</span>'
            return

        try:
            from phase_generators import PhaseGenerator
            from optics_utils import load_dict_from_json

            params = load_dict_from_json(self.config_path)
            params['M'] = self.w_test_m.value

            optimizer = PhaseGenerator(params)
            optimizer.generate(mode='fresnel')
            phase_8bit = optimizer.update_phase_8bit()

            self.hw.upload_slm(phase_8bit)
            self.w_test_status.value = f'<span style="color: green;">Fresnel M{params["M"]} uploaded</span>'

        except Exception as e:
            self.w_test_status.value = f'<span style="color: red;">Error: {e}</span>'

    def _on_set_rotation(self, btn):
        """Set rotation stage to appropriate angle."""
        if self.hw is None:
            self.w_test_status.value = '<span style="color: red;">Hardware not connected</span>'
            return

        try:
            m_value = self.w_test_m.value
            target_angle = self._get_rotation_angle(m_value)

            # Overshoot to eliminate gear backlash
            self.hw.rotation_move_to(target_angle + 10)
            self.hw.rotation_move_to(target_angle)

            actual_angle = self.hw.rotation_get_position()
            self.w_rot_pos.value = f'Rot: {actual_angle:.2f} deg'
            self.w_test_status.value = f'<span style="color: green;">Rotation set to {actual_angle:.2f} deg</span>'

        except Exception as e:
            self.w_test_status.value = f'<span style="color: red;">Error: {e}</span>'

    def _on_scan_files(self, btn):
        """Scan output directory for .npy files."""
        self.file_info_list = []

        if not os.path.exists(self.output_dir):
            self.w_file_info.value = f'<span style="color: red;">Directory not found: {self.output_dir}</span>'
            return

        # Find all .npy files in subdirectories
        for item in sorted(os.listdir(self.output_dir)):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                npy_file = os.path.join(item_path, f'{item}.npy')
                json_file = os.path.join(item_path, f'{item}.json')

                if os.path.exists(npy_file):
                    # Parse M value from filename
                    m_match = re.search(r'M(\d+)', item)
                    m_value = f'M{m_match.group(1)}' if m_match else None

                    # Check for randomness in JSON
                    has_randomness = False
                    if os.path.exists(json_file):
                        try:
                            with open(json_file, 'r') as f:
                                params = json.load(f)
                                has_randomness = params.get('randomness', 0) > 0
                        except:
                            pass

                    self.file_info_list.append({
                        'name': f'{item}.npy',
                        'path': npy_file,
                        'json_path': json_file if os.path.exists(json_file) else None,
                        'm_value': m_value,
                        'has_randomness': has_randomness
                    })

        # Update selector
        if self.file_info_list:
            options = []
            for info in self.file_info_list:
                prefix = 'R' if info['has_randomness'] else ' '
                m_str = info['m_value'] or '??'
                options.append((f"[{prefix}][{m_str}] {info['name']}", info['name']))
            self.w_file_select.options = options
            self.w_file_info.value = f'Found {len(self.file_info_list)} files ([R]=randomness, double-click to load)'
        else:
            self.w_file_select.options = []
            self.w_file_info.value = '<span style="color: orange;">No .npy files found</span>'

    def _on_file_select(self, change):
        """Handle file selection change."""
        selected = change['new']
        if not selected:
            self.w_load_selected_btn.disabled = True
            self.w_scan_summary.value = ''
            return

        self.w_load_selected_btn.disabled = False
        self._update_scan_summary()

        # Check for double-click (same single selection twice in quick succession)
        if len(selected) == 1:
            # Use a simple approach: if selecting same item, treat as double-click intent
            # User can click "Load Selected" button for explicit load
            pass

    def _on_load_selected(self, btn):
        """Load first selected pattern to SLM and set rotation."""
        selected = self.w_file_select.value
        if not selected:
            return

        # Get first selected file info
        info = self._get_file_info(selected[0])
        if not info:
            return

        self._load_pattern(info)

    def _load_pattern(self, info):
        """Load a pattern to SLM (without adjusting rotation stage)."""
        if self.hw is None:
            with self.w_status:
                clear_output()
                print("Hardware not connected!")
            return

        try:
            # Load and upload phase pattern
            pattern = np.load(info['path'])
            self.hw.upload_slm(pattern)

            with self.w_status:
                clear_output()
                print(f"Loaded: {info['name']}")
                print(f"M={info['m_value']}, Randomness={info['has_randomness']}")

        except Exception as e:
            with self.w_status:
                clear_output()
                print(f"Error loading pattern: {e}")

    def _on_use_current_z(self, btn):
        """Use current Z position as focal plane."""
        if self.hw is None:
            return
        try:
            z_pos = self.hw.stage_get_position()
            self.w_z_focal.value = z_pos
            self.initial_z_pos = z_pos
        except:
            pass

    def _update_scan_summary(self):
        """Update scan summary based on selected files."""
        selected = self.w_file_select.value
        if not selected:
            self.w_scan_summary.value = ''
            return

        # Count by M value and randomness
        m_counts = {}  # {m_value: {'standard': count, 'rand': count}}
        total_frames = 0

        for name in selected:
            info = self._get_file_info(name)
            if info and info['m_value']:
                m = info['m_value']
                if m not in m_counts:
                    m_counts[m] = {'standard': 0, 'rand': 0}
                if info['has_randomness']:
                    m_counts[m]['rand'] += 1
                else:
                    m_counts[m]['standard'] += 1
                if m in DEFAULT_SCAN_PARAMS:
                    total_frames += DEFAULT_SCAN_PARAMS[m]['num_steps']

        # Build summary HTML
        lines = [f'<b>Selected: {len(selected)} files, {total_frames} total frames</b><br>']
        for m in sorted(m_counts.keys()):
            counts = m_counts[m]
            params = DEFAULT_SCAN_PARAMS.get(m, {})
            steps = params.get('num_steps', '?')
            z_range = params.get('z_range', '?')
            angle = self.rotation_angles.get(m, '?')
            angle_rand = self.rotation_angles_rand.get(m, '?')
            z_half = z_range / 2 if isinstance(z_range, (int, float)) else '?'

            # Show counts for standard and randomness separately
            count_parts = []
            if counts['standard'] > 0:
                count_parts.append(f"{counts['standard']} std")
            if counts['rand'] > 0:
                count_parts.append(f"{counts['rand']} rand")
            count_str = '+'.join(count_parts)

            # Show both angles if different
            if angle != angle_rand:
                angle_str = f"rot {angle}/{angle_rand}deg"
            else:
                angle_str = f"rot {angle}deg"

            lines.append(f'{m}: {count_str} x {steps} steps, +/-{z_half}mm, {angle_str}')

        self.w_scan_summary.value = '<br>'.join(lines)

    def _get_file_info(self, name):
        """Get file info by name."""
        for info in self.file_info_list:
            if info['name'] == name:
                return info
        return None

    def display(self):
        """Display the GUI."""
        display(self.main_layout)

    def get_selected_files(self):
        """Get list of selected file names."""
        return list(self.w_file_select.value)

    def get_selected_paths(self):
        """Get list of selected file paths."""
        paths = []
        for name in self.w_file_select.value:
            info = self._get_file_info(name)
            if info:
                paths.append(info['path'])
        return paths

    def get_file_info_list(self):
        """Get detailed info for selected files."""
        result = []
        for name in self.w_file_select.value:
            info = self._get_file_info(name)
            if info:
                result.append(info)
        return result

    def get_z_focal(self):
        """Get focal plane Z position."""
        return self.w_z_focal.value

    def get_hardware(self):
        """Get hardware manager instance."""
        return self.hw

    def get_rotation_angles(self):
        """Get current rotation angle settings."""
        return self.rotation_angles.copy()

    def get_rotation_angles_rand(self):
        """Get current rotation angle settings for patterns with randomness."""
        return self.rotation_angles_rand.copy()


class AutoCaptureGUI:
    """
    GUI for automated PSF capture with progress display.
    """

    def __init__(self, setup_gui, save_dir=None, scan_params=None):
        """
        Initialize the auto capture GUI.

        Args:
            setup_gui: CaptureSetupGUI instance (provides hw, files, z_focal)
            save_dir: Directory to save captured data
            scan_params: Dict of scan parameters by M value (optional, uses defaults)
        """
        self.setup_gui = setup_gui
        self.save_dir = save_dir or r"Z:\\SLM_super_resolution\\data\\for_auto_scan\\"
        self.scan_params = scan_params or DEFAULT_SCAN_PARAMS

        self.is_running = False
        self.scan_info = None

        self._create_widgets()
        self._setup_layout()
        self._setup_callbacks()

    def _create_widgets(self):
        """Create GUI widgets."""
        # Save directory
        self.w_save_dir = widgets.Text(
            value=self.save_dir,
            description='Save Dir:',
            style={'description_width': '70px'},
            layout=widgets.Layout(width='500px')
        )

        # User prefix
        self.w_prefix = widgets.Text(
            value='PSF',
            description='Prefix:',
            style={'description_width': '70px'},
            layout=widgets.Layout(width='200px')
        )

        # Control buttons
        self.w_start_btn = widgets.Button(
            description='Start Capture',
            button_style='success',
            icon='play',
            layout=widgets.Layout(width='140px')
        )

        self.w_stop_btn = widgets.Button(
            description='Stop',
            button_style='danger',
            icon='stop',
            layout=widgets.Layout(width='100px'),
            disabled=True
        )

        # Progress display
        self.w_progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            style={'description_width': '70px'},
            layout=widgets.Layout(width='400px')
        )

        self.w_progress_text = widgets.HTML(value='')

        # Current status
        self.w_current_pattern = widgets.HTML(value='')
        self.w_time_info = widgets.HTML(value='')

        # Status output (minimal)
        self.w_status = widgets.Output(
            layout=widgets.Layout(height='80px', overflow='auto')
        )

    def _setup_layout(self):
        """Setup widget layout."""
        config_section = widgets.VBox([
            widgets.HTML('<h3>Capture Settings</h3>'),
            self.w_save_dir,
            self.w_prefix
        ])

        control_section = widgets.HBox([
            self.w_start_btn,
            self.w_stop_btn
        ])

        progress_section = widgets.VBox([
            widgets.HTML('<h3>Progress</h3>'),
            self.w_progress_bar,
            self.w_progress_text,
            self.w_current_pattern,
            self.w_time_info
        ])

        self.main_layout = widgets.VBox([
            config_section,
            control_section,
            progress_section,
            self.w_status
        ])

    def _setup_callbacks(self):
        """Setup widget callbacks."""
        self.w_start_btn.on_click(self._on_start)
        self.w_stop_btn.on_click(self._on_stop)

    def _on_start(self, btn):
        """Start capture process."""
        self.is_running = True
        self.w_start_btn.disabled = True
        self.w_stop_btn.disabled = False

        try:
            self._run_capture()
        except Exception as e:
            with self.w_status:
                print(f"Error: {e}")
        finally:
            self.is_running = False
            self.w_start_btn.disabled = False
            self.w_stop_btn.disabled = True

    def _on_stop(self, btn):
        """Stop capture process."""
        self.is_running = False
        self.w_stop_btn.disabled = True

    def _run_capture(self):
        """Run the capture process."""
        hw = self.setup_gui.get_hardware()
        if hw is None:
            with self.w_status:
                print("Hardware not connected!")
            return

        # Get selected files
        file_info_list = self.setup_gui.get_file_info_list()
        if not file_info_list:
            with self.w_status:
                print("No files selected!")
            return

        z_focal = self.setup_gui.get_z_focal()
        save_dir = self.w_save_dir.value
        rotation_angles = self.setup_gui.get_rotation_angles()
        rotation_angles_rand = self.setup_gui.get_rotation_angles_rand()

        # Calculate total frames
        total_frames = 0
        for info in file_info_list:
            m = info['m_value']
            if m in self.scan_params:
                total_frames += self.scan_params[m]['num_steps']

        # Capture click position
        with self.w_status:
            clear_output()
            print("Click camera trigger button...")

        click_pos = hw.capture_position()
        if click_pos is None:
            with self.w_status:
                print("Failed to capture click position!")
            return

        # Clean old files
        if os.path.exists(save_dir):
            old_files = [f for f in os.listdir(save_dir) if f.endswith((".tiff", ".tif"))]
            for f in old_files:
                os.remove(os.path.join(save_dir, f))

        # Initialize scan info
        scan_start_time = datetime.now()
        self.scan_info = {
            'start_time': scan_start_time.isoformat(),
            'z_focal_plane': z_focal,
            'scan_params_by_m': self.scan_params,
            'm_angles': rotation_angles,
            'm_angles_rand': rotation_angles_rand,
            'patterns': [],
            'save_dir': save_dir,
        }

        # Main capture loop
        frame_counter = 0
        current_m = None
        current_rand = None  # Track randomness state for rotation changes

        with self.w_status:
            clear_output()
            print(f"Starting: {len(file_info_list)} patterns, {total_frames} frames")

        for pattern_idx, info in enumerate(file_info_list):
            if not self.is_running:
                break

            m_value = info['m_value']
            has_rand = info['has_randomness']
            params = self.scan_params.get(m_value, self.scan_params['M5'])
            num_steps = params['num_steps']
            z_range = params['z_range']

            # Update current pattern display
            self.w_current_pattern.value = f'<b>[{pattern_idx+1}/{len(file_info_list)}]</b> {info["name"]}'

            # Calculate Z positions
            z_positions = np.linspace(
                z_focal - z_range / 2,
                z_focal + z_range / 2,
                num_steps
            )

            # Set rotation if M or randomness state changed
            if m_value != current_m or has_rand != current_rand:
                if has_rand:
                    target_angle = rotation_angles_rand.get(m_value, 260.0)
                else:
                    target_angle = rotation_angles.get(m_value, 260.0)
                hw.rotation_move_to(target_angle + 10)
                hw.rotation_move_to(target_angle)
                current_m = m_value
                current_rand = has_rand

            # Upload phase pattern
            pattern = np.load(info['path'])
            hw.upload_slm(pattern)

            # Record pattern info
            pattern_info = {
                'name': info['name'],
                'path': info['path'],
                'm_pattern': m_value,
                'has_randomness': has_rand,
                'num_steps': num_steps,
                'z_range': z_range,
                'z_positions': z_positions.tolist(),
                'frame_start': frame_counter + 1,
            }

            # Move to start position
            hw.stage_move_to(z_positions[0])
            time.sleep(0.2)

            # Z-scan loop
            for z_idx, z_pos in enumerate(z_positions):
                if not self.is_running:
                    break

                frame_counter += 1

                # Move and capture
                hw.stage_move_to(z_pos)
                time.sleep(0.2)
                hw.click_at()
                time.sleep(1.0)

                # Update progress
                progress = frame_counter / total_frames * 100
                self.w_progress_bar.value = progress
                self.w_progress_text.value = f'Frame {frame_counter}/{total_frames} ({progress:.1f}%) | Z={z_pos:.4f}mm'

                # Estimate remaining time
                elapsed = time.time() - scan_start_time.timestamp()
                if frame_counter > 0:
                    eta = elapsed / frame_counter * (total_frames - frame_counter)
                    eta_min = int(eta // 60)
                    eta_sec = int(eta % 60)
                    self.w_time_info.value = f'Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | ETA: {eta_min}m {eta_sec}s'

            pattern_info['frame_end'] = frame_counter
            self.scan_info['patterns'].append(pattern_info)

        # Complete
        scan_end_time = datetime.now()
        self.scan_info['end_time'] = scan_end_time.isoformat()
        self.scan_info['duration_seconds'] = (scan_end_time - scan_start_time).total_seconds()
        self.scan_info['total_frames'] = frame_counter

        self.w_progress_bar.value = 100
        self.w_current_pattern.value = '<b>Capture complete!</b>'
        duration = self.scan_info['duration_seconds']
        self.w_time_info.value = f'Total time: {int(duration//60)}m {int(duration%60)}s'

        with self.w_status:
            clear_output()
            print(f"Complete! {frame_counter} frames in {duration:.1f}s")

    def display(self):
        """Display the GUI."""
        display(self.main_layout)

    def get_scan_info(self):
        """Get scan info for data organization."""
        return self.scan_info

    def organize_data(self, user_prefix=None):
        """
        Organize captured TIFF files into folders.

        Args:
            user_prefix: Filename prefix (uses GUI value if not provided)
        """
        if self.scan_info is None:
            print("No scan info available. Run capture first.")
            return

        prefix = user_prefix or self.w_prefix.value
        save_dir = self.w_save_dir.value

        # Find TIFF files
        tiff_pattern = os.path.join(save_dir, "ss_single_*.tiff")
        tiff_files = sorted(
            glob.glob(tiff_pattern),
            key=lambda x: int(os.path.basename(x).replace('ss_single_', '').replace('.tiff', ''))
        )

        print(f"Found {len(tiff_files)} TIFF files")

        frame_idx = 0
        for pattern_info in self.scan_info['patterns']:
            npy_name = pattern_info['name']
            m_pattern = pattern_info['m_pattern']
            num_steps = pattern_info['num_steps']
            z_positions = pattern_info['z_positions']

            # Create folder
            pattern_basename = os.path.splitext(npy_name)[0]
            pattern_folder = os.path.join(save_dir, pattern_basename)
            os.makedirs(pattern_folder, exist_ok=True)

            # Move files
            for z_idx, z_pos in enumerate(z_positions):
                if frame_idx >= len(tiff_files):
                    break

                src_path = tiff_files[frame_idx]
                new_name = f"{prefix}_{m_pattern}_frame{z_idx+1:03d}_z{z_pos:.4f}mm.tiff"
                dst_path = os.path.join(pattern_folder, new_name)

                shutil.move(src_path, dst_path)
                frame_idx += 1

            # Save scan info JSON
            info_path = os.path.join(pattern_folder, f"{pattern_basename}_scan_info.json")
            with open(info_path, 'w') as f:
                json.dump(pattern_info, f, indent=2)

            print(f"Organized: {pattern_basename}/")

        print(f"Data organization complete!")


def create_capture_setup_gui(hw_manager=None, output_dir='./output', rotation_angles=None, rotation_angles_rand=None):
    """
    Create and display the capture setup GUI.

    Args:
        hw_manager: RemoteHardwareManager instance (optional)
        output_dir: Directory containing .npy files
        rotation_angles: Dict of M value to rotation angle, e.g., {'M5': 268.0}
        rotation_angles_rand: Dict of M value to rotation angle for patterns with randomness

    Returns:
        CaptureSetupGUI instance
    """
    gui = CaptureSetupGUI(hw_manager=hw_manager, output_dir=output_dir,
                          rotation_angles=rotation_angles, rotation_angles_rand=rotation_angles_rand)
    gui.display()
    return gui


def create_auto_capture_gui(setup_gui, save_dir=None):
    """
    Create and display the auto capture GUI.

    Args:
        setup_gui: CaptureSetupGUI instance
        save_dir: Directory to save captured data

    Returns:
        AutoCaptureGUI instance
    """
    gui = AutoCaptureGUI(setup_gui, save_dir=save_dir)
    gui.display()
    return gui
