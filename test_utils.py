# test_utils.py
"""
Utility functions and GUI for phase optimizer testing.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display, clear_output
import ipywidgets as widgets
from typing import Optional, Dict, Any

from optics_utils import compute_psf_centers, save_dict_as_json, load_dict_from_json
from phase_generators import PhaseGenerator
from visualization import plot_phase, plot_2d_comparisons, plot_2d_comparisons_interactive, plot_energy_distribution


# =============================================================================
# QuickTestGUI - Main GUI for testing
# =============================================================================

class QuickTestGUI:
    """
    GUI for quick testing of phase optimization with all parameters.
    Includes SLM connection, optimization, visualization, and config management.

    Parameters
    ----------
    config_path : str, optional
        Path to JSON config file for default parameters.
        If None, uses BASE_CONFIG_PATH (./config/base.json).
        If file doesn't exist, uses hardcoded defaults.
    """

    BASE_CONFIG_PATH = r"./config/base.json"

    def __init__(self, config_path: Optional[str] = None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.slm_manager = None
        self.sim_mode = True
        self.optimizer = None
        self.phase_8bit = None
        self.config_path = config_path or self.BASE_CONFIG_PATH

        # Load default config
        self.default_params = self._load_config(self.config_path)

        self._create_widgets()
        self._setup_layout()
        self._setup_callbacks()
        self._apply_config(self.default_params)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        import os
        if os.path.exists(path):
            try:
                params = load_dict_from_json(path)
                print(f"Loaded config from: {path}")
                return params
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")
        return {}

    def _apply_config(self, params: Dict[str, Any]):
        """Apply configuration to widgets."""
        if not params:
            return

        # Map config keys to widgets
        widget_map = {
            'M': self.w_M,
            'N': self.w_N,
            'roi_center_x': self.w_roi_center_x,
            'roi_center_y': self.w_roi_center_y,
            'focal_length': (self.w_focal_length, lambda v: v * 1000),  # m -> mm
            'two_pi_value': self.w_two_pi_value,
            'overlap_ratio': self.w_overlap_ratio,
            'airy_correction': self.w_airy_correction,
            'center_blend': self.w_center_blend,
            'randomness': self.w_randomness,
            'random_seed': self.w_random_seed,
            'dof_correction': self.w_dof_correction,
            'lr': self.w_lr,
            'ni': self.w_ni,
            'upsampling': self.w_upsampling,
            'mask_count': self.w_mask_count,
            'interleaving': self.w_interleaving,
        }

        # Apply values
        for key, widget_info in widget_map.items():
            if key in params:
                value = params[key]
                if isinstance(widget_info, tuple):
                    widget, transform = widget_info
                    value = transform(value)
                else:
                    widget = widget_info
                try:
                    widget.value = value
                except Exception:
                    pass  # Skip if value is incompatible

        # Handle depth_in_focus (list -> string)
        if 'depth_in_focus' in params:
            dif = params['depth_in_focus']
            if isinstance(dif, list) and len(dif) >= 2:
                self.w_depth_in_focus.value = f"{dif[0]}, {dif[1]}"

        # Handle weights
        if 'weights' in params:
            w = params['weights']
            if 'mse' in w:
                self.w_weight_mse.value = w['mse']
            if 'depth_in_focus' in w:
                self.w_weight_depth.value = w['depth_in_focus']
            if 'eff_mean' in w:
                self.w_weight_eff_mean.value = w['eff_mean']
            if 'eff_std' in w:
                self.w_weight_eff_std.value = w['eff_std']

        print(f"Applied {len(params)} config parameters")

    def _create_widgets(self):
        """Create all GUI widgets."""
        style = {'description_width': '120px'}
        layout_narrow = widgets.Layout(width='220px')
        layout_wide = widgets.Layout(width='280px')

        # === SLM Connection ===
        self.w_slm_mode = widgets.Dropdown(
            options=[('Simulation', 'sim'), ('Remote', 'remote'), ('Local', 'local')],
            value='sim',
            description='SLM Mode:',
            style=style, layout=layout_narrow
        )
        self.w_connect_btn = widgets.Button(
            description='Connect SLM',
            button_style='primary',
            layout=widgets.Layout(width='120px', height='30px')
        )
        self.w_slm_status = widgets.HTML(value='<span style="color:gray">Not connected</span>')

        # === Basic Parameters ===
        self.w_M = widgets.IntSlider(
            value=5, min=2, max=15, step=1,
            description='M (array size):',
            style=style, layout=layout_wide
        )
        self.w_N = widgets.IntSlider(
            value=850, min=64, max=1152, step=32,
            description='N (pixels):',
            style=style, layout=layout_wide
        )

        # === ROI Center Controls ===
        center_style = {'description_width': '70px'}
        center_layout = widgets.Layout(width='280px')
        self.w_roi_center_x = widgets.IntSlider(
            value=960, min=0, max=1920, step=1,
            description='Center X:',
            style=center_style, layout=center_layout
        )
        self.w_roi_center_y = widgets.IntSlider(
            value=576, min=0, max=1152, step=1,
            description='Center Y:',
            style=center_style, layout=center_layout
        )
        self.w_reset_center_btn = widgets.Button(
            description='Reset',
            button_style='',
            layout=widgets.Layout(width='60px', height='28px')
        )
        self.w_focal_length = widgets.FloatText(
            value=73.9, description='Focal (mm):',
            style=style, layout=layout_narrow
        )
        self.w_two_pi_value = widgets.IntText(
            value=210, description='2pi Value:',
            style=style, layout=layout_narrow
        )
        self.w_mode = widgets.RadioButtons(
            options=['fresnel', 'optimized'],
            value='fresnel',
            description='Mode:',
            style=style
        )

        # === Optimization Parameters ===
        self.w_overlap_ratio = widgets.FloatSlider(
            value=0.3, min=0.0, max=0.9, step=0.05,
            description='Overlap Ratio:',
            style=style, layout=layout_wide, readout_format='.2f'
        )
        self.w_airy_correction = widgets.FloatSlider(
            value=1.0, min=0.1, max=2.0, step=0.1,
            description='Airy Correction:',
            style=style, layout=layout_wide, readout_format='.1f'
        )
        self.w_center_blend = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.1,
            description='Center Blend:',
            style=style, layout=layout_wide, readout_format='.1f'
        )
        self.w_randomness = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.5, step=0.05,
            description='Randomness:',
            style=style, layout=layout_wide, readout_format='.2f'
        )
        self.w_random_seed = widgets.IntText(
            value=42, description='Random Seed:',
            style=style, layout=layout_narrow
        )
        self.w_use_seed = widgets.Checkbox(value=True, description='Use Seed', style=style)

        # === Depth Parameters ===
        self.w_depth_in_focus = widgets.Textarea(
            value='-0.5, 0.5', placeholder='e.g., -0.5, 0.5',
            description='Depth in Focus:', style=style,
            layout=widgets.Layout(width='300px', height='50px')
        )
        self.w_dof_correction = widgets.FloatSlider(
            value=1.0, min=0.1, max=3.0, step=0.1,
            description='DOF Correction:',
            style=style, layout=layout_wide, readout_format='.1f'
        )

        # === Training Parameters ===
        self.w_lr = widgets.FloatLogSlider(
            value=0.05, base=10, min=-2, max=0, step=0.1,
            description='Learning Rate:',
            style=style, layout=layout_wide, readout_format='.3f'
        )
        self.w_ni = widgets.IntSlider(
            value=500, min=10, max=1000, step=10,
            description='Iterations:',
            style=style, layout=layout_wide
        )
        self.w_upsampling = widgets.FloatSlider(
            value=2.0, min=1.0, max=4.0, step=0.5,
            description='Upsampling:',
            style=style, layout=layout_wide, readout_format='.1f'
        )

        # === Loss Weights ===
        weight_style = {'description_width': '80px'}
        weight_layout = widgets.Layout(width='150px')
        self.w_weight_mse = widgets.FloatText(value=1.0, description='MSE:', style=weight_style, layout=weight_layout)
        self.w_weight_depth = widgets.FloatText(value=1.0, description='Depth:', style=weight_style, layout=weight_layout)
        self.w_weight_eff_mean = widgets.FloatText(value=20.0, description='Eff Mean:', style=weight_style, layout=weight_layout)
        self.w_weight_eff_std = widgets.FloatText(value=50.0, description='Eff Std:', style=weight_style, layout=weight_layout)

        # === Mask Parameters ===
        self.w_mask_count = widgets.IntSlider(
            value=0, min=0, max=9, step=1,
            description='Mask Count:',
            style=style, layout=layout_wide
        )
        self.w_interleaving = widgets.Dropdown(
            options=['checkerboard', 'coarse1', 'coarse2', 'coarse3'],
            value='coarse1', description='Interleaving:',
            style=style, layout=layout_narrow
        )

        # === Action Buttons ===
        btn_layout = widgets.Layout(width='120px', height='32px')
        self.w_run_btn = widgets.Button(description='Run', button_style='success', layout=btn_layout)
        self.w_upload_btn = widgets.Button(description='Upload SLM', button_style='danger', layout=btn_layout)
        self.w_visualize_btn = widgets.Button(description='Visualize', button_style='info', layout=btn_layout)
        self.w_analyze_btn = widgets.Button(description='Efficiency', button_style='primary', layout=btn_layout)
        self.w_show_centers_btn = widgets.Button(description='PSF Centers', button_style='warning', layout=btn_layout)
        self.w_psf_slices_btn = widgets.Button(description='PSF Slices', button_style='', layout=btn_layout)
        self.w_psf_patches_btn = widgets.Button(description='PSF Patches', button_style='', layout=btn_layout)
        self.w_save_config_btn = widgets.Button(description='Save Config', button_style='', layout=btn_layout)
        self.w_load_config_btn = widgets.Button(description='Load Config', button_style='', layout=btn_layout)

        # === Output Areas ===
        self.w_info = widgets.HTML(value='', layout=widgets.Layout(width='100%'))
        self.w_log_output = widgets.Output(
            layout=widgets.Layout(width='100%', height='180px', overflow='auto', border='1px solid #ccc')
        )
        self.w_plot_output = widgets.Output(layout=widgets.Layout(width='100%'))

    def _setup_layout(self):
        """Setup the widget layout."""
        # SLM section
        slm_section = widgets.HBox([
            widgets.HTML('<b>SLM:</b> '),
            self.w_slm_mode, self.w_connect_btn, self.w_slm_status
        ])

        # Basic params
        basic_section = widgets.VBox([
            widgets.HTML('<h4>Basic</h4>'),
            self.w_M, self.w_N,
            widgets.HBox([self.w_roi_center_x, self.w_reset_center_btn]),
            self.w_roi_center_y,
            widgets.HBox([self.w_focal_length, self.w_two_pi_value]),
            self.w_mode
        ])

        # Optimization params
        opt_section = widgets.VBox([
            widgets.HTML('<h4>Optimization</h4>'),
            self.w_overlap_ratio, self.w_airy_correction, self.w_center_blend,
            self.w_randomness,
            widgets.HBox([self.w_random_seed, self.w_use_seed])
        ])

        # Depth params
        depth_section = widgets.VBox([
            widgets.HTML('<h4>Depth</h4>'),
            self.w_depth_in_focus, self.w_dof_correction
        ])

        # Training params
        train_section = widgets.VBox([
            widgets.HTML('<h4>Training</h4>'),
            self.w_lr, self.w_ni, self.w_upsampling
        ])

        # Weights
        weights_section = widgets.VBox([
            widgets.HTML('<h4>Weights</h4>'),
            widgets.HBox([self.w_weight_mse, self.w_weight_depth]),
            widgets.HBox([self.w_weight_eff_mean, self.w_weight_eff_std])
        ])

        # Mask params
        mask_section = widgets.VBox([
            widgets.HTML('<h4>Mask</h4>'),
            self.w_mask_count, self.w_interleaving
        ])

        # Actions
        action_section = widgets.VBox([
            widgets.HTML('<h4>Actions</h4>'),
            widgets.HBox([self.w_run_btn, self.w_upload_btn]),
            widgets.HBox([self.w_visualize_btn, self.w_analyze_btn]),
            widgets.HBox([self.w_show_centers_btn, self.w_psf_slices_btn, self.w_psf_patches_btn]),
            widgets.HBox([self.w_load_config_btn, self.w_save_config_btn])
        ])

        left_col = widgets.VBox([basic_section, opt_section, depth_section])
        right_col = widgets.VBox([train_section, weights_section, mask_section, action_section])
        params_panel = widgets.HBox([left_col, right_col])

        self.main_layout = widgets.VBox([
            widgets.HTML('<h2>Phase Optimizer Test</h2>'),
            slm_section,
            self.w_info,
            params_panel,
            widgets.HTML('<h4>Log</h4>'),
            self.w_log_output,
            widgets.HTML('<h4>Visualization</h4>'),
            self.w_plot_output
        ])

    def _setup_callbacks(self):
        """Setup widget callbacks."""
        self.w_connect_btn.on_click(self._on_connect)
        self.w_run_btn.on_click(self._on_run)
        self.w_upload_btn.on_click(self._on_upload)
        self.w_visualize_btn.on_click(self._on_visualize)
        self.w_analyze_btn.on_click(self._on_analyze)
        self.w_show_centers_btn.on_click(self._on_show_centers)
        self.w_psf_slices_btn.on_click(self._on_psf_slices)
        self.w_psf_patches_btn.on_click(self._on_psf_patches)
        self.w_save_config_btn.on_click(self._on_save_config)
        self.w_load_config_btn.on_click(self._on_load_config)
        self.w_reset_center_btn.on_click(self._on_reset_center)

        for w in [self.w_overlap_ratio, self.w_M, self.w_N, self.w_randomness]:
            w.observe(self._update_info, names='value')

    def _update_info(self, change=None):
        """Update info display."""
        M, N = self.w_M.value, self.w_N.value
        overlap, randomness = self.w_overlap_ratio.value, self.w_randomness.value

        if overlap > 0 and M > 0:
            alpha = (overlap * (N - N / M) + N / M) / (N / M)
            dof_info = f'alpha={alpha:.2f}, DOF x{1/alpha**2:.2f}'
        else:
            dof_info = 'Standard Fresnel'

        rand_info = f'sigma={randomness * N / M:.1f}px' if randomness > 0 else 'Off'
        slm_info = 'Sim' if self.sim_mode else ('Connected' if self.slm_manager else 'Disconnected')

        self.w_info.value = f'''
        <div style="background:#e8f4f8; padding:8px; border-radius:5px; margin:5px 0;">
            <b>Config:</b> {M}x{M}, {N}px | <b>Overlap:</b> {dof_info} |
            <b>Rand:</b> {rand_info} | <b>SLM:</b> {slm_info} | <b>Device:</b> {self.device}
        </div>'''

    def _on_connect(self, btn):
        """Connect to SLM."""
        mode = self.w_slm_mode.value
        with self.w_log_output:
            try:
                if mode == 'sim':
                    from hardware import SLMManager
                    self.slm_manager = SLMManager(sim_mode=True)
                    self.sim_mode = True
                    self.w_slm_status.value = '<span style="color:blue">Simulation</span>'
                    print("SLM: Simulation mode")
                elif mode == 'remote':
                    from hardware import RemoteSLMManager
                    self.slm_manager = RemoteSLMManager()
                    self.sim_mode = False
                    self.w_slm_status.value = '<span style="color:green">Remote Connected</span>'
                else:
                    from hardware import SLMManager
                    self.slm_manager = SLMManager(sim_mode=False)
                    self.sim_mode = False
                    self.w_slm_status.value = '<span style="color:green">Local Connected</span>'
            except Exception as e:
                self.w_slm_status.value = f'<span style="color:red">Error</span>'
                print(f"Connection error: {e}")
        self._update_info()

    def _get_params(self):
        """Get current parameters as dictionary."""
        try:
            depth_list = [float(x.strip()) for x in self.w_depth_in_focus.value.strip().split(',') if x.strip()]
        except:
            depth_list = [-0.5, 0.5]

        N = self.w_N.value
        return {
            'M': self.w_M.value, 'N': N, 'output_size': N,
            'focal_length': self.w_focal_length.value * 1e-3,
            'two_pi_value': self.w_two_pi_value.value,
            'overlap_ratio': self.w_overlap_ratio.value,
            'airy_correction': self.w_airy_correction.value,
            'center_blend': self.w_center_blend.value,
            'randomness': self.w_randomness.value,
            'random_seed': self.w_random_seed.value if self.w_use_seed.value else None,
            'depth_in_focus': depth_list, 'depth_out_focus': None,
            'dof_correction': self.w_dof_correction.value,
            'lr': self.w_lr.value, 'ni': self.w_ni.value,
            'show_iters': max(10, self.w_ni.value // 10),
            'mask_count': self.w_mask_count.value,
            'interleaving': self.w_interleaving.value,
            'masked_airy_correction': 1.0, 'focusing_eff_correction': 1.0, 'psf_energy_level': 1.0,
            'weights': {
                'mse': self.w_weight_mse.value,
                'depth_in_focus': self.w_weight_depth.value,
                'depth_out_focus': 0.0,
                'masked': 1.0 if self.w_mask_count.value > 0 else 0.0,
                'eff_mean': self.w_weight_eff_mean.value,
                'eff_std': self.w_weight_eff_std.value
            },
            'shape': [1152, 1920],
            'roi_center_x': self.w_roi_center_x.value,
            'roi_center_y': self.w_roi_center_y.value
        }

    def _set_params(self, params):
        """Set GUI from parameters dictionary."""
        self.w_M.value = params.get('M', 5)
        self.w_N.value = params.get('N', 850)
        self.w_roi_center_x.value = params.get('roi_center_x', 960)
        self.w_roi_center_y.value = params.get('roi_center_y', 576)
        self.w_focal_length.value = params.get('focal_length', 0.0739) * 1e3
        self.w_two_pi_value.value = params.get('two_pi_value', 210)
        self.w_overlap_ratio.value = params.get('overlap_ratio', 0.3)
        self.w_airy_correction.value = params.get('airy_correction', 1.0)
        self.w_center_blend.value = params.get('center_blend', 0.0)
        self.w_randomness.value = params.get('randomness', 0.0)
        seed = params.get('random_seed')
        self.w_use_seed.value = seed is not None
        if seed: self.w_random_seed.value = seed
        self.w_depth_in_focus.value = ', '.join(str(d) for d in params.get('depth_in_focus', [-0.5, 0.5]))
        self.w_dof_correction.value = params.get('dof_correction', 1.0)
        self.w_lr.value = params.get('lr', 0.05)
        self.w_ni.value = params.get('ni', 500)
        self.w_mask_count.value = params.get('mask_count', 0)
        self.w_interleaving.value = params.get('interleaving', 'coarse1')
        weights = params.get('weights', {})
        self.w_weight_mse.value = weights.get('mse', 1.0)
        self.w_weight_depth.value = weights.get('depth_in_focus', 1.0)
        self.w_weight_eff_mean.value = weights.get('eff_mean', 20.0)
        self.w_weight_eff_std.value = weights.get('eff_std', 50.0)

    def _on_save_config(self, btn):
        with self.w_log_output:
            try:
                save_dict_as_json(self._get_params(), self.BASE_CONFIG_PATH)
                print(f"Saved: {self.BASE_CONFIG_PATH}")
            except Exception as e:
                print(f"Save error: {e}")

    def _on_load_config(self, btn):
        with self.w_log_output:
            try:
                self._set_params(load_dict_from_json(self.BASE_CONFIG_PATH))
                print(f"Loaded: {self.BASE_CONFIG_PATH}")
                self._update_info()
            except Exception as e:
                print(f"Load error: {e}")

    def _on_reset_center(self, btn):
        """Reset center position to base.json defaults."""
        with self.w_log_output:
            try:
                params = load_dict_from_json(self.BASE_CONFIG_PATH)
                self.w_roi_center_x.value = params.get('roi_center_x', 960)
                self.w_roi_center_y.value = params.get('roi_center_y', 576)
                print(f"Center reset to: ({self.w_roi_center_x.value}, {self.w_roi_center_y.value})")
            except Exception as e:
                print(f"Reset error: {e}")

    def _on_run(self, btn):
        with self.w_log_output:
            clear_output()
            params = self._get_params()
            mode = self.w_mode.value
            print(f"Running {mode} mode: M={params['M']}, N={params['N']}")
            try:
                self.optimizer = PhaseGenerator(params, device=self.device, mode=mode)
                if mode == 'fresnel':
                    self.optimizer.generate(mode='fresnel')
                else:
                    self.optimizer.generate(mode='optimized', upsampling=self.w_upsampling.value, visualize=False)
                self.phase_8bit = self.optimizer.update_phase_8bit()
                print(f"Done. Phase shape: {self.phase_8bit.shape}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback; traceback.print_exc()
        self._update_info()

    def _on_upload(self, btn):
        with self.w_log_output:
            if self.phase_8bit is None:
                print("Run optimization first"); return
            if self.sim_mode:
                print("Simulation mode - no upload"); return
            if self.slm_manager is None:
                print("SLM not connected"); return
            try:
                self.slm_manager.upload(self.phase_8bit)
                print("Uploaded to SLM")
            except Exception as e:
                print(f"Upload error: {e}")

    def _on_visualize(self, btn):
        if self.optimizer is None:
            with self.w_log_output: print("Run first!"); return
        with self.w_plot_output:
            clear_output(wait=True)
            plt.close('all')
            plot_phase(self.optimizer); plt.show()
            # Use interactive version for adjustable slice position (important with randomness)
            plot_2d_comparisons_interactive(self.optimizer, upsampling=1.0)

    def _on_analyze(self, btn):
        if self.optimizer is None:
            with self.w_log_output: print("Run first!"); return
        with self.w_plot_output:
            clear_output(wait=True)
            plt.close('all')
            plot_energy_distribution(self.optimizer, upsampling=3); plt.show()
            if self.optimizer.history:
                print("\nLoss History:")
                for k, v in self.optimizer.history.items():
                    if v: print(f"  {k}: {v[0]:.4f} -> {v[-1]:.4f}")

    def _on_show_centers(self, btn):
        with self.w_plot_output:
            clear_output(wait=True)
            params = self._get_params()
            show_psf_centers_plot(
                params['M'], params['N'], params['overlap_ratio'],
                params['center_blend'], params['randomness'],
                params['random_seed'], self.device
            )

    def _on_psf_slices(self, btn):
        if self.optimizer is None:
            with self.w_log_output: print("Run first!"); return
        with self.w_plot_output:
            clear_output(wait=True)
            show_psf_slices_interactive(self.optimizer, upsampling=2.0)

    def _on_psf_patches(self, btn):
        if self.optimizer is None:
            with self.w_log_output: print("Run first!"); return
        with self.w_plot_output:
            clear_output(wait=True)
            show_psf_patches_interactive(self.optimizer, upsampling=2.0)

    def display(self):
        """Display the GUI."""
        self._update_info()
        display(self.main_layout)


# =============================================================================
# Visualization helper functions
# =============================================================================

def show_psf_centers_plot(
    M: int, N: int, overlap_ratio: float, center_blend: float,
    randomness: float, random_seed: Optional[int], device: torch.device
):
    """Show PSF centers visualization comparing reference vs randomized positions."""
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    result_ref = compute_psf_centers(
        M=M, overlap_ratio=overlap_ratio, center_blend=center_blend,
        z_ratio=1.0, N=N, output_size=N, device=device, randomness=0.0
    )
    centers_ref = result_ref['centers_pixel'].cpu().numpy()

    ax = axes[0]
    ax.scatter(centers_ref[:, 1], centers_ref[:, 0], c='blue', s=100, label='PSF Centers')
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Reference (No Randomness) - {N}x{N}')
    ax.grid(True, alpha=0.3); ax.legend()

    result_rand = compute_psf_centers(
        M=M, overlap_ratio=overlap_ratio, center_blend=center_blend,
        z_ratio=1.0, N=N, output_size=N, device=device,
        randomness=randomness, random_seed=random_seed
    )
    centers_rand = result_rand['centers_pixel'].cpu().numpy()

    ax = axes[1]
    ax.scatter(centers_ref[:, 1], centers_ref[:, 0], c='lightgray', s=100, marker='x', label='Reference')
    ax.scatter(centers_rand[:, 1], centers_rand[:, 0], c='red', s=100, label='Randomized')
    if randomness > 0:
        for i in range(len(centers_ref)):
            ax.annotate('', xy=(centers_rand[i, 1], centers_rand[i, 0]),
                       xytext=(centers_ref[i, 1], centers_ref[i, 0]),
                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Randomness = {randomness}, Seed = {random_seed}')
    ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout(); plt.show()

    if randomness > 0 and 'random_offsets' in result_rand:
        offsets = result_rand['random_offsets'].cpu().numpy()
        print(f"\nOffset Stats: mean=({offsets[:, 0].mean():.2f}, {offsets[:, 1].mean():.2f}), "
              f"std=({offsets[:, 0].std():.2f}, {offsets[:, 1].std():.2f}), max={np.abs(offsets).max():.2f} px")


def show_psf_slices_interactive(optimizer, upsampling: float = 2.0):
    """Show individual PSF slices with template overlay and adjustable slice position.

    Features:
    - Global intensity map with highlighted PSF regions
    - Fine control: Adjust slice position ±pixels around PSF center
    - Template overlay with different colors (orange=template, blue=actual)
    - Group navigation for viewing multiple PSFs
    """
    import matplotlib.patches as patches

    M, N = optimizer.M, optimizer.N
    N_up = int(N * upsampling)
    randomness = getattr(optimizer, 'randomness', 0.0)
    random_seed = getattr(optimizer, 'random_seed', None)

    center_info = compute_psf_centers(
        M=M, overlap_ratio=optimizer.overlap_ratio, center_blend=optimizer.center_blend,
        z_ratio=1.0, N=N_up, output_size=N_up, device=optimizer.device,
        randomness=randomness, random_seed=random_seed
    )
    centers = center_info['centers_pixel'].cpu().numpy()  # [M*M, 2] format: [x, y]

    with torch.no_grad():
        I_opt = optimizer.forward(upsampling=upsampling).cpu().numpy()

    # Get template if available
    template = None
    if hasattr(optimizer, 'total_psfs') and optimizer.total_psfs is not None:
        from scipy.ndimage import zoom as scipy_zoom
        t = optimizer.total_psfs.cpu().numpy()
        if t.shape[0] != N_up:
            template = scipy_zoom(t, upsampling, order=1)
        else:
            template = t

    pitch = N_up / M
    slice_half_width = int(pitch * 0.5)
    num_psfs = M * M
    psfs_per_group = 5
    num_groups = (num_psfs + psfs_per_group - 1) // psfs_per_group

    # Create widgets for slice position control
    style = {'description_width': '100px'}
    layout = widgets.Layout(width='350px')

    w_fine_offset = widgets.IntSlider(
        value=0, min=-int(pitch * 0.3), max=int(pitch * 0.3), step=1,
        description='Fine Adj (px):', style=style, layout=layout,
        continuous_update=False
    )
    w_log_range = widgets.FloatSlider(
        value=4, min=1, max=6, step=0.5,
        description='Log Range:', style=style, layout=widgets.Layout(width='280px'),
        continuous_update=False, readout_format='.1f'
    )
    w_position_label = widgets.HTML(value='<i>Slice at PSF center (adjust with Fine slider)</i>')

    group_selector = widgets.IntSlider(
        value=0, min=0, max=num_groups - 1, description='PSF Group:',
        layout=widgets.Layout(width='400px'), continuous_update=False
    )
    prev_btn = widgets.Button(description='◀ Prev', layout=widgets.Layout(width='80px'))
    next_btn = widgets.Button(description='Next ▶', layout=widgets.Layout(width='80px'))
    reset_btn = widgets.Button(description='Reset', layout=widgets.Layout(width='80px'))

    plot_out = widgets.Output()

    def compute_slices(fine_offset):
        """Compute PSF slices with given offset from center."""
        psf_slices = []
        for i in range(num_psfs):
            # centers format is [x, y], so centers[i, 0]=x, centers[i, 1]=y
            cx, cy = int(centers[i, 0]), int(centers[i, 1])
            # Apply fine offset to y (row) direction for horizontal slice
            cy_actual = cy + fine_offset
            x_start, x_end = max(0, cx - slice_half_width), min(N_up, cx + slice_half_width)

            if 0 <= cy_actual < N_up:
                slice_data = I_opt[cy_actual, x_start:x_end]
                if slice_data.max() > 0:
                    slice_data = slice_data / slice_data.max()
                template_slice = None
                if template is not None and 0 <= cy_actual < template.shape[0]:
                    template_slice = template[cy_actual, x_start:x_end]
                    if template_slice.max() > 0:
                        template_slice = template_slice / template_slice.max()
            else:
                slice_data = np.zeros(max(1, x_end - x_start))
                template_slice = None

            psf_slices.append({
                'data': slice_data, 'template': template_slice,
                'index': i, 'row': i // M, 'col': i % M,
                'cx': cx, 'cy': cy, 'cy_actual': cy_actual,
                'x_start': x_start, 'x_end': x_end
            })
        return psf_slices

    def plot_group(group_idx, fine_offset, log_range):
        plt.close('all')
        psf_slices = compute_slices(fine_offset)

        start_idx = group_idx * psfs_per_group
        end_idx = min((group_idx + 1) * psfs_per_group, num_psfs)
        group_psfs = psf_slices[start_idx:end_idx]
        n_cols = len(group_psfs)

        # Create figure with global view on left, slices on right
        fig = plt.figure(figsize=(5 + 3.5 * n_cols, 6))
        gs = fig.add_gridspec(2, n_cols + 1, width_ratios=[1.5] + [1] * n_cols)

        # Left: Global intensity map (spans both rows)
        ax_global = fig.add_subplot(gs[:, 0])
        vmin = I_opt.max() * (10 ** (-log_range))
        im = ax_global.imshow(I_opt, cmap='hot', norm=mcolors.LogNorm(vmin=vmin, vmax=I_opt.max()))
        ax_global.set_title(f'Global Intensity (log, {log_range:.0f} dec)', fontsize=10)
        ax_global.axis('off')

        # Draw thin dashed rectangles for current group PSFs
        colors = plt.cm.tab10(np.linspace(0, 1, n_cols))
        for j, psf in enumerate(group_psfs):
            # Rectangle around PSF region - thin dashed line
            rect = patches.Rectangle(
                (psf['x_start'], psf['cy_actual'] - 2),
                psf['x_end'] - psf['x_start'], 5,
                linewidth=1, linestyle='--', edgecolor=colors[j], facecolor='none'
            )
            ax_global.add_patch(rect)
            # Label
            ax_global.text(psf['x_start'], psf['cy_actual'] - 5, f"#{psf['index']}",
                          color=colors[j], fontsize=8, fontweight='bold')

        # Right: PSF slices
        for j, psf in enumerate(group_psfs):
            for row, (log_scale, ylabel) in enumerate([(False, 'Intensity'), (True, 'Log Intensity')]):
                ax = fig.add_subplot(gs[row, j + 1])
                x = np.arange(len(psf['data']))

                # Plot template first (background)
                if psf['template'] is not None:
                    if log_scale:
                        ax.semilogy(x, psf['template'] + 1e-6, 'orange', lw=2, alpha=0.8, label='Template')
                    else:
                        ax.fill_between(x, psf['template'], alpha=0.4, color='orange', label='Template')
                        ax.plot(x, psf['template'], 'orange', lw=1.5, alpha=0.8)

                # Plot actual PSF
                if log_scale:
                    ax.semilogy(x, psf['data'] + 1e-6, color=colors[j], lw=1.5, label='Actual')
                else:
                    ax.plot(x, psf['data'], color=colors[j], lw=1.5, label='Actual')
                    ax.fill_between(x, psf['data'], alpha=0.3, color=colors[j])

                if row == 0:
                    title = f"#{psf['index']} [{psf['row']},{psf['col']}]"
                    if fine_offset != 0:
                        title += f"\ny={psf['cy_actual']} (off={fine_offset:+d})"
                    ax.set_title(title, fontsize=9)
                    ax.legend(loc='upper right', fontsize=7)

                ax.set_xlabel('Pixels', fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if not log_scale:
                    ax.set_ylim(0, 1.15)
                else:
                    ax.set_ylim(1e-4, 2)

        offset_text = f" (offset={fine_offset:+d}px)" if fine_offset != 0 else ""
        fig.suptitle(f'PSF Slices - Group {group_idx + 1}/{num_groups}{offset_text}', fontweight='bold')
        plt.tight_layout()
        plt.show()

    def update_all(change=None):
        fine_offset = w_fine_offset.value
        log_range = w_log_range.value
        w_position_label.value = (
            f'<b>Slice offset:</b> {fine_offset:+d} px from PSF center'
            if fine_offset != 0 else '<i>Slice at PSF center</i>'
        )
        with plot_out:
            clear_output(wait=True)
            plot_group(group_selector.value, fine_offset, log_range)

    def on_prev(b):
        if group_selector.value > 0:
            group_selector.value -= 1

    def on_next(b):
        if group_selector.value < num_groups - 1:
            group_selector.value += 1

    def on_reset(b):
        w_fine_offset.value = 0
        group_selector.value = 0

    # Setup callbacks
    group_selector.observe(update_all, names='value')
    w_fine_offset.observe(update_all, names='value')
    w_log_range.observe(update_all, names='value')
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    reset_btn.on_click(on_reset)

    # Display controls
    controls = widgets.VBox([
        widgets.HTML('<h4>PSF Slice Viewer</h4>'),
        widgets.HBox([group_selector, prev_btn, next_btn]),
        widgets.HBox([w_fine_offset, reset_btn, w_log_range]),
        w_position_label
    ])
    display(controls)
    display(plot_out)

    # Initial plot
    update_all()


def show_psf_patches_interactive(optimizer, upsampling: float = 2.0):
    """Show PSF 2D patches with propagation analysis.

    Features:
    - Global intensity map with highlighted patch regions
    - 2D intensity patches for each PSF in a group
    - Propagation slice view (horizontal slice, z propagation)
    - Log scale and range controls
    - Adjustable patch size
    """
    import matplotlib.patches as mpatches

    M, N = optimizer.M, optimizer.N
    N_up = int(N * upsampling)
    pixel_size_up = optimizer.pixel_size / upsampling
    randomness = getattr(optimizer, 'randomness', 0.0)
    random_seed = getattr(optimizer, 'random_seed', None)

    center_info = compute_psf_centers(
        M=M, overlap_ratio=optimizer.overlap_ratio, center_blend=optimizer.center_blend,
        z_ratio=1.0, N=N_up, output_size=N_up, device=optimizer.device,
        randomness=randomness, random_seed=random_seed
    )
    centers = center_info['centers_pixel'].cpu().numpy()  # [M*M, 2] format: [x, y]

    # Precompute focal plane intensity
    with torch.no_grad():
        I_focal = optimizer.forward(upsampling=upsampling).cpu().numpy()

    pitch = N_up / M
    num_psfs = M * M
    psfs_per_group = 4  # Fewer per group since we show more info
    num_groups = (num_psfs + psfs_per_group - 1) // psfs_per_group

    # Create widgets
    style = {'description_width': '60px'}

    group_selector = widgets.IntSlider(
        value=0, min=0, max=num_groups - 1, description='Group:',
        style=style, layout=widgets.Layout(width='280px'), continuous_update=False
    )
    w_patch_size = widgets.FloatSlider(
        value=0.6, min=0.3, max=1.0, step=0.1,
        description='Patch:', style=style, layout=widgets.Layout(width='180px'),
        continuous_update=False, readout_format='.1f'
    )
    w_global_log = widgets.FloatSlider(
        value=4, min=1, max=6, step=0.5,
        description='Global:', style=style, layout=widgets.Layout(width='180px'),
        continuous_update=False, readout_format='.1f'
    )
    w_patch_log = widgets.Checkbox(value=True, indent=False)
    w_patch_range = widgets.FloatSlider(
        value=4, min=1, max=6, step=0.5,
        layout=widgets.Layout(width='120px'),
        continuous_update=False, readout_format='.1f'
    )
    w_z_range = widgets.FloatSlider(
        value=2.0, min=0.5, max=5.0, step=0.5,
        description='Z(DOF):', style=style, layout=widgets.Layout(width='180px'),
        continuous_update=False, readout_format='.1f'
    )
    prev_btn = widgets.Button(description='◀', layout=widgets.Layout(width='40px'))
    next_btn = widgets.Button(description='▶', layout=widgets.Layout(width='40px'))

    plot_out = widgets.Output()

    def compute_propagation_for_psf(cx, cy, z_values, half_width):
        """Compute propagation map for a single PSF location."""
        n_z = len(z_values)
        width = 2 * half_width
        intensity_map = np.zeros((n_z, width))

        with torch.no_grad():
            for i, z in enumerate(z_values):
                I = optimizer.forward(z=z, upsampling=upsampling).cpu().numpy()
                if 0 <= cy < I.shape[0]:
                    x_start = max(0, cx - half_width)
                    x_end = min(I.shape[1], cx + half_width)
                    slice_data = I[cy, x_start:x_end]
                    # Pad if needed
                    if len(slice_data) < width:
                        padded = np.zeros(width)
                        offset = (width - len(slice_data)) // 2
                        padded[offset:offset+len(slice_data)] = slice_data
                        slice_data = padded
                    intensity_map[i, :len(slice_data)] = slice_data[:width]
        return intensity_map

    def plot_group(group_idx, global_log, patch_log, patch_range, z_range_dof, patch_ratio):
        plt.close('all')

        patch_half = int(pitch * patch_ratio)
        start_idx = group_idx * psfs_per_group
        end_idx = min((group_idx + 1) * psfs_per_group, num_psfs)
        n_cols = end_idx - start_idx

        # Figure layout: Global view + n_cols columns for PSFs
        fig, axes = plt.subplots(2, n_cols + 1, figsize=(3.5 + 3 * n_cols, 7),
                                gridspec_kw={'width_ratios': [1.3] + [1] * n_cols})

        # Global intensity map (spans both rows visually by making row 1 empty)
        ax_global = axes[0, 0]
        axes[1, 0].axis('off')  # Hide bottom-left
        vmin_global = I_focal.max() * (10 ** (-global_log))
        ax_global.imshow(I_focal, cmap='hot',
                        norm=mcolors.LogNorm(vmin=vmin_global, vmax=I_focal.max()))
        ax_global.set_title(f'Global ({global_log:.0f} dec)', fontsize=9)
        ax_global.axis('off')

        # Draw patch boxes for current group
        colors = plt.cm.tab10(np.linspace(0, 1, n_cols))
        group_psfs = []
        for j in range(n_cols):
            idx = start_idx + j
            cx, cy = int(centers[idx, 0]), int(centers[idx, 1])
            x_start = max(0, cx - patch_half)
            y_start = max(0, cy - patch_half)
            x_end = min(N_up, cx + patch_half)
            y_end = min(N_up, cy + patch_half)

            group_psfs.append({
                'idx': idx, 'cx': cx, 'cy': cy,
                'x_start': x_start, 'x_end': x_end,
                'y_start': y_start, 'y_end': y_end,
                'row': idx // M, 'col': idx % M
            })

            # Draw dashed rectangle
            rect = mpatches.Rectangle(
                (x_start, y_start), x_end - x_start, y_end - y_start,
                linewidth=1.5, linestyle='--', edgecolor=colors[j], facecolor='none'
            )
            ax_global.add_patch(rect)
            ax_global.text(x_start, y_start - 3, f"#{idx}",
                          color=colors[j], fontsize=9, fontweight='bold')

        # Prepare z values for propagation
        focal_dist = optimizer.focal_length
        dof = optimizer.depth_of_focus
        z_min = focal_dist - z_range_dof * dof
        z_max = focal_dist + z_range_dof * dof
        n_z = 30
        z_values = torch.linspace(z_min, z_max, n_z)
        z_relative_mm = (z_values - focal_dist).numpy() * 1e3

        # Plot each PSF
        for j, psf in enumerate(group_psfs):
            # Row 0: 2D patch
            ax_patch = axes[0, j + 1]
            patch_data = I_focal[psf['y_start']:psf['y_end'], psf['x_start']:psf['x_end']]
            if patch_data.size > 0:
                if patch_log:
                    vmin_patch = I_focal.max() * (10 ** (-patch_range))
                    ax_patch.imshow(patch_data, cmap='hot',
                                   norm=mcolors.LogNorm(vmin=vmin_patch, vmax=I_focal.max()))
                else:
                    ax_patch.imshow(patch_data, cmap='hot')
            scale_text = f"log{patch_range:.0f}" if patch_log else "lin"
            ax_patch.set_title(f"#{psf['idx']} [{psf['row']},{psf['col']}]", fontsize=9, color=colors[j])
            ax_patch.axis('off')

            # Row 1: Propagation slice (always log for visibility)
            ax_prop = axes[1, j + 1]
            prop_map = compute_propagation_for_psf(psf['cx'], psf['cy'], z_values, patch_half)

            # Normalize and apply log
            prop_max = prop_map.max() if prop_map.max() > 0 else 1
            prop_norm = prop_map / prop_max
            prop_log_data = np.log10(prop_norm + 10**(-patch_range))
            prop_log_data = np.clip(prop_log_data, -patch_range, 0)

            extent = [-patch_half * pixel_size_up * 1e3, patch_half * pixel_size_up * 1e3,
                     z_relative_mm[-1], z_relative_mm[0]]
            ax_prop.imshow(prop_log_data, cmap='hot', aspect='auto', extent=extent)
            ax_prop.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.8)
            # DOF markers
            for mult in [0.5, 1.0]:
                ax_prop.axhline(y=mult * dof * 1e3, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
                ax_prop.axhline(y=-mult * dof * 1e3, color='white', linestyle='--', linewidth=0.5, alpha=0.5)

            ax_prop.set_xlabel('X (mm)', fontsize=8)
            if j == 0:
                ax_prop.set_ylabel('Δz (mm)', fontsize=8)
            ax_prop.tick_params(labelsize=7)

        scale_label = f"log{patch_range:.0f}" if patch_log else "linear"
        fig.suptitle(f'PSF Patches - Group {group_idx + 1}/{num_groups} (patch={patch_ratio:.1f}, {scale_label})',
                    fontweight='bold', fontsize=11)
        plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.15)
        plt.show()

    def update_all(change=None):
        with plot_out:
            clear_output(wait=True)
            plot_group(group_selector.value, w_global_log.value, w_patch_log.value,
                      w_patch_range.value, w_z_range.value, w_patch_size.value)

    def on_prev(b):
        if group_selector.value > 0:
            group_selector.value -= 1

    def on_next(b):
        if group_selector.value < num_groups - 1:
            group_selector.value += 1

    # Setup callbacks
    group_selector.observe(update_all, names='value')
    w_global_log.observe(update_all, names='value')
    w_patch_log.observe(update_all, names='value')
    w_patch_range.observe(update_all, names='value')
    w_z_range.observe(update_all, names='value')
    w_patch_size.observe(update_all, names='value')
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)

    # Display controls with HTML labels
    controls = widgets.VBox([
        widgets.HTML('<h4>PSF Patches Viewer</h4>'),
        widgets.HBox([group_selector, prev_btn, next_btn, w_patch_size]),
        widgets.HBox([
            widgets.HTML('<span style="margin-right:3px">Global:</span>'), w_global_log,
            widgets.HTML('<span style="margin:0 8px">Patch Log:</span>'), w_patch_log, w_patch_range,
            w_z_range
        ]),
    ])
    display(controls)
    display(plot_out)

    # Initial plot
    update_all()


def plot_atf_interactive(optimizer, atf: np.ndarray, pixel_size: float):
    """Interactive ATF visualization with lenslet selector."""
    M = optimizer.M
    extent_value = 1 / (pixel_size * 1e3)

    def plot_atf_single(row=0, col=0):
        plt.close('all')
        extent_range = [0, extent_value, 0, extent_value]
        idx = row * M + col
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        intensity = np.abs(atf[idx]).T
        ax1.imshow(np.angle(atf[idx].T), cmap='hsv', alpha=intensity / intensity.max(), extent=extent_range, origin='lower')
        ax1.set_title(f'Phase [{row},{col}]'); ax1.set_xlabel('X (mm^-1)'); ax1.set_ylabel('Y (mm^-1)')
        ax2.imshow(np.abs(atf[idx].T), extent=extent_range, origin='lower', cmap='viridis')
        ax2.set_title(f'Amplitude [{row},{col}]'); ax2.set_xlabel('X (mm^-1)'); ax2.set_ylabel('Y (mm^-1)')
        plt.tight_layout(); plt.show()

    from ipywidgets import interact, IntSlider
    interact(plot_atf_single, row=IntSlider(min=0, max=M-1, value=0, description='Row:'),
             col=IntSlider(min=0, max=M-1, value=0, description='Col:'))


def compute_and_save_atf(optimizer, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Compute ATF from optimizer and optionally save to .mat file."""
    N = optimizer.N * 5
    pixel_size = optimizer.pixel_size / 5
    atf = optimizer.compute_transfer_function(N=N, pixel_size=pixel_size, verbose=False)
    result = {
        'atf': atf, 'pixel_size': pixel_size, 'wavelength': optimizer.wavelength,
        'focal_length': optimizer.focal_length, 'M': optimizer.M, 'spot_radius': optimizer.airy_radius
    }
    if output_path:
        import scipy.io as sio
        sio.savemat(output_path, result)
        print(f"ATF saved: {output_path}")
    return result
