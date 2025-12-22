"""
Phase Optimizer GUI for Jupyter Notebook
Provides interactive widgets for configuring and managing optimization jobs.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import copy
from datetime import datetime
import numpy as np

class PhaseOptimizerGUI:
    """
    GUI class for managing phase optimization jobs in Jupyter notebook.
    """
    
    def __init__(self, default_params=None):
        """
        Initialize the GUI with optional default parameters.
        
        Args:
            default_params: dict, optional default parameters from config
        """
        self.default_params = default_params or {}
        self.job_list = []  # List of (job_title, params_dict) tuples
        self.custom_prefix = ''  # Store custom prefix
        
        self._create_widgets()
        self._setup_layout()
        self._setup_callbacks()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Common parameters
        self.w_M = widgets.IntText(
            value=5, description='M:', 
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        self.w_focal_length = widgets.FloatText(
            value=73.9, description='Focal (mm):', 
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        self.w_two_pi_value = widgets.IntText(
            value=210, description='2π Value:', 
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        self.w_N = widgets.IntText(
            value=850, description='N (ROI size):', 
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        # Mode selection
        self.w_mode = widgets.Dropdown(
            options=['fresnel', 'optimized'],
            value='fresnel',
            description='Mode:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        # Optimized mode specific parameters
        self.w_overlap_ratio = widgets.FloatText(
            value=0.25, description='Overlap Ratio:', 
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        # DOF info display (shown when overlap_ratio > 0)
        self.w_dof_info = widgets.HTML(
            value='',
            layout=widgets.Layout(width='400px')
        )
        
        self.w_airy_correction = widgets.FloatText(
            value=1.0, description='Airy Correction:', 
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        # Depth in focus - using Textarea for flexible list input
        self.w_depth_label = widgets.HTML(
            value='<b>Depth in Focus:</b> (comma-separated values)',
            layout=widgets.Layout(width='300px')
        )
        
        self.w_depth_in_focus = widgets.Textarea(
            value='-2.5, -1.5, -1, -0.5, 0.5, 1, 1.5, 2.5',
            placeholder='e.g., -2.5, -1.5, -1, -0.5, 0.5, 1, 1.5, 2.5',
            layout=widgets.Layout(width='400px', height='60px')
        )
        
        # === Depth Quick Set Buttons ===
        self.w_depth_default_btn = widgets.Button(
            description='Default [-0.5, 0.5]',
            button_style='info',
            layout=widgets.Layout(width='140px')
        )
        
        self.w_depth_remove_zero_btn = widgets.Button(
            description='Remove 0',
            button_style='warning',
            layout=widgets.Layout(width='100px')
        )
        
        # Range generator inputs
        self.w_depth_range = widgets.FloatText(
            value=2.0, description='Range ±:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='120px')
        )
        
        self.w_depth_layers = widgets.IntText(
            value=5, description='Layers:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='110px')
        )
        
        self.w_depth_generate_btn = widgets.Button(
            description='Generate',
            button_style='success',
            icon='cogs',
            layout=widgets.Layout(width='100px')
        )
        
        # Container for optimized mode widgets
        self.depth_quick_buttons = widgets.HBox([
            self.w_depth_default_btn,
            self.w_depth_remove_zero_btn
        ])
        
        self.depth_range_generator = widgets.HBox([
            self.w_depth_range,
            self.w_depth_layers,
            self.w_depth_generate_btn
        ])
        
        self.optimized_params_box = widgets.VBox([
            self.w_overlap_ratio,
            self.w_dof_info,  # Add DOF info display here
            self.w_airy_correction,
            self.w_depth_label,
            self.w_depth_in_focus,
            self.depth_quick_buttons,
            self.depth_range_generator
        ])
        
        # === Job Title Section ===
        self.w_job_title = widgets.Text(
            value='M5_fresnel_F73.9',
            description='Job Title:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Custom prefix input
        self.w_custom_prefix = widgets.Text(
            value='',
            description='Prefix:',
            placeholder='Optional custom prefix',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='200px')
        )
        
        # Template buttons
        self.w_template1_btn = widgets.Button(
            description='Template 1',
            button_style='info',
            tooltip='Basic: M{M}_mode_F{focal}',
            layout=widgets.Layout(width='100px')
        )
        
        self.w_template2_btn = widgets.Button(
            description='Template 2',
            button_style='info',
            tooltip='With date: Template1_MMDD',
            layout=widgets.Layout(width='100px')
        )
        
        # Buttons
        self.w_add_btn = widgets.Button(
            description='Add to Job List',
            button_style='primary',
            icon='plus',
            layout=widgets.Layout(width='150px')
        )
        
        self.w_remove_btn = widgets.Button(
            description='Remove Selected',
            button_style='danger',
            icon='trash',
            layout=widgets.Layout(width='150px')
        )
        
        self.w_clear_btn = widgets.Button(
            description='Clear All',
            button_style='warning',
            icon='times',
            layout=widgets.Layout(width='150px')
        )
        
        # Job list display
        self.w_job_select = widgets.Select(
            options=[],
            description='Jobs:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='400px', height='150px')
        )
        
        # Job details display
        self.w_job_details = widgets.Output(
            layout=widgets.Layout(
                width='400px', 
                height='200px', 
                border='1px solid #ccc',
                overflow='auto'
            )
        )
        
        # Status output
        self.w_status = widgets.Output()
    
    def _setup_layout(self):
        """Setup the widget layout."""
        # Common parameters section
        common_section = widgets.VBox([
            widgets.HTML('<h3>Common Parameters</h3>'),
            widgets.HBox([self.w_M, self.w_focal_length]),
            widgets.HBox([self.w_two_pi_value, self.w_N]),
            self.w_mode
        ])
        
        # Optimized parameters section (initially hidden)
        self.optimized_section = widgets.VBox([
            widgets.HTML('<h3>Optimized Mode Parameters</h3>'),
            self.optimized_params_box
        ])
        self.optimized_section.layout.display = 'none'
        
        # Job title and buttons with templates
        job_controls = widgets.VBox([
            widgets.HTML('<h3>Job Configuration</h3>'),
            widgets.HBox([self.w_custom_prefix, self.w_template1_btn, self.w_template2_btn]),
            self.w_job_title,
            widgets.HBox([self.w_add_btn, self.w_remove_btn, self.w_clear_btn])
        ])
        
        # Job list section
        job_list_section = widgets.VBox([
            widgets.HTML('<h3>Job List</h3>'),
            self.w_job_select,
            widgets.HTML('<b>Job Details:</b>'),
            self.w_job_details
        ])
        
        # Left panel (parameters)
        left_panel = widgets.VBox([
            common_section,
            self.optimized_section,
            job_controls
        ])
        
        # Right panel (job list)
        right_panel = job_list_section
        
        # Main layout
        self.main_layout = widgets.VBox([
            widgets.HBox([left_panel, right_panel]),
            self.w_status
        ])
    
    def _setup_callbacks(self):
        """Setup widget callbacks."""
        self.w_mode.observe(self._on_mode_change, names='value')
        self.w_M.observe(self._update_default_title, names='value')
        self.w_M.observe(self._update_dof_info, names='value')  # Also update DOF info
        self.w_N.observe(self._update_dof_info, names='value')  # Also update DOF info
        self.w_focal_length.observe(self._update_default_title, names='value')
        self.w_airy_correction.observe(self._update_default_title, names='value')
        self.w_depth_in_focus.observe(self._update_default_title, names='value')
        
        # Overlap ratio callback for DOF info
        self.w_overlap_ratio.observe(self._update_dof_info, names='value')
        self.w_overlap_ratio.observe(self._update_default_title, names='value')
        
        # Custom prefix callback
        self.w_custom_prefix.observe(self._on_prefix_change, names='value')
        
        self.w_add_btn.on_click(self._on_add_job)
        self.w_remove_btn.on_click(self._on_remove_job)
        self.w_clear_btn.on_click(self._on_clear_jobs)
        self.w_job_select.observe(self._on_job_select, names='value')
        
        # Depth quick buttons callbacks
        self.w_depth_default_btn.on_click(self._on_depth_default)
        self.w_depth_remove_zero_btn.on_click(self._on_depth_remove_zero)
        self.w_depth_generate_btn.on_click(self._on_depth_generate)
        
        # Template buttons callbacks
        self.w_template1_btn.on_click(self._on_template1)
        self.w_template2_btn.on_click(self._on_template2)
    
    def _update_dof_info(self, change):
        """Update DOF info display based on overlap ratio."""
        overlap_ratio = self.w_overlap_ratio.value
        
        if overlap_ratio > 0:
            M = self.w_M.value
            N = self.w_N.value
            
            if M > 0:
                # Calculate α = [overlap_ratio * (N - N/M) + N/M] / (N/M)
                # Simplified: α = overlap_ratio * (M - 1) + 1
                base_size = N / M
                extended_size = overlap_ratio * (N - N / M) + N / M
                alpha = extended_size / base_size
                
                fresnel_dof = alpha ** 2
                dof_reduction = 1 / (alpha ** 2)
                
                fresnel_airy = alpha
                airy_reduction = 1 / alpha
                
                self.w_dof_info.value = (
                    f'<div style="background-color: #e7f3ff; padding: 8px; '
                    f'border-radius: 4px; border-left: 3px solid #2196F3; margin: 5px 0;">'
                    f'<b>α = {alpha:.3f}</b><br>'
                    f'<b>Fresnel Airy:</b> α = {fresnel_airy:.3f} &nbsp;|&nbsp; '
                    f'<b>Airy reduction:</b> 1/α = {airy_reduction:.3f}<br>'
                    f'<b>Fresnel DOF:</b> α² = {fresnel_dof:.3f} &nbsp;|&nbsp; '
                    f'<b>DOF reduction:</b> 1/α² = {dof_reduction:.3f}'
                    f'</div>'
                )
            else:
                self.w_dof_info.value = ''
        else:
            self.w_dof_info.value = ''
    
    def _on_mode_change(self, change):
        """Handle mode selection change."""
        if change['new'] == 'optimized':
            self.optimized_section.layout.display = 'flex'
            # Update DOF info when switching to optimized mode
            self._update_dof_info(None)
        else:
            self.optimized_section.layout.display = 'none'
        self._update_default_title(None)
    
    def _on_prefix_change(self, change):
        """Handle custom prefix change - store it for persistence."""
        self.custom_prefix = change['new']
        self._update_default_title(None)
    
    def _get_base_title(self):
        """Get the base title without prefix."""
        M = self.w_M.value
        focal = self.w_focal_length.value
        mode = self.w_mode.value
        
        if mode == 'fresnel':
            return f'M{M}_fresnel_F{focal}'
        else:
            airy = self.w_airy_correction.value
            overlap = self.w_overlap_ratio.value
            depth_list = self._parse_depth_list()
            if depth_list:
                depth_range = (max(depth_list) - min(depth_list))
                return f'M{M}_depth_range_{depth_range}x_airy{airy}_over{overlap}'
            else:
                return f'M{M}_airy{airy}_over{overlap}_optimized'
    
    def _update_default_title(self, change):
        """Update default job title based on parameters."""
        base_title = self._get_base_title()
        
        # Apply custom prefix if set
        if self.custom_prefix:
            self.w_job_title.value = f'{self.custom_prefix}_{base_title}'
        else:
            self.w_job_title.value = base_title
    
    def _on_template1(self, btn):
        """Apply template 1: basic naming."""
        base_title = self._get_base_title()
        if self.custom_prefix:
            self.w_job_title.value = f'{self.custom_prefix}_{base_title}'
        else:
            self.w_job_title.value = base_title
    
    def _on_template2(self, btn):
        """Apply template 2: basic naming + date suffix."""
        base_title = self._get_base_title()
        date_suffix = datetime.now().strftime('%m%d')
        if self.custom_prefix:
            self.w_job_title.value = f'{self.custom_prefix}_{base_title}_{date_suffix}'
        else:
            self.w_job_title.value = f'{base_title}_{date_suffix}'
    
    def _on_depth_default(self, btn):
        """Reset depth to default [-0.5, 0.5]."""
        self.w_depth_in_focus.value = '-0.5, 0.5'
        with self.w_status:
            clear_output()
            print("✅ Depth reset to default: [-0.5, 0.5]")
    
    def _on_depth_remove_zero(self, btn):
        """Remove 0 from the depth list."""
        depth_list = self._parse_depth_list()
        if not depth_list:
            with self.w_status:
                clear_output()
                print("⚠️ Invalid depth list format.")
            return
        
        # Remove zeros (with tolerance for floating point)
        filtered = [x for x in depth_list if abs(x) > 1e-9]
        self.w_depth_in_focus.value = ', '.join(str(x) for x in filtered)
        
        with self.w_status:
            clear_output()
            if len(filtered) < len(depth_list):
                print(f"✅ Removed 0 from depth list. New list: {filtered}")
            else:
                print("ℹ️ No 0 found in the list.")
    
    def _on_depth_generate(self, btn):
        """Generate depth list from range and layers."""
        range_val = self.w_depth_range.value
        layers = self.w_depth_layers.value
        
        if layers < 2:
            with self.w_status:
                clear_output()
                print("⚠️ Layers must be at least 2.")
            return
        
        if range_val <= 0:
            with self.w_status:
                clear_output()
                print("⚠️ Range must be positive.")
            return
        
        # Generate evenly spaced values from -range to +range
        depth_list = np.linspace(-range_val, range_val, layers).tolist()
        # Round to avoid floating point artifacts
        depth_list = [round(x, 4) for x in depth_list]
        
        self.w_depth_in_focus.value = ', '.join(str(x) for x in depth_list)
        
        with self.w_status:
            clear_output()
            print(f"✅ Generated depth list: {depth_list}")
    
    def _parse_depth_list(self):
        """Parse depth in focus text to list of floats."""
        try:
            text = self.w_depth_in_focus.value.strip()
            if not text:
                return []
            values = [float(x.strip()) for x in text.split(',')]
            return values
        except ValueError:
            return []
    
    def _get_current_params(self):
        """Get current parameters as a dictionary."""
        params = copy.deepcopy(self.default_params)
        
        params['M'] = self.w_M.value
        params['focal_length'] = self.w_focal_length.value * 1e-3  # Convert mm to m
        params['two_pi_value'] = self.w_two_pi_value.value
        params['N'] = self.w_N.value
        params['output_size'] = self.w_N.value
        params['mode'] = self.w_mode.value
        
        if self.w_mode.value == 'optimized':
            params['overlap_ratio'] = self.w_overlap_ratio.value
            params['airy_correction'] = self.w_airy_correction.value
            params['depth_in_focus'] = self._parse_depth_list()
            params['mask_count'] = 0
        
        return params
    
    def _on_add_job(self, btn):
        """Add current configuration to job list."""
        job_title = self.w_job_title.value.strip()
        
        if not job_title:
            with self.w_status:
                clear_output()
                print("⚠️ Please enter a job title.")
            return
        
        # Check for duplicate titles
        existing_titles = [j[0] for j in self.job_list]
        if job_title in existing_titles:
            with self.w_status:
                clear_output()
                print(f"⚠️ Job '{job_title}' already exists. Please use a different name.")
            return
        
        params = self._get_current_params()
        self.job_list.append((job_title, params))
        
        self._update_job_list_display()
        
        with self.w_status:
            clear_output()
            print(f"✅ Added job: {job_title}")
    
    def _on_remove_job(self, btn):
        """Remove selected job from list."""
        selected = self.w_job_select.value
        if selected:
            self.job_list = [(t, p) for t, p in self.job_list if t != selected]
            self._update_job_list_display()
            with self.w_status:
                clear_output()
                print(f"🗑️ Removed job: {selected}")
    
    def _on_clear_jobs(self, btn):
        """Clear all jobs from list."""
        self.job_list = []
        self._update_job_list_display()
        with self.w_status:
            clear_output()
            print("🧹 Cleared all jobs.")
    
    def _update_job_list_display(self):
        """Update the job list widget."""
        titles = [j[0] for j in self.job_list]
        self.w_job_select.options = titles
        if titles:
            self.w_job_select.value = titles[-1]
    
    def _on_job_select(self, change):
        """Handle job selection - show details and load parameters."""
        selected = change['new']
        if not selected:
            self.w_job_details.clear_output()
            return
        
        # Find the job params
        params = None
        for title, p in self.job_list:
            if title == selected:
                params = p
                break
        
        if params is None:
            return
        
        # Display job details
        with self.w_job_details:
            clear_output()
            print(f"📋 Job: {selected}")
            print("-" * 40)
            print(f"Mode: {params.get('mode', 'unknown')}")
            print(f"M: {params.get('M')}")
            print(f"Focal Length: {params.get('focal_length', 0) * 1000:.1f} mm")
            print(f"2π Value: {params.get('two_pi_value')}")
            print(f"N: {params.get('N')}")
            
            if params.get('mode') == 'optimized':
                print(f"Overlap Ratio: {params.get('overlap_ratio')}")
                print(f"Airy Correction: {params.get('airy_correction')}")
                print(f"Depth in Focus: {params.get('depth_in_focus')}")
        
        # Load parameters into widgets
        self._load_params_to_widgets(params)
    
    def _load_params_to_widgets(self, params):
        """Load parameters from dict to widgets."""
        self.w_M.value = params.get('M', 5)
        self.w_focal_length.value = params.get('focal_length', 73.9e-3) * 1000  # m to mm
        self.w_two_pi_value.value = params.get('two_pi_value', 210)
        self.w_N.value = params.get('N', 850)
        self.w_mode.value = params.get('mode', 'fresnel')
        
        if params.get('mode') == 'optimized':
            self.w_overlap_ratio.value = params.get('overlap_ratio', 0.25)
            self.w_airy_correction.value = params.get('airy_correction', 1.0)
            depth_list = params.get('depth_in_focus', [])
            self.w_depth_in_focus.value = ', '.join(str(x) for x in depth_list)
    
    def display(self):
        """Display the GUI."""
        display(self.main_layout)
    
    def get_job_list(self):
        """
        Get the current job list for processing.
        
        Returns:
            list: List of (job_title, params_dict) tuples
        """
        return copy.deepcopy(self.job_list)


def create_optimizer_gui(default_params=None):
    """
    Create and display the optimizer GUI.
    
    Args:
        default_params: dict, optional default parameters from config
    
    Returns:
        PhaseOptimizerGUI: The GUI instance for accessing job list
    """
    gui = PhaseOptimizerGUI(default_params)
    gui.display()
    return gui