"""
Phase Optimizer GUI for Jupyter Notebook
Provides interactive widgets for configuring and managing optimization jobs.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import copy
import os
import json
from datetime import datetime
import numpy as np

class PhaseOptimizerGUI:
    """
    GUI class for managing phase optimization jobs in Jupyter notebook.
    """
    
    def __init__(self, default_params=None, output_dir='./output'):
        """
        Initialize the GUI with optional default parameters.

        Args:
            default_params: dict, optional default parameters from config
            output_dir: str, directory containing existing recipe outputs
        """
        self.default_params = default_params or {}
        self.output_dir = os.path.normpath(output_dir)
        self.job_list = []  # List of (job_title, params_dict) tuples
        self.recipe_list = []  # List of existing recipes from output folder
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

        # Randomness parameters for PSF target variation
        self.w_randomness = widgets.FloatText(
            value=0.0, description='Randomness:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )

        self.w_use_random_seed = widgets.Checkbox(
            value=False, description='Fixed:',
            indent=False,
            layout=widgets.Layout(width='60px')
        )

        self.w_random_seed = widgets.IntText(
            value=42, description='Seed:',
            style={'description_width': '35px'},
            layout=widgets.Layout(width='100px')
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
        
        # Randomness controls row
        self.randomness_controls = widgets.HBox([
            self.w_randomness,
            self.w_use_random_seed,
            self.w_random_seed
        ])

        self.optimized_params_box = widgets.VBox([
            self.w_overlap_ratio,
            self.w_dof_info,  # Add DOF info display here
            self.w_airy_correction,
            self.randomness_controls,
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
        
        # Job list header (dynamic)
        self.w_job_list_header = widgets.HTML(
            value='<h3>Job List (0 jobs)</h3>'
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
        
        # === Existing Recipe Browser ===
        self.w_scan_recipes_btn = widgets.Button(
            description='Scan Recipes',
            button_style='info',
            icon='search',
            layout=widgets.Layout(width='130px')
        )

        self.w_recipe_select = widgets.SelectMultiple(
            options=[],
            description='Recipes:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='350px', height='120px')
        )

        self.w_load_recipe_btn = widgets.Button(
            description='Load',
            button_style='success',
            icon='download',
            tooltip='Load first selected recipe params to GUI',
            layout=widgets.Layout(width='80px'),
            disabled=True
        )

        self.w_randomize_btn = widgets.Button(
            description='+ Randomize',
            button_style='primary',
            icon='random',
            tooltip='Add selected recipes to Job List with current Randomness params',
            layout=widgets.Layout(width='120px'),
            disabled=True
        )

        self.w_recipe_info = widgets.HTML(
            value='<i>Click "Scan Recipes" to find existing recipes</i>',
            layout=widgets.Layout(width='350px')
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
            self.w_job_list_header,
            self.w_job_select,
            widgets.HTML('<b>Job Details:</b>'),
            self.w_job_details
        ])

        # Existing recipe browser section
        recipe_browser_section = widgets.VBox([
            widgets.HTML('<h3>Load Existing Recipe</h3>'),
            widgets.HBox([self.w_scan_recipes_btn, self.w_load_recipe_btn, self.w_randomize_btn]),
            self.w_recipe_select,
            self.w_recipe_info
        ])

        # Left panel (parameters)
        left_panel = widgets.VBox([
            common_section,
            self.optimized_section,
            job_controls
        ])

        # Right panel (job list + recipe browser)
        right_panel = widgets.VBox([
            job_list_section,
            recipe_browser_section
        ])
        
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

        # Randomness callback for title update
        self.w_randomness.observe(self._update_default_title, names='value')
        
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

        # Recipe browser callbacks
        self.w_scan_recipes_btn.on_click(self._on_scan_recipes)
        self.w_load_recipe_btn.on_click(self._on_load_recipe)
        self.w_randomize_btn.on_click(self._on_randomize_recipes)
        self.w_recipe_select.observe(self._on_recipe_select, names='value')

    def _scan_recipes(self):
        """Scan output directory for existing recipes."""
        self.recipe_list = []

        if not os.path.exists(self.output_dir):
            return

        for item in sorted(os.listdir(self.output_dir)):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                json_file = os.path.join(item_path, f'{item}.json')
                npy_file = os.path.join(item_path, f'{item}.npy')

                has_json = os.path.exists(json_file)
                has_npy = os.path.exists(npy_file)

                if has_json or has_npy:
                    self.recipe_list.append({
                        'title': item,
                        'path': item_path,
                        'has_params': has_json,
                        'has_phase': has_npy,
                        'json_path': json_file if has_json else None
                    })

    def _on_scan_recipes(self, btn):
        """Handle scan recipes button click."""
        self._scan_recipes()

        # Update recipe selector
        if self.recipe_list:
            recipe_options = [
                (f"{'✓' if r['has_params'] else '○'} {r['title']}", r['title'])
                for r in self.recipe_list
            ]
            self.w_recipe_select.options = recipe_options
            self.w_recipe_info.value = f'<span style="color: green;">Found {len(self.recipe_list)} recipe(s)</span>'
        else:
            self.w_recipe_select.options = []
            self.w_recipe_info.value = f'<span style="color: orange;">No recipes found in {self.output_dir}</span>'

        with self.w_status:
            clear_output()
            print(f"Scanned {self.output_dir}: found {len(self.recipe_list)} recipe(s)")

    def _on_recipe_select(self, change):
        """Handle recipe selection change (supports multi-select)."""
        selected = change['new']  # tuple of selected values
        if not selected:
            self.w_load_recipe_btn.disabled = True
            self.w_randomize_btn.disabled = True
            self.w_recipe_info.value = '<i>Select recipe(s) to load or randomize</i>'
            return

        # Count selected recipes with params
        selected_with_params = []
        for title in selected:
            for r in self.recipe_list:
                if r['title'] == title and r['has_params']:
                    selected_with_params.append(r)
                    break

        # Enable buttons based on selection
        self.w_load_recipe_btn.disabled = len(selected_with_params) == 0
        self.w_randomize_btn.disabled = len(selected_with_params) == 0

        # Show info
        if len(selected) == 1:
            recipe = selected_with_params[0] if selected_with_params else None
            if recipe:
                info_parts = [f"<b>{recipe['title']}</b>"]
                info_parts.append('<span style="color: green;">Has params</span>')
                self.w_recipe_info.value = ' | '.join(info_parts)
            else:
                self.w_recipe_info.value = f'<span style="color: orange;">{selected[0]}: No params file</span>'
        else:
            self.w_recipe_info.value = f'<b>{len(selected)} selected</b> ({len(selected_with_params)} with params)'

    def _on_load_recipe(self, btn):
        """Load first selected recipe parameters into GUI."""
        selected = self.w_recipe_select.value
        if not selected:
            return

        # Find the first recipe with params
        recipe = None
        for title in selected:
            for r in self.recipe_list:
                if r['title'] == title and r['has_params']:
                    recipe = r
                    break
            if recipe:
                break

        if not recipe or not recipe['json_path']:
            with self.w_status:
                clear_output()
                print(f"Cannot load recipe: no JSON file found")
            return

        # Load parameters from JSON
        try:
            with open(recipe['json_path'], 'r') as f:
                params = json.load(f)

            # Load parameters to widgets
            self._load_params_to_widgets(params)

            with self.w_status:
                clear_output()
                print(f"Loaded recipe: {recipe['title']}")
                print(f"Mode: {params.get('mode', 'unknown')}, M: {params.get('M')}")

        except Exception as e:
            with self.w_status:
                clear_output()
                print(f"Error loading recipe: {e}")

    def _on_randomize_recipes(self, btn):
        """Add selected recipes to job list with current randomness params."""
        selected = self.w_recipe_select.value
        if not selected:
            return

        # Get current randomness params from GUI
        randomness = self.w_randomness.value
        random_seed = self.w_random_seed.value if self.w_use_random_seed.value else None

        added_count = 0
        skipped_count = 0

        for title in selected:
            # Find the recipe
            recipe = None
            for r in self.recipe_list:
                if r['title'] == title and r['has_params']:
                    recipe = r
                    break

            if not recipe or not recipe['json_path']:
                skipped_count += 1
                continue

            try:
                # Load original params
                with open(recipe['json_path'], 'r') as f:
                    params = json.load(f)

                # Update randomness params only
                params['randomness'] = randomness
                params['random_seed'] = random_seed

                # Generate new job title with randomness suffix
                if randomness > 0:
                    new_title = f"{recipe['title']}_rand{randomness}"
                else:
                    new_title = f"{recipe['title']}_rand0"

                # Check for duplicate titles
                existing_titles = [j[0] for j in self.job_list]
                if new_title in existing_titles:
                    # Add suffix to make unique
                    suffix = 1
                    while f"{new_title}_{suffix}" in existing_titles:
                        suffix += 1
                    new_title = f"{new_title}_{suffix}"

                # Add to job list
                self.job_list.append((new_title, params))
                added_count += 1

            except Exception as e:
                skipped_count += 1
                continue

        # Update display
        self._update_job_list_display()

        with self.w_status:
            clear_output()
            print(f"Added {added_count} job(s) with randomness={randomness}, seed={random_seed}")
            if skipped_count > 0:
                print(f"Skipped {skipped_count} recipe(s) (no params file)")

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
            randomness = self.w_randomness.value
            depth_list = self._parse_depth_list()
            if depth_list:
                depth_range = (max(depth_list) - min(depth_list))
                base = f'M{M}_depth_range_{depth_range}x_airy{airy}_over{overlap}'
            else:
                base = f'M{M}_airy{airy}_over{overlap}_optimized'
            # Add randomness suffix if enabled
            if randomness > 0:
                base += f'_rand{randomness}'
            return base
    
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
            params['randomness'] = self.w_randomness.value
            params['random_seed'] = self.w_random_seed.value if self.w_use_random_seed.value else None

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
        # Update header with job count
        count = len(self.job_list)
        self.w_job_list_header.value = f'<h3>Job List ({count} job{"s" if count != 1 else ""})</h3>'
    
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
                print(f"Randomness: {params.get('randomness', 0.0)}")
                seed = params.get('random_seed')
                print(f"Random Seed: {seed if seed is not None else 'None (random)'}")
        
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
            self.w_randomness.value = params.get('randomness', 0.0)
            seed = params.get('random_seed')
            if seed is not None:
                self.w_random_seed.value = seed
                self.w_use_random_seed.value = True
            else:
                self.w_use_random_seed.value = False
    
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


def create_optimizer_gui(default_params=None, output_dir='./output'):
    """
    Create and display the optimizer GUI.

    Args:
        default_params: dict, optional default parameters from config
        output_dir: str, directory containing existing recipe outputs (for loading)

    Returns:
        PhaseOptimizerGUI: The GUI instance for accessing job list
    """
    gui = PhaseOptimizerGUI(default_params, output_dir=output_dir)
    gui.display()
    return gui