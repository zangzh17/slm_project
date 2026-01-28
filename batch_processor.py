"""
Batch Processor for Phase Optimization Jobs
Processes jobs from the GUI and saves results.
"""

import os
import torch
import pickle
import json
import numpy as np
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output


def process_jobs(gui, 
                 slm_manager,
                 device=None,
                 output_dir='./output',
                 save_optimizer=True,
                 visualize=False,
                 upsampling=2.0):
    """
    Process all jobs in the GUI job list.
    
    Args:
        gui: PhaseOptimizerGUI instance
        slm_manager: SLMManager instance
        device: torch device (default: auto-detect)
        output_dir: Output directory path
        save_optimizer: Whether to save optimizer instances
        visualize: Whether to show visualization during processing
        upsampling: Upsampling factor for optimization
    
    Returns:
        dict: Results dictionary with job titles as keys
    """
    # Import here to avoid circular imports
    from phase_generators import PhaseGenerator
    from optics_utils import save_array, save_dict_as_json
    
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    job_list = gui.get_job_list()
    
    if not job_list:
        print("⚠️ No jobs in the queue. Please add jobs using the GUI.")
        return {}
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    total_jobs = len(job_list)
    
    print(f"{'='*60}")
    print(f"🚀 Starting batch processing: {total_jobs} job(s)")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖥️  Device: {device}")
    print(f"{'='*60}\n")
    
    for idx, (job_title, params) in enumerate(job_list, 1):
        print(f"\n[{idx}/{total_jobs}] Processing: {job_title}")
        print("-" * 50)
        
        try:
            # Update shape from SLM manager
            params['shape'] = slm_manager.shape
            
            mode = params.get('mode', 'fresnel')
            
            if mode == 'fresnel':
                # Fresnel mode - no optimization needed
                print("  Mode: Fresnel (direct generation)")
                
                optimizer = PhaseGenerator(params, device=device)
                optimizer.generate(mode='fresnel')
                phase_8bit = optimizer.update_phase_8bit()
                
            else:
                # Optimized mode
                print("  Mode: Optimized")
                print(f"  Airy correction: {params.get('airy_correction')}")
                print(f"  Depth in focus: {params.get('depth_in_focus')}")
                
                optimizer = PhaseGenerator(params, device=device, mode='optimized')
                optimizer._prepare_template()
                optimizer.generate(upsampling=upsampling, visualize=visualize)
                phase_8bit = optimizer.update_phase_8bit()
            
            # Create job-specific output directory
            job_output_dir = os.path.join(output_dir, job_title)
            os.makedirs(job_output_dir, exist_ok=True)
            
            # Save phase pattern (npy file)
            npy_path = os.path.join(job_output_dir, f'{job_title}.npy')
            save_array(phase_8bit, os.path.join(job_output_dir, job_title))
            print(f"  ✅ Saved phase pattern: {npy_path}")
            
            # Save parameters (json file)
            json_path = os.path.join(job_output_dir, f'{job_title}.json')
            # Make params JSON serializable
            params_to_save = _make_json_serializable(params)
            save_dict_as_json(params_to_save, json_path)
            print(f"  ✅ Saved parameters: {json_path}")
            
            # Save optimizer instance (pickle file)
            if save_optimizer:
                optimizer_path = os.path.join(job_output_dir, f'{job_title}_optimizer.pkl')
                with open(optimizer_path, 'wb') as f:
                    pickle.dump(optimizer, f)
                print(f"  ✅ Saved optimizer: {optimizer_path}")
            
            # Store results
            results[job_title] = {
                'status': 'success',
                'output_dir': job_output_dir,
                'npy_path': npy_path,
                'json_path': json_path,
                'optimizer_path': optimizer_path if save_optimizer else None,
                'optimizer': optimizer
            }
            
            print(f"  ✅ Job completed successfully!")
            
        except Exception as e:
            print(f"  ❌ Error processing job: {str(e)}")
            results[job_title] = {
                'status': 'error',
                'error': str(e)
            }
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Batch Processing Summary")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    error_count = sum(1 for r in results.values() if r['status'] == 'error')
    
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {error_count}")
    print(f"📁 Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    return results


def _make_json_serializable(obj):
    """Convert non-serializable objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


class JobBrowserGUI:
    """
    Interactive GUI for browsing and visualizing saved optimization jobs.
    
    Usage in Jupyter:
        browser = browse_jobs(output_dir='./output')
        # Click on a job to visualize it
        
        # Access loaded optimizer:
        optimizer = browser.get_current_optimizer()
        
        # With SLM auto-upload:
        browser = browse_jobs(output_dir='./output', slm_manager=slm_manager)
        # Clicking a job will automatically upload the phase pattern to SLM
    """
    
    def __init__(self, output_dir='./output', upsampling=3, slm_manager=None):
        """
        Initialize the job browser.
        
        Args:
            output_dir: Directory containing job subdirectories
            upsampling: Upsampling factor for visualization
            slm_manager: Optional SLMManager instance for auto-uploading phase patterns
        """
        self.output_dir = os.path.normpath(output_dir)
        self.upsampling = upsampling
        self.slm_manager = slm_manager
        self.current_optimizer = None
        self.current_job = None
        self.current_phase_8bit = None
        self.job_list = []
        
        # Scan for jobs
        self._scan_jobs()
        
        # Create widgets
        self._create_widgets()
    
    def _scan_jobs(self):
        """Scan output directory for valid jobs."""
        self.job_list = []
        
        if not os.path.exists(self.output_dir):
            print(f"⚠️ Output directory does not exist: {self.output_dir}")
            return
        
        for item in sorted(os.listdir(self.output_dir)):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                # Check for required files
                pkl_file = os.path.join(item_path, f'{item}_optimizer.pkl')
                json_file = os.path.join(item_path, f'{item}.json')
                npy_file = os.path.join(item_path, f'{item}.npy')
                
                has_pkl = os.path.exists(pkl_file)
                has_json = os.path.exists(json_file)
                has_npy = os.path.exists(npy_file)
                
                if has_pkl or has_json or has_npy:
                    self.job_list.append({
                        'title': item,
                        'path': item_path,
                        'has_optimizer': has_pkl,
                        'has_params': has_json,
                        'has_phase': has_npy
                    })
        
        print(f"✅ Found {len(self.job_list)} job(s) in {self.output_dir}")
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Job selector
        job_options = [(f"{'✓' if j['has_optimizer'] else '○'} {j['title']}", j['title']) 
                       for j in self.job_list]
        
        self.job_selector = widgets.Select(
            options=job_options,
            value=None,
            rows=min(15, max(5, len(self.job_list))),
            description='Jobs:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '50px'}
        )
        
        # Info display
        self.info_display = widgets.Textarea(
            value='Select a job to view details...',
            description='Info:',
            layout=widgets.Layout(width='400px', height='120px'),
            style={'description_width': '50px'},
            disabled=True
        )
        
        # Buttons
        self.visualize_btn = widgets.Button(
            description='📊 Visualize',
            button_style='primary',
            layout=widgets.Layout(width='120px'),
            disabled=True
        )
        
        self.refresh_btn = widgets.Button(
            description='🔄 Refresh',
            button_style='info',
            layout=widgets.Layout(width='100px')
        )
        
        # Upload button (only shown if slm_manager is provided)
        self.upload_btn = widgets.Button(
            description='📤 Upload to SLM',
            button_style='success',
            layout=widgets.Layout(width='140px'),
            disabled=True
        )
        
        # Visualization options
        self.upsampling_slider = widgets.FloatSlider(
            value=self.upsampling,
            min=1.0,
            max=5.0,
            step=0.5,
            description='Upsample:',
            layout=widgets.Layout(width='250px'),
            style={'description_width': '70px'}
        )
        
        # Auto-upload checkbox (only relevant if slm_manager is provided)
        self.auto_upload_checkbox = widgets.Checkbox(
            value=True if self.slm_manager else False,
            description='Auto-upload on select',
            indent=False,
            disabled=self.slm_manager is None
        )
        
        self.viz_checkboxes = widgets.VBox([
            widgets.Checkbox(value=True, description='Phase Pattern', indent=False),
            widgets.Checkbox(value=True, description='2D Comparisons', indent=False),
            widgets.Checkbox(value=True, description='Cross Sections', indent=False),
            widgets.Checkbox(value=True, description='Energy Distribution', indent=False),
        ])
        
        # Output area for visualization
        self.viz_output = widgets.Output()
        
        # Status label
        self.status_label = widgets.Label(value='')
        
        # Attach callbacks
        self.job_selector.observe(self._on_job_select, names='value')
        self.visualize_btn.on_click(self._on_visualize)
        self.refresh_btn.on_click(self._on_refresh)
        self.upload_btn.on_click(self._on_upload)
    
    def _on_job_select(self, change):
        """Handle job selection."""
        job_title = change['new']
        if job_title is None:
            return
        
        # Find job info
        job_info = next((j for j in self.job_list if j['title'] == job_title), None)
        if job_info is None:
            return
        
        self.current_job = job_title
        self.current_phase_8bit = None
        
        # Update info display
        info_lines = [
            f"Job: {job_title}",
            f"Path: {job_info['path']}",
            f"─" * 30,
            f"Optimizer (.pkl): {'✓' if job_info['has_optimizer'] else '✗'}",
            f"Parameters (.json): {'✓' if job_info['has_params'] else '✗'}",
            f"Phase (.npy): {'✓' if job_info['has_phase'] else '✗'}",
        ]
        
        # Load and show params if available
        if job_info['has_params']:
            try:
                json_path = os.path.join(job_info['path'], f'{job_title}.json')
                with open(json_path, 'r') as f:
                    params = json.load(f)
                info_lines.append(f"─" * 30)
                for key in ['mode', 'airy_correction', 'depth_in_focus', 'focal_length']:
                    if key in params:
                        info_lines.append(f"{key}: {params[key]}")
            except Exception as e:
                info_lines.append(f"Error loading params: {e}")
        
        self.info_display.value = '\n'.join(info_lines)
        
        # Enable visualize button if optimizer exists
        self.visualize_btn.disabled = not job_info['has_optimizer']
        
        # Enable upload button if phase file exists and slm_manager is available
        self.upload_btn.disabled = not (job_info['has_phase'] and self.slm_manager is not None)
        
        # Load phase and auto-upload if enabled
        if job_info['has_phase']:
            try:
                npy_path = os.path.join(job_info['path'], f'{job_title}.npy')
                self.current_phase_8bit = np.load(npy_path)
                
                # Auto-upload if slm_manager is available and auto-upload is enabled
                if self.slm_manager is not None and self.auto_upload_checkbox.value:
                    self.slm_manager.upload(self.current_phase_8bit)
                    self.status_label.value = f"✅ Selected & uploaded: {job_title}"
                else:
                    self.status_label.value = f"Selected: {job_title}"
            except Exception as e:
                self.status_label.value = f"⚠️ Error loading phase: {e}"
                self.current_phase_8bit = None
        else:
            self.status_label.value = f"Selected: {job_title} (no phase file)"
    
    def _on_upload(self, button):
        """Handle manual upload button click."""
        if self.current_phase_8bit is None:
            self.status_label.value = "⚠️ No phase pattern loaded"
            return
        
        if self.slm_manager is None:
            self.status_label.value = "⚠️ No SLM manager available"
            return
        
        try:
            self.slm_manager.upload(self.current_phase_8bit)
            self.status_label.value = f"✅ Uploaded: {self.current_job}"
        except Exception as e:
            self.status_label.value = f"❌ Upload error: {e}"
    
    def _on_visualize(self, button):
        """Handle visualize button click."""
        if self.current_job is None:
            return
        
        with self.viz_output:
            clear_output(wait=True)
            
            try:
                print(f"🔄 Loading optimizer for: {self.current_job}")
                
                # Load optimizer
                optimizer_path = os.path.join(
                    self.output_dir, 
                    self.current_job, 
                    f'{self.current_job}_optimizer.pkl'
                )
                
                with open(optimizer_path, 'rb') as f:
                    self.current_optimizer = pickle.load(f)
                
                print(f"✅ Optimizer loaded successfully")
                print(f"\n📊 Generating visualizations...")
                print("-" * 40)
                
                # Import visualization functions
                from visualization import (
                    plot_phase, plot_2d_comparisons, plot_2d_comparisons_interactive,
                    plot_cross_sections, plot_energy_distribution
                )
                
                upsampling = self.upsampling_slider.value
                checkboxes = self.viz_checkboxes.children
                
                if checkboxes[0].value:
                    plot_phase(self.current_optimizer)
                
                if checkboxes[1].value:
                    plot_2d_comparisons_interactive(self.current_optimizer)
                
                if checkboxes[2].value:
                    plot_cross_sections(self.current_optimizer, upsampling=upsampling)
                
                if checkboxes[3].value:
                    plot_energy_distribution(self.current_optimizer, upsampling=upsampling)
                
                self.status_label.value = f"✅ Visualized: {self.current_job}"
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                self.status_label.value = f"❌ Error visualizing {self.current_job}"
    
    def _on_refresh(self, button):
        """Handle refresh button click."""
        self._scan_jobs()
        
        job_options = [(f"{'✓' if j['has_optimizer'] else '○'} {j['title']}", j['title']) 
                       for j in self.job_list]
        self.job_selector.options = job_options
        self.job_selector.value = None
        self.current_job = None
        self.current_optimizer = None
        self.current_phase_8bit = None
        self.info_display.value = 'Select a job to view details...'
        self.visualize_btn.disabled = True
        self.upload_btn.disabled = True
        self.status_label.value = f"🔄 Refreshed: {len(self.job_list)} jobs found"
    
    def display(self):
        """Display the GUI."""
        print("📁 Job Browser - Click a job to view details, then Visualize")
        print("   ✓ = has optimizer, ○ = no optimizer")
        if self.slm_manager:
            print("   📤 SLM Manager connected - auto-upload enabled\n")
        else:
            print("\n")
        
        # Left panel: selector and buttons
        button_row = widgets.HBox([self.visualize_btn, self.refresh_btn, self.upload_btn] 
                                   if self.slm_manager else [self.visualize_btn, self.refresh_btn])
        left_panel = widgets.VBox([
            self.job_selector,
            button_row,
            self.status_label
        ])
        
        # Right panel: info and options
        options_children = [
            widgets.Label('Visualization Options:'),
            self.upsampling_slider,
            self.viz_checkboxes
        ]
        if self.slm_manager:
            options_children.insert(1, self.auto_upload_checkbox)
        
        options_box = widgets.VBox(options_children)
        right_panel = widgets.VBox([self.info_display, options_box])
        
        # Main layout
        top_panel = widgets.HBox([left_panel, right_panel])
        
        display(top_panel)
        display(self.viz_output)
    
    def get_current_optimizer(self):
        """
        Get the currently loaded optimizer.
        
        Returns:
            PhaseGenerator or None: The loaded optimizer instance
        """
        return self.current_optimizer
    
    def get_current_job(self):
        """
        Get the currently selected job title.
        
        Returns:
            str or None: Current job title
        """
        return self.current_job
    
    def get_current_phase(self):
        """
        Get the currently loaded phase pattern.
        
        Returns:
            np.ndarray or None: The loaded 8-bit phase pattern
        """
        return self.current_phase_8bit
    
    def get_job_list(self):
        """
        Get list of all available jobs.
        
        Returns:
            list: List of job info dictionaries
        """
        return self.job_list


def browse_jobs(output_dir='./output', upsampling=3, slm_manager=None):
    """
    Quick function to create and display job browser GUI.
    
    Args:
        output_dir: Directory containing job subdirectories
        upsampling: Default upsampling factor for visualization
        slm_manager: Optional SLMManager instance for auto-uploading phase patterns
    
    Returns:
        JobBrowserGUI: The browser instance
    
    Example:
        # Basic usage
        browser = browse_jobs('./output')
        # After clicking a job and visualizing:
        optimizer = browser.get_current_optimizer()
        
        # With SLM auto-upload
        browser = browse_jobs('./output', slm_manager=slm_manager)
        # Clicking a job will automatically upload the phase to SLM
    """
    browser = JobBrowserGUI(output_dir, upsampling, slm_manager=slm_manager)
    browser.display()
    return browser


def list_saved_jobs(output_dir='./output'):
    """
    List all saved jobs in the output directory.
    
    Args:
        output_dir: Output directory path
    
    Returns:
        list: List of job titles
    """
    if not os.path.exists(output_dir):
        print(f"⚠️ Output directory does not exist: {output_dir}")
        return []
    
    jobs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains expected files
            json_file = os.path.join(item_path, f'{item}.json')
            if os.path.exists(json_file):
                jobs.append(item)
    
    print(f"📁 Found {len(jobs)} saved job(s) in {output_dir}:")
    for job in jobs:
        print(f"  - {job}")
    
    return jobs