"""
NPY File Selection GUI Module
Provides interactive widget for selecting .npy files from job subdirectories
with M%d pattern extraction and statistics

Adapted for new output structure where each job has its own subdirectory:
    ./output/
        job_title_1/
            job_title_1.npy
            job_title_1.json
            job_title_1_optimizer.pkl
        job_title_2/
            job_title_2.npy
            ...
"""

import ipywidgets as widgets
from IPython.display import display
import os
import glob
import re
from natsort import natsorted
from collections import Counter


class NPYFileSelector:
    """
    Interactive GUI for selecting multiple .npy files from job subdirectories
    with automatic M%d pattern extraction and filtering
    
    Attributes:
        output_dir (str): Directory containing job subdirectories
        selected_files (list): List of currently selected filenames (filtered)
        confirmed (bool): Whether selection has been confirmed
        m_patterns (list): List of extracted M%d patterns from selected files
        m_pattern_stats (dict): Statistics of M%d patterns {pattern: count}
        file_selector (widgets.SelectMultiple): Multi-select widget
        selection_display (widgets.Textarea): Text area showing selections
        stats_display (widgets.Textarea): Text area showing M%d statistics
        confirm_button (widgets.Button): Button to confirm selection
        status_label (widgets.Label): Label showing current status
    """
    
    # Regex pattern to match M followed by digits (e.g., M2, M11, M123)
    M_PATTERN_REGEX = re.compile(r'M(\d+)')
    
    def __init__(self, output_dir="./output", max_rows=15):
        """
        Initialize the NPY file selector
        
        Args:
            output_dir (str): Directory path containing job subdirectories
            max_rows (int): Maximum rows to display in selector widget
        """
        # Normalize path
        self.output_dir = os.path.normpath(output_dir)
        self.max_rows = max_rows
        self.selected_files = []  # Filtered files (only those with M%d)
        self.excluded_files = []  # Files without M%d pattern
        self.confirmed = False
        self.m_patterns = []  # Extracted M%d patterns
        self.m_pattern_stats = {}  # {pattern: count}
        self.file_to_pattern = {}  # {filename: pattern}
        self.file_to_path = {}  # {filename: full_path}
        
        # Scan for .npy files in subdirectories
        self._scan_npy_files()
        
        # Initialize widgets
        self._create_widgets()
    
    def _scan_npy_files(self):
        """Scan job subdirectories for .npy files."""
        self.npy_files = []
        self.npy_names = []
        self.file_to_path = {}
        
        if not os.path.exists(self.output_dir):
            return
        
        # Iterate through subdirectories
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                # Look for .npy files in this subdirectory
                npy_pattern = os.path.join(item_path, "*.npy")
                for npy_file in glob.glob(npy_pattern):
                    self.npy_files.append(npy_file)
                    fname = os.path.basename(npy_file)
                    self.npy_names.append(fname)
                    self.file_to_path[fname] = npy_file
        
        # Natural sort
        sorted_pairs = natsorted(zip(self.npy_names, self.npy_files), key=lambda x: x[0])
        if sorted_pairs:
            self.npy_names, self.npy_files = zip(*sorted_pairs)
            self.npy_names = list(self.npy_names)
            self.npy_files = list(self.npy_files)
        else:
            self.npy_names = []
            self.npy_files = []
    
    @staticmethod
    def extract_m_pattern(filename):
        """
        Extract M%d pattern from filename, finding the maximum match
        
        Args:
            filename (str): Filename to extract pattern from
            
        Returns:
            str or None: Extracted pattern (e.g., 'M11') or None if not found
        """
        matches = NPYFileSelector.M_PATTERN_REGEX.findall(filename)
        if not matches:
            return None
        # Find the maximum numeric value and return corresponding pattern
        max_num = max(int(m) for m in matches)
        return f'M{max_num}'
        
    def _create_widgets(self):
        """Create and configure the GUI widgets"""
        if not self.npy_names:
            print(f"⚠️ No .npy files found in subdirectories of {self.output_dir}")
        else:
            print(f"✅ Found {len(self.npy_names)} .npy files in {self.output_dir}")
        
        # Create multi-select widget
        self.file_selector = widgets.SelectMultiple(
            options=self.npy_names,
            value=[],
            rows=min(self.max_rows, max(5, len(self.npy_names))),
            description='NPY Files:',
            layout=widgets.Layout(width='400px'),
            style={'description_width': '80px'}
        )
        
        # Text area to display filtered selection
        self.selection_display = widgets.Textarea(
            value='',
            placeholder='Selected files (with M%d pattern) will appear here...',
            description='Valid:',
            layout=widgets.Layout(width='450px', height='120px'),
            style={'description_width': '60px'}
        )
        
        # Text area to display M%d statistics
        self.stats_display = widgets.Textarea(
            value='',
            placeholder='M%d pattern statistics will appear here...',
            description='Stats:',
            layout=widgets.Layout(width='450px', height='100px'),
            style={'description_width': '60px'}
        )
        
        # Text area to display excluded files
        self.excluded_display = widgets.Textarea(
            value='',
            placeholder='Excluded files (no M%d pattern) will appear here...',
            description='Excluded:',
            layout=widgets.Layout(width='450px', height='60px'),
            style={'description_width': '60px'}
        )
        
        # Confirm button
        self.confirm_button = widgets.Button(
            description='✓ Confirm Selection',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        self.confirm_button.on_click(self._on_confirm)
        
        # Refresh button
        self.refresh_button = widgets.Button(
            description='🔄 Refresh',
            button_style='info',
            layout=widgets.Layout(width='100px')
        )
        self.refresh_button.on_click(lambda b: self.refresh())
        
        # Status label
        self.status_label = widgets.Label(
            value='',
            layout=widgets.Layout(width='500px')
        )
        
        # Attach observer
        self.file_selector.observe(self._on_selection_change, names='value')
    
    def _process_selection(self, filenames):
        """
        Process selection: extract M%d patterns and filter files
        
        Args:
            filenames (list): List of selected filenames
            
        Returns:
            tuple: (valid_files, excluded_files, patterns, stats, file_to_pattern)
        """
        valid_files = []
        excluded_files = []
        patterns = []
        file_to_pattern = {}
        
        for fname in filenames:
            pattern = self.extract_m_pattern(fname)
            if pattern:
                valid_files.append(fname)
                patterns.append(pattern)
                file_to_pattern[fname] = pattern
            else:
                excluded_files.append(fname)
        
        # Calculate statistics
        stats = dict(Counter(patterns))
        # Sort by pattern number
        stats = dict(sorted(stats.items(), key=lambda x: int(x[0][1:])))
        
        return valid_files, excluded_files, patterns, stats, file_to_pattern
    
    def _on_selection_change(self, change):
        """Callback for selection changes"""
        raw_selection = list(change['new'])
        
        # Process and filter selection
        (self.selected_files, 
         self.excluded_files, 
         self.m_patterns, 
         self.m_pattern_stats,
         self.file_to_pattern) = self._process_selection(raw_selection)
        
        # Update selection display (valid files with their patterns)
        display_lines = [f"{fname}  →  {self.file_to_pattern[fname]}" 
                        for fname in self.selected_files]
        self.selection_display.value = '\n'.join(display_lines)
        
        # Update statistics display
        if self.m_pattern_stats:
            stats_lines = [f"{pattern}: {count} file(s)" 
                          for pattern, count in self.m_pattern_stats.items()]
            total_line = f"─────────────\nTotal: {len(self.selected_files)} valid files"
            self.stats_display.value = '\n'.join(stats_lines) + '\n' + total_line
        else:
            self.stats_display.value = 'No valid M%d patterns found'
        
        # Update excluded display
        if self.excluded_files:
            self.excluded_display.value = '\n'.join(self.excluded_files)
        else:
            self.excluded_display.value = '(none)'
        
        # Update status
        self.confirmed = False
        status_parts = [f'🔹 {len(self.selected_files)} valid']
        if self.excluded_files:
            status_parts.append(f'⚠️ {len(self.excluded_files)} excluded')
        status_parts.append('(not confirmed)')
        self.status_label.value = ' | '.join(status_parts)
    
    def _on_confirm(self, button):
        """Callback for confirm button"""
        self.confirmed = True
        self.status_label.value = f'✅ Confirmed {len(self.selected_files)} files ({len(self.m_pattern_stats)} M-types)'
        print(f"\n✅ Selection confirmed: {len(self.selected_files)} valid files")
        if self.excluded_files:
            print(f"⚠️  Excluded {len(self.excluded_files)} files without M%d pattern")
        print(f"\n📊 M-pattern statistics:")
        for pattern, count in self.m_pattern_stats.items():
            print(f"   {pattern}: {count} file(s)")
        print(f"\n📁 Access methods:")
        print(f"   file_selector_widget.get_selected_files()   → filtered filenames")
        print(f"   file_selector_widget.get_selected_paths()   → full paths")
        print(f"   file_selector_widget.get_m_patterns()       → M%d pattern list")
        print(f"   file_selector_widget.get_m_pattern_stats()  → {{pattern: count}}")
        print(f"   file_selector_widget.get_unique_m_patterns() → unique patterns")
        print(f"   file_selector_widget.get_output_dir()       → output directory")
    
    def display(self):
        """Display the selector GUI"""
        print("Select .npy files for scanning (Ctrl+Click for multiple):")
        print("Files without M%d pattern will be automatically excluded.\n")
        
        # Left panel: file selector and buttons
        button_row = widgets.HBox([self.confirm_button, self.refresh_button])
        left_panel = widgets.VBox([self.file_selector, button_row])
        
        # Right panel: displays and status
        right_panel = widgets.VBox([
            self.selection_display, 
            self.stats_display,
            self.excluded_display,
            self.status_label
        ])
        
        display(widgets.HBox([left_panel, right_panel]))
    
    def get_selected_files(self):
        """
        Get list of selected filenames (filtered, only those with M%d)
        
        Returns:
            list: List of selected .npy filenames with M%d pattern
        """
        return self.selected_files
    
    def get_selected_paths(self):
        """
        Get full paths of selected files (filtered)
        
        Returns:
            list: List of full file paths for selected files
        """
        return [self.file_to_path[fname] for fname in self.selected_files]
    
    def get_m_patterns(self):
        """
        Get list of M%d patterns corresponding to selected files
        
        Returns:
            list: List of M%d patterns (e.g., ['M2', 'M2', 'M11', 'M3'])
                  Same length and order as get_selected_files()
        """
        return self.m_patterns
    
    def get_m_pattern_stats(self):
        """
        Get statistics of M%d patterns
        
        Returns:
            dict: Dictionary mapping pattern to count (e.g., {'M2': 3, 'M11': 2})
        """
        return self.m_pattern_stats
    
    def get_unique_m_patterns(self):
        """
        Get list of unique M%d patterns (sorted by numeric value)
        
        Returns:
            list: Sorted list of unique patterns (e.g., ['M2', 'M3', 'M11'])
        """
        return list(self.m_pattern_stats.keys())
    
    def get_file_to_pattern_map(self):
        """
        Get mapping from filename to its M%d pattern
        
        Returns:
            dict: Dictionary mapping filename to pattern
        """
        return self.file_to_pattern
    
    def get_file_to_path_map(self):
        """
        Get mapping from filename to its full path
        
        Returns:
            dict: Dictionary mapping filename to full path
        """
        return {fname: self.file_to_path[fname] for fname in self.selected_files}
    
    def get_excluded_files(self):
        """
        Get list of files that were excluded (no M%d pattern)
        
        Returns:
            list: List of excluded filenames
        """
        return self.excluded_files
        
    def get_output_dir(self):
        """
        Get the output directory path
        
        Returns:
            str: Output directory path
        """
        return self.output_dir
    
    def is_confirmed(self):
        """
        Check if selection has been confirmed
        
        Returns:
            bool: True if selection confirmed
        """
        return self.confirmed
    
    def set_selection(self, filenames):
        """
        Programmatically set selected files
        
        Args:
            filenames (list): List of filenames to select
        """
        valid_files = [f for f in filenames if f in self.npy_names]
        self.file_selector.value = valid_files
    
    def refresh(self):
        """Refresh the file list from directory"""
        self._scan_npy_files()
        self.file_selector.options = self.npy_names
        self.file_selector.value = []
        self.selected_files = []
        self.excluded_files = []
        self.m_patterns = []
        self.m_pattern_stats = {}
        self.file_to_pattern = {}
        self.confirmed = False
        self.selection_display.value = ''
        self.stats_display.value = ''
        self.excluded_display.value = ''
        self.status_label.value = ''
        print(f"🔄 Refreshed: Found {len(self.npy_names)} .npy files")


# Convenience function for quick usage
def select_npy_files(output_dir="./output", max_rows=15):
    """
    Quick function to create and display NPY file selector
    
    Args:
        output_dir (str): Directory containing job subdirectories with .npy files
        max_rows (int): Maximum rows in selector widget
        
    Returns:
        NPYFileSelector: Configured selector instance
    
    Example:
        # in cell#1
        file_selector_widget = select_npy_files(output_dir="./output")
        
        # in cell#2
        selected_npy_files = file_selector_widget.get_selected_files()
        output_dir = file_selector_widget.get_output_dir()
        
        # Get full paths (important for new directory structure)
        selected_paths = file_selector_widget.get_selected_paths()
    """
    selector = NPYFileSelector(output_dir, max_rows)
    selector.display()
    return selector