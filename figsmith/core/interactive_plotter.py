"""
Interactive plotter with data loading, column selection, and COMPLETE figure editing
Includes ALL tunable parameters from the original system
"""

import matplotlib
# Use Agg backend to prevent automatic figure display in Jupyter
# This ensures figures only display when we explicitly render them
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import mathtext as _mt
import threading
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output, Image
from collections.abc import Sequence

from ..io import DataLoader
from .property_manager import PropertyManager
from ..utils.helpers import (
    to_float_array as _to_float_array,
    filter_valid_vectors as _filter_valid_vectors,
    try_reshape_grid as _try_reshape_grid,
    CMAP_OPTIONS
)


class InteractivePlotter:
    """
    Unified interactive plotter with COMPLETE parameter control:
    - Data loading from .dat files
    - Column selection and initial plot creation
    - Per-line tunable parameters (each line individually)
    - Complete global figure parameters (ALL from basic_controls)
    - Field/contour specific controls
    - Save functionality with multiple format options
    """

    def __init__(self):
        """Initialize the interactive plotter"""
        self.loader = None
        self._in_memory_mode = False
        self.fig = None
        self.ax = None
        self.plot_type = 'line'
        self.selected_data = {}
        self.line_styles = {}
        self.prop_manager = None
        self.colorbar = None
        self.colorbar_ax = None  # Separate axes for colorbar
        # Global mathtext toggle for label rendering
        self.use_mathtext = False
        # MathText parser for validation (safe fallback if unavailable)
        try:
            self._mathtext_parser = _mt.MathTextParser('agg')
        except Exception:
            self._mathtext_parser = None
        # Pending labels store: {ax_id: {category: {'raw','validated','applied'}}}
        self.pending_labels = {}

        # Colorbar positioning defaults
        self.colorbar_gap = 0.02  # Gap between plot and colorbar
        self.colorbar_width = 0.02  # Width of colorbar

        # Subplot support
        self.subplot_mode = False  # Whether subplots are enabled
        self.num_subplots = 1
        self.subplot_rows = 1
        self.subplot_cols = 1
        self.axes_list = []  # List of all subplot axes
        self.loaders_list = []  # List of DataLoader instances for each subplot
        self.subplot_configs = []  # Configuration for each subplot
        self.colorbars = []  # List of colorbars for each subplot
        self.colorbar_axes = []  # List of colorbar axes
        # Per-subplot persistent colorbar settings (label, font sizes, weight, etc.)
        self.subplot_colorbar_settings = []
        # Per-subplot legend visibility state
        self.subplot_legend_visible = []
        # Single-plot legend visibility
        self.legend_visible = True
        self.subplot_contour_data = []  # Contour data for each subplot (for redrawing)
        self.subplot_scatter_data = []  # Scatter data for each subplot (for redrawing)
        self.vector_data = None  # Vector field data for single plot
        self.vector_quiver = None
        self.vector_overlay = None
        self.vector_overlay_colorbar = None
        self.vector_quiver_colorbar = None
        self.subplot_vector_data = []
        self.subplot_vector_quivers = []
        self.subplot_vector_overlays = []
        self.subplot_vector_colorbars = []
        self.subplot_vector_quiver_colorbars = []
        self.vector_style = self._default_vector_style()
        self.subplot_vector_styles = []

        # Build UI
        self._build_ui()

    # ===== Helper utilities (centralized) =====
    def _default_vector_style(self):
        """Return baseline vector plotting parameters."""
        return {
            'scale': 1.0,
            'width': 0.0025,
            'pivot': 'middle',
            'cmap': 'viridis',
            'alpha': 0.8,
            'arrow_color': 'black',
            'decimation': 1,
            'colorbar': True,
            'colorbar_label': None,
            'colorbar_label_fontsize': 10,
            'colorbar_tick_fontsize': 9,
            'colorbar_fontweight': 'normal',
            'colorbar_gap': 0.02,
            'colorbar_width': 0.02,
            'overlay_type': 'contourf (filled)',
            'overlay_levels': 50,
            'overlay_log': False,
            'overlay_cmap': 'RdBu_r',
            'overlay_alpha': 1.0,
            'overlay_line_thickness': 0.5,
            'overlay_line_color': 'black',
            'overlay_show_colorbar': True,
            'overlay_label': None,
            'overlay_label_fontsize': 10,
            'overlay_tick_fontsize': 9,
            'overlay_fontweight': 'normal',
            'overlay_colorbar_gap': 0.04,
            'overlay_colorbar_width': 0.03
        }

    def _set_in_memory_mode(self, active, message=None):
        """Enable/disable file input widgets depending on data source."""
        self._in_memory_mode = bool(active)
        if hasattr(self, 'file_input'):
            self.file_input.disabled = self._in_memory_mode
        if hasattr(self, 'load_btn'):
            self.load_btn.disabled = self._in_memory_mode
        if hasattr(self, 'data_mode_info'):
            if self._in_memory_mode:
                note = message or "Using in-memory data"
                self.data_mode_info.value = (
                    f"<span style='color:#2E5090;font-weight:bold;'>⚡ {note}</span>"
                )
            else:
                self.data_mode_info.value = ""

    def _reset_before_data_ingest(self):
        """Clear interactive state before loading new data."""
        self.line_styles = {}
        self.line_style_container.children = []
        self.editing_controls_container.layout.display = 'none'
        self.editing_controls_container.children = []
        self.save_section.layout.display = 'none'
        self.vector_data = None
        self.vector_quiver = None
        self.vector_overlay = None
        self.vector_overlay_colorbar = None
        self.vector_quiver_colorbar = None
        self.vector_style = self._default_vector_style()
        self._y_columns_selected = []

        if self.fig is not None:
            try:
                plt.close(self.fig)
            except Exception:
                pass
            self.fig = None
            self.ax = None
            self.colorbar = None
            self.colorbar_ax = None
            with self.fig_output:
                clear_output(wait=True)

    def _infer_x_column(self, columns, preferred=None):
        """Choose a sensible default x column."""
        if preferred and preferred in columns:
            return preferred
        for col in columns:
            if str(col).lower() == 'x':
                return col
        return columns[0] if columns else None

    def _infer_default_y_columns(self, columns, x_col):
        """Choose baseline y columns distinct from x."""
        for col in columns:
            if col != x_col:
                return [col]
        return []
    def _get_subplot_settings(self, subplot_idx):
        """Return persisted settings dict for a subplot (create if missing)."""
        while len(self.subplot_colorbar_settings) <= subplot_idx:
            self.subplot_colorbar_settings.append({})
        if self.subplot_colorbar_settings[subplot_idx] is None:
            self.subplot_colorbar_settings[subplot_idx] = {}
        return self.subplot_colorbar_settings[subplot_idx]

    def _get_subplot_legend_visible(self, subplot_idx):
        while len(self.subplot_legend_visible) <= subplot_idx:
            self.subplot_legend_visible.append(True)
        return self.subplot_legend_visible[subplot_idx]

    def _set_subplot_legend_visible(self, subplot_idx, value: bool):
        while len(self.subplot_legend_visible) <= subplot_idx:
            self.subplot_legend_visible.append(True)
        self.subplot_legend_visible[subplot_idx] = bool(value)

    # ===== Label formatting helpers =====
    def _store_axis_labels(self, ax, raw_dict):
        """Store raw label text in axes for persistence across redraws."""
        if not hasattr(ax, '_ff_labels'):
            ax._ff_labels = {}
        for key, value in raw_dict.items():
            if value is not None:
                ax._ff_labels[key] = str(value)

    def _reapply_user_labels(self, ax):
        """Re-apply user-entered labels/title stored on the axes, using mathtext policy.

        We store raw text in ax._ff_labels during UI edits and use it here to
        restore labels after redraws so they don't regress to sanitized text.
        """
        try:
            store = getattr(ax, '_ff_labels', None)
            if not isinstance(store, dict):
                return
                
            # Apply each stored label if it exists
            if store.get('xlabel') is not None:
                formatted = self._apply_mathtext(store['xlabel'])
                if formatted is not None:
                    ax.set_xlabel(formatted)
                    
            if store.get('ylabel') is not None:
                formatted = self._apply_mathtext(store['ylabel'])
                if formatted is not None:
                    ax.set_ylabel(formatted)
                    
            if store.get('title') is not None:
                formatted = self._apply_mathtext(store['title'])
                if formatted is not None:
                    ax.set_title(formatted)
        except Exception:
            pass

    def _store_legend_label(self, artist, raw_text):
        """Attach raw legend label to an artist for later reapplication."""
        if raw_text is not None:
            try:
                setattr(artist, '_ff_legend_raw', str(raw_text))
            except Exception:
                pass

    def _reapply_legend_labels(self, ax):
        """Re-apply stored legend labels on all artists in an axes using mathtext policy."""
        try:
            for ln in ax.get_lines():
                raw = getattr(ln, '_ff_legend_raw', None)
                if raw is not None:
                    try:
                        ln.set_label(self._apply_mathtext(raw))
                    except Exception:
                        ln.set_label(str(raw))
            for coll in ax.collections:
                if hasattr(coll, 'get_offsets'):
                    raw = getattr(coll, '_ff_legend_raw', None)
                    if raw is not None:
                        try:
                            coll.set_label(self._apply_mathtext(raw))
                        except Exception:
                            coll.set_label(str(raw))
            # If a legend is currently visible, refresh it to pick up changes
            leg = ax.get_legend()
            if leg is not None:
                try:
                    ax.legend()
                except Exception:
                    pass
        except Exception:
            pass

    def _reapply_all_stored_labels_on_figure(self):
        """Re-apply stored axes labels and legend labels across the whole figure."""
        if not self.fig:
            return

        try:
            # Process all axes in the figure
            for ax in list(self.fig.axes):
                # Skip colorbar axes
                if ax in self.colorbar_axes or ax == self.colorbar_ax:
                    continue
                    
                # Reapply axis labels and title
                self._reapply_user_labels(ax)
                
                # Reapply legend labels and update legend if visible
                self._reapply_legend_labels(ax)
            
            # Handle colorbars separately
            self._relabel_all_colorbars_mathtext()
            
        except Exception:
            # Last resort: attempt to redraw with plain text if mathtext fails
            if self.use_mathtext:
                temp_state = self.use_mathtext
                self.use_mathtext = False
                try:
                    for ax in list(self.fig.axes):
                        if ax not in self.colorbar_axes and ax != self.colorbar_ax:
                            self._reapply_user_labels(ax)
                            self._reapply_legend_labels(ax)
                    self._relabel_all_colorbars_mathtext()
                finally:
                    self.use_mathtext = temp_state
    def _validate_math(self, expr: str) -> bool:
        """Validate if a math expression can be parsed by matplotlib."""
        if self._mathtext_parser is None:
            return True  # Fallback to accepting all input
        try:
            self._mathtext_parser.parse(expr, dpi=100)
            return True
        except Exception:
            return False

    def _get_ax_key(self, ax):
        try:
            return int(id(ax))
        except Exception:
            return ax

    def _get_pending_slot(self, ax, category):
        key = self._get_ax_key(ax)
        if key not in self.pending_labels:
            self.pending_labels[key] = {}
        if category not in self.pending_labels[key]:
            self.pending_labels[key][category] = {'raw': None, 'validated': None, 'applied': None}
        return self.pending_labels[key][category]

    def _validate_format_math(self, raw: str):
        """Return (is_valid, formatted_text) for a raw input under current math policy."""
        if raw is None:
            return True, ''
        s = str(raw)
        if s.strip() == '':
            return True, ''
        if not self.use_mathtext:
            return True, s.replace('$', '')
        # Normalize dollars
        if s.startswith('$$') and s.endswith('$$') and len(s) >= 4:
            expr = s[2:-2]
        elif s.startswith('$') and s.endswith('$') and len(s) >= 2:
            expr = s[1:-1]
        else:
            expr = s
        if self._validate_math(expr):
            return True, f'${expr}$'
        return False, s.replace('$', '')

    def _validate_and_store_label(self, ax, category: str, raw_text: str):
        slot = self._get_pending_slot(ax, category)
        slot['raw'] = raw_text
        ok, fmt = self._validate_format_math(raw_text)
        slot['validated'] = fmt if ok else None
        return ok

    def _apply_stored_label(self, ax, category: str):
        slot = self._get_pending_slot(ax, category)
        text = slot.get('validated')
        if text is None:
            return False
        try:
            if category == 'xlabel':
                ax.set_xlabel(text)
            elif category == 'ylabel':
                ax.set_ylabel(text)
            elif category == 'title':
                ax.set_title(text)
            slot['applied'] = text
            # Persist raw on axes for redraws
            if not hasattr(ax, '_ff_labels') or not isinstance(getattr(ax, '_ff_labels'), dict):
                ax._ff_labels = {'xlabel': None, 'ylabel': None, 'title': None}
            ax._ff_labels[category] = slot.get('raw')
            return True
        except Exception:
            return False

    def _extract_math_expr(self, text: str) -> str:
        """Extract math expression from potentially dollar-wrapped text."""
        if text.startswith('$$') and text.endswith('$$') and len(text) >= 4:
            return text[2:-2]
        elif text.startswith('$') and text.endswith('$') and len(text) >= 2:
            return text[1:-1]
        return text

    def _apply_mathtext(self, text: str):
        """Return a label formatted for mathtext when toggle is ON, no $ required.

        Policy:
        - Mathtext OFF: return plain text, strip any dollar signs.
        - Mathtext ON: try to parse as math by auto-wrapping in $...$.
            * If input is already wrapped with $...$ or $$...$$, normalize and validate.
            * If validation succeeds, return a single-wrapped $expr$ string.
            * If validation fails, fall back to plain text (dollars stripped).
        """
        if text is None:
            return ''
        s = str(text)
        if s.strip() == '':
            return ''
        # When disabled, always render plain text
        if not self.use_mathtext:
            return s.replace('$', '')
            
        # Extract expression without outer dollars
        expr = self._extract_math_expr(s)
        
        # Validate and format
        if expr.strip() == '':
            return ''
        if self._validate_math(expr):
            return f'${expr}$'
        else:
            # Invalid math expression - return sanitized plain text
            return s.replace('$', '')

    def _relabel_all_colorbars_mathtext(self):
        """Re-apply mathtext formatting to all existing colorbars based on toggle."""
        # Single-plot colorbar
        if getattr(self, 'colorbar', None):
            try:
                lbl = self.colorbar.ax.get_ylabel()
                self.colorbar.set_label(self._apply_mathtext(lbl))
            except Exception:
                pass
        # Subplot colorbars
        for cbar in getattr(self, 'colorbars', []) or []:
            if cbar is not None:
                try:
                    lbl = cbar.ax.get_ylabel()
                    cbar.set_label(self._apply_mathtext(lbl))
                except Exception:
                    pass

    def _remove_collections_of_type(self, ax, type_keyword):
        """Remove artists from ax.collections whose type name contains type_keyword."""
        to_remove = [c for c in ax.collections if type_keyword in str(type(c))]
        for c in to_remove:
            try:
                c.remove()
            except Exception:
                pass

    def _remove_subplot_colorbar(self, subplot_idx):
        """Remove subplot colorbar and its axes if present."""
        if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
            try:
                self.colorbars[subplot_idx].remove()
            except Exception:
                pass
            self.colorbars[subplot_idx] = None
        if subplot_idx < len(self.colorbar_axes) and self.colorbar_axes[subplot_idx]:
            try:
                self.fig.delaxes(self.colorbar_axes[subplot_idx])
            except Exception:
                pass
            self.colorbar_axes[subplot_idx] = None

    def _create_subplot_colorbar(self, subplot_idx, mappable, default_label,
                                 gap=None, width=None):
        """Create colorbar for a subplot next to its axes and apply persisted styles."""
        # Ensure arrays sized
        while len(self.colorbars) <= subplot_idx:
            self.colorbars.append(None)
        while len(self.colorbar_axes) <= subplot_idx:
            self.colorbar_axes.append(None)

        ax = self.axes_list[subplot_idx]
        pos = ax.get_position()
        # Pull persisted position if not provided
        settings = self._get_subplot_settings(subplot_idx)
        use_gap = settings.get('gap', 0.01) if gap is None else gap
        use_width = settings.get('width', 0.01) if width is None else width

        cbar_ax = self.fig.add_axes([pos.x1 + use_gap, pos.y0, use_width, pos.height])
        cbar = self.fig.colorbar(mappable, cax=cbar_ax)

        # Apply persisted styles
        label_text = settings.get('label', default_label)
        label_size = settings.get('label_fontsize', None)
        tick_size = settings.get('tick_fontsize', None)
        fontweight = settings.get('fontweight', None)

        label_text_fmt = self._apply_mathtext(label_text)
        if label_size is not None:
            cbar.set_label(label_text_fmt, fontsize=label_size)
        else:
            cbar.set_label(label_text_fmt)
        if tick_size is not None:
            try:
                cbar.ax.tick_params(labelsize=tick_size)
            except Exception:
                pass
        if fontweight is not None:
            try:
                cbar.ax.yaxis.label.set_fontweight(fontweight)
            except Exception:
                pass

        # Track
        self.colorbars[subplot_idx] = cbar
        self.colorbar_axes[subplot_idx] = cbar_ax
        return cbar

    def _infer_subplot_plot_type(self, ax):
        """Infer actual plot type present on an axes based on artists."""
        # Prioritize quiver (vector) over contour overlays
        # Vector field plots may include a contour/contourf overlay; identify as 'vector' if any quiver present
        quiver_colls = [c for c in ax.collections if 'Quiver' in str(type(c))]
        if quiver_colls:
            return 'vector'

        # Then check for contour collections
        contour_colls = [c for c in ax.collections if 'Contour' in str(type(c))]
        if contour_colls:
            return 'contour'

        # Scatter collections (non-contour)
        scatter_colls = [c for c in ax.collections if hasattr(c, 'get_offsets') and 'Contour' not in str(type(c))]
        if scatter_colls:
            return 'scatter'

        # Lines
        if ax.get_lines():
            return 'line'

        return 'line'

    def _apply_grid(self, ax, enabled, linestyle, alpha, linewidth):
        """Robustly apply grid visibility and style without side effects."""
        try:
            ax.grid(enabled, linestyle=linestyle, alpha=alpha, linewidth=linewidth)
        except TypeError:
            # Older Matplotlib: fall back to separate tweaks
            ax.grid(enabled)
        # Explicitly style existing lines (both axes) and ensure visibility
        for line in list(ax.xaxis.get_gridlines()) + list(ax.yaxis.get_gridlines()):
            try:
                line.set_linestyle(linestyle)
                line.set_alpha(alpha)
                line.set_linewidth(linewidth)
                line.set_visible(enabled)
            except Exception:
                pass

    @property
    def y_columns(self):
        """Property to mimic y_columns widget interface"""
        class YColumnsProxy:
            def __init__(self, parent):
                self.parent = parent

            @property
            def value(self):
                return tuple(self.parent._y_columns_selected)

            def observe(self, callback, names):
                # Store callback for later use
                self.parent._y_columns_callback = callback

        return YColumnsProxy(self)

    def _build_ui(self):
        """Build the complete user interface"""

        # ========== DATA LOADING SECTION ==========
        self.file_input = widgets.Textarea(
            placeholder='Paste or type file path here (e.g., example_data/cfd_results.dat)',
            description='Data File:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='600px', height='60px')
        )

        self.load_btn = widgets.Button(
            description='Load Data',
            button_style='primary',
            icon='upload'
        )
        self.load_btn.on_click(self._load_data)
        self.data_mode_info = widgets.HTML(
            value='',
            layout=widgets.Layout(margin='5px 0 0 0')
        )

        # ========== SUBPLOT CONTROLS ==========
        self.enable_subplots = widgets.Checkbox(
            value=False,
            description='Enable Subplots',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.enable_subplots.observe(self._on_enable_subplots_change, 'value')

        self.num_subplots_slider = widgets.IntSlider(
            value=2,
            min=1,
            max=12,
            step=1,
            description='# Subplots:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='650px', display='none')
        )
        self.num_subplots_slider.observe(self._on_num_subplots_change, 'value')

        self.auto_layout = widgets.Checkbox(
            value=True,
            description='Auto Layout',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px', display='none')
        )
        self.auto_layout.observe(self._on_auto_layout_change, 'value')

        self.subplot_rows_slider = widgets.IntSlider(
            value=1,
            min=1,
            max=4,
            step=1,
            description='Rows:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='650px', display='none'),
            disabled=True
        )
        self.subplot_rows_slider.observe(self._on_layout_change, 'value')

        self.subplot_cols_slider = widgets.IntSlider(
            value=2,
            min=1,
            max=4,
            step=1,
            description='Cols:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='650px', display='none'),
            disabled=True
        )
        self.subplot_cols_slider.observe(self._on_layout_change, 'value')

        # Container for per-subplot configuration tabs
        self.subplot_tabs_container = widgets.VBox([], layout=widgets.Layout(display='none'))

        # Create all subplots button
        self.create_subplots_btn = widgets.Button(
            description='Create All Subplots',
            button_style='success',
            icon='chart-area',
            layout=widgets.Layout(display='none', width='240px')
        )
        self.create_subplots_btn.on_click(self._create_plot)

        # Plot type selector
        self.plot_type_dropdown = widgets.Dropdown(
            options=['Line Plot', 'Scatter Plot', 'Contour Plot', 'Tricontour Plot', 'Cylindrical Contour', 'Vector Field'],
            value='Line Plot',
            description='Plot Type:',
            style={'description_width': '100px'}
        )
        self.plot_type_dropdown.observe(self._on_plot_type_change, 'value')

        # Column selectors for line/scatter plots
        self.x_column = widgets.Dropdown(
            options=[],
            description='X Axis:',
            style={'description_width': '100px'},
            disabled=True
        )

        # Y column selector - use horizontal checkboxes for cleaner UI
        self.y_columns_label = widgets.HTML('<b>Y Axes (select multiple):</b>')
        self.y_columns_container = widgets.HBox([], layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            width='100%'
        ))
        self._y_column_checkboxes = {}  # Store checkboxes by column name
        self._y_columns_selected = []  # Track selected columns

        # Separate column selectors for contour plots
        self.x_column_contour = widgets.Dropdown(
            options=[],
            description='X Axis:',
            style={'description_width': '100px'},
            disabled=True
        )

        self.y_column_contour = widgets.Dropdown(
            options=[],
            description='Y Axis:',
            style={'description_width': '100px'},
            disabled=True
        )

        self.field_column = widgets.Dropdown(
            options=[],
            description='Field Data:',
            style={'description_width': '100px'},
            disabled=True
        )

        # Cylindrical coordinate selectors
        self.r_column = widgets.Dropdown(
            options=[],
            description='R (radius):',
            style={'description_width': '100px'},
            disabled=True
        )

        self.theta_column = widgets.Dropdown(
            options=[],
            description='θ (theta):',
            style={'description_width': '100px'},
            disabled=True
        )

        self.field_column_cyl = widgets.Dropdown(
            options=[],
            description='Field Data:',
            style={'description_width': '100px'},
            disabled=True
        )

        # Vector field selectors
        self.vector_x_column = widgets.Dropdown(
            options=[],
            description='X Coord:',
            style={'description_width': '100px'},
            disabled=True
        )

        self.vector_y_column = widgets.Dropdown(
            options=[],
            description='Y Coord:',
            style={'description_width': '100px'},
            disabled=True
        )

        self.vector_u_column = widgets.Dropdown(
            options=[],
            description='U Component:',
            style={'description_width': '120px'},
            disabled=True
        )

        self.vector_v_column = widgets.Dropdown(
            options=[],
            description='V Component:',
            style={'description_width': '120px'},
            disabled=True
        )

        self.vector_color_field = widgets.Dropdown(
            options=['None'],
            description='Color Field:',
            style={'description_width': '110px'},
            disabled=True,
            value='None'
        )

        self.vector_overlay_field = widgets.Dropdown(
            options=['None'],
            description='Overlay Field:',
            style={'description_width': '120px'},
            disabled=True,
            value='None'
        )

        # Scatter plot color column selector
        self.scatter_color_column = widgets.Dropdown(
            options=['None'],
            description='Color By:',
            style={'description_width': '100px'},
            disabled=True,
            value='None'
        )

        # Line style customization container
        self.line_style_container = widgets.VBox([])

        # Plot button
        self.plot_btn = widgets.Button(
            description='Create Plot',
            button_style='success',
            icon='line-chart',
            disabled=True
        )
        self.plot_btn.on_click(self._create_plot)

        # Figure output
        self.fig_output = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                height='auto',
                border='1px solid #ddd',
                padding='10px'
            )
        )
        # Dedicated image widget to avoid duplicate rendering from Jupyter front-end
        self.fig_image_widget = widgets.Image(format='png', layout=widgets.Layout(width='100%'))
        with self.fig_output:
            clear_output(wait=True)
            display(self.fig_image_widget)

        # ========== PLOT EDITING SECTION (Initially hidden) ==========
        self.editing_controls_container = widgets.VBox([], layout=widgets.Layout(display='none'))

        # Save section
        self.save_format = widgets.Dropdown(
            options=['.svg', '.png', '.jpg', '.pdf'],
            value='.svg',
            description='Format:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='150px')
        )

        self.save_filename = widgets.Text(
            placeholder='figure',
            description='Filename:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='250px')
        )

        self.save_btn = widgets.Button(
            description='Save Figure',
            button_style='info',
            icon='save'
        )
        self.save_btn.on_click(self._save_figure)

        self.save_section = widgets.HBox([
            self.save_filename,
            self.save_format,
            self.save_btn
        ], layout=widgets.Layout(padding='10px', display='none'))

        # ========== LAYOUT ==========
        file_controls = widgets.VBox([
            widgets.HBox([self.file_input, self.load_btn]),
            self.data_mode_info
        ])

        self.column_selector_container = widgets.VBox([
            widgets.HTML('<b>Select Data Columns:</b>'),
            self.x_column,
            self.y_columns_label,
            self.y_columns_container,
            self.scatter_color_column,
            self.x_column_contour,
            self.y_column_contour,
            self.field_column,
            self.r_column,
            self.theta_column,
            self.field_column_cyl,
            self.vector_x_column,
            self.vector_y_column,
            self.vector_u_column,
            self.vector_v_column,
            self.vector_color_field,
            self.vector_overlay_field,
            self.line_style_container,
            self.plot_btn
        ], layout=widgets.Layout(padding='10px', display='none'))

        data_loading_section = widgets.VBox([
            # Combined headers with gradient background and reduced spacing
            # widgets.HTML('''
            #     <div style="
            #         background: #50589C;
            #         border: 16px solid #F5D3C4;
            #         padding: 20px;
            #         border-radius: 0px;
            #         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            #         margin-bottom: 15px;
            #     ">
            #         <h3 style="
            #             font-family: 'Monaco', monospace;
            #             font-size: 20px;
            #             color: white;
            #             text-align: center;
            #             margin: 0;
            #             text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            #         ">FigSmith: An Interactive Visualization Widget for Scientific Computing Datasets</h3>
            #         <h4 style="
            #             font-family: 'Lucida Console', monospace;
            #             font-size: 12px;
            #             color: white;
            #             text-align: center;
            #             margin: 5px 0 0 0;
            #         ">Jupyter-notebook-friendly lightweight matplotlib wrapper using ipywidgets</h4>
            #     </div>
            # '''),

            widgets.HTML('''
            <div style="padding: 0; margin-bottom: 16px;">
                <div style="border: 20px solid #37353E; padding: 0px; border-radius: 0px;">
                    <div style="
                        background: linear-gradient(to right, #00b4db, #0083b0);
                        padding: 18px 20px;
                        border-radius: 0px;
                        color: #ffffff;
                        text-align: center;
                        box-shadow:
                        0 8px 18px rgba(0,0,0,0.15),
                        inset 0 0 0 2px rgba(255,255,255,0.15);
                    ">
                        <h3 style="
                        font-family: Monaco, Menlo, Consolas, 'Courier New', monospace;
                        font-size: 20px;
                        margin: 0;
                        letter-spacing: 0.2px;
                        ">FigSmith: Interactive Visualization for Scientific Computing</h3>
                        <p style="
                        font-family: 'Lucida Console', Monaco, monospace;
                        font-size: 12px;
                        margin: 6px 0 0 0;
                        opacity: 0.95;
                        ">Jupyter‑friendly Matplotlib wrapper powered by ipywidgets</p>
                    </div>
                    </div>
                </div>
                </div>
            </div>
            '''),

            # File selection row with enhanced styling
            widgets.HBox([
                file_controls
            ], layout=widgets.Layout(
                margin='10px 0',
                display='flex',
                justify_content='center'
            )),
            # Subplot controls
            widgets.VBox([
                widgets.HTML('<h4 style="margin: 0 0 10px 0; color: #2E5090;">📊 Subplot Configuration</h4>'),
                self.enable_subplots,
                widgets.HBox([
                    self.num_subplots_slider,
                    self.auto_layout
                ], layout=widgets.Layout(margin='8px 0')),
                widgets.HBox([
                    self.subplot_rows_slider,
                    self.subplot_cols_slider
                ], layout=widgets.Layout(margin='8px 0'))
            ], layout=widgets.Layout(
                margin='15px 0',
                padding='20px',
                border='2px solid #2E5090',
                border_radius='8px',
                background_color='#F0F4F8'
            )),
            # Per-subplot configuration tabs
            self.subplot_tabs_container,
            # Create subplots button
            widgets.HBox([self.create_subplots_btn], layout=widgets.Layout(
                margin='10px 0',
                display='flex',
                justify_content='center'
            )),
            # Plot type and column selector with better spacing
            widgets.VBox([
                self.plot_type_dropdown,
                self.column_selector_container
            ], layout=widgets.Layout(
                margin='10px 0',
                padding='15px'
            ))
        ], layout=widgets.Layout(
            width='100%',
            padding='20px',
            background_color='#F8F9FA',
            border='1px solid #DEE2E6',
            border_radius='12px'
        ))

        self.layout = widgets.VBox([
            data_loading_section,
            self.fig_output,
            self.save_section,
            self.editing_controls_container
        ])

    def _create_y_column_checkboxes(self, columns):
        """Create horizontal checkboxes for Y column selection"""
        self._y_column_checkboxes = {}
        checkboxes = []

        for col in columns:
            checkbox = widgets.Checkbox(
                value=False,
                description=col,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='auto', margin='2px 10px 2px 0px')
            )

            def make_checkbox_callback(col_name):
                def callback(change):
                    if change.new:
                        if col_name not in self._y_columns_selected:
                            self._y_columns_selected.append(col_name)
                    else:
                        if col_name in self._y_columns_selected:
                            self._y_columns_selected.remove(col_name)
                    # Trigger callback if exists
                    if hasattr(self, '_y_columns_callback'):
                        self._y_columns_callback({'new': tuple(self._y_columns_selected)})
                    # Update line style controls for line plots
                    if self.plot_type == 'line' and self.loader:
                        self._create_line_style_controls()
                return callback

            checkbox.observe(make_checkbox_callback(col), 'value')
            self._y_column_checkboxes[col] = checkbox
            checkboxes.append(checkbox)

        self.y_columns_container.children = checkboxes

    def _on_enable_subplots_change(self, change):
        """Handle enable subplots checkbox change"""
        self.subplot_mode = change.new

        if self.subplot_mode:
            # Show subplot controls
            self.num_subplots_slider.layout.display = 'block'
            self.auto_layout.layout.display = 'block'
            self.subplot_rows_slider.layout.display = 'block'
            self.subplot_cols_slider.layout.display = 'block'
            self.create_subplots_btn.layout.display = 'block'

            # Build subplot tabs
            self._build_subplot_tabs()

            # Hide single plot controls
            self.column_selector_container.layout.display = 'none'
            self.plot_type_dropdown.layout.display = 'none'
            self.plot_btn.layout.display = 'none'
        else:
            # Hide subplot controls
            self.num_subplots_slider.layout.display = 'none'
            self.auto_layout.layout.display = 'none'
            self.subplot_rows_slider.layout.display = 'none'
            self.subplot_cols_slider.layout.display = 'none'
            self.subplot_tabs_container.layout.display = 'none'
            self.create_subplots_btn.layout.display = 'none'

            # Show single plot controls
            self.column_selector_container.layout.display = 'block'
            self.plot_type_dropdown.layout.display = 'block'
            self.plot_btn.layout.display = 'block'

    def _on_num_subplots_change(self, change):
        """Handle number of subplots change"""
        self.num_subplots = change.new

        # Auto-calculate layout if enabled
        if self.auto_layout.value:
            self._calculate_auto_layout()

        # Rebuild subplot tabs
        self._build_subplot_tabs()

    def _on_auto_layout_change(self, change):
        """Handle auto layout checkbox change"""
        if change.new:
            # Disable manual row/col sliders
            self.subplot_rows_slider.disabled = True
            self.subplot_cols_slider.disabled = True
            self._calculate_auto_layout()
        else:
            # Enable manual row/col sliders
            self.subplot_rows_slider.disabled = False
            self.subplot_cols_slider.disabled = False

    def _on_layout_change(self, change):
        """Handle manual layout change"""
        if not self.auto_layout.value:
            self.subplot_rows = self.subplot_rows_slider.value
            self.subplot_cols = self.subplot_cols_slider.value

    def _calculate_auto_layout(self):
        """Automatically calculate optimal subplot layout"""
        import math
        n = self.num_subplots

        # Try to make as square as possible
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        self.subplot_rows = rows
        self.subplot_cols = cols

        # Update sliders
        self.subplot_rows_slider.value = rows
        self.subplot_cols_slider.value = cols

    def _build_subplot_tabs(self):
        """Build per-subplot configuration tabs"""
        if not self.subplot_mode:
            return

        # Initialize subplot configs if needed
        while len(self.subplot_configs) < self.num_subplots:
            self.subplot_configs.append({
                'file_path': '',
                'loader': None,
                'plot_type': 'line',
                'x_column': None,
                'y_columns': [],
                'x_column_contour': None,
                'y_column_contour': None,
                'field_column': None,
                'r_column': None,
                'theta_column': None,
                'field_column_cyl': None,
                'scatter_color_column': 'None',
                'vector_x_column': None,
                'vector_y_column': None,
                'vector_u_column': None,
                'vector_v_column': None,
                'vector_color_field': 'None',
                'vector_overlay_field': 'None'
            })

        target_len = len(self.subplot_configs)
        while len(self.subplot_vector_data) < target_len:
            self.subplot_vector_data.append(None)
        while len(self.subplot_vector_quivers) < target_len:
            self.subplot_vector_quivers.append(None)
        while len(self.subplot_vector_overlays) < target_len:
            self.subplot_vector_overlays.append(None)
        while len(self.subplot_vector_colorbars) < target_len:
            self.subplot_vector_colorbars.append(None)
        while len(self.subplot_vector_quiver_colorbars) < target_len:
            self.subplot_vector_quiver_colorbars.append(None)
        while len(self.subplot_vector_styles) < target_len:
            self.subplot_vector_styles.append(self.vector_style.copy())

        # Trim extra configs
        self.subplot_configs = self.subplot_configs[:self.num_subplots]
        self.subplot_vector_data = self.subplot_vector_data[:self.num_subplots]
        self.subplot_vector_quivers = self.subplot_vector_quivers[:self.num_subplots]
        self.subplot_vector_overlays = self.subplot_vector_overlays[:self.num_subplots]
        self.subplot_vector_colorbars = self.subplot_vector_colorbars[:self.num_subplots]
        self.subplot_vector_styles = self.subplot_vector_styles[:self.num_subplots]
        self.subplot_vector_quiver_colorbars = self.subplot_vector_quiver_colorbars[:self.num_subplots]

        # Build tab for each subplot
        subplot_tabs = []
        for i in range(self.num_subplots):
            subplot_tab = self._build_single_subplot_tab(i)
            subplot_tabs.append(subplot_tab)

        # Create Tab widget
        tabs = widgets.Tab(children=subplot_tabs)
        for i in range(self.num_subplots):
            tabs.set_title(i, f'Subplot {i+1}')

        # Show the tabs
        self.subplot_tabs_container.children = [tabs]
        self.subplot_tabs_container.layout.display = 'block'

    def _build_single_subplot_tab(self, subplot_idx):
        """Build configuration UI for a single subplot"""
        config = self.subplot_configs[subplot_idx]

        # File input for this subplot
        file_input = widgets.Textarea(
            placeholder='Paste or type file path here',
            description='Data File:',
            value=config['file_path'],
            style={'description_width': '100px'},
            layout=widgets.Layout(width='600px', height='60px')
        )

        load_btn = widgets.Button(
            description='Load Data',
            button_style='primary',
            icon='upload'
        )

        # Plot type selector
        plot_type_dropdown = widgets.Dropdown(
            options=['Line Plot', 'Scatter Plot', 'Contour Plot', 'Tricontour Plot', 'Cylindrical Contour', 'Vector Field'],
            value='Line Plot',
            description='Plot Type:',
            style={'description_width': '100px'}
        )

        # Column selectors
        x_column = widgets.Dropdown(
            options=[],
            description='X Axis:',
            style={'description_width': '100px'},
            disabled=True
        )

        y_columns_label = widgets.HTML('<b>Y Axes (select multiple):</b>')
        y_columns_container = widgets.HBox([], layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            width='100%'
        ))

        # Contour selectors
        x_column_contour = widgets.Dropdown(
            options=[],
            description='X Axis:',
            style={'description_width': '100px'},
            disabled=True
        )

        y_column_contour = widgets.Dropdown(
            options=[],
            description='Y Axis:',
            style={'description_width': '100px'},
            disabled=True
        )

        field_column = widgets.Dropdown(
            options=[],
            description='Field Data:',
            style={'description_width': '100px'},
            disabled=True
        )

        # Cylindrical selectors
        r_column = widgets.Dropdown(
            options=[],
            description='R (radius):',
            style={'description_width': '100px'},
            disabled=True
        )

        theta_column = widgets.Dropdown(
            options=[],
            description='θ (theta):',
            style={'description_width': '100px'},
            disabled=True
        )

        field_column_cyl = widgets.Dropdown(
            options=[],
            description='Field Data:',
            style={'description_width': '100px'},
            disabled=True
        )

        scatter_color_column = widgets.Dropdown(
            options=['None'],
            description='Color By:',
            style={'description_width': '100px'},
            disabled=True,
            value='None'
        )

        vector_x_column = widgets.Dropdown(
            options=[],
            description='X Coord:',
            style={'description_width': '110px'},
            disabled=True
        )

        vector_y_column = widgets.Dropdown(
            options=[],
            description='Y Coord:',
            style={'description_width': '110px'},
            disabled=True
        )

        vector_u_column = widgets.Dropdown(
            options=[],
            description='U Component:',
            style={'description_width': '130px'},
            disabled=True
        )

        vector_v_column = widgets.Dropdown(
            options=[],
            description='V Component:',
            style={'description_width': '130px'},
            disabled=True
        )

        vector_color_field = widgets.Dropdown(
            options=['None'],
            description='Color Field:',
            style={'description_width': '120px'},
            disabled=True,
            value='None'
        )

        vector_overlay_field = widgets.Dropdown(
            options=['None'],
            description='Overlay Field:',
            style={'description_width': '120px'},
            disabled=True,
            value='None'
        )

        # Store widgets in config for later access
        config['widgets'] = {
            'file_input': file_input,
            'load_btn': load_btn,
            'plot_type_dropdown': plot_type_dropdown,
            'x_column': x_column,
            'y_columns_container': y_columns_container,
            'y_columns_checkboxes': {},
            'y_columns_selected': [],
            'x_column_contour': x_column_contour,
            'y_column_contour': y_column_contour,
            'field_column': field_column,
            'r_column': r_column,
            'theta_column': theta_column,
            'field_column_cyl': field_column_cyl,
            'scatter_color_column': scatter_color_column,
            'vector_x_column': vector_x_column,
            'vector_y_column': vector_y_column,
            'vector_u_column': vector_u_column,
            'vector_v_column': vector_v_column,
            'vector_color_field': vector_color_field,
            'vector_overlay_field': vector_overlay_field
        }

        # Define callbacks
        def on_load_data(btn):
            self._load_subplot_data(subplot_idx)

        def on_plot_type_change(change):
            self._on_subplot_plot_type_change(subplot_idx, change)

        load_btn.on_click(on_load_data)
        plot_type_dropdown.observe(on_plot_type_change, 'value')

        # Build layout
        column_selector_area = widgets.VBox([
            x_column,
            y_columns_label,
            y_columns_container,
            scatter_color_column,
            x_column_contour,
            y_column_contour,
            field_column,
            r_column,
            theta_column,
            field_column_cyl,
            vector_x_column,
            vector_y_column,
            vector_u_column,
            vector_v_column,
            vector_color_field,
            vector_overlay_field
        ])

        # Initially hide column selectors based on plot type
        self._update_subplot_column_visibility(subplot_idx)

        tab_content = widgets.VBox([
            widgets.HBox([file_input, load_btn]),
            plot_type_dropdown,
            column_selector_area
        ], layout=widgets.Layout(padding='10px'))

        return tab_content

    def _load_subplot_data(self, subplot_idx):
        """Load data for a specific subplot"""
        config = self.subplot_configs[subplot_idx]
        widgets_dict = config['widgets']
        filepath = widgets_dict['file_input'].value.strip()

        if not filepath:
            print(f"❌ Subplot {subplot_idx+1}: Please enter a file path")
            return

        try:
            # Load data
            loader = DataLoader(filepath)
            config['loader'] = loader
            config['file_path'] = filepath

            # Reset subplot-specific cached data when loading new file
            widgets_dict['y_columns_selected'] = []
            widgets_dict['y_columns_checkboxes'] = {}
            while len(self.subplot_scatter_data) <= subplot_idx:
                self.subplot_scatter_data.append(None)
            self.subplot_scatter_data[subplot_idx] = None
            while len(self.subplot_contour_data) <= subplot_idx:
                self.subplot_contour_data.append(None)
            self.subplot_contour_data[subplot_idx] = None

            print(f"✓ Subplot {subplot_idx+1}: Loaded {filepath}")
            print(f"  Columns: {loader.columns}")
            print(f"  Rows: {len(loader.df)}")

            # Populate column selectors
            columns = list(loader.columns)
            widgets_dict['x_column'].options = columns
            widgets_dict['x_column_contour'].options = columns
            widgets_dict['y_column_contour'].options = columns
            widgets_dict['field_column'].options = columns
            widgets_dict['r_column'].options = columns
            widgets_dict['theta_column'].options = columns
            widgets_dict['field_column_cyl'].options = columns
            widgets_dict['scatter_color_column'].options = ['None'] + columns
            widgets_dict['vector_x_column'].options = columns
            widgets_dict['vector_y_column'].options = columns
            widgets_dict['vector_u_column'].options = columns
            widgets_dict['vector_v_column'].options = columns
            widgets_dict['vector_color_field'].options = ['None'] + columns
            widgets_dict['vector_overlay_field'].options = ['None'] + columns

            # Create Y column checkboxes
            self._create_subplot_y_column_checkboxes(subplot_idx, columns)

            # Enable selectors
            widgets_dict['x_column'].disabled = False
            widgets_dict['x_column_contour'].disabled = False
            widgets_dict['y_column_contour'].disabled = False
            widgets_dict['field_column'].disabled = False
            widgets_dict['r_column'].disabled = False
            widgets_dict['theta_column'].disabled = False
            widgets_dict['field_column_cyl'].disabled = False
            widgets_dict['scatter_color_column'].disabled = False
            widgets_dict['vector_x_column'].disabled = False
            widgets_dict['vector_y_column'].disabled = False
            widgets_dict['vector_u_column'].disabled = False
            widgets_dict['vector_v_column'].disabled = False
            widgets_dict['vector_color_field'].disabled = False
            widgets_dict['vector_overlay_field'].disabled = False

            # Set defaults
            if len(columns) >= 2:
                widgets_dict['x_column'].value = columns[0]
                widgets_dict['x_column_contour'].value = columns[0]
                widgets_dict['y_column_contour'].value = columns[1]
                widgets_dict['r_column'].value = columns[0]
                widgets_dict['theta_column'].value = columns[1]
                # Select second column by default
                if columns[1] in widgets_dict['y_columns_checkboxes']:
                    widgets_dict['y_columns_checkboxes'][columns[1]].value = True
                widgets_dict['vector_x_column'].value = columns[0]
                widgets_dict['vector_y_column'].value = columns[1]

            if len(columns) >= 3:
                widgets_dict['field_column'].value = columns[2]
                widgets_dict['field_column_cyl'].value = columns[2]
                widgets_dict['vector_u_column'].value = columns[2]

            if len(columns) >= 4:
                widgets_dict['vector_v_column'].value = columns[3]

            widgets_dict['vector_color_field'].value = 'None'
            widgets_dict['vector_overlay_field'].value = 'None'

            # Reset subplot vector caches
            while len(self.subplot_vector_data) <= subplot_idx:
                self.subplot_vector_data.append(None)
            self.subplot_vector_data[subplot_idx] = None
            while len(self.subplot_vector_quivers) <= subplot_idx:
                self.subplot_vector_quivers.append(None)
            self.subplot_vector_quivers[subplot_idx] = None
            while len(self.subplot_vector_overlays) <= subplot_idx:
                self.subplot_vector_overlays.append(None)
            self.subplot_vector_overlays[subplot_idx] = None
            while len(self.subplot_vector_colorbars) <= subplot_idx:
                self.subplot_vector_colorbars.append(None)
            self.subplot_vector_colorbars[subplot_idx] = None
            while len(self.subplot_vector_quiver_colorbars) <= subplot_idx:
                self.subplot_vector_quiver_colorbars.append(None)
            self.subplot_vector_quiver_colorbars[subplot_idx] = None
            while len(self.subplot_vector_styles) <= subplot_idx:
                self.subplot_vector_styles.append(self.vector_style.copy())
            self.subplot_vector_styles[subplot_idx] = self.vector_style.copy()

        except Exception as e:
            print(f"❌ Subplot {subplot_idx+1}: Error loading data: {e}")

    def _create_subplot_y_column_checkboxes(self, subplot_idx, columns):
        """Create Y column checkboxes for a specific subplot"""
        config = self.subplot_configs[subplot_idx]
        widgets_dict = config['widgets']

        checkboxes = []
        widgets_dict['y_columns_checkboxes'] = {}

        for col in columns:
            checkbox = widgets.Checkbox(
                value=False,
                description=col,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='auto', margin='2px 10px 2px 0px')
            )

            def make_callback(col_name):
                def callback(change):
                    if change.new:
                        if col_name not in widgets_dict['y_columns_selected']:
                            widgets_dict['y_columns_selected'].append(col_name)
                    else:
                        if col_name in widgets_dict['y_columns_selected']:
                            widgets_dict['y_columns_selected'].remove(col_name)
                return callback

            checkbox.observe(make_callback(col), 'value')
            widgets_dict['y_columns_checkboxes'][col] = checkbox
            checkboxes.append(checkbox)

        widgets_dict['y_columns_container'].children = checkboxes

    def _on_subplot_plot_type_change(self, subplot_idx, change):
        """Handle plot type change for a specific subplot"""
        config = self.subplot_configs[subplot_idx]
        plot_type_str = change.new

        # Map plot type string to internal type
        type_map = {
            'Line Plot': 'line',
            'Scatter Plot': 'scatter',
            'Contour Plot': 'contour',
            'Tricontour Plot': 'tricontour',
            'Cylindrical Contour': 'cylindrical',
            'Vector Field': 'vector'
        }
        config['plot_type'] = type_map.get(plot_type_str, 'line')

        # Update visibility
        self._update_subplot_column_visibility(subplot_idx)

    def _update_subplot_column_visibility(self, subplot_idx):
        """Update column selector visibility for a subplot based on plot type"""
        config = self.subplot_configs[subplot_idx]
        widgets_dict = config['widgets']
        plot_type = config['plot_type']

        # Hide all first
        widgets_dict['x_column'].layout.display = 'none'
        widgets_dict['y_columns_container'].layout.display = 'none'
        widgets_dict['scatter_color_column'].layout.display = 'none'
        widgets_dict['x_column_contour'].layout.display = 'none'
        widgets_dict['y_column_contour'].layout.display = 'none'
        widgets_dict['field_column'].layout.display = 'none'
        widgets_dict['r_column'].layout.display = 'none'
        widgets_dict['theta_column'].layout.display = 'none'
        widgets_dict['field_column_cyl'].layout.display = 'none'
        widgets_dict['vector_x_column'].layout.display = 'none'
        widgets_dict['vector_y_column'].layout.display = 'none'
        widgets_dict['vector_u_column'].layout.display = 'none'
        widgets_dict['vector_v_column'].layout.display = 'none'
        widgets_dict['vector_color_field'].layout.display = 'none'
        widgets_dict['vector_overlay_field'].layout.display = 'none'

        # Show relevant selectors
        if plot_type == 'line':
            widgets_dict['x_column'].layout.display = 'block'
            widgets_dict['y_columns_container'].layout.display = 'flex'
        elif plot_type == 'scatter':
            widgets_dict['x_column'].layout.display = 'block'
            widgets_dict['y_columns_container'].layout.display = 'flex'
            widgets_dict['scatter_color_column'].layout.display = 'block'
        elif plot_type in ['contour', 'tricontour']:
            widgets_dict['x_column_contour'].layout.display = 'block'
            widgets_dict['y_column_contour'].layout.display = 'block'
            widgets_dict['field_column'].layout.display = 'block'
        elif plot_type == 'cylindrical':
            widgets_dict['r_column'].layout.display = 'block'
            widgets_dict['theta_column'].layout.display = 'block'
            widgets_dict['field_column_cyl'].layout.display = 'block'
        elif plot_type == 'vector':
            widgets_dict['vector_x_column'].layout.display = 'block'
            widgets_dict['vector_y_column'].layout.display = 'block'
            widgets_dict['vector_u_column'].layout.display = 'block'
            widgets_dict['vector_v_column'].layout.display = 'block'
            widgets_dict['vector_color_field'].layout.display = 'block'
            widgets_dict['vector_overlay_field'].layout.display = 'block'

    def _apply_loader(self, loader, *, source_label=None, preselect=None, in_memory=False, reset_state=True):
        """Populate UI widgets from a prepared DataLoader."""
        if loader is None or getattr(loader, 'df', None) is None:
            raise ValueError("Loader requires a populated DataFrame")

        if reset_state:
            self._reset_before_data_ingest()

        self.loader = loader
        columns = list(loader.columns)
        if not columns:
            raise ValueError("Loaded data contains no columns")

        preselect = preselect or {}
        # Update file widgets based on source
        in_memory_note = None
        if in_memory:
            in_memory_note = "Using in-memory data"
            if source_label:
                in_memory_note = f"{in_memory_note} ({source_label})"
        self._set_in_memory_mode(in_memory, message=in_memory_note)

        # Configure dropdown options
        self.x_column.options = columns
        self.x_column_contour.options = columns
        self.y_column_contour.options = columns
        self.field_column.options = columns
        self.r_column.options = columns
        self.theta_column.options = columns
        self.field_column_cyl.options = columns
        self.scatter_color_column.options = ['None'] + columns
        self.vector_x_column.options = columns
        self.vector_y_column.options = columns
        self.vector_u_column.options = columns
        self.vector_v_column.options = columns
        self.vector_color_field.options = ['None'] + columns
        self.vector_overlay_field.options = ['None'] + columns

        # Refresh Y-column checkboxes
        self._y_columns_selected = []
        self._create_y_column_checkboxes(columns)

        def _set_dropdown(widget, value=None, fallback_idx=None):
            if value in columns:
                widget.value = value
            elif fallback_idx is not None and 0 <= fallback_idx < len(columns):
                widget.value = columns[fallback_idx]

        # Line/scatter selections
        x_choice = self._infer_x_column(columns, preselect.get('x'))
        if x_choice is not None:
            self.x_column.value = x_choice

        y_pref = preselect.get('y') or []
        if isinstance(y_pref, str):
            y_pref = [y_pref]
        y_pref = list(dict.fromkeys(y_pref))  # keep order, drop duplicates
        selected_y = [col for col in y_pref if col in columns]
        if not selected_y:
            selected_y = self._infer_default_y_columns(columns, x_choice)
        for col in selected_y:
            checkbox = self._y_column_checkboxes.get(col)
            if checkbox is not None:
                checkbox.value = True

        # Contour/cylindrical selections
        field_pref = preselect.get('field') or {}
        _set_dropdown(self.x_column_contour, field_pref.get('x'), fallback_idx=0)
        _set_dropdown(self.y_column_contour, field_pref.get('y'), fallback_idx=1)
        _set_dropdown(self.field_column, field_pref.get('field'), fallback_idx=2)
        _set_dropdown(self.r_column, field_pref.get('r'), fallback_idx=0)
        _set_dropdown(self.theta_column, field_pref.get('theta'), fallback_idx=1)
        _set_dropdown(self.field_column_cyl, field_pref.get('field'), fallback_idx=2)

        # Vector selections
        vector_pref = preselect.get('vector') or {}
        _set_dropdown(self.vector_x_column, vector_pref.get('x'), fallback_idx=0)
        _set_dropdown(self.vector_y_column, vector_pref.get('y'), fallback_idx=1)
        _set_dropdown(self.vector_u_column, vector_pref.get('u'), fallback_idx=2)
        _set_dropdown(self.vector_v_column, vector_pref.get('v'), fallback_idx=3)

        color_value = vector_pref.get('c')
        overlay_value = vector_pref.get('overlay')
        self.vector_color_field.value = color_value if color_value in self.vector_color_field.options else 'None'
        self.vector_overlay_field.value = overlay_value if overlay_value in self.vector_overlay_field.options else 'None'

        self.scatter_color_column.value = 'None'
        scatter_pref = preselect.get('scatter_color')
        if scatter_pref in self.scatter_color_column.options:
            self.scatter_color_column.value = scatter_pref

        # Enable widgets now that we have data
        controls = [
            self.x_column,
            self.x_column_contour,
            self.y_column_contour,
            self.field_column,
            self.r_column,
            self.theta_column,
            self.field_column_cyl,
            self.scatter_color_column,
            self.vector_x_column,
            self.vector_y_column,
            self.vector_u_column,
            self.vector_v_column,
            self.vector_color_field,
            self.vector_overlay_field
        ]
        for widget in controls:
            widget.disabled = False

        self.plot_btn.disabled = False
        self.column_selector_container.layout.display = 'block'

        # Refresh dependent UI (line styles/visibility)
        self._on_plot_type_change(None)

        descriptor = source_label or ('in-memory data' if in_memory else '')
        prefix = "✓ Using in-memory data" if in_memory else "✓ Loaded data"
        if descriptor:
            prefix = f"{prefix} ({descriptor})"
        print(prefix)
        print(f"  Columns: {columns}")
        print(f"  Rows: {len(loader.df)}")

    def _load_data(self, btn=None):
        """Load data from file"""
        filepath = self.file_input.value.strip()

        if not filepath:
            print("❌ Please enter a file path")
            return

        try:
            self._set_in_memory_mode(False)
            self._reset_before_data_ingest()
            loader = DataLoader(filepath)
            self._apply_loader(loader, source_label=filepath, in_memory=False, reset_state=False)

        except Exception as e:
            print(f"❌ Error loading file: {e}")

    def _on_plot_type_change(self, change):
        """Handle plot type changes"""
        plot_type_map = {
            'Line Plot': 'line',
            'Scatter Plot': 'scatter',
            'Contour Plot': 'contour',
            'Tricontour Plot': 'tricontour',
            'Cylindrical Contour': 'cylindrical',
            'Vector Field': 'vector'
        }
        self.plot_type = plot_type_map.get(self.plot_type_dropdown.value, 'line')

        # Hide all selectors first
        self.x_column.layout.display = 'none'
        self.y_columns_label.layout.display = 'none'
        self.y_columns_container.layout.display = 'none'
        self.scatter_color_column.layout.display = 'none'
        self.x_column_contour.layout.display = 'none'
        self.y_column_contour.layout.display = 'none'
        self.field_column.layout.display = 'none'
        self.r_column.layout.display = 'none'
        self.theta_column.layout.display = 'none'
        self.field_column_cyl.layout.display = 'none'
        self.vector_x_column.layout.display = 'none'
        self.vector_y_column.layout.display = 'none'
        self.vector_u_column.layout.display = 'none'
        self.vector_v_column.layout.display = 'none'
        self.vector_color_field.layout.display = 'none'
        self.vector_overlay_field.layout.display = 'none'

        # Show appropriate selectors based on plot type
        if self.plot_type in ['line', 'scatter']:
            self.x_column.layout.display = 'block'
            self.y_columns_label.layout.display = 'block'
            self.y_columns_container.layout.display = 'flex'

            # Show/hide scatter color column selector
            if self.plot_type == 'scatter':
                self.scatter_color_column.layout.display = 'block'

            # Create line style customizers for line plots only
            if self.plot_type == 'line' and self.loader:
                self._create_line_style_controls()
            else:
                self.line_style_container.children = []

        elif self.plot_type in ['contour', 'tricontour']:
            self.x_column_contour.layout.display = 'block'
            self.y_column_contour.layout.display = 'block'
            self.field_column.layout.display = 'block'
            self.line_style_container.children = []

        elif self.plot_type == 'cylindrical':
            self.r_column.layout.display = 'block'
            self.theta_column.layout.display = 'block'
            self.field_column_cyl.layout.display = 'block'
            self.line_style_container.children = []
        elif self.plot_type == 'vector':
            self.vector_x_column.layout.display = 'block'
            self.vector_y_column.layout.display = 'block'
            self.vector_u_column.layout.display = 'block'
            self.vector_v_column.layout.display = 'block'
            self.vector_color_field.layout.display = 'block'
            self.vector_overlay_field.layout.display = 'block'
            self.line_style_container.children = []

    def _on_y_columns_change(self, change):
        """Handle Y column selection changes"""
        if self.plot_type == 'line' and self.loader:
            self._create_line_style_controls()

    def _create_line_style_controls(self):
        """Create controls for customizing individual lines (line plots only)"""
        if not self.loader or not self.y_columns.value:
            return

        # Only show for line plots, not scatter plots
        if self.plot_type != 'line':
            self.line_style_container.children = []
            return

        controls = [widgets.HTML('<b>Initial Line Styles:</b>')]

        colors = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

        for i, y_col in enumerate(self.y_columns.value):
            # Create style controls for each selected y column
            color_picker = widgets.Dropdown(
                options=colors,
                value=colors[i % len(colors)],
                description=f'{y_col} Color:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='300px')
            )

            style_picker = widgets.Dropdown(
                options=[('Solid', '-'), ('Dashed', '--'), ('Dotted', ':'), ('Dash-dot', '-.')],
                value='-',
                description=f'{y_col} Style:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='300px')
            )

            marker_picker = widgets.Dropdown(
                options=[('None', ''), ('Circle', 'o'), ('Square', 's'), ('Triangle', '^'), ('Star', '*')],
                value='',
                description=f'{y_col} Marker:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='300px')
            )

            # Store references
            self.line_styles[y_col] = {
                'color': color_picker,
                'style': style_picker,
                'marker': marker_picker
            }

            controls.extend([color_picker, style_picker, marker_picker,
                           widgets.HTML('<hr style="margin: 5px 0;">')])

        self.line_style_container.children = controls

    def _create_plot(self, btn=None):
        """Create the plot based on selections"""
        # Check if we're in subplot mode
        if self.subplot_mode:
            self._create_subplots()
            return

        # Single plot mode
        if not self.loader:
            print("❌ No data loaded")
            return

        try:
            # Clear previous plot
            if self.fig:
                plt.close(self.fig)

            # Create figure (static display)
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(7, 5))

            if self.plot_type == 'line':
                self._plot_line()
            elif self.plot_type == 'scatter':
                self._plot_scatter()
            elif self.plot_type == 'contour':
                self._plot_contour()
            elif self.plot_type == 'tricontour':
                self._plot_tricontour()
            elif self.plot_type == 'cylindrical':
                self._plot_cylindrical_contour()
            elif self.plot_type == 'vector':
                self._plot_vector()

            # Initialize property manager for editing
            self.prop_manager = PropertyManager(self.fig, self.ax)

            # Ensure grid is disabled for contour plots (after PropertyManager init)
            if self.plot_type in ['contour', 'tricontour', 'cylindrical']:
                self.ax.grid(False)

            # Display figure (static)
            self._render_current_figure()

            print("✓ Plot created!")

            # Show editing controls
            self._build_editing_controls()
            self.editing_controls_container.layout.display = 'block'
            self.save_section.layout.display = 'flex'

            print("  📊 Scroll down to see all tunable parameter tabs.")

        except Exception as e:
            print(f"❌ Error creating plot: {e}")
            import traceback
            traceback.print_exc()

    def _create_subplots(self):
        """Create subplot grid based on configurations"""
        try:
            # Sync subplot counts and layout from widgets to avoid stale values
            try:
                self.num_subplots = int(self.num_subplots_slider.value)
            except Exception:
                self.num_subplots = max(1, self.num_subplots)

            if self.auto_layout.value:
                self._calculate_auto_layout()
            else:
                try:
                    rows = int(self.subplot_rows_slider.value)
                except Exception:
                    rows = max(1, self.subplot_rows)
                rows = max(1, rows)
                import math
                cols = math.ceil(self.num_subplots / rows)
                self.subplot_rows = rows
                self.subplot_cols = max(1, int(cols))

            # Validate that at least one subplot has data
            has_data = False
            for config in self.subplot_configs:
                if config.get('loader') is not None:
                    has_data = True
                    break

            if not has_data:
                print("❌ No data loaded for any subplot. Please load data first.")
                return

            # Clear previous plot
            if self.fig:
                plt.close(self.fig)

            # Clear previous colorbars
            self.colorbars = []
            self.colorbar_axes = []

            # Calculate figure size based on subplot grid
            fig_width = self.subplot_cols * 5
            fig_height = self.subplot_rows * 4

            # Create subplot grid
            plt.ioff()
            self.fig, axes = plt.subplots(
                int(self.subplot_rows),
                int(self.subplot_cols),
                figsize=(fig_width, fig_height)
            )

            # Handle axes being a single object or array; always store as a Python list
            if self.subplot_rows * self.subplot_cols == 1:
                self.axes_list = [axes]
            elif self.subplot_rows == 1 or self.subplot_cols == 1:
                self.axes_list = list(axes)
            else:
                # Ensure list type (not numpy.ndarray) so membership/indexing works
                self.axes_list = list(axes.flatten())

            # Plot each subplot
            for i, config in enumerate(self.subplot_configs):
                if i >= len(self.axes_list):
                    break

                self.ax = self.axes_list[i]  # Set current axis for plotting methods
                self.loader = config.get('loader')  # Set current loader

                if self.loader is None:
                    # Clear empty subplot
                    self.ax.text(0.5, 0.5, f'Subplot {i+1}\n(No data loaded)',
                                ha='center', va='center', transform=self.ax.transAxes,
                                fontsize=12, color='gray')
                    self.ax.set_xticks([])
                    self.ax.set_yticks([])
                    continue

                # Plot based on configured type
                plot_type = config['plot_type']
                self._plot_subplot(i, config, plot_type)

            # Remove unused subplots so the display does not duplicate panels
            total_axes = len(self.axes_list)
            for i in range(self.num_subplots, total_axes):
                try:
                    self.fig.delaxes(self.axes_list[i])
                except Exception:
                    self.axes_list[i].axis('off')
            self.axes_list = self.axes_list[:self.num_subplots]

            # Apply default spacing
            self.fig.subplots_adjust(hspace=0.3, wspace=0.3)

            # Display figure
            self._render_current_figure()

            print(f"✓ Created {len(self.subplot_configs)} subplot(s)!")

            # Show editing controls (these will be modified for subplots)
            self._build_editing_controls()
            self.editing_controls_container.layout.display = 'block'
            self.save_section.layout.display = 'flex'

            print("  📊 Scroll down to see all tunable parameter tabs.")

        except Exception as e:
            print(f"❌ Error creating subplots: {e}")
            import traceback
            traceback.print_exc()

    def _plot_subplot(self, subplot_idx, config, plot_type):
        """Plot a single subplot based on configuration"""
        widgets_dict = config['widgets']

        try:
            if plot_type == 'line':
                self._plot_subplot_line(subplot_idx, config, widgets_dict)
            elif plot_type == 'scatter':
                self._plot_subplot_scatter(subplot_idx, config, widgets_dict)
            elif plot_type == 'contour':
                self._plot_subplot_contour(subplot_idx, config, widgets_dict)
            elif plot_type == 'tricontour':
                self._plot_subplot_tricontour(subplot_idx, config, widgets_dict)
            elif plot_type == 'cylindrical':
                self._plot_subplot_cylindrical(subplot_idx, config, widgets_dict)
            elif plot_type == 'vector':
                self._plot_subplot_vector(subplot_idx, config, widgets_dict)

        except Exception as e:
            print(f"❌ Error plotting subplot {subplot_idx+1}: {e}")
            self.ax.text(0.5, 0.5, f'Error in subplot {subplot_idx+1}',
                        ha='center', va='center', transform=self.ax.transAxes,
                        fontsize=10, color='red')

    def _plot_subplot_line(self, subplot_idx, config, widgets_dict):
        """Create line plot for subplot"""
        x_col = widgets_dict['x_column'].value
        y_cols = widgets_dict['y_columns_selected']

        if not x_col or not y_cols:
            return

        x_data = self.loader.get_column(x_col)

        for y_col in y_cols:
            y_data = self.loader.get_column(y_col)
            self.ax.plot(x_data, y_data, label=y_col)

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel('Value')
        if self._get_subplot_legend_visible(subplot_idx):
            self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _plot_subplot_scatter(self, subplot_idx, config, widgets_dict):
        """Create scatter plot for subplot"""
        x_col = widgets_dict['x_column'].value
        y_cols = widgets_dict['y_columns_selected']
        color_col = widgets_dict['scatter_color_column'].value

        if not x_col or not y_cols:
            return

        x_data = self.loader.get_column(x_col)
        use_color = color_col != 'None'

        # STORE DATA FOR REDRAWING (only if using color mapping)
        if use_color and len(y_cols) > 0:
            # For simplicity, store data for first y_col (most common case)
            y_col = y_cols[0]
            y_data = self.loader.get_column(y_col)
            color_data = self.loader.get_column(color_col)

            scatter_data = {
                'x': x_data,
                'y': y_data,
                'color_data': color_data,
                'x_label': x_col,
                'y_label': y_col,
                'color_column': color_col,
                'cmap': 'viridis',
                'vmin': float(color_data.min()),
                'vmax': float(color_data.max())
            }

            # Store data
            while len(self.subplot_scatter_data) <= subplot_idx:
                self.subplot_scatter_data.append(None)
            self.subplot_scatter_data[subplot_idx] = scatter_data

        for y_col in y_cols:
            y_data = self.loader.get_column(y_col)

            if use_color:
                color_data = self.loader.get_column(color_col)
                scatter = self.ax.scatter(x_data, y_data, c=color_data,
                                        cmap='viridis', label=y_col)
                # Add colorbar for this subplot using centralized helper
                self._create_subplot_colorbar(subplot_idx, scatter, self._apply_mathtext(color_col), gap=0.01, width=0.01)
            else:
                self.ax.scatter(x_data, y_data, label=y_col)

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel('Value')
        if len(y_cols) > 0 and self._get_subplot_legend_visible(subplot_idx):
            self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _plot_subplot_contour(self, subplot_idx, config, widgets_dict):
        """Create contour plot for subplot"""
        x_col = widgets_dict['x_column_contour'].value
        y_col = widgets_dict['y_column_contour'].value
        field_col = widgets_dict['field_column'].value

        if not x_col or not y_col or not field_col:
            return

        x_data = self.loader.get_column(x_col)
        y_data = self.loader.get_column(y_col)
        field_data = self.loader.get_column(field_col)

        # Determine if regular grid
        unique_x = np.unique(x_data)
        unique_y = np.unique(y_data)
        is_regular = len(unique_x) * len(unique_y) == len(field_data)

        # STORE DATA FOR REDRAWING
        contour_data = {
            'x': x_data,
            'y': y_data,
            'z': field_data,
            'x_label': x_col,
            'y_label': y_col,
            'field_label': field_col,
            'is_regular_grid': is_regular,
            'cylindrical': False
        }

        if is_regular:
            X = x_data.reshape(len(unique_y), len(unique_x))
            Y = y_data.reshape(len(unique_y), len(unique_x))
            Z = field_data.reshape(len(unique_y), len(unique_x))
            contour_data['X'] = X
            contour_data['Y'] = Y
            contour_data['Z'] = Z

        # Store data
        while len(self.subplot_contour_data) <= subplot_idx:
            self.subplot_contour_data.append(None)
        self.subplot_contour_data[subplot_idx] = contour_data

        # Try regular contour, fall back to tricontour if needed
        try:
            if is_regular:
                # Regular grid
                X = contour_data['X']
                Y = contour_data['Y']
                Z = contour_data['Z']
                contour = self.ax.contourf(X, Y, Z, levels=15, cmap='viridis')
            else:
                # Irregular grid - use tricontour
                contour = self.ax.tricontourf(x_data, y_data, field_data,
                                             levels=15, cmap='viridis')
        except:
            # Fallback to tricontour
            contour = self.ax.tricontourf(x_data, y_data, field_data,
                                         levels=15, cmap='viridis')

        # Add colorbar using centralized helper
        self._create_subplot_colorbar(subplot_idx, contour, self._apply_mathtext(field_col), gap=0.01, width=0.01)

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.set_aspect('equal', adjustable='box')

    def _plot_subplot_tricontour(self, subplot_idx, config, widgets_dict):
        """Create tricontour plot for subplot"""
        x_col = widgets_dict['x_column_contour'].value
        y_col = widgets_dict['y_column_contour'].value
        field_col = widgets_dict['field_column'].value

        if not x_col or not y_col or not field_col:
            return

        x_data = self.loader.get_column(x_col)
        y_data = self.loader.get_column(y_col)
        field_data = self.loader.get_column(field_col)

        # STORE DATA FOR REDRAWING
        contour_data = {
            'x': x_data,
            'y': y_data,
            'z': field_data,
            'x_label': x_col,
            'y_label': y_col,
            'field_label': field_col,
            'is_regular_grid': False,
            'cylindrical': False
        }

        # Store data
        while len(self.subplot_contour_data) <= subplot_idx:
            self.subplot_contour_data.append(None)
        self.subplot_contour_data[subplot_idx] = contour_data

        contour = self.ax.tricontourf(x_data, y_data, field_data,
                                     levels=15, cmap='viridis')

        # Add colorbar using centralized helper
        self._create_subplot_colorbar(subplot_idx, contour, self._apply_mathtext(field_col), gap=0.01, width=0.01)

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.set_aspect('equal', adjustable='box')

    def _plot_subplot_cylindrical(self, subplot_idx, config, widgets_dict):
        """Create cylindrical contour plot for subplot"""
        r_col = widgets_dict['r_column'].value
        theta_col = widgets_dict['theta_column'].value
        field_col = widgets_dict['field_column_cyl'].value

        if not r_col or not theta_col or not field_col:
            return

        r_data = self.loader.get_column(r_col)
        theta_data = self.loader.get_column(theta_col)
        field_data = self.loader.get_column(field_col)

        # Convert cylindrical to Cartesian
        x_data = r_data * np.cos(theta_data)
        y_data = r_data * np.sin(theta_data)

        # STORE DATA FOR REDRAWING
        contour_data = {
            'x': x_data,
            'y': y_data,
            'z': field_data,
            'x_label': 'X (from r, θ)',
            'y_label': 'Y (from r, θ)',
            'field_label': field_col,
            'is_regular_grid': False,
            'cylindrical': True
        }

        # Store data
        while len(self.subplot_contour_data) <= subplot_idx:
            self.subplot_contour_data.append(None)
        self.subplot_contour_data[subplot_idx] = contour_data

        # Create triangulation and contour
        contour = self.ax.tricontourf(x_data, y_data, field_data,
                                     levels=15, cmap='viridis')

        # Add colorbar using centralized helper
        self._create_subplot_colorbar(subplot_idx, contour, field_col, gap=0.01, width=0.01)

        self.ax.set_xlabel('X (from r, θ)')
        self.ax.set_ylabel('Y (from r, θ)')
        self.ax.set_aspect('equal', adjustable='box')

    def _redraw_subplot_contour(self, subplot_idx, contour_type, levels, cmap, alpha,
                               show_colorbar, line_width=0.5, line_color='black',
                               colorbar_gap=0.01, colorbar_width=0.01):
        """Redraw a specific subplot's contour with new settings"""

        if subplot_idx >= len(self.subplot_contour_data):
            print(f"❌ No contour data for subplot {subplot_idx+1}")
            return

        data = self.subplot_contour_data[subplot_idx]
        if data is None:
            return

        ax = self.axes_list[subplot_idx]

        # Remove only existing contour collections without clearing the whole axes
        existing_contours = [c for c in ax.collections if 'Contour' in str(type(c))]
        for c in existing_contours:
            try:
                c.remove()
            except Exception:
                pass

        # Remove old colorbar via helper
        self._remove_subplot_colorbar(subplot_idx)

        # Redraw based on type
        if data['is_regular_grid']:
            X, Y, Z = data['X'], data['Y'], data['Z']

            if contour_type == 'contourf (filled)':
                cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)
            elif contour_type == 'contour (lines)':
                cs = ax.contour(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)
            else:  # both (filled + lines)
                cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)
                ax.contour(X, Y, Z, levels=levels, colors=line_color,
                          linewidths=line_width, alpha=0.8)
        else:  # Triangulated
            from matplotlib import tri
            triang = tri.Triangulation(data['x'], data['y'])

            if contour_type == 'contourf (filled)':
                cs = ax.tricontourf(triang, data['z'], levels=levels, cmap=cmap, alpha=alpha)
            elif contour_type == 'contour (lines)':
                cs = ax.tricontour(triang, data['z'], levels=levels, cmap=cmap, alpha=alpha)
            else:  # both (filled + lines)
                cs = ax.tricontourf(triang, data['z'], levels=levels, cmap=cmap, alpha=alpha)
                ax.tricontour(triang, data['z'], levels=levels, colors=line_color,
                             linewidths=line_width, alpha=0.8)

        # Recreate colorbar if needed
        if show_colorbar:
            self._create_subplot_colorbar(
                subplot_idx,
                cs,
                self._apply_mathtext(data['field_label']),
                gap=colorbar_gap,
                width=colorbar_width,
            )

        # Re-apply any stored user labels for this axes
        self._reapply_user_labels(ax)

        self.refresh()

    def _redraw_subplot_scatter(self, subplot_idx, cmap, vmin, vmax, show_colorbar,
                                colorbar_gap=0.01, colorbar_width=0.01):
        """Redraw a specific subplot's scatter plot with new settings"""

        if subplot_idx >= len(self.subplot_scatter_data):
            print(f"❌ No scatter data for subplot {subplot_idx+1}")
            return

        data = self.subplot_scatter_data[subplot_idx]
        if data is None:
            return

        ax = self.axes_list[subplot_idx]

        # Update existing scatter collections in place (preserve per-series edits)
        scatter_colls = [c for c in ax.collections if hasattr(c, 'get_offsets') and 'Contour' not in str(type(c))]
        for sc in scatter_colls:
            try:
                sc.set_cmap(cmap)
                sc.set_clim(vmin, vmax)
                if data.get('color_data') is not None:
                    sc.set_array(data['color_data'])
            except Exception:
                pass

        # Remove old colorbar
        if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
            try:
                self.colorbars[subplot_idx].remove()
            except:
                pass
            self.colorbars[subplot_idx] = None

        if subplot_idx < len(self.colorbar_axes) and self.colorbar_axes[subplot_idx]:
            try:
                self.fig.delaxes(self.colorbar_axes[subplot_idx])
            except:
                pass
            self.colorbar_axes[subplot_idx] = None

        # Choose an existing scatter to bind colorbar, or create one if missing
        if scatter_colls:
            scatter = scatter_colls[0]
        else:
            scatter = ax.scatter(data['x'], data['y'], c=data['color_data'],
                                 cmap=cmap, vmin=vmin, vmax=vmax, label=data['y_label'])

        # Recreate colorbar if needed
        if show_colorbar:
            self._create_subplot_colorbar(
                subplot_idx,
                scatter,
                data['color_column'],
                gap=colorbar_gap,
                width=colorbar_width,
            )

        # Refresh legend to reflect new handles, only if user wants it
        if self._get_subplot_legend_visible(subplot_idx):
            ax.legend()

        self.refresh()

    def _plot_subplot_vector(self, subplot_idx, config, widgets_dict):
        """Create vector field plot for subplot"""
        loader = config.get('loader')
        if loader is None:
            return

        x_col = widgets_dict['vector_x_column'].value
        y_col = widgets_dict['vector_y_column'].value
        u_col = widgets_dict['vector_u_column'].value
        v_col = widgets_dict['vector_v_column'].value

        if not all([x_col, y_col, u_col, v_col]):
            print(f"❌ Subplot {subplot_idx+1}: Please select X/Y/U/V columns")
            return

        x = _to_float_array(loader.get_column(x_col))
        y = _to_float_array(loader.get_column(y_col))
        u = _to_float_array(loader.get_column(u_col))
        v = _to_float_array(loader.get_column(v_col))
        color_field_name = widgets_dict['vector_color_field'].value
        if color_field_name and color_field_name not in (None, 'None'):
            colors = _to_float_array(loader.get_column(color_field_name))
        else:
            colors = None
            color_field_name = None

        x, y, u, v, colors, valid_mask = _filter_valid_vectors(x, y, u, v, colors)
        if x.size == 0:
            print(f"❌ Subplot {subplot_idx+1}: No valid vector data after filtering")
            return
        magnitude = np.sqrt(u**2 + v**2)
        grid_result = _try_reshape_grid(x, y, u, v, colors, magnitude)
        if grid_result:
            grid_X, grid_Y, (grid_U, grid_V, grid_C, grid_M) = grid_result
        else:
            grid_X = grid_Y = grid_U = grid_V = grid_C = grid_M = None
        grid_result = _try_reshape_grid(x, y, u, v, colors, magnitude)
        if grid_result:
            grid_X, grid_Y, (grid_U, grid_V, grid_C, grid_M) = grid_result
        else:
            grid_X = grid_Y = grid_U = grid_V = grid_C = grid_M = None

        # Ensure style storage for this subplot
        while len(self.subplot_vector_styles) <= subplot_idx:
            self.subplot_vector_styles.append(self.vector_style.copy())
        style = self.subplot_vector_styles[subplot_idx]

        # Clean up previous overlays and colorbars
        while len(self.subplot_vector_overlays) <= subplot_idx:
            self.subplot_vector_overlays.append(None)
        while len(self.subplot_vector_colorbars) <= subplot_idx:
            self.subplot_vector_colorbars.append(None)
        while len(self.subplot_vector_quivers) <= subplot_idx:
            self.subplot_vector_quivers.append(None)
        while len(self.subplot_vector_data) <= subplot_idx:
            self.subplot_vector_data.append(None)

        prev_overlay = self.subplot_vector_overlays[subplot_idx]
        if prev_overlay is not None:
            try:
                for coll in getattr(prev_overlay, 'collections', []):
                    coll.remove()
            except Exception:
                try:
                    prev_overlay.remove()
                except Exception:
                    pass
            self.subplot_vector_overlays[subplot_idx] = None

        prev_colorbar = self.subplot_vector_colorbars[subplot_idx]
        if prev_colorbar is not None:
            try:
                prev_colorbar.remove()
            except Exception:
                pass
            self.subplot_vector_colorbars[subplot_idx] = None

        prev_quiver_cbar = None
        if subplot_idx < len(self.subplot_vector_quiver_colorbars):
            prev_quiver_cbar = self.subplot_vector_quiver_colorbars[subplot_idx]
        if prev_quiver_cbar is not None:
            try:
                prev_quiver_cbar.remove()
            except Exception:
                pass
            self.subplot_vector_quiver_colorbars[subplot_idx] = None

        self._remove_subplot_colorbar(subplot_idx)

        cmap = style.get('cmap', 'viridis')
        scale_setting = max(0.01, style.get('scale', 1.0))
        width_setting = max(0.0001, style.get('width', 0.0025))
        pivot = style.get('pivot', 'middle')
        scale_value = 40.0 / scale_setting
        arrow_alpha = float(style.get('alpha', 0.8))

        ax = self.axes_list[subplot_idx]
        try:
            ax.xaxis.set_units(None)
            ax.yaxis.set_units(None)
            ax.xaxis.converter = None
            ax.yaxis.converter = None
        except Exception:
            pass
        # ===== IMPORTANT: Draw overlay FIRST, then quiver on top (same as single plot) =====
        # Step 1: Draw overlay contour field (if specified) - background layer
        overlay_field_name = widgets_dict['vector_overlay_field'].value
        overlay_field_store = overlay_field_name if overlay_field_name not in (None, 'None') else None
        overlay = None
        if overlay_field_name and overlay_field_name != 'None':
            overlay_data = _to_float_array(loader.get_column(overlay_field_name))[valid_mask]
            levels_setting = max(3, int(style.get('overlay_levels', 20)))
            alpha = float(style.get('overlay_alpha', 1.0))
            overlay_cmap = style.get('overlay_cmap', 'plasma')
            overlay_log = bool(style.get('overlay_log', False))
            overlay_type = style.get('overlay_type', 'contourf (filled)')
            overlay_gap = style.get('overlay_colorbar_gap', self.colorbar_gap)
            overlay_width = style.get('overlay_colorbar_width', self.colorbar_width)
            overlay_fontweight = style.get('overlay_fontweight', 'normal')

            levels = levels_setting
            if overlay_log:
                positive = overlay_data[overlay_data > 0]
                if positive.size > 0:
                    min_val = positive.min()
                    max_val = positive.max()
                    if min_val > 0 and max_val > min_val:
                        levels = np.geomspace(min_val, max_val, levels_setting)

            overlay_grid_data = None
            if grid_result:
                overlay_grid_result = _try_reshape_grid(x, y, overlay_data)
                if overlay_grid_result:
                    _, _, [overlay_grid_data] = overlay_grid_result

            # Get line style settings for 'both' mode
            line_width = float(style.get('overlay_line_thickness', 0.5))
            line_color = style.get('overlay_line_color', 'black')

            try:
                if overlay_grid_data is not None:
                    if overlay_type == 'contour (lines)':
                        overlay = ax.contour(grid_X, grid_Y, overlay_grid_data, levels=levels, cmap=overlay_cmap)
                    elif overlay_type == 'both (filled + lines)':
                        overlay = ax.contourf(grid_X, grid_Y, overlay_grid_data, levels=levels, cmap=overlay_cmap, alpha=alpha)
                        ax.contour(grid_X, grid_Y, overlay_grid_data, levels=levels, colors=line_color, linewidths=line_width)
                    else:
                        overlay = ax.contourf(grid_X, grid_Y, overlay_grid_data, levels=levels, cmap=overlay_cmap, alpha=alpha)
                else:
                    if overlay_type == 'contour (lines)':
                        overlay = ax.tricontour(x, y, overlay_data, levels=levels, cmap=overlay_cmap)
                    elif overlay_type == 'both (filled + lines)':
                        overlay = ax.tricontourf(x, y, overlay_data, levels=levels, cmap=overlay_cmap, alpha=alpha)
                        ax.tricontour(x, y, overlay_data, levels=levels, colors=line_color, linewidths=line_width)
                    else:
                        overlay = ax.tricontourf(x, y, overlay_data, levels=levels, cmap=overlay_cmap, alpha=alpha)
            except Exception:
                overlay = ax.scatter(x, y, c=overlay_data, cmap=overlay_cmap, s=5, alpha=alpha)

            self.subplot_vector_overlays[subplot_idx] = overlay
            if overlay is not None and style.get('overlay_show_colorbar', True):
                old_gap = getattr(self, 'colorbar_gap', 0.02)
                old_width = getattr(self, 'colorbar_width', 0.02)
                self.colorbar_gap = overlay_gap
                self.colorbar_width = overlay_width
                # Pass explicit gap/width to honor vector overlay controls in subplot mode
                overlay_cbar = self._create_subplot_colorbar(
                    subplot_idx,
                    overlay,
                    self._apply_mathtext(style.get('overlay_label') or overlay_field_name),
                    gap=overlay_gap,
                    width=overlay_width
                )
                self.colorbar_gap = old_gap
                self.colorbar_width = old_width
                if overlay_cbar:
                    overlay_cbar.set_label(
                        self._apply_mathtext(style.get('overlay_label') or overlay_field_name),
                        fontsize=style.get('overlay_label_fontsize', 10),
                        weight=overlay_fontweight
                    )
                    overlay_cbar.ax.tick_params(labelsize=style.get('overlay_tick_fontsize', 9))
                self.subplot_vector_colorbars[subplot_idx] = overlay_cbar

        # Step 2: Draw quiver arrows ON TOP of overlay
        quiver_kwargs = dict(scale=scale_value, width=width_setting, pivot=pivot, alpha=arrow_alpha)
        quiver_x = grid_X if grid_X is not None else x
        quiver_y = grid_Y if grid_Y is not None else y
        quiver_u = grid_U if grid_U is not None else u
        quiver_v = grid_V if grid_V is not None else v
        quiver_colors = grid_C if grid_C is not None else colors

        # Apply decimation (plot every Nth arrow)
        decimation = int(style.get('decimation', 1))
        if decimation > 1:
            if grid_X is not None:
                # For gridded data, slice both dimensions
                quiver_x = quiver_x[::decimation, ::decimation]
                quiver_y = quiver_y[::decimation, ::decimation]
                quiver_u = quiver_u[::decimation, ::decimation]
                quiver_v = quiver_v[::decimation, ::decimation]
                if quiver_colors is not None:
                    quiver_colors = quiver_colors[::decimation, ::decimation]
            else:
                # For scattered data, slice the array
                quiver_x = quiver_x[::decimation]
                quiver_y = quiver_y[::decimation]
                quiver_u = quiver_u[::decimation]
                quiver_v = quiver_v[::decimation]
                if quiver_colors is not None:
                    quiver_colors = quiver_colors[::decimation]

        # Get arrow color from style
        arrow_color = style.get('arrow_color', 'black')

        if quiver_colors is not None:
            quiver = ax.quiver(quiver_x, quiver_y, quiver_u, quiver_v, quiver_colors, cmap=cmap, **quiver_kwargs)
        else:
            quiver = ax.quiver(quiver_x, quiver_y, quiver_u, quiver_v, color=arrow_color, **quiver_kwargs)

        self.subplot_vector_quivers[subplot_idx] = quiver

        quiver_show_cb = style.get('colorbar', True) and quiver_colors is not None
        quiver_label = style.get('colorbar_label') or (color_field_name if color_field_name else '')
        quiver_label_size = style.get('colorbar_label_fontsize', 10)
        quiver_tick_size = style.get('colorbar_tick_fontsize', 9)
        quiver_fontweight = style.get('colorbar_fontweight', 'normal')
        quiver_gap = style.get('colorbar_gap', self.colorbar_gap)
        quiver_width = style.get('colorbar_width', self.colorbar_width)

        if quiver_show_cb:
            old_gap = getattr(self, 'colorbar_gap', 0.02)
            old_width = getattr(self, 'colorbar_width', 0.02)
            self.colorbar_gap = quiver_gap
            self.colorbar_width = quiver_width
            # Pass explicit gap/width to honor vector quiver CB controls in subplot mode
            quiver_cbar = self._create_subplot_colorbar(
                subplot_idx,
                quiver,
                self._apply_mathtext(quiver_label),
                gap=quiver_gap,
                width=quiver_width
            )
            self.colorbar_gap = old_gap
            self.colorbar_width = old_width
            if quiver_cbar:
                quiver_cbar.set_label(self._apply_mathtext(quiver_label), fontsize=quiver_label_size, weight=quiver_fontweight)
                quiver_cbar.ax.tick_params(labelsize=quiver_tick_size)
            self.subplot_vector_quiver_colorbars[subplot_idx] = quiver_cbar
        else:
            self.subplot_vector_quiver_colorbars[subplot_idx] = None

        self.subplot_vector_data[subplot_idx] = {
            'x': x,
            'y': y,
            'u': u,
            'v': v,
            'magnitude': magnitude,
            'colors': quiver_colors,
            'color_field': color_field_name,
            'overlay_field': overlay_field_store
        }

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title('Vector Field')
        ax.grid(True, alpha=0.3)
        if grid_result:
            try:
                ax.set_aspect('equal', adjustable='box')
            except Exception:
                pass

    def _plot_line(self):
        """Create line plot"""
        x_data = self.loader.get_column(self.x_column.value)

        for y_col in self.y_columns.value:
            y_data = self.loader.get_column(y_col)

            # Get line styles if available
            if y_col in self.line_styles:
                styles = self.line_styles[y_col]
                color = styles['color'].value
                linestyle = styles['style'].value
                marker = styles['marker'].value
            else:
                color = None
                linestyle = '-'
                marker = ''

            self.ax.plot(x_data, y_data, label=y_col,
                        color=color, linestyle=linestyle, marker=marker)

        self.ax.set_xlabel(self.x_column.value)
        self.ax.set_ylabel('Value')
        self.ax.set_title('Line Plot')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _plot_scatter(self):
        """Create scatter plot"""
        x_data = self.loader.get_column(self.x_column.value)

        # Check if color intensity column is selected
        use_color_intensity = (hasattr(self, 'scatter_color_column') and
                               self.scatter_color_column.value != 'None')

        if use_color_intensity:
            color_data = self.loader.get_column(self.scatter_color_column.value)
            # Store scatter data for later manipulation
            self.scatter_data = {
                'x': x_data,
                'y_columns': list(self.y_columns.value),
                'color_column': self.scatter_color_column.value,
                'color_data': color_data
            }

        for i, y_col in enumerate(self.y_columns.value):
            y_data = self.loader.get_column(y_col)

            if use_color_intensity:
                # Create scatter plot with color intensity
                scatter = self.ax.scatter(x_data, y_data, c=color_data,
                                         cmap='viridis', label=y_col,
                                         alpha=0.6, s=50)
                # Add colorbar using separate axes for first series only
                if i == 0:
                    self._add_colorbar_with_separate_axes(scatter,
                                                          label=self.scatter_color_column.value)
            else:
                # Regular scatter plot without color intensity
                self.ax.scatter(x_data, y_data, label=y_col, alpha=0.6, s=50)

        self.ax.set_xlabel(self.x_column.value)
        self.ax.set_ylabel('Value')
        self.ax.set_title('Scatter Plot')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _plot_vector(self):
        """Create vector field plot"""
        x_col = self.vector_x_column.value
        y_col = self.vector_y_column.value
        u_col = self.vector_u_column.value
        v_col = self.vector_v_column.value

        if not all([x_col, y_col, u_col, v_col]):
            print("❌ Please select X, Y, U, and V columns for vector plot")
            return

        x = _to_float_array(self.loader.get_column(x_col))
        y = _to_float_array(self.loader.get_column(y_col))
        u = _to_float_array(self.loader.get_column(u_col))
        v = _to_float_array(self.loader.get_column(v_col))
        color_field_name = self.vector_color_field.value
        if color_field_name and color_field_name not in (None, 'None'):
            colors = _to_float_array(self.loader.get_column(color_field_name))
        else:
            colors = None
            color_field_name = None

        x, y, u, v, colors, valid_mask = _filter_valid_vectors(x, y, u, v, colors)
        if x.size == 0:
            print("❌ Vector plot: no valid data after filtering")
            return

        magnitude = np.sqrt(u**2 + v**2)
        grid_result = _try_reshape_grid(x, y, u, v, colors, magnitude)
        if grid_result:
            grid_X, grid_Y, (grid_U, grid_V, grid_C, grid_M) = grid_result
        else:
            grid_X = grid_Y = grid_U = grid_V = grid_C = grid_M = None

        # Remove previous overlays/colorbars if present
        for artist in [self.vector_overlay]:
            if artist is not None:
                try:
                    for coll in getattr(artist, 'collections', []):
                        coll.remove()
                except Exception:
                    try:
                        artist.remove()
                    except Exception:
                        pass
        self.vector_overlay = None

        for cbar in [self.vector_overlay_colorbar, self.vector_quiver_colorbar]:
            if cbar is not None:
                try:
                    cbar.remove()
                except Exception:
                    pass
        self.vector_overlay_colorbar = None
        self.vector_quiver_colorbar = None

        if self.colorbar_ax is not None:
            try:
                self.fig.delaxes(self.colorbar_ax)
            except Exception:
                pass
            self.colorbar_ax = None
        self.colorbar = None

        cmap = self.vector_style.get('cmap', 'viridis')
        scale_setting = max(0.01, self.vector_style.get('scale', 1.0))
        width_setting = max(0.0001, self.vector_style.get('width', 0.0025))
        pivot = self.vector_style.get('pivot', 'middle')
        arrow_alpha = float(self.vector_style.get('alpha', 0.8))
        scale_value = 40.0 / scale_setting

        # Reset any stale categorical converters on axes
        try:
            self.ax.xaxis.set_units(None)
            self.ax.yaxis.set_units(None)
            self.ax.xaxis.converter = None
            self.ax.yaxis.converter = None
        except Exception:
            pass

        # ===== IMPORTANT: Draw overlay FIRST, then quiver on top =====
        # Step 1: Draw overlay contour field (if specified) - this goes in the background
        overlay_field_name = self.vector_overlay_field.value
        overlay_field_store = overlay_field_name if overlay_field_name not in (None, 'None') else None
        overlay = None
        if overlay_field_name and overlay_field_name != 'None':
            overlay_data = _to_float_array(self.loader.get_column(overlay_field_name))[valid_mask]
            levels_setting = max(3, int(self.vector_style.get('overlay_levels', 20)))
            alpha = float(self.vector_style.get('overlay_alpha', 0.6))
            overlay_cmap = self.vector_style.get('overlay_cmap', 'plasma')
            overlay_log = bool(self.vector_style.get('overlay_log', False))
            overlay_type = self.vector_style.get('overlay_type', 'contourf (filled)')
            overlay_gap = self.vector_style.get('overlay_colorbar_gap', self.colorbar_gap)
            overlay_width = self.vector_style.get('overlay_colorbar_width', self.colorbar_width)
            overlay_fontweight = self.vector_style.get('overlay_fontweight', 'normal')

            levels = levels_setting
            if overlay_log:
                positive = overlay_data[overlay_data > 0]
                if positive.size > 0:
                    min_val = positive.min()
                    max_val = positive.max()
                    if min_val > 0 and max_val > min_val:
                        levels = np.geomspace(min_val, max_val, levels_setting)

            overlay_grid_data = None
            if grid_result:
                grid_overlay_result = _try_reshape_grid(x, y, overlay_data)
                if grid_overlay_result:
                    _, _, [overlay_grid_data] = grid_overlay_result

            # Get line style settings for 'both' mode
            line_width = float(self.vector_style.get('overlay_line_thickness', 0.5))
            line_color = self.vector_style.get('overlay_line_color', 'black')

            try:
                if overlay_grid_data is not None:
                    if overlay_type == 'contour (lines)':
                        overlay = self.ax.contour(grid_X, grid_Y, overlay_grid_data, levels=levels, cmap=overlay_cmap)
                    elif overlay_type == 'both (filled + lines)':
                        overlay = self.ax.contourf(grid_X, grid_Y, overlay_grid_data, levels=levels, cmap=overlay_cmap, alpha=alpha)
                        self.ax.contour(grid_X, grid_Y, overlay_grid_data, levels=levels, colors=line_color, linewidths=line_width)
                    else:
                        overlay = self.ax.contourf(grid_X, grid_Y, overlay_grid_data, levels=levels, cmap=overlay_cmap, alpha=alpha)
                else:
                    if overlay_type == 'contour (lines)':
                        overlay = self.ax.tricontour(x, y, overlay_data, levels=levels, cmap=overlay_cmap)
                    elif overlay_type == 'both (filled + lines)':
                        overlay = self.ax.tricontourf(x, y, overlay_data, levels=levels, cmap=overlay_cmap, alpha=alpha)
                        self.ax.tricontour(x, y, overlay_data, levels=levels, colors=line_color, linewidths=line_width)
                    else:
                        overlay = self.ax.tricontourf(x, y, overlay_data, levels=levels, cmap=overlay_cmap, alpha=alpha)
            except Exception:
                overlay = self.ax.scatter(x, y, c=overlay_data, cmap=overlay_cmap, s=5, alpha=alpha)

            self.vector_overlay = overlay
            if overlay is not None and self.vector_style.get('overlay_show_colorbar', True):
                old_gap = getattr(self, 'colorbar_gap', 0.02)
                old_width = getattr(self, 'colorbar_width', 0.02)
                self.colorbar_gap = overlay_gap
                self.colorbar_width = overlay_width
                overlay_cbar = self._add_colorbar_with_separate_axes(overlay, label=self._apply_mathtext(self.vector_style.get('overlay_label') or overlay_field_name))
                self.colorbar_gap = old_gap
                self.colorbar_width = old_width
                if overlay_cbar:
                    overlay_cbar.set_label(self._apply_mathtext(self.vector_style.get('overlay_label') or overlay_field_name),
                                            fontsize=self.vector_style.get('overlay_label_fontsize', 10),
                                            weight=overlay_fontweight)
                    overlay_cbar.ax.tick_params(labelsize=self.vector_style.get('overlay_tick_fontsize', 9))
                self.vector_overlay_colorbar = overlay_cbar

        # Step 2: Draw quiver arrows ON TOP of overlay
        quiver_kwargs = dict(scale=scale_value, width=width_setting, pivot=pivot, alpha=arrow_alpha)
        quiver_x = grid_X if grid_X is not None else x
        quiver_y = grid_Y if grid_Y is not None else y
        quiver_u = grid_U if grid_U is not None else u
        quiver_v = grid_V if grid_V is not None else v
        quiver_colors = grid_C if grid_C is not None else colors

        # Apply decimation (plot every Nth arrow)
        decimation = int(self.vector_style.get('decimation', 1))
        if decimation > 1:
            if grid_X is not None:
                # For gridded data, slice both dimensions
                quiver_x = quiver_x[::decimation, ::decimation]
                quiver_y = quiver_y[::decimation, ::decimation]
                quiver_u = quiver_u[::decimation, ::decimation]
                quiver_v = quiver_v[::decimation, ::decimation]
                if quiver_colors is not None:
                    quiver_colors = quiver_colors[::decimation, ::decimation]
            else:
                # For scattered data, slice the array
                quiver_x = quiver_x[::decimation]
                quiver_y = quiver_y[::decimation]
                quiver_u = quiver_u[::decimation]
                quiver_v = quiver_v[::decimation]
                if quiver_colors is not None:
                    quiver_colors = quiver_colors[::decimation]

        # Get arrow color from style
        arrow_color = self.vector_style.get('arrow_color', 'black')

        if quiver_colors is not None:
            quiver = self.ax.quiver(quiver_x, quiver_y, quiver_u, quiver_v, quiver_colors, cmap=cmap, **quiver_kwargs)
        else:
            quiver = self.ax.quiver(quiver_x, quiver_y, quiver_u, quiver_v, color=arrow_color, **quiver_kwargs)

        self.vector_quiver = quiver

        quiver_show_cb = self.vector_style.get('colorbar', True) and quiver_colors is not None
        quiver_label = self.vector_style.get('colorbar_label') or (color_field_name if color_field_name else '')
        quiver_label_size = self.vector_style.get('colorbar_label_fontsize', 10)
        quiver_tick_size = self.vector_style.get('colorbar_tick_fontsize', 9)
        quiver_fontweight = self.vector_style.get('colorbar_fontweight', 'normal')
        quiver_gap = self.vector_style.get('colorbar_gap', self.colorbar_gap)
        quiver_width = self.vector_style.get('colorbar_width', self.colorbar_width)

        if quiver_show_cb:
            old_gap = getattr(self, 'colorbar_gap', 0.02)
            old_width = getattr(self, 'colorbar_width', 0.02)
            self.colorbar_gap = quiver_gap
            self.colorbar_width = quiver_width
            quiver_cbar = self._add_colorbar_with_separate_axes(quiver, label=self._apply_mathtext(quiver_label))
            self.colorbar_gap = old_gap
            self.colorbar_width = old_width
            if quiver_cbar:
                quiver_cbar.set_label(self._apply_mathtext(quiver_label), fontsize=quiver_label_size, weight=quiver_fontweight)
                quiver_cbar.ax.tick_params(labelsize=quiver_tick_size)
            self.vector_quiver_colorbar = quiver_cbar
        else:
            self.vector_quiver_colorbar = None

        self.vector_data = {
            'x': x,
            'y': y,
            'u': u,
            'v': v,
            'magnitude': magnitude,
            'colors': quiver_colors,
            'color_field': color_field_name,
            'overlay_field': overlay_field_store
        }

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.set_title('Vector Field')
        self.ax.grid(True, alpha=0.3)
        if grid_result:
            try:
                self.ax.set_aspect('equal', adjustable='box')
            except Exception:
                pass

    def _redraw_vector(self, subplot_idx=None):
        """Redraw vector field with current style settings."""
        if subplot_idx is None:
            if self.plot_type != 'vector' or not self.loader or self.ax is None:
                return

            # Preserve text properties before clearing
            current_xlabel = self.ax.xaxis.label.get_text()
            current_ylabel = self.ax.yaxis.label.get_text()
            current_title = self.ax.title.get_text()
            title_fontsize = self.ax.title.get_fontsize()
            label_fontsize = self.ax.xaxis.label.get_fontsize()
            tick_labels = self.ax.xaxis.get_ticklabels()
            tick_fontsize = tick_labels[0].get_fontsize() if tick_labels else 10
            title_fontweight = self.ax.title.get_fontweight()
            label_fontweight = self.ax.xaxis.label.get_fontweight()
            title_fontfamily = self.ax.title.get_fontfamily()
            label_fontfamily = self.ax.xaxis.label.get_fontfamily()
            title_color = self.ax.title.get_color()
            label_color = self.ax.xaxis.label.get_color()
            # Preserve limits, aspect, and grid style
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            try:
                current_aspect = self.ax.get_aspect()
            except Exception:
                current_aspect = None
            grid_lines = list(self.ax.xaxis.get_gridlines()) + list(self.ax.yaxis.get_gridlines())
            grid_enabled = any(line.get_visible() for line in grid_lines) if grid_lines else False
            # Try to infer current grid style from an existing line
            if grid_lines:
                gl = grid_lines[0]
                grid_linestyle = getattr(gl, 'get_linestyle', lambda: '--')()
                grid_alpha = getattr(gl, 'get_alpha', lambda: 0.3)() or 0.3
                grid_linewidth = getattr(gl, 'get_linewidth', lambda: 0.8)()
            else:
                grid_linestyle, grid_alpha, grid_linewidth = '--', 0.3, 0.8

            self.ax.clear()
            self._plot_vector()

            # Restore text properties after redraw
            # Prefer user-entered labels stored on the axes
            self._reapply_user_labels(self.ax)
            # Fallback to previously captured labels if no stored values
            if not getattr(self.ax, '_ff_labels', None):
                if current_xlabel:
                    self.ax.set_xlabel(self._apply_mathtext(current_xlabel))
                if current_ylabel:
                    self.ax.set_ylabel(self._apply_mathtext(current_ylabel))
                if current_title:
                    self.ax.set_title(self._apply_mathtext(current_title))

            self.ax.title.set_fontsize(title_fontsize)
            self.ax.xaxis.label.set_fontsize(label_fontsize)
            self.ax.yaxis.label.set_fontsize(label_fontsize)
            self.ax.tick_params(labelsize=tick_fontsize)
            self.ax.title.set_fontweight(title_fontweight)
            self.ax.xaxis.label.set_fontweight(label_fontweight)
            self.ax.yaxis.label.set_fontweight(label_fontweight)
            self.ax.title.set_fontfamily(title_fontfamily)
            self.ax.xaxis.label.set_fontfamily(label_fontfamily)
            self.ax.yaxis.label.set_fontfamily(label_fontfamily)
            self.ax.title.set_color(title_color)
            self.ax.xaxis.label.set_color(label_color)
            self.ax.yaxis.label.set_color(label_color)
            # Restore limits, aspect, and grid
            try:
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            except Exception:
                pass
            if current_aspect is not None:
                try:
                    self.ax.set_aspect(current_aspect)
                except Exception:
                    pass
            self._apply_grid(self.ax, grid_enabled, grid_linestyle, grid_alpha, grid_linewidth)

            self.refresh()
        else:
            if subplot_idx >= len(self.subplot_configs):
                return
            config = self.subplot_configs[subplot_idx]
            loader = config.get('loader')
            if loader is None or subplot_idx >= len(self.axes_list):
                return
            ax = self.axes_list[subplot_idx]
            # Preserve text/font/limits/grid properties before clearing
            current_xlabel = ax.xaxis.label.get_text()
            current_ylabel = ax.yaxis.label.get_text()
            current_title = ax.title.get_text()
            title_fontsize = ax.title.get_fontsize()
            label_fontsize = ax.xaxis.label.get_fontsize()
            tick_labels = ax.xaxis.get_ticklabels()
            tick_fontsize = tick_labels[0].get_fontsize() if tick_labels else 10
            title_fontweight = ax.title.get_fontweight()
            label_fontweight = ax.xaxis.label.get_fontweight()
            title_fontfamily = ax.title.get_fontfamily()
            label_fontfamily = ax.xaxis.label.get_fontfamily()
            title_color = ax.title.get_color()
            label_color = ax.xaxis.label.get_color()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            try:
                current_aspect = ax.get_aspect()
            except Exception:
                current_aspect = None
            grid_lines = list(ax.xaxis.get_gridlines()) + list(ax.yaxis.get_gridlines())
            grid_enabled = any(line.get_visible() for line in grid_lines) if grid_lines else False
            if grid_lines:
                gl = grid_lines[0]
                grid_linestyle = getattr(gl, 'get_linestyle', lambda: '--')()
                grid_alpha = getattr(gl, 'get_alpha', lambda: 0.3)() or 0.3
                grid_linewidth = getattr(gl, 'get_linewidth', lambda: 0.8)()
            else:
                grid_linestyle, grid_alpha, grid_linewidth = '--', 0.3, 0.8

            ax.clear()
            prev_ax = self.ax
            prev_loader = self.loader
            self.ax = ax
            self.loader = loader
            self._plot_subplot_vector(subplot_idx, config, config['widgets'])
            self.ax = prev_ax
            self.loader = prev_loader
            # Restore text/font/limits/grid after redraw
            # Prefer stored user-entered labels for this axes
            self._reapply_user_labels(ax)
            if not getattr(ax, '_ff_labels', None):
                if current_xlabel:
                    ax.set_xlabel(self._apply_mathtext(current_xlabel))
                if current_ylabel:
                    ax.set_ylabel(self._apply_mathtext(current_ylabel))
                if current_title:
                    ax.set_title(self._apply_mathtext(current_title))
            ax.title.set_fontsize(title_fontsize)
            ax.xaxis.label.set_fontsize(label_fontsize)
            ax.yaxis.label.set_fontsize(label_fontsize)
            ax.tick_params(labelsize=tick_fontsize)
            ax.title.set_fontweight(title_fontweight)
            ax.xaxis.label.set_fontweight(label_fontweight)
            ax.yaxis.label.set_fontweight(label_fontweight)
            ax.title.set_fontfamily(title_fontfamily)
            ax.xaxis.label.set_fontfamily(label_fontfamily)
            ax.yaxis.label.set_fontfamily(label_fontfamily)
            ax.title.set_color(title_color)
            ax.xaxis.label.set_color(label_color)
            ax.yaxis.label.set_color(label_color)
            try:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            except Exception:
                pass
            if current_aspect is not None:
                try:
                    ax.set_aspect(current_aspect)
                except Exception:
                    pass
            self._apply_grid(ax, grid_enabled, grid_linestyle, grid_alpha, grid_linewidth)
            self.refresh()

    def _plot_contour(self):
        """Create contour plot"""
        x_data = self.loader.get_column(self.x_column_contour.value)
        y_data = self.loader.get_column(self.y_column_contour.value)
        field_data = self.loader.get_column(self.field_column.value)

        # Try to reshape data for regular grid
        try:
            # Assume data is on a regular grid
            n_unique_x = len(np.unique(x_data))
            n_unique_y = len(np.unique(y_data))

            X = x_data.reshape(n_unique_y, n_unique_x)
            Y = y_data.reshape(n_unique_y, n_unique_x)
            Z = field_data.reshape(n_unique_y, n_unique_x)

            # Store data for redrawing
            self.contour_data = {
                'X': X, 'Y': Y, 'Z': Z,
                'x_label': self.x_column_contour.value,
                'y_label': self.y_column_contour.value,
                'field_label': self.field_column.value,
                'is_regular_grid': True
            }

            # Create contour plot
            cs = self.ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r')

            # Add colorbar using separate axes (prevents shrinking)
            self._add_colorbar_with_separate_axes(cs, label=self._apply_mathtext(self.field_column.value))

        except:
            # Fallback to tricontourf if reshape fails
            print("Regular grid reshape failed, using tricontourf...")
            self._plot_tricontour()
            return

        # Preserve any custom labels entered via UI
        self._reapply_user_labels(self.ax)
        if not getattr(self.ax, '_ff_labels', None):
            self.ax.set_xlabel(self.x_column_contour.value)
            self.ax.set_ylabel(self.y_column_contour.value)
        self.ax.set_title(f'Contour Plot: {self.field_column.value}')
        self.ax.set_aspect('equal')
        self.ax.grid(False)  # Disable grid for contour plots by default

    def _plot_tricontour(self):
        """Create triangulated contour plot for irregular data"""
        x_data = self.loader.get_column(self.x_column_contour.value)
        y_data = self.loader.get_column(self.y_column_contour.value)
        field_data = self.loader.get_column(self.field_column.value)

        # Create triangulation
        triang = tri.Triangulation(x_data, y_data)

        # Store data for redrawing
        self.contour_data = {
            'x': x_data, 'y': y_data, 'z': field_data,
            'x_label': self.x_column_contour.value,
            'y_label': self.y_column_contour.value,
            'field_label': self.field_column.value,
            'is_regular_grid': False
        }

        # Plot tricontourf
        cs = self.ax.tricontourf(triang, field_data, levels=20, cmap='RdBu_r')

        # Add colorbar using separate axes (prevents shrinking)
        self._add_colorbar_with_separate_axes(cs, label=self._apply_mathtext(self.field_column.value))

        self._reapply_user_labels(self.ax)
        if not getattr(self.ax, '_ff_labels', None):
            self.ax.set_xlabel(self.x_column_contour.value)
            self.ax.set_ylabel(self.y_column_contour.value)
        self.ax.set_title(f'Tricontour Plot: {self.field_column.value}')
        self.ax.set_aspect('equal')
        self.ax.grid(False)  # Disable grid for contour plots by default

    def _plot_cylindrical_contour(self):
        """Create contour plot from cylindrical coordinates (r, θ) → (x, y)"""
        r_data = self.loader.get_column(self.r_column.value)
        theta_data = self.loader.get_column(self.theta_column.value)
        field_data = self.loader.get_column(self.field_column_cyl.value)

        # Convert cylindrical to Cartesian coordinates
        x_data = r_data * np.cos(theta_data)
        y_data = r_data * np.sin(theta_data)

        # Create triangulation for irregular grid
        triang = tri.Triangulation(x_data, y_data)

        # Store data for redrawing
        self.contour_data = {
            'x': x_data, 'y': y_data, 'z': field_data,
            'x_label': 'X',
            'y_label': 'Y',
            'field_label': self.field_column_cyl.value,
            'is_regular_grid': False,
            'cylindrical': True,
            'r_label': self.r_column.value,
            'theta_label': self.theta_column.value
        }

        # Plot tricontourf
        cs = self.ax.tricontourf(triang, field_data, levels=20, cmap='RdBu_r')

        # Add colorbar
        self._add_colorbar_with_separate_axes(cs, label=self._apply_mathtext(self.field_column_cyl.value))

        self.ax.set_xlabel('X (from r·cos(θ))')
        self.ax.set_ylabel('Y (from r·sin(θ))')
        self.ax.set_title(f'Cylindrical Contour: {self.field_column_cyl.value}')
        self.ax.set_aspect('equal')
        self.ax.grid(False)

    def _add_colorbar_with_separate_axes(self, mappable, label=''):
        """Add colorbar using separate axes to prevent figure shrinking"""
        # Remove old colorbar and its axes if exists
        if self.colorbar:
            try:
                self.colorbar.remove()
            except (KeyError, AttributeError, ValueError):
                pass
            self.colorbar = None

        if hasattr(self, 'colorbar_ax') and self.colorbar_ax:
            try:
                # Try to remove from figure's axes list manually to avoid KeyError
                if self.colorbar_ax in self.fig.axes:
                    self.fig.delaxes(self.colorbar_ax)
            except (ValueError, AttributeError, KeyError):
                # If that fails, try direct removal
                try:
                    self.colorbar_ax.remove()
                except (ValueError, AttributeError, KeyError):
                    pass
            finally:
                self.colorbar_ax = None

        # Get main axes position
        pos = self.ax.get_position()

        # Create axes for colorbar
        # Position: [left, bottom, width, height]
        # Use stored values or defaults
        cbar_width = getattr(self, 'colorbar_width', 0.02)  # Width of colorbar
        gap = getattr(self, 'colorbar_gap', 0.02)  # Gap between plot and colorbar (increased from 0.01)

        cax = self.fig.add_axes([
            pos.x1 + gap,  # Left edge: right edge of main axes + gap
            pos.y0,        # Bottom: same as main axes
            cbar_width,    # Width: thin colorbar
            pos.height     # Height: same as main axes
        ])

        self.colorbar_ax = cax
        self.colorbar = self.fig.colorbar(mappable, cax=cax, label=self._apply_mathtext(label))

        return self.colorbar

    def _redraw_contour(self):
        """Redraw contour plot with updated settings"""
        if not hasattr(self, 'contour_data'):
            print("❌ No contour data available for redrawing")
            return

        try:
            # Get current settings from widgets
            contour_type = self.contour_type_dropdown.value if hasattr(self, 'contour_type_dropdown') else 'contourf (filled)'

            # Get colormap from dropdown (or default)
            cmap = self.cmap_dropdown.value if hasattr(self, 'cmap_dropdown') else 'RdBu_r'

            # Get alpha/transparency from slider (or default)
            alpha = self.contour_alpha_slider.value if hasattr(self, 'contour_alpha_slider') else 1.0

            # Get line properties for "both" mode
            line_width = self.contour_line_thickness.value if hasattr(self, 'contour_line_thickness') else 0.5
            line_color = self.contour_line_color.value if hasattr(self, 'contour_line_color') else 'black'

            # Get levels - handle log scale
            if hasattr(self, 'levels_log_toggle') and self.levels_log_toggle.value:
                levels = int(10 ** self.levels_slider.value)
            else:
                levels = int(self.levels_slider.value) if hasattr(self, 'levels_slider') else 20

            # Check colorbar visibility setting
            show_colorbar = self.colorbar_toggle.value if hasattr(self, 'colorbar_toggle') else True

            # Remove only existing contour collections; avoid clearing entire axes
            existing_contours = [c for c in self.ax.collections if 'Contour' in str(type(c))]
            for c in existing_contours:
                try:
                    c.remove()
                except Exception:
                    pass

            # Remove colorbar first
            if self.colorbar:
                try:
                    self.colorbar.remove()
                except (KeyError, AttributeError, ValueError):
                    pass
                self.colorbar = None

            # Then remove colorbar axes thoroughly
            if hasattr(self, 'colorbar_ax') and self.colorbar_ax:
                try:
                    # Try multiple removal methods to ensure it's gone
                    if self.colorbar_ax in self.fig.axes:
                        self.fig.delaxes(self.colorbar_ax)
                except (ValueError, AttributeError, KeyError):
                    try:
                        self.colorbar_ax.remove()
                    except (ValueError, AttributeError, KeyError):
                        pass
                self.colorbar_ax = None

            # Redraw based on type
            data = self.contour_data

            if data['is_regular_grid']:
                X, Y, Z = data['X'], data['Y'], data['Z']

                if contour_type == 'contourf (filled)':
                    cs = self.ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)
                elif contour_type == 'contour (lines)':
                    cs = self.ax.contour(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)
                else:  # both
                    cs = self.ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)
                    self.ax.contour(X, Y, Z, levels=levels, colors=line_color, linewidths=line_width, alpha=0.8)

            else:  # Triangulated
                triang = tri.Triangulation(data['x'], data['y'])

                if contour_type == 'contourf (filled)':
                    cs = self.ax.tricontourf(triang, data['z'], levels=levels, cmap=cmap, alpha=alpha)
                elif contour_type == 'contour (lines)':
                    cs = self.ax.tricontour(triang, data['z'], levels=levels, cmap=cmap, alpha=alpha)
                else:  # both
                    cs = self.ax.tricontourf(triang, data['z'], levels=levels, cmap=cmap, alpha=alpha)
                    self.ax.tricontour(triang, data['z'], levels=levels, colors=line_color, linewidths=line_width, alpha=0.8)

            # Recreate colorbar only if toggle is on (using separate axes to prevent shrinking)
            if show_colorbar:
                self._add_colorbar_with_separate_axes(cs, label=self._apply_mathtext(data['field_label']))

                # Restore colorbar font sizes if they were previously set
                if hasattr(self, 'colorbar_label_fontsize') and self.colorbar:
                    self.colorbar.set_label(self._apply_mathtext(data['field_label']), fontsize=self.colorbar_label_fontsize)

                if hasattr(self, 'colorbar_tick_fontsize') and self.colorbar:
                    self.colorbar.ax.tick_params(labelsize=self.colorbar_tick_fontsize)
            else:
                self.colorbar = None
                self.colorbar_ax = None

            # Preserve any user-modified labels, title, aspect, and grid settings
            # Store current text properties to preserve user changes
            current_xlabel = self.ax.xaxis.label.get_text()
            current_ylabel = self.ax.yaxis.label.get_text()
            current_title = self.ax.title.get_text()

            # Store font properties
            title_fontsize = self.ax.title.get_fontsize()
            label_fontsize = self.ax.xaxis.label.get_fontsize()
            tick_labels = self.ax.xaxis.get_ticklabels()
            tick_fontsize = tick_labels[0].get_fontsize() if tick_labels else 10

            # Store font style properties
            title_fontweight = self.ax.title.get_fontweight()
            label_fontweight = self.ax.xaxis.label.get_fontweight()
            title_fontfamily = self.ax.title.get_fontfamily()
            label_fontfamily = self.ax.xaxis.label.get_fontfamily()
            title_color = self.ax.title.get_color()
            label_color = self.ax.xaxis.label.get_color()

            # Restore labels
            if current_xlabel:
                self.ax.set_xlabel(current_xlabel)
            if current_ylabel:
                self.ax.set_ylabel(current_ylabel)
            if current_title:
                self.ax.set_title(current_title)

            # Restore font sizes
            self.ax.title.set_fontsize(title_fontsize)
            self.ax.xaxis.label.set_fontsize(label_fontsize)
            self.ax.yaxis.label.set_fontsize(label_fontsize)
            self.ax.tick_params(labelsize=tick_fontsize)

            # Restore font styles
            self.ax.title.set_fontweight(title_fontweight)
            self.ax.xaxis.label.set_fontweight(label_fontweight)
            self.ax.yaxis.label.set_fontweight(label_fontweight)
            self.ax.title.set_fontfamily(title_fontfamily)
            self.ax.xaxis.label.set_fontfamily(label_fontfamily)
            self.ax.yaxis.label.set_fontfamily(label_fontfamily)
            self.ax.title.set_color(title_color)
            self.ax.xaxis.label.set_color(label_color)
            self.ax.yaxis.label.set_color(label_color)

            # Refresh display (canvas.draw is now handled in refresh method)
            self.refresh()

            print(f"✓ Contour redrawn with {levels} levels (type: {contour_type}, cmap: {cmap}, alpha: {alpha})")

        except Exception as e:
            print(f"❌ Error redrawing contour: {e}")
            import traceback
            traceback.print_exc()

    def _build_editing_controls(self):
        """Build COMPLETE figure editing controls after plot creation"""

        # Header (Refresh button removed - all controls auto-refresh)
        header = widgets.HTML(
            '<h3 style="border-bottom: 2px solid #3498db; padding: 10px; margin: 0 0 10px 0;">📊 Fine-Tune All Plot Parameters</h3>'
        )

        # Check if we're in subplot mode
        if self.subplot_mode and len(self.axes_list) > 1:
            # Build per-subplot editing controls
            control_tabs = self._build_subplot_editing_controls()
        else:
            # Build single-plot editing controls (original behavior)
            control_tabs = self._build_single_plot_editing_controls()

        self.editing_controls_container.children = [header, control_tabs]

    def _build_subplot_editing_controls(self):
        """Build editing controls for subplot mode - one tab per subplot"""
        subplot_tabs = []

        # Create a tab for each subplot
        for i, ax in enumerate(self.axes_list[:len(self.subplot_configs)]):
            subplot_controls = self._build_controls_for_subplot(i, ax)
            subplot_tabs.append(subplot_controls)

        # Add a global figure tab
        figure_tab = widgets.VBox([
            self._build_figure_controls()
        ], layout=widgets.Layout(padding='10px'))
        subplot_tabs.append(figure_tab)

        # Create outer tabs
        outer_tabs = widgets.Tab(children=subplot_tabs)
        for i in range(len(self.subplot_configs)):
            outer_tabs.set_title(i, f'Subplot {i+1}')
        outer_tabs.set_title(len(subplot_tabs) - 1, '📐 Global Figure')

        outer_tabs.layout = widgets.Layout(
            width='100%',
            min_height='300px',
            max_height='500px',
            overflow_y='auto'
        )

        return outer_tabs

    def _build_controls_for_subplot(self, subplot_idx, ax):
        """Build editing controls for a specific subplot"""
        config = self.subplot_configs[subplot_idx]
        subplot_plot_type = config.get('plot_type', 'line')
        inferred_type = self._infer_subplot_plot_type(ax)
        if inferred_type and inferred_type != subplot_plot_type:
            subplot_plot_type = inferred_type
            config['plot_type'] = inferred_type

        # Build all control sections for this specific subplot
        tab_children = []
        tab_titles = []

        # Per-line controls (for line/scatter plots ONLY)
        if subplot_plot_type in ['line', 'scatter']:
            per_line_controls = self._build_per_line_controls(ax)
            if per_line_controls:
                tab_children.append(per_line_controls)
                tab_titles.append('📈 Per-Line Settings')

        # Figure controls (only in single plot mode, not per-subplot)
        # Skip figure controls here - they're in global tab

        # Axes controls for this subplot (ALWAYS)
        tab_children.append(self._build_axes_controls(ax))
        tab_titles.append('📊 Axes')

        # Text/Font controls for this subplot (ALWAYS)
        tab_children.append(self._build_text_controls(ax))
        tab_titles.append('🔤 Text & Fonts')

        # Legend controls for this subplot (ALWAYS)
        tab_children.append(self._build_legend_controls(ax))
        tab_titles.append('📝 Legend')

        # Field/Contour controls (ONLY for contour/tricontour/cylindrical/scatter-with-color)
        # Use EXACT same logic as single-plot mode
        field_controls = self._build_field_controls_for_subplot(ax, subplot_idx, subplot_plot_type)
        if field_controls:
            tab_children.append(field_controls)
            if subplot_plot_type in ['contour', 'tricontour', 'cylindrical']:
                tab_titles.append('🌊 Field/Contour')
            elif subplot_plot_type == 'scatter':
                tab_titles.append('🎨 Scatter Color')

        if subplot_plot_type == 'vector':
            vector_field_controls = self._build_vector_field_controls(subplot_idx=subplot_idx)
            if vector_field_controls:
                tab_children.append(vector_field_controls)
                tab_titles.append('🌊 Field/Contour')
            vector_controls = self._build_vector_controls(subplot_idx=subplot_idx)
            if vector_controls:
                tab_children.append(vector_controls)
                tab_titles.append('🧭 Vector Field')

        # Create inner tabs for this subplot
        inner_tabs = widgets.Tab(children=tab_children)
        for i, title in enumerate(tab_titles):
            inner_tabs.set_title(i, title)

        inner_tabs.layout = widgets.Layout(
            width='100%',
            min_height='250px',
            overflow_y='auto'
        )

        return inner_tabs

    def _build_single_plot_editing_controls(self):
        """Build editing controls for single plot mode (original behavior)"""
        tab_children = []
        tab_titles = []

        # Per-line controls (for line/scatter plots)
        if self.plot_type in ['line', 'scatter']:
            per_line_controls = self._build_per_line_controls(self.ax)
            if per_line_controls:
                tab_children.append(per_line_controls)
                tab_titles.append('📈 Per-Line Settings')

        # Figure controls
        tab_children.append(self._build_figure_controls())
        tab_titles.append('📐 Figure')

        # Axes controls
        tab_children.append(self._build_axes_controls(self.ax))
        tab_titles.append('📊 Axes')

        # Text/Font controls
        tab_children.append(self._build_text_controls(self.ax))
        tab_titles.append('🔤 Text & Fonts')

        # Legend controls
        tab_children.append(self._build_legend_controls(self.ax))
        tab_titles.append('📝 Legend')

        # Field/Contour/Scatter controls (if applicable)
        field_controls = self._build_field_controls(self.ax)
        if field_controls:
            tab_children.append(field_controls)
            if self.plot_type in ['contour', 'tricontour', 'cylindrical']:
                tab_titles.append('🌊 Field/Contour')
            elif self.plot_type == 'scatter' and hasattr(self, 'scatter_data') and self.scatter_data is not None:
                tab_titles.append('🎨 Scatter Color')

        if self.plot_type == 'vector':
            vector_field_controls = self._build_vector_field_controls()
            if vector_field_controls:
                tab_children.append(vector_field_controls)
                tab_titles.append('🌊 Field/Contour')

        if self.plot_type == 'vector':
            vector_controls = self._build_vector_controls()
            if vector_controls:
                tab_children.append(vector_controls)
                tab_titles.append('🧭 Vector Field')

        # Create tabs
        control_tabs = widgets.Tab(children=tab_children)
        for i, title in enumerate(tab_titles):
            control_tabs.set_title(i, title)

        control_tabs.layout = widgets.Layout(
            width='100%',
            min_height='300px',
            max_height='500px',
            overflow_y='auto'
        )

        return control_tabs

    def _build_per_line_controls(self, ax=None):
        """Build per-line/per-series editing controls for line and scatter plots"""
        if ax is None:
            ax = self.ax
        if not ax:
            return None

        # Handle both line plots and scatter plots
        lines = ax.get_lines()
        # Filter scatter plots only (exclude contour collections)
        scatters = [c for c in ax.collections
                   if hasattr(c, 'get_offsets') and 'Contour' not in str(type(c))]

        if not lines and not scatters:
            return None

        per_series_accordions = []

        # Build controls for line plots
        for i, line in enumerate(lines):
            label = line.get_label()
            if label.startswith('_'):
                label = f"Line {i+1}"
            # Initialize stored legend raw if not present
            if getattr(line, '_ff_legend_raw', None) is None:
                try:
                    self._store_legend_label(line, label)
                except Exception:
                    pass

            # Controls for this line
            line_controls = []

            # Color
            color_dropdown = widgets.Dropdown(
                options=['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow'],
                value=self._get_closest_color(line.get_color()),
                description='Color:',
                style={'description_width': '120px'}
            )

            def make_color_callback(line_obj):
                def callback(change):
                    line_obj.set_color(change.new)
                    self.refresh()
                return callback

            color_dropdown.observe(make_color_callback(line), 'value')
            line_controls.append(color_dropdown)

            # Line width
            linewidth_slider = widgets.FloatSlider(
                value=line.get_linewidth(),
                min=0.5, max=5, step=0.1,
                description='Line Width:',
                style={'description_width': '120px'},
                readout_format='.1f'
            )

            def make_linewidth_callback(line_obj):
                def callback(change):
                    line_obj.set_linewidth(change.new)
                    self.refresh()
                return callback

            linewidth_slider.observe(make_linewidth_callback(line), 'value')
            line_controls.append(linewidth_slider)

            # Line style
            linestyle_dropdown = widgets.Dropdown(
                options=[('Solid', '-'), ('Dashed', '--'), ('Dotted', ':'), ('Dash-dot', '-.')],
                value=line.get_linestyle(),
                description='Line Style:',
                style={'description_width': '120px'}
            )

            def make_linestyle_callback(line_obj):
                def callback(change):
                    line_obj.set_linestyle(change.new)
                    self.refresh()
                return callback

            linestyle_dropdown.observe(make_linestyle_callback(line), 'value')
            line_controls.append(linestyle_dropdown)

            # Marker
            current_marker = line.get_marker()
            if current_marker == 'None':
                current_marker = ''

            marker_dropdown = widgets.Dropdown(
                options=[('None', ''), ('Circle', 'o'), ('Square', 's'), ('Triangle', '^'), ('Star', '*'), ('Plus', '+'), ('X', 'x'), ('Diamond', 'D')],
                value=current_marker,
                description='Marker:',
                style={'description_width': '120px'}
            )

            def make_marker_callback(line_obj):
                def callback(change):
                    line_obj.set_marker(change.new if change.new else 'None')
                    self.refresh()
                return callback

            marker_dropdown.observe(make_marker_callback(line), 'value')
            line_controls.append(marker_dropdown)

            # Marker size
            markersize_slider = widgets.FloatSlider(
                value=line.get_markersize(),
                min=1, max=20, step=0.5,
                description='Marker Size:',
                style={'description_width': '120px'},
                readout_format='.1f'
            )

            def make_markersize_callback(line_obj):
                def callback(change):
                    line_obj.set_markersize(change.new)
                    self.refresh()
                return callback

            markersize_slider.observe(make_markersize_callback(line), 'value')
            line_controls.append(markersize_slider)

            # Alpha/Transparency
            alpha_slider = widgets.FloatSlider(
                value=line.get_alpha() if line.get_alpha() is not None else 1.0,
                min=0, max=1, step=0.05,
                description='Transparency:',
                style={'description_width': '120px'},
                readout_format='.2f'
            )

            def make_alpha_callback(line_obj):
                def callback(change):
                    line_obj.set_alpha(change.new)
                    self.refresh()
                return callback

            alpha_slider.observe(make_alpha_callback(line), 'value')
            line_controls.append(alpha_slider)

            # Antialiasing
            antialiased_toggle = widgets.Checkbox(
                value=line.get_antialiased(),
                description='Antialiasing'
            )

            def make_antialiased_callback(line_obj):
                def callback(change):
                    line_obj.set_antialiased(change.new)
                    self.refresh()
                return callback

            antialiased_toggle.observe(make_antialiased_callback(line), 'value')
            line_controls.append(antialiased_toggle)

            # Legend label (customizable) with Apply gating
            legend_label_input = widgets.Text(
                value=line.get_label() if not line.get_label().startswith('_') else label,
                description='Legend Label:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='70%')
            )
            legend_label_apply = widgets.Button(description='Apply', button_style='')

            def on_legend_apply(btn, line_obj=line, ax_ref=ax, input_widget=legend_label_input):
                raw = input_widget.value
                # Validate/format
                ok, fmt = self._validate_format_math(raw)
                if not ok:
                    btn.description = '× Invalid'
                    btn.button_style = 'danger'
                else:
                    self._store_legend_label(line_obj, raw)
                    try:
                        line_obj.set_label(fmt)
                    except Exception:
                        line_obj.set_label(raw)
                    # Refresh legend if visible
                    show_legend = False
                    if self.subplot_mode and ax_ref in self.axes_list:
                        try:
                            idx = self.axes_list.index(ax_ref)
                            show_legend = self._get_subplot_legend_visible(idx)
                        except Exception:
                            show_legend = False
                    else:
                        show_legend = bool(self.legend_visible)
                    if show_legend:
                        try:
                            ax_ref.legend()
                        except Exception:
                            try:
                                line_obj.set_label(fmt.replace('$', ''))
                                ax_ref.legend()
                            except Exception:
                                pass
                    btn.description = '✓ Applied'
                    btn.button_style = 'success'
                    self.refresh()
                # Reset after 1s
                def _reset():
                    try:
                        btn.description = 'Apply'
                        btn.button_style = ''
                    except Exception:
                        pass
                threading.Timer(1.0, _reset).start()

            legend_label_apply.on_click(on_legend_apply)
            line_controls.append(widgets.HBox([legend_label_input, legend_label_apply]))

            # Create accordion for this line
            line_vbox = widgets.VBox(line_controls, layout=widgets.Layout(padding='10px'))
            per_series_accordions.append((label, line_vbox))

        # Build controls for scatter plots
        for i, scatter in enumerate(scatters):
            label = scatter.get_label()
            if label.startswith('_') or not label:
                label = f"Scatter {i+1}"
            if getattr(scatter, '_ff_legend_raw', None) is None:
                try:
                    self._store_legend_label(scatter, label)
                except Exception:
                    pass

            scatter_controls = []

            # Color
            current_colors = scatter.get_facecolors()
            if len(current_colors) > 0:
                current_color = current_colors[0]
            else:
                current_color = [0, 0, 1, 1]  # Default blue

            color_dropdown = widgets.Dropdown(
                options=['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow'],
                value='black',
                description='Color:',
                style={'description_width': '120px'}
            )

            def make_scatter_color_callback(scatter_obj):
                def callback(change):
                    scatter_obj.set_facecolors(change.new)
                    self.refresh()
                return callback

            color_dropdown.observe(make_scatter_color_callback(scatter), 'value')
            scatter_controls.append(color_dropdown)

            # Marker size
            current_sizes = scatter.get_sizes()
            current_size = current_sizes[0] if len(current_sizes) > 0 else 50

            size_slider = widgets.FloatSlider(
                value=current_size,
                min=1, max=50, step=1,
                description='Marker Size:',
                style={'description_width': '120px'},
                readout_format='.0f'
            )

            def make_size_callback(scatter_obj):
                def callback(change):
                    scatter_obj.set_sizes([change.new])
                    self.refresh()
                return callback

            size_slider.observe(make_size_callback(scatter), 'value')
            scatter_controls.append(size_slider)

            # Marker style
            marker_dropdown = widgets.Dropdown(
                options=[('Circle', 'o'), ('Square', 's'), ('Triangle', '^'), ('Star', '*'), ('Plus', '+'), ('X', 'x'), ('Diamond', 'D'), ('Pentagon', 'p')],
                value='o',
                description='Marker:',
                style={'description_width': '120px'}
            )

            def make_marker_callback(scatter_obj):
                def callback(change):
                    scatter_obj.set_paths([plt.matplotlib.markers.MarkerStyle(change.new).get_path()])
                    self.refresh()
                return callback

            marker_dropdown.observe(make_marker_callback(scatter), 'value')
            scatter_controls.append(marker_dropdown)

            # Legend label (customizable) with Apply gating
            scatter_legend_input = widgets.Text(
                value=scatter.get_label() if scatter.get_label() and not scatter.get_label().startswith('_') else label,
                description='Legend Label:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='70%')
            )
            scatter_legend_apply = widgets.Button(description='Apply', button_style='')

            def on_scatter_legend_apply(btn, scatter_obj=scatter, ax_ref=ax, input_widget=scatter_legend_input):
                raw = input_widget.value
                ok, fmt = self._validate_format_math(raw)
                if not ok:
                    btn.description = '× Invalid'
                    btn.button_style = 'danger'
                else:
                    self._store_legend_label(scatter_obj, raw)
                    try:
                        scatter_obj.set_label(fmt)
                    except Exception:
                        scatter_obj.set_label(raw)
                    show_legend = False
                    if self.subplot_mode and ax_ref in self.axes_list:
                        try:
                            idx = self.axes_list.index(ax_ref)
                            show_legend = self._get_subplot_legend_visible(idx)
                        except Exception:
                            show_legend = False
                    else:
                        show_legend = bool(self.legend_visible)
                    if show_legend:
                        try:
                            ax_ref.legend()
                        except Exception:
                            try:
                                scatter_obj.set_label(fmt.replace('$', ''))
                                ax_ref.legend()
                            except Exception:
                                pass
                    btn.description = '✓ Applied'
                    btn.button_style = 'success'
                    self.refresh()
                def _reset():
                    try:
                        btn.description = 'Apply'
                        btn.button_style = ''
                    except Exception:
                        pass
                threading.Timer(1.0, _reset).start()

            scatter_legend_apply.on_click(on_scatter_legend_apply)
            scatter_controls.append(widgets.HBox([scatter_legend_input, scatter_legend_apply]))

            # Transparency
            current_alpha = scatter.get_alpha()
            if current_alpha is None:
                current_alpha = 1.0

            alpha_slider = widgets.FloatSlider(
                value=current_alpha,
                min=0, max=1, step=0.05,
                description='Transparency:',
                style={'description_width': '120px'},
                readout_format='.2f'
            )

            def make_scatter_alpha_callback(scatter_obj):
                def callback(change):
                    scatter_obj.set_alpha(change.new)
                    self.refresh()
                return callback

            alpha_slider.observe(make_scatter_alpha_callback(scatter), 'value')
            scatter_controls.append(alpha_slider)

            # Edge color
            edge_color_dropdown = widgets.Dropdown(
                options=['none', 'black', 'white', 'blue', 'red', 'green', 'gray'],
                value='black',
                description='Edge Color:',
                style={'description_width': '120px'}
            )

            def make_edge_color_callback(scatter_obj):
                def callback(change):
                    scatter_obj.set_edgecolors(change.new)
                    self.refresh()
                return callback

            edge_color_dropdown.observe(make_edge_color_callback(scatter), 'value')
            scatter_controls.append(edge_color_dropdown)

            # Edge width
            edge_width_slider = widgets.FloatSlider(
                value=1.0,
                min=0, max=5, step=0.1,
                description='Edge Width:',
                style={'description_width': '120px'},
                readout_format='.1f'
            )

            def make_edge_width_callback(scatter_obj):
                def callback(change):
                    scatter_obj.set_linewidths(change.new)
                    self.refresh()
                return callback

            edge_width_slider.observe(make_edge_width_callback(scatter), 'value')
            scatter_controls.append(edge_width_slider)

            # Create accordion for this scatter series
            scatter_vbox = widgets.VBox(scatter_controls, layout=widgets.Layout(padding='10px'))
            per_series_accordions.append((label, scatter_vbox))

        # Create accordion containing all series (lines + scatters)
        accordion = widgets.Accordion(children=[item[1] for item in per_series_accordions])
        for i, (label, _) in enumerate(per_series_accordions):
            accordion.set_title(i, label)
        accordion.selected_index = 0

        plot_type_name = "series" if scatters else "lines"
        container = widgets.VBox([
            widgets.HTML(f'<p style="margin: 10px; font-style: italic;">Customize each {plot_type_name} individually:</p>'),
            accordion
        ])

        return container

    def _build_figure_controls(self):
        """Build COMPLETE figure property controls"""
        controls = []

        # Figure size
        width_slider = widgets.FloatSlider(
            value=self.fig.get_size_inches()[0],
            min=3, max=16, step=0.5,
            description='Width (in):',
            style={'description_width': '120px'},
            readout_format='.1f'
        )

        height_slider = widgets.FloatSlider(
            value=self.fig.get_size_inches()[1],
            min=3, max=16, step=0.5,
            description='Height (in):',
            style={'description_width': '120px'},
            readout_format='.1f'
        )

        def on_size_change(change):
            self.fig.set_size_inches(width_slider.value, height_slider.value)
            self.refresh()

        width_slider.observe(on_size_change, 'value')
        height_slider.observe(on_size_change, 'value')

        # DPI
        dpi_slider = widgets.IntSlider(
            value=int(self.fig.dpi),
            min=50, max=300, step=10,
            description='DPI:',
            style={'description_width': '120px'}
        )

        def on_dpi_change(change):
            self.fig.dpi = change.new
            self.refresh()

        dpi_slider.observe(on_dpi_change, 'value')

        # Tight layout
        tight_toggle = widgets.Checkbox(
            value=False,
            description='Tight Layout',
            tooltip='Automatically adjust layout (may reset some text properties)'
        )

        def on_tight_change(change):
            if change.new:
                try:
                    self.fig.tight_layout()
                    # CRITICAL: Reposition colorbars after tight_layout
                    # tight_layout can move axes, so colorbars need to follow
                    self._reposition_all_colorbars()
                except Exception as e:
                    print(f"⚠️ Tight layout failed: {e}")
            self.refresh()

        tight_toggle.observe(on_tight_change, 'value')

        controls.extend([width_slider, height_slider, dpi_slider, tight_toggle])

        # Subplot spacing controls (only show if in subplot mode)
        if self.subplot_mode and len(self.axes_list) > 1:
            controls.append(widgets.HTML('<hr><b>Subplot Spacing:</b>'))

            hspace_slider = widgets.FloatSlider(
                value=0.3,
                min=0.0, max=1.0, step=0.05,
                description='Vertical Gap:',
                style={'description_width': '120px'},
                readout_format='.2f'
            )

            wspace_slider = widgets.FloatSlider(
                value=0.3,
                min=0.0, max=1.0, step=0.05,
                description='Horizontal Gap:',
                style={'description_width': '120px'},
                readout_format='.2f'
            )

            def on_spacing_change(change):
                self.fig.subplots_adjust(
                    hspace=hspace_slider.value,
                    wspace=wspace_slider.value
                )
                # CRITICAL: Reposition colorbars after layout change
                # Colorbar axes don't automatically follow their parent axes
                self._reposition_all_colorbars()
                self.refresh()

            hspace_slider.observe(on_spacing_change, 'value')
            wspace_slider.observe(on_spacing_change, 'value')

            controls.extend([hspace_slider, wspace_slider])

        # Nudge colorbars button (helps fix any stale positions in display)
        nudge_btn = widgets.Button(
            description='Nudge Colorbars',
            button_style='warning',
            tooltip='Force colorbars to reposition to match current layout'
        )


        def on_nudge(btn):
            # Force draw to update positions
            try:
                self.fig.canvas.draw()
            except Exception:
                pass

            if self.subplot_mode and self.axes_list:
                # For each subplot, prefer a type-aware redraw to preserve styles
                for i, ax_i in enumerate(self.axes_list):
                    inferred = self._infer_subplot_plot_type(ax_i)
                    if inferred == 'vector':
                        # Full redraw preserves vector overlay/quiver styles and colorbar settings
                        self._redraw_vector(i)
                    else:
                        # Fallback to old toggle behavior for non-vector plots
                        label_text = None
                        if i < len(self.colorbars) and self.colorbars[i]:
                            try:
                                label_text = self.colorbars[i].ax.get_ylabel()
                            except Exception:
                                label_text = None

                        # Remove existing
                        self._remove_subplot_colorbar(i)

                        # Find a mappable to bind
                        mappable = None
                        contour_colls = [c for c in ax_i.collections if 'Contour' in str(type(c))]
                        if contour_colls:
                            mappable = contour_colls[0]
                            if label_text is None and i < len(self.subplot_contour_data) and self.subplot_contour_data[i]:
                                label_text = self.subplot_contour_data[i].get('field_label')
                        else:
                            scatter_colls = [c for c in ax_i.collections if hasattr(c, 'get_offsets') and 'Contour' not in str(type(c))]
                            if scatter_colls:
                                mappable = scatter_colls[0]
                                if label_text is None and i < len(self.subplot_scatter_data) and self.subplot_scatter_data[i]:
                                    label_text = self.subplot_scatter_data[i].get('color_column')

                        if mappable is not None:
                            self._create_subplot_colorbar(i, mappable, self._apply_mathtext(label_text or ''))
                # Ensure alignment after any changes
                self._reposition_all_colorbars()
            else:
                # Single plot: simulate toggle by removing and re-adding colorbar
                # Capture label if any
                if self.plot_type == 'vector':
                    # Use full vector redraw to preserve CB styles
                    self._redraw_vector()
                else:
                    label_text = None
                    if getattr(self, 'colorbar', None):
                        try:
                            label_text = self.colorbar.ax.get_ylabel()
                        except Exception:
                            label_text = None

                    # Remove
                    if getattr(self, 'colorbar', None):
                        try:
                            self.colorbar.remove()
                        except Exception:
                            pass
                        self.colorbar = None
                    if getattr(self, 'colorbar_ax', None):
                        try:
                            self.fig.delaxes(self.colorbar_ax)
                        except Exception:
                            try:
                                self.colorbar_ax.remove()
                            except Exception:
                                pass
                        self.colorbar_ax = None

                    # Find a mappable: prefer contour, else first scatter
                    mappable = None
                    contour_colls = [c for c in self.ax.collections if 'Contour' in str(type(c))]
                    if contour_colls:
                        mappable = contour_colls[0]
                        if label_text is None and hasattr(self, 'contour_data'):
                            label_text = getattr(self, 'contour_data', {}).get('field_label')
                    else:
                        scatter_colls = [c for c in self.ax.collections if hasattr(c, 'get_offsets') and 'Contour' not in str(type(c))]
                        if scatter_colls:
                            mappable = scatter_colls[0]
                            if label_text is None and hasattr(self, 'scatter_data'):
                                label_text = getattr(self, 'scatter_data', {}).get('color_column')

                    if mappable is not None:
                        # Use existing helper for single plot
                        self._add_colorbar_with_separate_axes(mappable, label=self._apply_mathtext(label_text or ''))
            self.refresh()

        nudge_btn.on_click(on_nudge)
        
        controls.append(widgets.HTML('<hr>'))
        controls.append(nudge_btn)

        return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

    def _build_axes_controls(self, ax=None):
        """Build COMPLETE axes property controls"""
        if ax is None:
            ax = self.ax
        if not ax:
            return widgets.HTML('<p>No axes found</p>')

        controls = []

        # Labels and title (Apply workflow)
        # Initialize per-axes store for user-entered labels
        if not hasattr(ax, '_ff_labels') or not isinstance(getattr(ax, '_ff_labels'), dict):
            try:
                ax._ff_labels = {
                    'xlabel': ax.get_xlabel(),
                    'ylabel': ax.get_ylabel(),
                    'title': ax.get_title(),
                }
            except Exception:
                ax._ff_labels = {}

        xlabel_input = widgets.Text(
            value=ax._ff_labels.get('xlabel', ax.get_xlabel()),
            description='X Label:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='70%')
        )
        xlabel_apply = widgets.Button(description='Apply', button_style='')

        ylabel_input = widgets.Text(
            value=ax._ff_labels.get('ylabel', ax.get_ylabel()),
            description='Y Label:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='70%')
        )
        ylabel_apply = widgets.Button(description='Apply', button_style='')

        title_input = widgets.Text(
            value=ax._ff_labels.get('title', ax.get_title()),
            description='Title:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='70%')
        )
        title_apply = widgets.Button(description='Apply', button_style='')

        # Store raw on typing (no figure update)
        xlabel_input.observe(lambda ch: self._validate_and_store_label(ax, 'xlabel', ch.new), 'value')
        ylabel_input.observe(lambda ch: self._validate_and_store_label(ax, 'ylabel', ch.new), 'value')
        title_input.observe(lambda ch: self._validate_and_store_label(ax, 'title', ch.new), 'value')

        def _apply_clicked(category, btn):
            widget = {'xlabel': xlabel_input, 'ylabel': ylabel_input, 'title': title_input}[category]
            ok = self._validate_and_store_label(ax, category, widget.value)
            if ok and self._apply_stored_label(ax, category):
                btn.description = '✓ Applied'
                btn.button_style = 'success'
                self.refresh()
            else:
                btn.description = '× Invalid'
                btn.button_style = 'danger'
            def _reset():
                try:
                    btn.description = 'Apply'
                    btn.button_style = ''
                except Exception:
                    pass
            threading.Timer(1.0, _reset).start()

        xlabel_apply.on_click(lambda b: _apply_clicked('xlabel', b))
        ylabel_apply.on_click(lambda b: _apply_clicked('ylabel', b))
        title_apply.on_click(lambda b: _apply_clicked('title', b))

        # Hint for mathtext usage
        math_hint = widgets.HTML('<p style="margin:6px 0;color:#666;font-size:0.9em;">Tip: With MathText ON, type TeX-like input (e.g., \\omega). Click Apply to update.</p>')
        controls.extend([
            widgets.HBox([xlabel_input, xlabel_apply]),
            widgets.HBox([ylabel_input, ylabel_apply]),
            widgets.HBox([title_input, title_apply]),
            math_hint,
            widgets.HTML('<hr>')
        ])

        # Axis limits
        controls.append(widgets.HTML('<b>Axis Limits:</b>'))

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xlim_min = widgets.FloatText(value=xlim[0], description='X Min:', style={'description_width': '60px'}, layout=widgets.Layout(width='180px'))
        xlim_max = widgets.FloatText(value=xlim[1], description='X Max:', style={'description_width': '60px'}, layout=widgets.Layout(width='180px'))
        ylim_min = widgets.FloatText(value=ylim[0], description='Y Min:', style={'description_width': '60px'}, layout=widgets.Layout(width='180px'))
        ylim_max = widgets.FloatText(value=ylim[1], description='Y Max:', style={'description_width': '60px'}, layout=widgets.Layout(width='180px'))

        def on_xlim_change(change):
            ax.set_xlim(xlim_min.value, xlim_max.value)
            self.refresh()

        def on_ylim_change(change):
            ax.set_ylim(ylim_min.value, ylim_max.value)
            self.refresh()

        xlim_min.observe(on_xlim_change, 'value')
        xlim_max.observe(on_xlim_change, 'value')
        ylim_min.observe(on_ylim_change, 'value')
        ylim_max.observe(on_ylim_change, 'value')

        controls.extend([widgets.HBox([xlim_min, xlim_max]), widgets.HBox([ylim_min, ylim_max])])
        controls.append(widgets.HTML('<hr>'))

        # Grid controls
        controls.append(widgets.HTML('<b>Grid Settings:</b>'))

        # Safely check if grid is visible
        try:
            # Prefer direct inspection of current gridline artists on both axes
            lines = list(ax.xaxis.get_gridlines()) + list(ax.yaxis.get_gridlines())
            grid_visible = any(line.get_visible() for line in lines) if lines else False
        except Exception:
            # Fallback to private attr if available
            grid_visible = getattr(ax.xaxis, "_gridOnMajor", False)

        grid_toggle = widgets.Checkbox(
            value=grid_visible,
            description='Show Grid'
        )

        def on_grid_change(change):
            # Apply grid using centralized helper to avoid side effects
            self._apply_grid(
                ax,
                change.new,
                grid_linestyle.value,
                grid_alpha.value,
                grid_linewidth.value,
            )
            self.refresh()

        grid_toggle.observe(on_grid_change, 'value')

        grid_linestyle = widgets.Dropdown(
            options=[('Solid', '-'), ('Dashed', '--'), ('Dotted', ':'), ('Dash-dot', '-.')],
            value='--',
            description='Grid Style:',
            style={'description_width': '120px'}
        )

        def on_grid_linestyle_change(change):
            # Do not force-enable grid when adjusting style; respect toggle
            self._apply_grid(
                ax,
                grid_toggle.value,
                change.new,
                grid_alpha.value,
                grid_linewidth.value,
            )
            self.refresh()

        grid_linestyle.observe(on_grid_linestyle_change, 'value')

        grid_alpha = widgets.FloatSlider(
            value=0.5,
            min=0.1, max=1.0, step=0.05,
            description='Grid Alpha:',
            style={'description_width': '120px'},
            readout_format='.2f'
        )

        def on_grid_alpha_change(change):
            # Do not force-enable grid when adjusting alpha; respect toggle
            self._apply_grid(
                ax,
                grid_toggle.value,
                grid_linestyle.value,
                change.new,
                grid_linewidth.value,
            )
            self.refresh()

        grid_alpha.observe(on_grid_alpha_change, 'value')

        grid_linewidth = widgets.FloatSlider(
            value=0.8,
            min=0.1, max=3.0, step=0.1,
            description='Grid Width:',
            style={'description_width': '120px'},
            readout_format='.1f'
        )

        def on_grid_linewidth_change(change):
            # Do not force-enable grid when adjusting width; respect toggle
            self._apply_grid(
                ax,
                grid_toggle.value,
                grid_linestyle.value,
                grid_alpha.value,
                change.new,
            )
            self.refresh()

        grid_linewidth.observe(on_grid_linewidth_change, 'value')

        controls.extend([grid_toggle, grid_linestyle, grid_alpha, grid_linewidth])
        controls.append(widgets.HTML('<hr>'))

        # Scale
        controls.append(widgets.HTML('<b>Scale & Appearance:</b>'))

        xscale_dropdown = widgets.Dropdown(
            options=['linear', 'log', 'symlog', 'logit'],
            value=ax.get_xscale(),
            description='X Scale:',
            style={'description_width': '120px'}
        )

        yscale_dropdown = widgets.Dropdown(
            options=['linear', 'log', 'symlog', 'logit'],
            value=ax.get_yscale(),
            description='Y Scale:',
            style={'description_width': '120px'}
        )

        def on_xscale_change(change):
            ax.set_xscale(change.new)
            self.refresh()

        def on_yscale_change(change):
            ax.set_yscale(change.new)
            self.refresh()

        xscale_dropdown.observe(on_xscale_change, 'value')
        yscale_dropdown.observe(on_yscale_change, 'value')

        # Aspect ratio
        aspect_dropdown = widgets.Dropdown(
            options=['auto', 'equal'],
            value='auto',
            description='Aspect:',
            style={'description_width': '120px'}
        )

        def on_aspect_change(change):
            ax.set_aspect(change.new)
            self.refresh()

        aspect_dropdown.observe(on_aspect_change, 'value')

        # Background color
        axes_bgcolor = widgets.Dropdown(
            options=['white', 'lightgray', 'lightblue', 'lightyellow', 'none'],
            value='white',
            description='BG Color:',
            style={'description_width': '120px'}
        )

        def on_axes_bgcolor_change(change):
            ax.set_facecolor(change.new)
            self.refresh()

        axes_bgcolor.observe(on_axes_bgcolor_change, 'value')

        # Tick direction
        tick_direction = widgets.Dropdown(
            options=['in', 'out', 'inout'],
            value='out',
            description='Tick Dir:',
            style={'description_width': '120px'}
        )

        def on_tick_direction_change(change):
            ax.tick_params(direction=change.new)
            self.refresh()

        tick_direction.observe(on_tick_direction_change, 'value')

        controls.extend([xscale_dropdown, yscale_dropdown, aspect_dropdown, axes_bgcolor, tick_direction])

        return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

    def _build_text_controls(self, ax=None):
        """Build COMPLETE text/font property controls"""
        if ax is None:
            ax = self.ax
        if not ax:
            return widgets.HTML('<p>No axes found</p>')

        controls = []

        # MathText toggle
        mathtext_toggle = widgets.Checkbox(
            value=self.use_mathtext,
            description='Use MathText for labels',
            tooltip='Enable to render TeX-like inputs (e.g., \\omega, \\nu) without $...$'
        )

        def on_mathtext_toggle(change):
            self.use_mathtext = bool(change.new)
            # Re-apply to current axes labels and title
            try:
                ax.set_xlabel(self._apply_mathtext(ax.get_xlabel()))
                ax.set_ylabel(self._apply_mathtext(ax.get_ylabel()))
                ax.set_title(self._apply_mathtext(ax.get_title()))
            except Exception:
                pass
            # Update any existing colorbars
            self._relabel_all_colorbars_mathtext()
            # Re-apply legend labels across the whole figure (global toggle)
            try:
                for ax_all in list(self.fig.axes):
                    self._reapply_legend_labels(ax_all)
            except Exception:
                pass
            self.refresh()

        mathtext_toggle.observe(on_mathtext_toggle, 'value')

        controls.append(mathtext_toggle)
        controls.append(widgets.HTML('<hr>'))

        # Font sizes
        controls.append(widgets.HTML('<b>Font Sizes:</b>'))

        title_fontsize = widgets.IntSlider(
            value=int(ax.title.get_fontsize()),
            min=6, max=40, step=1,
            description='Title Size:',
            style={'description_width': '120px'}
        )

        label_fontsize = widgets.IntSlider(
            value=int(ax.xaxis.label.get_fontsize()),
            min=6, max=40, step=1,
            description='Label Size:',
            style={'description_width': '120px'}
        )

        tick_fontsize = widgets.IntSlider(
            value=int(ax.xaxis.get_ticklabels()[0].get_fontsize()) if ax.xaxis.get_ticklabels() else 10,
            min=6, max=40, step=1,
            description='Tick Size:',
            style={'description_width': '120px'}
        )

        def on_title_fontsize_change(change):
            ax.title.set_fontsize(change.new)
            self.refresh()

        def on_label_fontsize_change(change):
            ax.xaxis.label.set_fontsize(change.new)
            ax.yaxis.label.set_fontsize(change.new)
            self.refresh()

        def on_tick_fontsize_change(change):
            ax.tick_params(labelsize=change.new)
            self.refresh()

        title_fontsize.observe(on_title_fontsize_change, 'value')
        label_fontsize.observe(on_label_fontsize_change, 'value')
        tick_fontsize.observe(on_tick_fontsize_change, 'value')

        controls.extend([title_fontsize, label_fontsize, tick_fontsize])
        controls.append(widgets.HTML('<hr>'))

        # Font family
        controls.append(widgets.HTML('<b>Font Style:</b>'))

        fontfamily_dropdown = widgets.Dropdown(
            options=['sans-serif', 'serif', 'monospace', 'cursive', 'fantasy'],
            value='sans-serif',
            description='Font Family:',
            style={'description_width': '120px'}
        )

        def on_fontfamily_change(change):
            ax.title.set_fontfamily(change.new)
            ax.xaxis.label.set_fontfamily(change.new)
            ax.yaxis.label.set_fontfamily(change.new)
            self.refresh()

        fontfamily_dropdown.observe(on_fontfamily_change, 'value')

        # Font weight
        fontweight_dropdown = widgets.Dropdown(
            options=['normal', 'bold', 'light', 'heavy'],
            value='normal',
            description='Font Weight:',
            style={'description_width': '120px'}
        )

        def on_fontweight_change(change):
            ax.title.set_fontweight(change.new)
            ax.xaxis.label.set_fontweight(change.new)
            ax.yaxis.label.set_fontweight(change.new)
            self.refresh()

        fontweight_dropdown.observe(on_fontweight_change, 'value')

        # Text color
        text_color_dropdown = widgets.Dropdown(
            options=['black', 'gray', 'darkblue', 'darkred', 'darkgreen', 'white'],
            value='black',
            description='Text Color:',
            style={'description_width': '120px'}
        )

        def on_text_color_change(change):
            ax.title.set_color(change.new)
            ax.xaxis.label.set_color(change.new)
            ax.yaxis.label.set_color(change.new)
            self.refresh()

        text_color_dropdown.observe(on_text_color_change, 'value')

        controls.extend([fontfamily_dropdown, fontweight_dropdown, text_color_dropdown])

        return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

    def _build_legend_controls(self, ax=None):
        """Build COMPLETE legend property controls"""
        if ax is None:
            ax = self.ax
        if not ax:
            return widgets.HTML('<p>No axes found</p>')

        controls = []

        # Legend visibility
        # Init legend visibility state
        init_visible = ax.get_legend() is not None
        if self.subplot_mode and ax in self.axes_list:
            idx = self.axes_list.index(ax)
            self._set_subplot_legend_visible(idx, init_visible)
        else:
            self.legend_visible = init_visible

        legend_toggle = widgets.Checkbox(
            value=init_visible,
            description='Show Legend'
        )

        def on_legend_toggle(change):
            # Persist visibility and apply/remove
            if self.subplot_mode and ax in self.axes_list:
                idx = self.axes_list.index(ax)
                self._set_subplot_legend_visible(idx, change.new)
            else:
                self.legend_visible = change.new

            if change.new:
                ax.legend()
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
            self.refresh()

        legend_toggle.observe(on_legend_toggle, 'value')

        # Legend location
        location_dropdown = widgets.Dropdown(
            options=['best', 'upper right', 'upper left', 'lower left', 'lower right',
                    'right', 'center left', 'center right', 'lower center', 'upper center', 'center'],
            value='best',
            description='Location:',
            style={'description_width': '120px'}
        )

        # Legend font size
        legend_fontsize = widgets.IntSlider(
            value=10,
            min=6, max=40, step=1,
            description='Font Size:',
            style={'description_width': '120px'}
        )

        # Number of columns
        legend_ncol = widgets.IntSlider(
            value=1,
            min=1, max=5, step=1,
            description='Columns:',
            style={'description_width': '120px'}
        )

        # Shadow
        legend_shadow = widgets.Checkbox(
            value=False,
            description='Shadow'
        )

        def on_legend_shadow_change(change):
            legend = ax.get_legend()
            if legend:
                legend.set_shadow(change.new)
                self.refresh()

        legend_shadow.observe(on_legend_shadow_change, 'value')

        # Border width
        legend_borderwidth = widgets.FloatSlider(
            value=1.0,
            min=0, max=5, step=0.5,
            description='Border Width:',
            style={'description_width': '120px'},
            readout_format='.1f'
        )

        def on_legend_borderwidth_change(change):
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_linewidth(change.new)
                self.refresh()

        legend_borderwidth.observe(on_legend_borderwidth_change, 'value')

        # Advanced positioning
        controls.append(widgets.HTML('<hr><b>Advanced Positioning:</b>'))

        outside_plot_toggle = widgets.Checkbox(
            value=False,
            description='Place Outside Plot'
        )

        bbox_x_slider = widgets.FloatSlider(
            value=1.0,
            min=-0.5, max=2.0, step=0.05,
            description='X Position:',
            style={'description_width': '120px'},
            readout_format='.2f',
            disabled=True
        )

        bbox_y_slider = widgets.FloatSlider(
            value=1.0,
            min=-0.5, max=2.0, step=0.05,
            description='Y Position:',
            style={'description_width': '120px'},
            readout_format='.2f',
            disabled=True
        )

        # Helper to recreate legend with all current settings
        def recreate_legend(**kwargs):
            legend = ax.get_legend()
            if legend:
                legend.remove()

            legend_kwargs = kwargs.copy()

            if outside_plot_toggle.value:
                # Use bbox_to_anchor for precise positioning
                legend_kwargs['bbox_to_anchor'] = (bbox_x_slider.value, bbox_y_slider.value)
                legend_kwargs['loc'] = 'upper left'

                # Create legend
                ax.legend(**legend_kwargs)

                # CRITICAL: Apply tight_layout to prevent cutoff when legend is outside
                # BUT: Don't use tight_layout in subplot mode - it resets text properties!
                if not (self.subplot_mode and len(self.axes_list) > 1):
                    try:
                        self.fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right
                        self._reposition_all_colorbars()  # Reposition after tight_layout
                    except Exception:
                        # Fallback if tight_layout fails
                        self.fig.subplots_adjust(right=0.85)
                        self._reposition_all_colorbars()
                else:
                    # In subplot mode, manually adjust
                    self.fig.subplots_adjust(right=0.85)
                    self._reposition_all_colorbars()
            else:
                # Use standard location
                legend_kwargs['loc'] = location_dropdown.value
                ax.legend(**legend_kwargs)

                # Reset layout for inside legends
                # BUT: Don't use tight_layout in subplot mode - it resets text properties!
                if not (self.subplot_mode and len(self.axes_list) > 1):
                    try:
                        self.fig.tight_layout()
                        self._reposition_all_colorbars()  # Reposition after tight_layout
                    except Exception:
                        pass

            self.refresh()

        # Now update all callbacks to use recreate_legend
        def on_location_change_updated(change):
            if not outside_plot_toggle.value:
                recreate_legend()

        location_dropdown.observe(on_location_change_updated, 'value')

        def on_legend_fontsize_change_updated(change):
            recreate_legend(fontsize=change.new)

        legend_fontsize.observe(on_legend_fontsize_change_updated, 'value')

        def on_legend_ncol_change_updated(change):
            recreate_legend(ncol=change.new)

        legend_ncol.observe(on_legend_ncol_change_updated, 'value')

        def on_outside_toggle(change):
            bbox_x_slider.disabled = not change.new
            bbox_y_slider.disabled = not change.new
            recreate_legend()

        def update_legend_position(change=None):
            recreate_legend()

        outside_plot_toggle.observe(on_outside_toggle, 'value')
        bbox_x_slider.observe(update_legend_position, 'value')
        bbox_y_slider.observe(update_legend_position, 'value')

        controls.extend([legend_toggle, location_dropdown, legend_fontsize, legend_ncol, legend_shadow, legend_borderwidth])

        controls.extend([
            outside_plot_toggle,
            widgets.HTML('<p style="margin: 5px 0; font-size: 0.9em; color: #666;"><i>Tip: (1.0, 1.0) = top-right corner, (0.5, 0.5) = center</i></p>'),
            bbox_x_slider,
            bbox_y_slider
        ])

        return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

    def _build_field_controls_for_subplot(self, ax, subplot_idx, plot_type):
        """Build EXACT same field/contour controls as single-plot mode for a specific subplot"""

        controls = []
        config = self.subplot_configs[subplot_idx]

        # === SCATTER PLOT CONTROLS ===
        if plot_type == 'scatter':
            # Check if scatter data exists (i.e., color mapping was used)
            has_scatter_data = (subplot_idx < len(self.subplot_scatter_data) and
                              self.subplot_scatter_data[subplot_idx] is not None)

            if not has_scatter_data:
                # No color mapping - no controls needed
                return None

            controls.append(widgets.HTML(f'<b>Scatter Color Settings for Subplot {subplot_idx+1}:</b><br>'))

            # Get scatter collections from this axis
            scatter_collections = [c for c in ax.collections
                                 if hasattr(c, 'get_offsets') and 'Contour' not in str(type(c))]

            if not scatter_collections:
                controls.append(widgets.HTML('<p>No scatter plot found.</p>'))
                return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

            data = self.subplot_scatter_data[subplot_idx]

            # Colormap dropdown
            cmap_dropdown = widgets.Dropdown(
                options=['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                        'Greys', 'Greys_r', 'Blues', 'Blues_r', 'Reds', 'Reds_r',
                        'Greens', 'Greens_r', 'RdBu', 'RdBu_r', 'coolwarm', 'seismic'],
                value=data.get('cmap', 'viridis'),
                description='Colormap:',
                style={'description_width': '120px'}
            )

            # vmin/vmax sliders
            vmin_default = data['vmin']
            vmax_default = data['vmax']

            vmin_slider = widgets.FloatSlider(
                value=vmin_default,
                min=vmin_default - abs(vmin_default) * 0.5,
                max=vmax_default,
                step=(vmax_default - vmin_default) / 100 if vmax_default != vmin_default else 0.01,
                description='Color Min:',
                style={'description_width': '120px'},
                readout_format='.3f'
            )

            vmax_slider = widgets.FloatSlider(
                value=vmax_default,
                min=vmin_default,
                max=vmax_default + abs(vmax_default) * 0.5,
                step=(vmax_default - vmin_default) / 100 if vmax_default != vmin_default else 0.01,
                description='Color Max:',
                style={'description_width': '120px'},
                readout_format='.3f'
            )

            controls.extend([cmap_dropdown, vmin_slider, vmax_slider])
            controls.append(widgets.HTML('<hr>'))

            # Colorbar toggle
            controls.append(widgets.HTML('<b>Colorbar Display:</b>'))
            colorbar_toggle = widgets.Checkbox(
                value=(subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx] is not None),
                description='Show Colorbar'
            )
            controls.append(colorbar_toggle)
            controls.append(widgets.HTML('<hr>'))

            # Declare gap/width widgets early
            gap_widget_ref = [0.01]
            width_widget_ref = [0.01]

            # Helper function to trigger redraw
            def _trigger_scatter_redraw():
                cmap = cmap_dropdown.value
                vmin = vmin_slider.value
                vmax = vmax_slider.value
                show_colorbar = colorbar_toggle.value
                gap = gap_widget_ref[0]
                width = width_widget_ref[0]

                self._redraw_subplot_scatter(subplot_idx, cmap, vmin, vmax, show_colorbar, gap, width)

            # Connect callbacks
            cmap_dropdown.observe(lambda change: _trigger_scatter_redraw(), 'value')
            vmin_slider.observe(lambda change: _trigger_scatter_redraw(), 'value')
            vmax_slider.observe(lambda change: _trigger_scatter_redraw(), 'value')
            colorbar_toggle.observe(lambda change: _trigger_scatter_redraw(), 'value')

            # Colorbar customization
            if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
                controls.append(widgets.HTML('<b>Colorbar Customization:</b>'))

                colorbar_label = widgets.Text(
                    value=self.colorbars[subplot_idx].ax.get_ylabel(),
                    description='Label:',
                    style={'description_width': '120px'}
                )

                # Ensure settings store exists
                while len(self.subplot_colorbar_settings) <= subplot_idx:
                    self.subplot_colorbar_settings.append({})

                def on_colorbar_label_change(change):
                    self.subplot_colorbar_settings[subplot_idx]['label'] = change.new
                    if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
                        self.colorbars[subplot_idx].set_label(self._apply_mathtext(change.new))
                        self.refresh()

                colorbar_label.observe(on_colorbar_label_change, 'value')

                colorbar_fontsize = widgets.IntSlider(
                    value=10, min=6, max=40, step=1,
                    description='Label Size:',
                    style={'description_width': '120px'}
                )

                def on_colorbar_fontsize_change(change):
                    self.subplot_colorbar_settings[subplot_idx]['label_fontsize'] = change.new
                    if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
                        cbar = self.colorbars[subplot_idx]
                        cbar.set_label(self._apply_mathtext(cbar.ax.get_ylabel()), fontsize=change.new)
                        self.refresh()

                colorbar_fontsize.observe(on_colorbar_fontsize_change, 'value')

                colorbar_tick_fontsize = widgets.IntSlider(
                    value=9, min=6, max=40, step=1,
                    description='Tick Size:',
                    style={'description_width': '120px'}
                )

                def on_colorbar_tick_fontsize_change(change):
                    self.subplot_colorbar_settings[subplot_idx]['tick_fontsize'] = change.new
                    if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
                        self.colorbars[subplot_idx].ax.tick_params(labelsize=change.new)
                        self.refresh()

                colorbar_tick_fontsize.observe(on_colorbar_tick_fontsize_change, 'value')

                colorbar_fontweight = widgets.Dropdown(
                    options=['normal', 'bold', 'light', 'heavy'],
                    value='normal',
                    description='Label Weight:',
                    style={'description_width': '120px'}
                )

                def on_colorbar_fontweight_change(change):
                    self.subplot_colorbar_settings[subplot_idx]['fontweight'] = change.new
                    if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
                        cbar = self.colorbars[subplot_idx]
                        current_label = cbar.ax.get_ylabel()
                        current_fontsize = cbar.ax.yaxis.label.get_fontsize()
                        cbar.set_label(self._apply_mathtext(current_label), fontsize=current_fontsize, weight=change.new)
                        self.refresh()

                colorbar_fontweight.observe(on_colorbar_fontweight_change, 'value')

                # Gap and Width controls
                colorbar_gap = widgets.FloatSlider(
                    value=0.01, min=0.00, max=0.15, step=0.005,
                    description='Gap:',
                    style={'description_width': '120px'},
                    readout_format='.3f',
                    tooltip='Horizontal gap between plot and colorbar'
                )

                def on_colorbar_gap_change(change):
                    gap_widget_ref[0] = change.new
                    settings = self._get_subplot_settings(subplot_idx)
                    settings['gap'] = change.new
                    _trigger_scatter_redraw()

                colorbar_gap.observe(on_colorbar_gap_change, 'value')

                colorbar_width = widgets.FloatSlider(
                    value=0.01, min=0.01, max=0.05, step=0.005,
                    description='Width:',
                    style={'description_width': '120px'},
                    readout_format='.3f',
                    tooltip='Width of the colorbar'
                )

                def on_colorbar_width_change(change):
                    width_widget_ref[0] = change.new
                    settings = self._get_subplot_settings(subplot_idx)
                    settings['width'] = change.new
                    _trigger_scatter_redraw()

                colorbar_width.observe(on_colorbar_width_change, 'value')

                controls.extend([colorbar_label, colorbar_fontsize, colorbar_tick_fontsize,
                               colorbar_fontweight, colorbar_gap, colorbar_width])

            return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

        # === CONTOUR PLOT CONTROLS ===
        # Only show for contour plots
        if plot_type not in ['contour', 'tricontour', 'cylindrical']:
            return None

        # NOTE: For subplot mode, we provide the EXACT same controls as single-plot mode

        controls.append(widgets.HTML(f'<b>Contour Settings for Subplot {subplot_idx+1}:</b><br>'))

        # Get contour collections from this specific axis
        contour_colls = [c for c in ax.collections if 'Contour' in str(type(c))]

        if not contour_colls:
            controls.append(widgets.HTML('<p>No contour found. Create plot first.</p>'))
            return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

        # ===== EXACT SAME CONTROLS AS SINGLE-PLOT MODE =====

        # 1. CONTOUR TYPE (filled/lines/both) - WORKING
        controls.append(widgets.HTML('<b>Contour Type:</b>'))

        contour_type_dropdown = widgets.Dropdown(
            options=['contourf (filled)', 'contour (lines)', 'both (filled + lines)'],
            value='contourf (filled)',
            description='Type:',
            style={'description_width': '120px'}
        )

        # Line controls for "both" mode
        line_thickness = widgets.FloatSlider(
            value=0.5, min=0.1, max=5.0, step=0.1,
            description='Line Width:', style={'description_width': '80px'},
            layout=widgets.Layout(width='350px')
        )

        line_color = widgets.Dropdown(
            options=['black', 'white', 'red', 'blue', 'green'],
            value='black', description='Line Color:',
            style={'description_width': '80px'}, layout=widgets.Layout(width='250px')
        )

        line_controls_box = widgets.HBox([line_thickness, line_color],
                                        layout=widgets.Layout(display='none'))

        controls.append(contour_type_dropdown)
        controls.append(line_controls_box)
        controls.append(widgets.HTML('<hr>'))

        # 2. COLORBAR DISPLAY
        controls.append(widgets.HTML('<b>Colorbar Display:</b>'))

        colorbar_toggle = widgets.Checkbox(
            value=(subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx] is not None),
            description='Show Colorbar'
        )
        controls.append(colorbar_toggle)
        controls.append(widgets.HTML('<hr>'))

        # 3. CONTOUR LEVELS - WORKING
        controls.append(widgets.HTML('<b>Contour Levels:</b>'))

        levels_slider = widgets.IntSlider(
            value=15, min=1, max=200, step=1,
            description='Levels:', style={'description_width': '120px'}
        )

        levels_log_toggle = widgets.Checkbox(
            value=False,
            description='Logarithmic scale'
        )

        controls.extend([levels_slider, levels_log_toggle])
        controls.append(widgets.HTML('<hr>'))

        # 4. COLORMAP SELECTOR (EXACT same as single-plot mode)
        controls.append(widgets.HTML('<b>Colormap:</b>'))

        cmap_dropdown = widgets.Dropdown(
            options=[
                ('─── Sequential ───', 'viridis'),
                ('Greys', 'Greys'),
                ('Greys (reversed)', 'Greys_r'),
                ('Reds', 'Reds'),
                ('Reds (reversed)', 'Reds_r'),
                ('Blues', 'Blues'),
                ('Blues (reversed)', 'Blues_r'),
                ('Greens', 'Greens'),
                ('Greens (reversed)', 'Greens_r'),
                ('Viridis', 'viridis'),
                ('Plasma', 'plasma'),
                ('Inferno', 'inferno'),
                ('Magma', 'magma'),
                ('Cividis', 'cividis'),
                ('─── Diverging ───', 'RdBu_r'),
                ('Red-Blue', 'RdBu'),
                ('Red-Blue (reversed)', 'RdBu_r'),
                ('Red-Yellow-Blue', 'RdYlBu'),
                ('Cool-Warm', 'coolwarm'),
                ('Seismic', 'seismic'),
                ('─── Miscellaneous ───', 'jet'),
                ('Jet', 'jet'),
                ('Rainbow', 'rainbow'),
                ('Turbo', 'turbo'),
                ('Twilight', 'twilight')
            ],
            value='viridis',
            description='Colormap:',
            style={'description_width': '120px'}
        )

        def on_cmap_change(change):
            _trigger_redraw()

        cmap_dropdown.observe(on_cmap_change, 'value')
        controls.append(cmap_dropdown)

        # Transparency slider (EXACT same as single-plot mode)
        alpha_slider = widgets.FloatSlider(
            value=1.0,
            min=0, max=1, step=0.05,
            description='Transparency:',
            style={'description_width': '120px'},
            readout_format='.2f'
        )

        def on_alpha_change(change):
            _trigger_redraw()

        alpha_slider.observe(on_alpha_change, 'value')
        controls.append(alpha_slider)

        # Declare gap/width widgets early so they can be referenced in _trigger_redraw
        # Will be populated after the colorbar section
        gap_widget_ref = [0.01]  # Using list for mutability in closure
        width_widget_ref = [0.01]

        # Helper function to trigger redraw
        def _trigger_redraw():
            # Get current settings
            contour_type = contour_type_dropdown.value

            if levels_log_toggle.value:
                levels = int(10 ** levels_slider.value)
            else:
                levels = int(levels_slider.value)

            cmap = cmap_dropdown.value
            alpha = alpha_slider.value
            show_colorbar = colorbar_toggle.value
            line_width = line_thickness.value
            line_color_val = line_color.value
            gap = gap_widget_ref[0]
            width = width_widget_ref[0]

            # Call redraw method
            self._redraw_subplot_contour(subplot_idx, contour_type, levels, cmap, alpha,
                                        show_colorbar, line_width, line_color_val,
                                        gap, width)

        # Callbacks for contour type and levels
        def on_contour_type_change(change):
            is_both = change.new == 'both (filled + lines)'
            line_controls_box.layout.display = 'flex' if is_both else 'none'
            _trigger_redraw()

        def on_colorbar_toggle(change):
            _trigger_redraw()

        def on_levels_change(change):
            _trigger_redraw()

        def on_log_toggle(change):
            if change.new:
                levels_slider.min = 1
                levels_slider.max = 3
                levels_slider.step = 0.1
                levels_slider.value = 1.3
            else:
                levels_slider.min = 1
                levels_slider.max = 200
                levels_slider.step = 1
                levels_slider.value = 15
            _trigger_redraw()

        # Connect all observers
        contour_type_dropdown.observe(on_contour_type_change, 'value')
        colorbar_toggle.observe(on_colorbar_toggle, 'value')
        levels_slider.observe(on_levels_change, 'value')
        levels_log_toggle.observe(on_log_toggle, 'value')
        line_thickness.observe(lambda change: _trigger_redraw(), 'value')
        line_color.observe(lambda change: _trigger_redraw(), 'value')

        # Colorbar controls (EXACT same as single-plot mode)
        if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
            controls.append(widgets.HTML('<hr><b>Colorbar:</b>'))

            colorbar_label = widgets.Text(
                value=self.colorbars[subplot_idx].ax.get_ylabel(),
                description='Label:',
                style={'description_width': '120px'}
            )

            # Ensure settings store exists
            while len(self.subplot_colorbar_settings) <= subplot_idx:
                self.subplot_colorbar_settings.append({})

            def on_colorbar_label_change(change):
                # Persist label text
                self.subplot_colorbar_settings[subplot_idx]['label'] = change.new
                if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
                    self.colorbars[subplot_idx].set_label(change.new)
                    self.refresh()

            colorbar_label.observe(on_colorbar_label_change, 'value')
            controls.append(colorbar_label)

            # Label Size (EXACT same as single-plot mode)
            colorbar_fontsize = widgets.IntSlider(
                value=10,
                min=6, max=40, step=1,
                description='Label Size:',
                style={'description_width': '120px'}
            )

            def on_colorbar_fontsize_change(change):
                # Persist label font size
                self.subplot_colorbar_settings[subplot_idx]['label_fontsize'] = change.new
                if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
                    cbar = self.colorbars[subplot_idx]
                    cbar.set_label(cbar.ax.get_ylabel(), fontsize=change.new)
                    self.refresh()

            colorbar_fontsize.observe(on_colorbar_fontsize_change, 'value')
            controls.append(colorbar_fontsize)

            # Tick Size (EXACT same as single-plot mode)
            colorbar_tick_fontsize = widgets.IntSlider(
                value=9,
                min=6, max=40, step=1,
                description='Tick Size:',
                style={'description_width': '120px'}
            )

            def on_colorbar_tick_fontsize_change(change):
                # Persist tick font size
                self.subplot_colorbar_settings[subplot_idx]['tick_fontsize'] = change.new
                if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
                    self.colorbars[subplot_idx].ax.tick_params(labelsize=change.new)
                    self.refresh()

            colorbar_tick_fontsize.observe(on_colorbar_tick_fontsize_change, 'value')
            controls.append(colorbar_tick_fontsize)

            # Label Weight (EXACT same as single-plot mode)
            colorbar_fontweight = widgets.Dropdown(
                options=['normal', 'bold', 'light', 'heavy'],
                value='normal',
                description='Label Weight:',
                style={'description_width': '120px'}
            )

            def on_colorbar_fontweight_change(change):
                # Persist weight
                self.subplot_colorbar_settings[subplot_idx]['fontweight'] = change.new
                if subplot_idx < len(self.colorbars) and self.colorbars[subplot_idx]:
                    cbar = self.colorbars[subplot_idx]
                    current_label = cbar.ax.get_ylabel()
                    current_fontsize = cbar.ax.yaxis.label.get_fontsize()
                    cbar.set_label(current_label, fontsize=current_fontsize, weight=change.new)
                    self.refresh()

            colorbar_fontweight.observe(on_colorbar_fontweight_change, 'value')
            controls.append(colorbar_fontweight)

            # Colorbar position controls (Gap and Width)
            colorbar_gap = widgets.FloatSlider(
                value=0.01,  # Default gap for subplots
                min=0.00, max=0.15, step=0.005,
                description='Gap:',
                style={'description_width': '120px'},
                readout_format='.3f',
                tooltip='Horizontal gap between plot and colorbar'
            )

            def on_colorbar_gap_change(change):
                # Persist and trigger redraw
                gap_widget_ref[0] = change.new
                settings = self._get_subplot_settings(subplot_idx)
                settings['gap'] = change.new
                _trigger_redraw()

            colorbar_gap.observe(on_colorbar_gap_change, 'value')

            colorbar_width = widgets.FloatSlider(
                value=0.01,  # Default width for subplots
                min=0.01, max=0.05, step=0.005,
                description='Width:',
                style={'description_width': '120px'},
                readout_format='.3f',
                tooltip='Width of the colorbar'
            )

            def on_colorbar_width_change(change):
                # Persist and trigger redraw
                width_widget_ref[0] = change.new
                settings = self._get_subplot_settings(subplot_idx)
                settings['width'] = change.new
                _trigger_redraw()

            colorbar_width.observe(on_colorbar_width_change, 'value')

            controls.extend([colorbar_gap, colorbar_width])

        return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

    def _build_field_controls(self, ax=None, plot_type=None, subplot_mode=False, subplot_idx=None):
        """Build field/contour and scatter specific controls"""
        if ax is None:
            ax = self.ax
        if plot_type is None:
            plot_type = self.plot_type

        # Check if this is a scatter plot with color intensity
        is_scatter_with_color = (plot_type == 'scatter' and
                                 hasattr(self, 'scatter_data') and
                                 self.scatter_data is not None)

        # Only show for contour plots or scatter with color intensity
        if plot_type not in ['contour', 'tricontour', 'cylindrical'] and not is_scatter_with_color:
            return None

        controls = []

        # In subplot mode, provide simplified controls
        if subplot_mode and plot_type in ['contour', 'tricontour', 'cylindrical']:
            return self._build_simplified_contour_controls_for_subplot(ax, subplot_idx)

        # In subplot mode for scatter, provide simplified controls
        if subplot_mode and is_scatter_with_color:
            controls.append(widgets.HTML('<b>Note:</b> Scatter color controls work best in single-plot mode.<br>Basic colorbar is shown.'))
            return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

        # === SCATTER PLOT COLOR CONTROLS ===
        if is_scatter_with_color:
            controls.append(widgets.HTML('<b>Scatter Color Settings:</b>'))

            # Get first scatter collection
            scatter_collections = [c for c in ax.collections
                                  if hasattr(c, 'get_offsets') and 'Contour' not in str(type(c))]

            if scatter_collections:
                scatter = scatter_collections[0]

                # Colormap selector
                cmap_dropdown = widgets.Dropdown(
                    options=['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                            'Greys', 'Greys_r', 'Blues', 'Blues_r', 'Reds', 'Reds_r', 'Greens', 'Greens_r'
                            'RdBu', 'RdBu_r', 'coolwarm', 'seismic'],
                    value='viridis',
                    description='Colormap:',
                    style={'description_width': '120px'}
                )

                def on_scatter_cmap_change(change):
                    for sc in scatter_collections:
                        sc.set_cmap(change.new)
                    if self.colorbar:
                        self.colorbar.update_normal(scatter)
                    self.refresh()

                cmap_dropdown.observe(on_scatter_cmap_change, 'value')
                controls.append(cmap_dropdown)

                # vmin/vmax sliders for color range
                color_data = self.scatter_data['color_data']
                vmin_default = float(color_data.min())
                vmax_default = float(color_data.max())

                vmin_slider = widgets.FloatSlider(
                    value=vmin_default,
                    min=vmin_default - abs(vmin_default) * 0.5,
                    max=vmax_default,
                    step=(vmax_default - vmin_default) / 100,
                    description='Color Min:',
                    style={'description_width': '120px'},
                    readout_format='.3f'
                )

                vmax_slider = widgets.FloatSlider(
                    value=vmax_default,
                    min=vmin_default,
                    max=vmax_default + abs(vmax_default) * 0.5,
                    step=(vmax_default - vmin_default) / 100,
                    description='Color Max:',
                    style={'description_width': '120px'},
                    readout_format='.3f'
                )

                def on_vmin_change(change):
                    for sc in scatter_collections:
                        sc.set_clim(vmin=change.new)
                    self.refresh()

                def on_vmax_change(change):
                    for sc in scatter_collections:
                        sc.set_clim(vmax=change.new)
                    self.refresh()

                vmin_slider.observe(on_vmin_change, 'value')
                vmax_slider.observe(on_vmax_change, 'value')

                controls.extend([vmin_slider, vmax_slider])
                controls.append(widgets.HTML('<hr>'))

                # Colorbar customization controls for scatter
                if self.colorbar:
                    controls.append(widgets.HTML('<b>Colorbar Customization:</b>'))

                    colorbar_label = widgets.Text(
                        value=self.colorbar.ax.get_ylabel(),
                        description='Label:',
                        style={'description_width': '120px'}
                    )

                    def on_scatter_colorbar_label_change(change):
                        self.colorbar.set_label(self._apply_mathtext(change.new))
                        self.refresh()

                    colorbar_label.observe(on_scatter_colorbar_label_change, 'value')

                    colorbar_fontsize = widgets.IntSlider(
                        value=10,
                        min=6, max=40, step=1,
                        description='Label Size:',
                        style={'description_width': '120px'}
                    )

                    def on_scatter_colorbar_fontsize_change(change):
                        self.colorbar.set_label(self.colorbar.ax.get_ylabel(), fontsize=change.new)
                        self.refresh()

                    colorbar_fontsize.observe(on_scatter_colorbar_fontsize_change, 'value')

                    colorbar_tick_fontsize = widgets.IntSlider(
                        value=9,
                        min=6, max=40, step=1,
                        description='Tick Size:',
                        style={'description_width': '120px'}
                    )

                    def on_scatter_colorbar_tick_fontsize_change(change):
                        self.colorbar.ax.tick_params(labelsize=change.new)
                        self.refresh()

                    colorbar_tick_fontsize.observe(on_scatter_colorbar_tick_fontsize_change, 'value')

                    # NEW: Colorbar label font weight/style
                    colorbar_fontweight = widgets.Dropdown(
                        options=['normal', 'bold', 'light', 'heavy'],
                        value='normal',
                        description='Label Weight:',
                        style={'description_width': '120px'}
                    )

                    def on_scatter_colorbar_fontweight_change(change):
                        if self.colorbar:
                            # Re-apply label with new weight
                            current_label = self.colorbar.ax.get_ylabel()
                            current_fontsize = self.colorbar.ax.yaxis.label.get_fontsize()
                            self.colorbar.set_label(self._apply_mathtext(current_label), fontsize=current_fontsize, weight=change.new)
                            self.refresh()

                    colorbar_fontweight.observe(on_scatter_colorbar_fontweight_change, 'value')

                    # Colorbar position controls
                    colorbar_gap = widgets.FloatSlider(
                        value=getattr(self, 'colorbar_gap', 0.02),
                        min=0.00, max=0.15, step=0.005,
                        description='Gap:',
                        style={'description_width': '120px'},
                        readout_format='.3f',
                        tooltip='Horizontal gap between plot and colorbar'
                    )

                    def on_scatter_colorbar_gap_change(change):
                        self.colorbar_gap = change.new
                        # Recreate the scatter plot colorbar with new position
                        if hasattr(self, 'scatter_data') and len(scatter_collections) > 0:
                            self._add_colorbar_with_separate_axes(
                                scatter_collections[0],
                                label=self._apply_mathtext(self.scatter_data['color_column'])
                            )
                        self.refresh()

                    colorbar_gap.observe(on_scatter_colorbar_gap_change, 'value')

                    colorbar_width = widgets.FloatSlider(
                        value=getattr(self, 'colorbar_width', 0.02),
                        min=0.01, max=0.05, step=0.005,
                        description='Width:',
                        style={'description_width': '120px'},
                        readout_format='.3f',
                        tooltip='Width of the colorbar'
                    )

                    def on_scatter_colorbar_width_change(change):
                        self.colorbar_width = change.new
                        # Recreate the scatter plot colorbar with new width
                        if hasattr(self, 'scatter_data') and len(scatter_collections) > 0:
                            self._add_colorbar_with_separate_axes(
                                scatter_collections[0],
                                label=self._apply_mathtext(self.scatter_data['color_column'])
                            )
                        self.refresh()

                    colorbar_width.observe(on_scatter_colorbar_width_change, 'value')

                    controls.extend([colorbar_label, colorbar_fontsize, colorbar_tick_fontsize,
                                   colorbar_fontweight, colorbar_gap, colorbar_width])

            return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

        # === CONTOUR PLOT CONTROLS (existing code) ===

        # Contour type selector
        controls.append(widgets.HTML('<b>Contour Type:</b>'))

        contour_type_dropdown = widgets.Dropdown(
            options=['contourf (filled)', 'contour (lines)', 'both (filled + lines)'],
            value='contourf (filled)',
            description='Type:',
            style={'description_width': '120px'}
        )

        # Store reference for redrawing
        self.contour_type_dropdown = contour_type_dropdown

        # Line controls for "both" mode (initially hidden, displayed horizontally)
        contour_line_thickness = widgets.FloatSlider(
            value=0.5,
            min=0.1, max=5.0, step=0.1,
            description='Line Width:',
            style={'description_width': '80px'},
            readout_format='.1f',
            layout=widgets.Layout(width='350px')
        )

        contour_line_color = widgets.Dropdown(
            options=['black', 'white', 'red', 'blue', 'green', 'cyan', 'magenta', 'yellow'],
            value='black',
            description='Line Color:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='250px')
        )

        # Put line controls in horizontal layout
        line_controls_box = widgets.HBox([contour_line_thickness, contour_line_color],
                                         layout=widgets.Layout(display='none'))

        # Single unified callback for contour type changes
        def on_contour_type_change(change):
            is_both = change.new == 'both (filled + lines)'
            line_controls_box.layout.display = 'flex' if is_both else 'none'
            print(f"⚠️ Contour type changed to: {change.new}")
            print("   Redrawing plot...")
            self._redraw_contour()

        contour_type_dropdown.observe(on_contour_type_change, 'value')
        controls.append(contour_type_dropdown)

        # Store line control references
        self.contour_line_thickness = contour_line_thickness
        self.contour_line_color = contour_line_color

        # Add line controls callbacks
        def on_line_thickness_change(change):
            if contour_type_dropdown.value == 'both (filled + lines)':
                self._redraw_contour()

        def on_line_color_change(change):
            if contour_type_dropdown.value == 'both (filled + lines)':
                self._redraw_contour()

        contour_line_thickness.observe(on_line_thickness_change, 'value')
        contour_line_color.observe(on_line_color_change, 'value')

        controls.append(line_controls_box)

        controls.append(widgets.HTML('<hr>'))

        # Colorbar toggle
        controls.append(widgets.HTML('<b>Colorbar Display:</b>'))

        colorbar_toggle = widgets.Checkbox(
            value=True,
            description='Show Colorbar'
        )

        def on_colorbar_toggle(change):
            self._redraw_contour()

        colorbar_toggle.observe(on_colorbar_toggle, 'value')
        controls.append(colorbar_toggle)

        # Store reference for redrawing
        self.colorbar_toggle = colorbar_toggle

        controls.append(widgets.HTML('<hr>'))

        # Contour levels slider
        controls.append(widgets.HTML('<b>Contour Levels:</b>'))

        levels_slider = widgets.IntSlider(
            value=20,
            min=1, max=200, step=1,
            description='Levels:',
            style={'description_width': '120px'},
            readout_format='d'
        )

        # For better UX with large range, add a logarithmic option
        levels_log_toggle = widgets.Checkbox(
            value=False,
            description='Logarithmic scale (for high level counts)'
        )

        def on_levels_change(change):
            print(f"⚠️ Contour levels changed to: {change.new}")
            print("   Redrawing plot...")
            self._redraw_contour()

        levels_slider.observe(on_levels_change, 'value')

        def on_log_toggle(change):
            if change.new:
                # Switch to log scale
                levels_slider.min = 1
                levels_slider.max = 3  # 10^3 = 1000
                levels_slider.step = 0.1
                levels_slider.value = 1.3  # ~20 levels
                levels_slider.readout_format = '.1f'
                levels_slider.description = 'Log Levels:'
            else:
                # Switch back to linear
                levels_slider.min = 1
                levels_slider.max = 200
                levels_slider.step = 1
                levels_slider.value = 20
                levels_slider.readout_format = 'd'
                levels_slider.description = 'Levels:'
            # Redraw plot with new scale
            print("⚠️ Level scale changed")
            print("   Redrawing plot...")
            self._redraw_contour()

        levels_log_toggle.observe(on_log_toggle, 'value')

        controls.extend([levels_slider, levels_log_toggle])

        # Store references for redrawing
        self.levels_slider = levels_slider
        self.levels_log_toggle = levels_log_toggle

        controls.append(widgets.HTML('<hr>'))

        # Colormap selector
        controls.append(widgets.HTML('<b>Colormap:</b>'))

        # Organize colormaps by category using display names
        cmap_dropdown = widgets.Dropdown(
            options=[
                ('─── Sequential ───', 'viridis'),  # Separator
                ('Greys', 'Greys'),
                ('Greys (reversed)', 'Greys_r'),
                ('Reds', 'Reds'),
                ('Reds (reversed)', 'Reds_r'),
                ('Blues', 'Blues'),
                ('Blues (reversed)', 'Blues_r'),
                ('Greens', 'Greens'),
                ('Greens (reversed)', 'Greens_r'),
                ('Viridis', 'viridis'),
                ('Plasma', 'plasma'),
                ('Inferno', 'inferno'),
                ('Magma', 'magma'),
                ('Cividis', 'cividis'),
                ('─── Diverging ───', 'RdBu_r'),  # Separator
                ('Red-Blue', 'RdBu'),
                ('Red-Blue (reversed)', 'RdBu_r'),
                ('Red-Yellow-Blue', 'RdYlBu'),
                ('Cool-Warm', 'coolwarm'),
                ('Seismic', 'seismic'),
                ('─── Miscellaneous ───', 'jet'),  # Separator
                ('Jet', 'jet'),
                ('Rainbow', 'rainbow'),
                ('Turbo', 'turbo'),
                ('Twilight', 'twilight')
            ],
            value='RdBu_r',
            description='Colormap:',
            style={'description_width': '120px'}
        )

        def on_cmap_change(change):
            print(f"⚠️ Colormap changed to: {change.new}")
            print("   Redrawing plot...")
            self._redraw_contour()

        cmap_dropdown.observe(on_cmap_change, 'value')

        controls.append(cmap_dropdown)

        # Store reference for redrawing
        self.cmap_dropdown = cmap_dropdown

        # Contour alpha
        alpha_slider = widgets.FloatSlider(
            value=1.0,
            min=0, max=1, step=0.05,
            description='Transparency:',
            style={'description_width': '120px'},
            readout_format='.2f'
        )

        def on_alpha_change(change):
            print(f"⚠️ Transparency changed to: {change.new:.2f}")
            print("   Redrawing plot...")
            self._redraw_contour()

        alpha_slider.observe(on_alpha_change, 'value')

        controls.append(alpha_slider)

        # Store reference for redrawing
        self.contour_alpha_slider = alpha_slider
        controls.append(widgets.HTML('<hr>'))

        # Colorbar controls
        if self.colorbar:
            controls.append(widgets.HTML('<b>Colorbar:</b>'))

            colorbar_label = widgets.Text(
                value=self.colorbar.ax.get_ylabel(),
                description='Label:',
                style={'description_width': '120px'}
            )

            def on_colorbar_label_change(change):
                self.colorbar.set_label(self._apply_mathtext(change.new))
                self.refresh()

            colorbar_label.observe(on_colorbar_label_change, 'value')

            colorbar_fontsize = widgets.IntSlider(
                value=10,
                min=6, max=40, step=1,
                description='Label Size:',
                style={'description_width': '120px'}
            )
            # Initialize stored value
            self.colorbar_label_fontsize = 10

            def on_colorbar_fontsize_change(change):
                self.colorbar_label_fontsize = change.new  # Store for later use
                if self.colorbar:
                    self.colorbar.set_label(self._apply_mathtext(self.colorbar.ax.get_ylabel()), fontsize=change.new)
                    self.refresh()

            colorbar_fontsize.observe(on_colorbar_fontsize_change, 'value')
            # Store reference to widget
            self.colorbar_fontsize_widget = colorbar_fontsize

            colorbar_tick_fontsize = widgets.IntSlider(
                value=9,
                min=6, max=40, step=1,
                description='Tick Size:',
                style={'description_width': '120px'}
            )
            # Initialize stored value
            self.colorbar_tick_fontsize = 9

            def on_colorbar_tick_fontsize_change(change):
                self.colorbar_tick_fontsize = change.new  # Store for later use
                if self.colorbar:
                    self.colorbar.ax.tick_params(labelsize=change.new)
                    self.refresh()

            colorbar_tick_fontsize.observe(on_colorbar_tick_fontsize_change, 'value')
            # Store reference to widget
            self.colorbar_tick_fontsize_widget = colorbar_tick_fontsize

            # NEW: Colorbar label font weight/style for contour plots
            colorbar_fontweight = widgets.Dropdown(
                options=['normal', 'bold', 'light', 'heavy'],
                value='normal',
                description='Label Weight:',
                style={'description_width': '120px'}
            )

            def on_colorbar_fontweight_change(change):
                if self.colorbar:
                    # Re-apply label with new weight
                    current_label = self.colorbar.ax.get_ylabel()
                    current_fontsize = self.colorbar_label_fontsize
                    self.colorbar.set_label(self._apply_mathtext(current_label), fontsize=current_fontsize, weight=change.new)
                    self.refresh()

            colorbar_fontweight.observe(on_colorbar_fontweight_change, 'value')

            # Colorbar position controls
            colorbar_gap = widgets.FloatSlider(
                value=getattr(self, 'colorbar_gap', 0.02),
                min=0.00, max=0.15, step=0.005,
                description='Gap:',
                style={'description_width': '120px'},
                readout_format='.3f',
                tooltip='Horizontal gap between plot and colorbar'
            )

            def on_colorbar_gap_change(change):
                self.colorbar_gap = change.new
                self._redraw_contour()

            colorbar_gap.observe(on_colorbar_gap_change, 'value')

            colorbar_width = widgets.FloatSlider(
                value=getattr(self, 'colorbar_width', 0.02),
                min=0.01, max=0.05, step=0.005,
                description='Width:',
                style={'description_width': '120px'},
                readout_format='.3f',
                tooltip='Width of the colorbar'
            )

            def on_colorbar_width_change(change):
                self.colorbar_width = change.new
                self._redraw_contour()

            colorbar_width.observe(on_colorbar_width_change, 'value')

            controls.extend([colorbar_label, colorbar_fontsize, colorbar_tick_fontsize,
                           colorbar_fontweight, colorbar_gap, colorbar_width])

        return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

    def _build_vector_controls(self, subplot_idx=None):
        """Build controls specific to vector field plots."""
        controls = []

        if subplot_idx is None:
            style = self.vector_style
            color_field_selected = self.vector_color_field.value not in (None, 'None')
        else:
            while len(self.subplot_vector_styles) <= subplot_idx:
                self.subplot_vector_styles.append(self.vector_style.copy())
            style = self.subplot_vector_styles[subplot_idx]
            config = self.subplot_configs[subplot_idx]
            color_field_selected = config['widgets']['vector_color_field'].value not in (None, 'None')

        scale_slider = widgets.FloatSlider(
            value=style.get('scale', 1.0),
            min=0.1,
            max=10.0,
            step=0.1,
            description='Scale:',
            style={'description_width': '100px'},
            readout_format='.1f'
        )

        def on_scale_change(change):
            style['scale'] = change.new
            self._redraw_vector(subplot_idx)

        scale_slider.observe(on_scale_change, 'value')

        width_slider = widgets.FloatSlider(
            value=style.get('width', 0.0025),
            min=0.0005,
            max=0.02,
            step=0.0005,
            description='Arrow Width:',
            style={'description_width': '120px'},
            readout_format='.4f'
        )

        def on_width_change(change):
            style['width'] = change.new
            self._redraw_vector(subplot_idx)

        width_slider.observe(on_width_change, 'value')

        alpha_slider = widgets.FloatSlider(
            value=float(style.get('alpha', 0.8)),
            min=0.1,
            max=1.0,
            step=0.05,
            description='Arrow Alpha:',
            style={'description_width': '120px'},
            readout_format='.2f'
        )

        def on_alpha_change(change):
            style['alpha'] = change.new
            self._redraw_vector(subplot_idx)

        alpha_slider.observe(on_alpha_change, 'value')

        pivot_dropdown = widgets.Dropdown(
            options=[('Tail', 'tail'), ('Middle', 'middle'), ('Tip', 'tip')],
            value=style.get('pivot', 'middle'),
            description='Pivot:',
            style={'description_width': '100px'}
        )

        def on_pivot_change(change):
            style['pivot'] = change.new
            self._redraw_vector(subplot_idx)

        pivot_dropdown.observe(on_pivot_change, 'value')

        # Uniform arrow color (used when no color field)
        arrow_color_dropdown = widgets.Dropdown(
            options=['black', 'blue', 'red', 'green', 'gray', 'darkblue', 'darkred', 'darkgreen'],
            value=style.get('arrow_color', 'black'),
            description='Arrow Color:',
            style={'description_width': '110px'},
            disabled=color_field_selected  # Only enabled when no color field
        )

        def on_arrow_color_change(change):
            style['arrow_color'] = change.new
            self._redraw_vector(subplot_idx)

        arrow_color_dropdown.observe(on_arrow_color_change, 'value')

        # Arrow decimation (plot every Nth arrow)
        decimation_slider = widgets.IntSlider(
            value=int(style.get('decimation', 1)),
            min=1,
            max=20,
            step=1,
            description='Every N Arrow:',
            style={'description_width': '120px'},
            readout_format='d'
        )

        def on_decimation_change(change):
            style['decimation'] = change.new
            self._redraw_vector(subplot_idx)

        decimation_slider.observe(on_decimation_change, 'value')

        cmap_dropdown = widgets.Dropdown(
            options=CMAP_OPTIONS,
            value=style.get('cmap', 'viridis'),
            description='Colormap:',
            style={'description_width': '110px'},
            disabled=not color_field_selected  # Only enabled when color field selected
        )

        def on_cmap_change(change):
            style['cmap'] = change.new
            self._redraw_vector(subplot_idx)

        cmap_dropdown.observe(on_cmap_change, 'value')

        controls.append(widgets.HTML('<b>Arrow Styling:</b>'))
        controls.extend([scale_slider, width_slider, alpha_slider, pivot_dropdown,
                        arrow_color_dropdown, decimation_slider, cmap_dropdown])

        # Quiver colorbar controls
        controls.append(widgets.HTML('<hr><b>Quiver Colorbar:</b>'))
        if not color_field_selected:
            controls.append(widgets.HTML('<em>Select a color field to enable these controls.</em>'))

        colorbar_toggle = widgets.Checkbox(
            value=style.get('colorbar', True),
            description='Show Colorbar',
            disabled=not color_field_selected
        )

        def on_colorbar_toggle(change):
            style['colorbar'] = change.new
            self._redraw_vector(subplot_idx)

        colorbar_toggle.observe(on_colorbar_toggle, 'value')

        colorbar_label = widgets.Text(
            value=style.get('colorbar_label') or '',
            description='Label:',
            style={'description_width': '100px'},
            disabled=not color_field_selected
        )
        cb_hint = widgets.HTML('<p style="margin:4px 0;color:#666;font-size:0.9em;">MathText ON: type \\nu, \\alpha, etc. No $...$ required.</p>')

        def on_colorbar_label(change):
            style['colorbar_label'] = change.new or None
            self._redraw_vector(subplot_idx)

        colorbar_label.observe(on_colorbar_label, 'value')

        colorbar_label_size = widgets.IntSlider(
            value=int(style.get('colorbar_label_fontsize', 10)),
            min=6,
            max=24,
            step=1,
            description='Label Size:',
            style={'description_width': '110px'},
            disabled=not color_field_selected
        )

        def on_colorbar_label_size(change):
            style['colorbar_label_fontsize'] = change.new
            self._redraw_vector(subplot_idx)

        colorbar_label_size.observe(on_colorbar_label_size, 'value')

        colorbar_tick_size = widgets.IntSlider(
            value=int(style.get('colorbar_tick_fontsize', 9)),
            min=6,
            max=24,
            step=1,
            description='Tick Size:',
            style={'description_width': '110px'},
            disabled=not color_field_selected
        )

        def on_colorbar_tick_size(change):
            style['colorbar_tick_fontsize'] = change.new
            self._redraw_vector(subplot_idx)

        colorbar_tick_size.observe(on_colorbar_tick_size, 'value')

        colorbar_fontweight = widgets.Dropdown(
            options=['normal', 'bold', 'light', 'heavy'],
            value=style.get('colorbar_fontweight', 'normal'),
            description='Label Weight:',
            style={'description_width': '120px'},
            disabled=not color_field_selected
        )

        def on_cb_fontweight(change):
            style['colorbar_fontweight'] = change.new
            self._redraw_vector(subplot_idx)

        colorbar_fontweight.observe(on_cb_fontweight, 'value')

        cb_gap_slider = widgets.FloatSlider(
            value=float(style.get('colorbar_gap', 0.02)),
            min=0.0,
            max=0.15,
            step=0.005,
            description='CB Gap:',
            style={'description_width': '110px'},
            readout_format='.3f',
            disabled=not color_field_selected
        )

        def on_cb_gap(change):
            style['colorbar_gap'] = change.new
            self._redraw_vector(subplot_idx)

        cb_gap_slider.observe(on_cb_gap, 'value')

        cb_width_slider = widgets.FloatSlider(
            value=float(style.get('colorbar_width', 0.02)),
            min=0.005,
            max=0.08,
            step=0.002,
            description='CB Width:',
            style={'description_width': '110px'},
            readout_format='.3f',
            disabled=not color_field_selected
        )

        def on_cb_width(change):
            style['colorbar_width'] = change.new
            self._redraw_vector(subplot_idx)

        cb_width_slider.observe(on_cb_width, 'value')

        controls.extend([colorbar_toggle, colorbar_label, cb_hint, colorbar_label_size, colorbar_tick_size, colorbar_fontweight, cb_gap_slider, cb_width_slider])

        return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

    def _build_vector_field_controls(self, subplot_idx=None):
        """Build contour-style controls for vector overlay fields."""
        controls = []

        if subplot_idx is None:
            style = self.vector_style
            overlay_field_name = self.vector_overlay_field.value
        else:
            while len(self.subplot_vector_styles) <= subplot_idx:
                self.subplot_vector_styles.append(self.vector_style.copy())
            style = self.subplot_vector_styles[subplot_idx]
            config = self.subplot_configs[subplot_idx]
            overlay_field_name = config['widgets']['vector_overlay_field'].value

        overlay_available = overlay_field_name not in (None, 'None')

        type_dropdown = widgets.Dropdown(
            options=['contourf (filled)', 'contour (lines)', 'both (filled + lines)'],
            value=style.get('overlay_type', 'contourf (filled)'),
            description='Type:',
            style={'description_width': '120px'},
            disabled=not overlay_available
        )

        # Line controls for "both" mode (initially hidden)
        line_thickness = widgets.FloatSlider(
            value=float(style.get('overlay_line_thickness', 0.5)),
            min=0.1,
            max=5.0,
            step=0.1,
            description='Line Width:',
            style={'description_width': '90px'},
            readout_format='.1f',
            disabled=not overlay_available,
            layout=widgets.Layout(width='48%')
        )

        line_color = widgets.Dropdown(
            options=['black', 'white', 'red', 'blue', 'green', 'cyan', 'magenta', 'yellow'],
            value=style.get('overlay_line_color', 'black'),
            description='Line Color:',
            style={'description_width': '90px'},
            disabled=not overlay_available,
            layout=widgets.Layout(width='48%')
        )

        # Container for line controls (side by side)
        line_controls_box = widgets.HBox(
            [line_thickness, line_color],
            layout=widgets.Layout(display='none', justify_content='space-between')
        )

        def on_line_thickness_change(change):
            style['overlay_line_thickness'] = change.new
            self._redraw_vector(subplot_idx)

        def on_line_color_change(change):
            style['overlay_line_color'] = change.new
            self._redraw_vector(subplot_idx)

        line_thickness.observe(on_line_thickness_change, 'value')
        line_color.observe(on_line_color_change, 'value')

        def on_type_change(change):
            # Show/hide line controls based on type
            is_both = change.new == 'both (filled + lines)'
            line_controls_box.layout.display = 'flex' if is_both else 'none'
            style['overlay_type'] = change.new
            self._redraw_vector(subplot_idx)

        type_dropdown.observe(on_type_change, 'value')

        # Show line controls if current type is "both"
        if style.get('overlay_type', 'contourf (filled)') == 'both (filled + lines)':
            line_controls_box.layout.display = 'flex'

        colorbar_toggle = widgets.Checkbox(
            value=style.get('overlay_show_colorbar', True),
            description='Show Colorbar',
            disabled=not overlay_available
        )

        def on_overlay_cb(change):
            style['overlay_show_colorbar'] = change.new
            self._redraw_vector(subplot_idx)

        colorbar_toggle.observe(on_overlay_cb, 'value')

        levels_slider = widgets.IntSlider(
            value=int(style.get('overlay_levels', 20)),
            min=5,
            max=200,
            step=1,
            description='Levels:',
            style={'description_width': '120px'},
            disabled=not overlay_available
        )

        def on_levels_change(change):
            style['overlay_levels'] = change.new
            self._redraw_vector(subplot_idx)

        levels_slider.observe(on_levels_change, 'value')

        log_toggle = widgets.Checkbox(
            value=style.get('overlay_log', False),
            description='Logarithmic scale (for high levels)',
            disabled=not overlay_available
        )

        def on_log_change(change):
            style['overlay_log'] = change.new
            self._redraw_vector(subplot_idx)

        log_toggle.observe(on_log_change, 'value')

        cmap_dropdown = widgets.Dropdown(
            options=CMAP_OPTIONS,
            value=style.get('overlay_cmap', 'plasma'),
            description='Colormap:',
            style={'description_width': '120px'},
            disabled=not overlay_available
        )

        def on_cmap_change(change):
            style['overlay_cmap'] = change.new
            self._redraw_vector(subplot_idx)

        cmap_dropdown.observe(on_cmap_change, 'value')

        alpha_slider = widgets.FloatSlider(
            value=float(style.get('overlay_alpha', 1.0)),
            min=0.05,
            max=1.0,
            step=0.05,
            description='Transparency:',
            style={'description_width': '120px'},
            readout_format='.2f',
            disabled=not overlay_available
        )

        def on_alpha_change(change):
            style['overlay_alpha'] = change.new
            self._redraw_vector(subplot_idx)

        alpha_slider.observe(on_alpha_change, 'value')

        label_text = widgets.Text(
            value=style.get('overlay_label') or '',
            description='Colorbar Label:',
            style={'description_width': '140px'},
            disabled=not overlay_available
        )
        overlay_hint = widgets.HTML('<p style="margin:4px 0;color:#666;font-size:0.9em;">MathText ON: type \\alpha, \\omega, etc. No $...$ required.</p>')

        def on_label_change(change):
            style['overlay_label'] = change.new or None
            self._redraw_vector(subplot_idx)

        label_text.observe(on_label_change, 'value')

        label_size_slider = widgets.IntSlider(
            value=int(style.get('overlay_label_fontsize', 10)),
            min=6,
            max=30,
            step=1,
            description='Label Size:',
            style={'description_width': '120px'},
            disabled=not overlay_available
        )

        def on_label_size(change):
            style['overlay_label_fontsize'] = change.new
            self._redraw_vector(subplot_idx)

        label_size_slider.observe(on_label_size, 'value')

        tick_size_slider = widgets.IntSlider(
            value=int(style.get('overlay_tick_fontsize', 9)),
            min=6,
            max=30,
            step=1,
            description='Tick Size:',
            style={'description_width': '120px'},
            disabled=not overlay_available
        )

        def on_tick_size(change):
            style['overlay_tick_fontsize'] = change.new
            self._redraw_vector(subplot_idx)

        tick_size_slider.observe(on_tick_size, 'value')

        fontweight_dropdown = widgets.Dropdown(
            options=['normal', 'bold', 'light', 'heavy'],
            value=style.get('overlay_fontweight', 'normal'),
            description='Label Weight:',
            style={'description_width': '120px'},
            disabled=not overlay_available
        )

        def on_fw_change(change):
            style['overlay_fontweight'] = change.new
            self._redraw_vector(subplot_idx)

        fontweight_dropdown.observe(on_fw_change, 'value')

        gap_slider = widgets.FloatSlider(
            value=float(style.get('overlay_colorbar_gap', 0.02)),
            min=0.0,
            max=0.15,
            step=0.005,
            description='CB Gap:',
            style={'description_width': '120px'},
            readout_format='.3f',
            disabled=not overlay_available
        )

        def on_gap_change(change):
            style['overlay_colorbar_gap'] = change.new
            self._redraw_vector(subplot_idx)

        gap_slider.observe(on_gap_change, 'value')

        width_slider = widgets.FloatSlider(
            value=float(style.get('overlay_colorbar_width', 0.02)),
            min=0.005,
            max=0.08,
            step=0.002,
            description='CB Width:',
            style={'description_width': '120px'},
            readout_format='.3f',
            disabled=not overlay_available
        )

        def on_width_change(change):
            style['overlay_colorbar_width'] = change.new
            self._redraw_vector(subplot_idx)

        width_slider.observe(on_width_change, 'value')

        if not overlay_available:
            controls.append(widgets.HTML('<b>Select an overlay field to enable contour controls.</b>'))

        controls.append(widgets.HTML('<b>Contour Type:</b>'))
        controls.append(type_dropdown)
        controls.append(line_controls_box)

        controls.append(widgets.HTML('<hr><b>Colorbar Display:</b>'))
        controls.append(colorbar_toggle)

        controls.append(widgets.HTML('<hr><b>Contour Levels:</b>'))
        controls.extend([levels_slider, log_toggle])

        controls.append(widgets.HTML('<hr><b>Colormap:</b>'))
        controls.extend([cmap_dropdown, alpha_slider])

        controls.append(widgets.HTML('<hr><b>Colorbar:</b>'))
        controls.extend([label_text, overlay_hint, label_size_slider, tick_size_slider, fontweight_dropdown, gap_slider, width_slider])

        return widgets.VBox(controls, layout=widgets.Layout(padding='10px'))

    def _get_closest_color(self, color):
        """Get the closest named color from a matplotlib color"""
        color_map = {
            'b': 'blue', 'g': 'green', 'r': 'red', 'c': 'cyan',
            'm': 'magenta', 'y': 'yellow', 'k': 'black', 'w': 'white'
        }

        if isinstance(color, str):
            if color in color_map:
                return color_map[color]
            elif color in ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow']:
                return color

        return 'blue'  # Default

    def _reposition_all_colorbars(self):
        """Reposition all colorbar axes after layout changes (e.g., subplots_adjust)"""
        if not (self.subplot_mode and len(self.axes_list) > 1):
            return  # Only needed in subplot mode

        # Ensure axes positions are up-to-date before measuring/repositioning
        try:
            self.fig.canvas.draw()
        except Exception:
            pass

        # Reposition each colorbar to match its parent axis's new position
        for i, cbar_ax in enumerate(self.colorbar_axes):
            if cbar_ax is not None and i < len(self.axes_list):
                parent_ax = self.axes_list[i]
                parent_pos = parent_ax.get_position()

                # Get current colorbar position to preserve user's gap/width settings
                current_cbar_pos = cbar_ax.get_position()

                # Calculate current gap and width (preserve user's settings if they changed them)
                # If colorbar was moved by user, this preserves their custom gap/width
                current_gap = current_cbar_pos.x0 - parent_pos.x1
                current_width = current_cbar_pos.width

                # Ensure reasonable defaults if calculation gives weird values
                if current_gap < 0 or current_gap > 0.2:
                    current_gap = 0.01
                if current_width < 0.005 or current_width > 0.1:
                    current_width = 0.01

                # Apply new position with preserved gap/width
                cbar_ax.set_position([
                    parent_pos.x1 + current_gap,  # Preserve gap
                    parent_pos.y0,                 # Match parent y
                    current_width,                 # Preserve width
                    parent_pos.height              # Match parent height
                ])

    def refresh(self):
        """Refresh the figure display"""
        if not self.fig:
            return

        try:
            # Force canvas draw to apply all pending property changes
            self.fig.canvas.draw()

            # Force flush of any pending events (ensures properties are applied)
            self.fig.canvas.flush_events()

            # Redisplay in output widget (single image to avoid duplication)
            self._render_current_figure()
        except Exception as e:
            # Attempt recovery by sanitizing any mathtext labels to plain text
            try:
                self._sanitize_all_mathtext()
                # Retry draw after sanitization
                try:
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                except Exception:
                    pass
                # Attempt to re-apply stored labels (axes + legends) and redraw once
                try:
                    self._reapply_all_stored_labels_on_figure()
                    try:
                        self.fig.canvas.draw()
                        self.fig.canvas.flush_events()
                    except Exception:
                        pass
                except Exception:
                    pass
                # Attempt to display again
                try:
                    self._render_current_figure()
                except Exception:
                    pass
            except Exception:
                # As a last resort, swallow the error to avoid UI collapse
                pass

    def _render_current_figure(self):
        """Render the current figure as a PNG into the dedicated image widget.

        This avoids any duplicate static rendering that can occur with direct display(self.fig)
        in some notebook front-ends.
        """
        if not self.fig:
            return
        import io
        buf = io.BytesIO()
        try:
            self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.fig.dpi)
            buf.seek(0)
            if hasattr(self, 'fig_image_widget') and self.fig_image_widget is not None:
                self.fig_image_widget.value = buf.getvalue()
            else:
                with self.fig_output:
                    clear_output(wait=True)
                    display(Image(data=buf.getvalue()))
        except Exception:
            with self.fig_output:
                clear_output(wait=True)
                display(self.fig)

    def _sanitize_all_mathtext(self):
        """Strip mathtext delimiters from all labels/titles and colorbar labels to prevent crashes.

        This is used as a recovery path when drawing fails due to invalid math expressions.
        """
        if not self.fig:
            return
        def _strip(s):
            try:
                return s.replace('$', '') if isinstance(s, str) else s
            except Exception:
                return s
        try:
            for ax in list(self.fig.axes):
                try:
                    ax.set_xlabel(_strip(ax.get_xlabel()))
                except Exception:
                    pass
                try:
                    ax.set_ylabel(_strip(ax.get_ylabel()))
                except Exception:
                    pass
                try:
                    ax.set_title(_strip(ax.get_title()))
                except Exception:
                    pass
                # Sanitize legend texts and artist labels
                try:
                    # Update artist labels (lines and scatters)
                    for ln in ax.get_lines():
                        try:
                            ln.set_label(_strip(ln.get_label()))
                        except Exception:
                            pass
                    for coll in ax.collections:
                        # Only those used as scatter (have get_offsets)
                        if hasattr(coll, 'get_offsets'):
                            try:
                                lbl = getattr(coll, 'get_label', lambda: '')()
                                if lbl:
                                    coll.set_label(_strip(lbl))
                            except Exception:
                                pass
                    # Update existing legend text objects
                    leg = ax.get_legend()
                    if leg is not None:
                        for txt in leg.get_texts():
                            try:
                                txt.set_text(_strip(txt.get_text()))
                            except Exception:
                                pass
                        try:
                            ax.legend()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        # Sanitize known colorbars
        try:
            if getattr(self, 'colorbar', None):
                try:
                    lbl = self.colorbar.ax.get_ylabel()
                    self.colorbar.set_label(_strip(lbl))
                except Exception:
                    pass
            for cbar in getattr(self, 'colorbars', []) or []:
                if cbar is not None:
                    try:
                        lbl = cbar.ax.get_ylabel()
                        cbar.set_label(_strip(lbl))
                    except Exception:
                        pass
        except Exception:
            pass

    def _save_figure(self, btn=None):
        """Save the figure to file"""
        if not self.fig:
            print("❌ No figure to save")
            return

        filename = self.save_filename.value.strip()
        if not filename:
            filename = 'figure'

        # Add extension
        full_filename = filename + self.save_format.value

        try:
            # Save with high quality
            self.fig.savefig(full_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to: {full_filename}")
        except Exception as e:
            print(f"❌ Error saving figure: {e}")

    def _copy_code(self, btn=None):
        """Generate and display Python code to reproduce the current figure"""
        if not self.fig or not self.loader:
            print("❌ No figure to generate code for")
            return

        code = self._generate_code()

        # Create a textarea widget for easy copying
        code_textarea = widgets.Textarea(
            value=code,
            description='',
            layout=widgets.Layout(width='100%', height='400px'),
            style={'description_width': '0px'}
        )

        # Create copy instruction
        copy_instruction = widgets.HTML(
            '<p style="color: green; font-weight: bold;">✓ Code generated! Select all (Ctrl/Cmd+A) and copy (Ctrl/Cmd+C):</p>'
        )

        # Display in a nice box
        code_box = widgets.VBox([
            widgets.HTML('<h4>📋 Reproducible Python Code:</h4>'),
            copy_instruction,
            code_textarea,
            widgets.HTML('<p style="color: #666; font-style: italic;">Tip: This code can be run standalone to reproduce your figure exactly.</p>')
        ], layout=widgets.Layout(
            border='2px solid #3498db',
            padding='15px',
            margin='10px 0px'
        ))

        display(code_box)

        # Also try to copy to clipboard using JavaScript (works in most browsers)
        try:
            from IPython.display import Javascript
            js_code = f"""
            navigator.clipboard.writeText(`{code}`).then(function() {{
                console.log('Code copied to clipboard!');
            }}, function() {{
                console.log('Clipboard copy failed, please use Ctrl+C');
            }});
            """
            display(Javascript(js_code))
            print("✓ Code also copied to clipboard (if browser allows)!")
        except:
            print("Note: Please manually select and copy the code from the box above.")

    def _generate_code(self):
        """Generate Python code that reproduces the current figure"""
        code_lines = []

        # Imports
        code_lines.append("# Generated code to reproduce FigSmith figure")
        code_lines.append("import numpy as np")
        code_lines.append("import matplotlib.pyplot as plt")
        if self.plot_type in ['contour', 'tricontour']:
            code_lines.append("from matplotlib import tri")
        code_lines.append("")

        # Load data
        code_lines.append("# Load data")
        code_lines.append(f"data = np.loadtxt('{self.file_input.value}', skiprows=1)")
        code_lines.append("")

        # Extract columns
        code_lines.append("# Extract columns")
        col_idx = {col: i for i, col in enumerate(self.loader.columns)}
        x_idx = col_idx.get(self.x_column.value, 0)
        code_lines.append(f"x = data[:, {x_idx}]")

        if self.plot_type in ['line', 'scatter']:
            for y_col in self._y_columns_selected:
                y_idx = col_idx.get(y_col, 1)
                code_lines.append(f"{y_col.replace(' ', '_')} = data[:, {y_idx}]")
        elif self.plot_type in ['contour', 'tricontour']:
            y_idx = col_idx.get(self.y_column_contour.value, 1)
            field_idx = col_idx.get(self.field_column.value, 2)
            code_lines.append(f"y = data[:, {y_idx}]")
            code_lines.append(f"field = data[:, {field_idx}]")
        code_lines.append("")

        # Create figure
        code_lines.append("# Create figure")
        figsize = self.fig.get_size_inches()
        code_lines.append(f"fig, ax = plt.subplots(figsize=({figsize[0]:.1f}, {figsize[1]:.1f}))")
        code_lines.append("")

        # Plot data
        code_lines.append("# Plot data")
        if self.plot_type == 'line':
            for y_col in self._y_columns_selected:
                line_label = y_col.replace(' ', '_')
                code_lines.append(f"ax.plot(x, {line_label}, label='{y_col}')")
        elif self.plot_type == 'scatter':
            for y_col in self._y_columns_selected:
                line_label = y_col.replace(' ', '_')
                code_lines.append(f"ax.scatter(x, {line_label}, label='{y_col}', alpha=0.6, s=50)")
        elif self.plot_type == 'contour':
            code_lines.append(f"# Reshape for regular grid (assuming structured data)")
            code_lines.append(f"# You may need to adjust these reshape dimensions")
            code_lines.append(f"cs = ax.contourf(x.reshape(-1, 1), y.reshape(-1, 1), field.reshape(-1, 1), levels=20, cmap='RdBu_r')")
            code_lines.append(f"fig.colorbar(cs, ax=ax, label='{self.field_column.value}')")

        code_lines.append("")

        # Axes properties
        code_lines.append("# Customize axes")
        code_lines.append(f"ax.set_xlabel('{self.ax.get_xlabel()}')")
        code_lines.append(f"ax.set_ylabel('{self.ax.get_ylabel()}')")
        code_lines.append(f"ax.set_title('{self.ax.get_title()}')")

        # Grid
        if self.ax.xaxis.get_gridlines():
            grid_visible = any(line.get_visible() for line in self.ax.xaxis.get_gridlines())
            if grid_visible:
                code_lines.append("ax.grid(True, alpha=0.3)")

        # Legend
        legend = self.ax.get_legend()
        if legend:
            code_lines.append("ax.legend()")

        code_lines.append("")

        # Save
        code_lines.append("# Save figure")
        code_lines.append("fig.savefig('figure.png', dpi=300, bbox_inches='tight')")
        code_lines.append("plt.show()")

        return "\n".join(code_lines)

    def display(self):
        """Display the interactive plotter"""
        display(self.layout)


# ===== In-memory ingestion helpers =====
def _is_series_like(obj):
    return isinstance(obj, (pd.Series, pd.Index))


def _is_all_strings_sequence(value, expected_len=None):
    if isinstance(value, str):
        return expected_len in (None, 1)
    if isinstance(value, Sequence) and not isinstance(value, (np.ndarray, pd.Series)):
        if expected_len is not None and len(value) != expected_len:
            return False
        return all(isinstance(item, str) for item in value)
    return False


def _value_has_array_payload(value):
    if value is None:
        return False
    if isinstance(value, str):
        return False
    if isinstance(value, np.ndarray):
        return True
    if _is_series_like(value):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        # Empty lists don't carry useful data
        if len(value) == 0:
            return False
        # Avoid treating pure lists of column names as data
        if all(isinstance(item, str) for item in value):
            return False
        return True
    return False


def _ensure_1d_vector(name, values):
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    # For higher dimensions, expect a single column that can be squeezed
    reshaped = arr.reshape(arr.shape[0], -1)
    if reshaped.shape[1] != 1:
        raise ValueError(f"{name} must be 1D")
    return reshaped[:, 0]


def _flatten_array(name, values):
    arr = np.asarray(values)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    return arr.reshape(-1)


def _validate_equal_length(arrays):
    lengths = {np.asarray(arr).shape[0] if np.asarray(arr).ndim == 1 else np.asarray(arr).size for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("All provided arrays must share the same length")


def _coerce_general_loader(data):
    if data is None:
        raise ValueError("data cannot be None")
    if isinstance(data, pd.DataFrame):
        return DataLoader.from_dataframe(data)
    if isinstance(data, dict):
        return DataLoader.from_dict(data)
    if isinstance(data, np.ndarray):
        return DataLoader.from_numpy(data)
    if _is_series_like(data):
        return DataLoader.from_numpy(np.asarray(data))
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) > 0 and all(hasattr(item, '__len__') and not isinstance(item, (str, bytes)) for item in data):
            arrays = [_ensure_1d_vector(f"data[{idx}]", item) for idx, item in enumerate(data)]
            _validate_equal_length(arrays)
            stacked = np.column_stack(arrays)
            columns = [f"col_{i}" for i in range(len(arrays))]
            return DataLoader.from_numpy(stacked, columns=columns)
        return DataLoader.from_numpy(np.asarray(data))
    return DataLoader.from_numpy(np.asarray(data))


def _normalize_y_arrays(y):
    if y is None:
        raise ValueError("`y` data is required when supplying in-memory arrays")
    if _is_all_strings_sequence(y):
        raise ValueError("When providing column names for `y`, please also supply `data=`.")
    if isinstance(y, (np.ndarray,)) or _is_series_like(y):
        return [np.asarray(y)]
    if isinstance(y, Sequence) and not isinstance(y, (str, bytes)):
        if len(y) > 0 and all(_value_has_array_payload(item) for item in y):
            return [_ensure_1d_vector(f'y[{idx}]', item) for idx, item in enumerate(y)]
    return [_ensure_1d_vector('y', y)]


def _coerce_line_loader(x, y):
    y_arrays = _normalize_y_arrays(y)
    _validate_equal_length(y_arrays)
    length = y_arrays[0].shape[0]
    if x is None:
        x_array = np.arange(length)
    else:
        if _is_all_strings_sequence(x):
            raise ValueError("When providing column names for `x`, also pass the dataset via `data=`.")
        x_array = _ensure_1d_vector('x', x)
        if x_array.shape[0] != length:
            raise ValueError("`x` and `y` arrays must have the same length")

    data = {'x': x_array}
    if len(y_arrays) == 1:
        data['y'] = y_arrays[0]
        y_names = ['y']
    else:
        y_names = []
        for idx, arr in enumerate(y_arrays):
            col_name = f'y{idx}'
            data[col_name] = arr
            y_names.append(col_name)

    loader = DataLoader.from_dataframe(pd.DataFrame(data))
    return loader, {'x': 'x', 'y': y_names}


def _canonical_field_key(key):
    if key is None:
        return None
    lower = str(key).lower()
    if lower == 'x':
        return 'x'
    if lower == 'y':
        return 'y'
    if lower in ('z', 'field'):
        return 'field'
    if lower == 'r':
        return 'r'
    if lower == 'theta':
        return 'theta'
    return None


def _auto_field_mapping(columns):
    lowered = {str(col).lower(): col for col in columns}
    x_col = lowered.get('x')
    y_col = lowered.get('y')
    field_col = lowered.get('z') or lowered.get('field')
    if field_col is None:
        for col in columns:
            if col not in {x_col, y_col}:
                field_col = col
                break
    if x_col and y_col and field_col:
        return {'x': x_col, 'y': y_col, 'field': field_col}
    return None


def _field_contains_array_payload(field):
    if field is None:
        return False
    if isinstance(field, pd.DataFrame):
        return True
    if isinstance(field, (list, tuple)) and len(field) == 3:
        return True
    if isinstance(field, dict):
        return any(_value_has_array_payload(v) for v in field.values())
    return False


def _build_field_loader_from_arrays(field):
    if isinstance(field, pd.DataFrame):
        loader = DataLoader.from_dataframe(field)
        mapping = _auto_field_mapping(loader.columns) or {}
        return loader, mapping

    if isinstance(field, (list, tuple)) and len(field) == 3:
        x_arr = _flatten_array('field[0]', field[0])
        y_arr = _flatten_array('field[1]', field[1])
        z_arr = _flatten_array('field[2]', field[2])
        _validate_equal_length([x_arr, y_arr, z_arr])
        df = pd.DataFrame({'x': x_arr, 'y': y_arr, 'field': z_arr})
        loader = DataLoader.from_dataframe(df)
        return loader, {'x': 'x', 'y': 'y', 'field': 'field'}

    if isinstance(field, dict):
        normalized = {}
        for key, value in field.items():
            canonical = _canonical_field_key(key)
            if canonical in {'x', 'y', 'field'}:
                normalized[canonical] = _flatten_array(f'field[{key}]', value)
        if {'x', 'y', 'field'} <= normalized.keys():
            _validate_equal_length([normalized['x'], normalized['y'], normalized['field']])
            df = pd.DataFrame({'x': normalized['x'], 'y': normalized['y'], 'field': normalized['field']})
            loader = DataLoader.from_dataframe(df)
            return loader, {'x': 'x', 'y': 'y', 'field': 'field'}

    raise ValueError("Field input must supply x, y, and field/z arrays of equal length")


def _resolve_field_hint(field, columns):
    if field is None:
        return None
    column_set = set(columns)
    mapping = {}
    if isinstance(field, dict):
        for key, value in field.items():
            canonical = _canonical_field_key(key)
            if canonical in {'x', 'y', 'field'} and isinstance(value, str) and value in column_set:
                mapping[canonical] = value
    elif _is_all_strings_sequence(field, expected_len=3):
        keys = ['x', 'y', 'field']
        mapping = {key: name for key, name in zip(keys, field) if name in column_set}
    if {'x', 'y', 'field'} <= mapping.keys():
        return mapping
    return None


def _canonical_vector_key(key):
    if key is None:
        return None
    lower = str(key).lower()
    if lower in {'x', 'y', 'u', 'v'}:
        return lower
    if lower in {'c', 'color', 'color_field'}:
        return 'c'
    if lower in {'overlay', 'overlay_field'}:
        return 'overlay'
    return None


def _auto_vector_mapping(columns):
    lowered = {str(col).lower(): col for col in columns}
    mapping = {}
    for key in ['x', 'y', 'u', 'v']:
        col = lowered.get(key)
        if col:
            mapping[key] = col
    if not {'x', 'y', 'u', 'v'} <= mapping.keys():
        return None
    if lowered.get('c'):
        mapping['c'] = lowered['c']
    if lowered.get('color'):
        mapping['c'] = lowered['color']
    if lowered.get('overlay'):
        mapping['overlay'] = lowered['overlay']
    return mapping


def _vector_contains_array_payload(vector):
    if vector is None:
        return False
    if isinstance(vector, pd.DataFrame):
        return True
    if isinstance(vector, dict):
        return any(_value_has_array_payload(v) for v in vector.values())
    return False


def _build_vector_loader_from_arrays(vector):
    if isinstance(vector, pd.DataFrame):
        loader = DataLoader.from_dataframe(vector)
        mapping = _auto_vector_mapping(loader.columns) or {}
        return loader, mapping

    if isinstance(vector, dict):
        normalized = {}
        for key, value in vector.items():
            canonical = _canonical_vector_key(key)
            if canonical:
                normalized[canonical] = _flatten_array(f'vector[{key}]', value)

        required = {'x', 'y', 'u', 'v'}
        if not required <= normalized.keys():
            raise ValueError("Vector input must provide x, y, u, and v arrays")
        _validate_equal_length([normalized[k] for k in required])

        data = {k: normalized[k] for k in ('x', 'y', 'u', 'v')}
        if 'c' in normalized:
            data['c'] = normalized['c']
        if 'overlay' in normalized:
            data['overlay'] = normalized['overlay']

        loader = DataLoader.from_dataframe(pd.DataFrame(data))
        selectors = {'x': 'x', 'y': 'y', 'u': 'u', 'v': 'v'}
        if 'c' in data:
            selectors['c'] = 'c'
        if 'overlay' in data:
            selectors['overlay'] = 'overlay'
        return loader, selectors

    raise ValueError("Vector input must be a dict or DataFrame containing x, y, u, v data")


def _resolve_vector_hint(vector, columns):
    if vector is None:
        return None
    column_set = set(columns)
    mapping = {}
    if isinstance(vector, dict):
        for key, value in vector.items():
            canonical = _canonical_vector_key(key)
            if canonical and isinstance(value, str) and value in column_set:
                mapping[canonical] = value
    elif isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)):
        ordered_keys = ['x', 'y', 'u', 'v', 'c', 'overlay']
        if len(vector) >= 4 and all(isinstance(item, str) for item in vector):
            mapping = {key: name for key, name in zip(ordered_keys, vector) if name in column_set}
    if {'x', 'y', 'u', 'v'} <= mapping.keys():
        return mapping
    return None


def _resolve_line_hints(x, y, columns):
    hints = {}
    column_set = set(columns)
    if isinstance(x, str) and x in column_set:
        hints['x'] = x
    if _is_all_strings_sequence(y):
        if isinstance(y, str):
            names = [y]
        else:
            names = list(dict.fromkeys(y))
        names = [name for name in names if name in column_set]
        if names:
            hints['y'] = names
    elif isinstance(y, str) and y in column_set:
        hints['y'] = [y]
    return hints


def _ingest_in_memory_data(*, data=None, x=None, y=None, field=None, vector=None):
    preselect = {}
    loader = None
    source_label = None
    base_source = None

    if data is not None:
        loader = _coerce_general_loader(data)
        base_source = 'data'
        source_label = f"{type(data).__name__} input"
    elif _value_has_array_payload(y):
        loader, line_pref = _coerce_line_loader(x, y)
        preselect.update(line_pref)
        base_source = 'line'
        source_label = "x/y arrays"
    elif _field_contains_array_payload(field):
        loader, field_pref = _build_field_loader_from_arrays(field)
        if field_pref:
            preselect['field'] = field_pref
        base_source = 'field'
        source_label = "field arrays"
    elif _vector_contains_array_payload(vector):
        loader, vector_pref = _build_vector_loader_from_arrays(vector)
        if vector_pref:
            preselect['vector'] = vector_pref
        base_source = 'vector'
        source_label = "vector arrays"
    else:
        raise ValueError("Please provide `data`, (`x`,`y`), `field`, or `vector` inputs for in-memory usage.")

    if loader is None:
        raise ValueError("Failed to ingest in-memory data")

    columns = list(loader.columns)
    if base_source == 'data':
        if _value_has_array_payload(x):
            raise ValueError("When `data` is provided, `x` must reference an existing column name.")
        if _value_has_array_payload(y):
            raise ValueError("When `data` is provided, `y` must reference existing column names.")

    line_hints = _resolve_line_hints(x, y, columns)
    if line_hints:
        preselect.update(line_hints)

    if 'field' not in preselect:
        field_hint = _resolve_field_hint(field, columns)
        if field_hint:
            preselect['field'] = field_hint
    if 'vector' not in preselect:
        vector_hint = _resolve_vector_hint(vector, columns)
        if vector_hint:
            preselect['vector'] = vector_hint

    return loader, preselect, source_label or "in-memory data"


# Convenience function
def load_and_plot(filepath=None, *, data=None, x=None, y=None, field=None, vector=None):
    """
    Create a unified interactive plotter for loading, plotting, and editing data from .dat files
    or in-memory columnar inputs.

    This combines:
    - Data loading and column selection
    - Initial plot creation with styling
    - COMPLETE per-line tunable parameters
    - COMPLETE global figure parameters
    - Field/contour specific controls
    - Save functionality with multiple formats

    Parameters
    ----------
    filepath : str, optional
        Path to .dat file to load initially.
    data : pandas.DataFrame, numpy.ndarray, dict, or sequence, optional
        In-memory tabular data to ingest instead of reading from disk.
    x : array-like or str, optional
        When arrays are provided (with y), supplies the x-axis values.
        When combined with `data`, acts as a column-name hint.
    y : array-like, sequence of arrays, or column name(s), optional
        In-memory y-series for line/scatter use cases, or column-name hints when used
        alongside `data`. If arrays are provided without `x`, FigSmith uses sample indices.
    field : dict, DataFrame, or tuple, optional
        Field/contour hints. Accepts dict/tuple of arrays (x, y, field) or column-name hints.
    vector : dict or DataFrame, optional
        Vector field hints. Accepts arrays for x, y, u, v (and optional color/overlay) or
        column-name hints referencing `data`.

    Notes
    -----
    If any in-memory arguments (`data`, `x`, `y`, `field`, `vector`) are provided,
    `filepath` is ignored and FigSmith stays in in-memory mode.

    Returns
    -------
    InteractivePlotter
        Interactive plotter instance

    Examples
    --------
    >>> import figsmith as ff
    >>> plotter = ff.load_and_plot('example_data/trig_functions.dat')
    >>> plotter = ff.load_and_plot(data=df)
    >>> plotter = ff.load_and_plot(x=x, y=[np.sin(x), np.cos(x)])
    >>> plotter = ff.load_and_plot(data=df, field={'x': 'x', 'y': 'y', 'z': 'temperature'})
    >>> plotter = ff.load_and_plot(vector={'x': x, 'y': y, 'u': u, 'v': v, 'c': speed})
    """
    plotter = InteractivePlotter()

    use_in_memory = any(arg is not None for arg in (data, x, y, field, vector))

    if use_in_memory:
        if filepath:
            print("ℹ️ Ignoring `filepath` because in-memory data was supplied.")
        loader, preselect, description = _ingest_in_memory_data(
            data=data,
            x=x,
            y=y,
            field=field,
            vector=vector
        )
        plotter._apply_loader(
            loader,
            source_label=description,
            preselect=preselect,
            in_memory=True,
            reset_state=True
        )
    elif filepath:
        plotter.file_input.value = filepath
        plotter._load_data()
    else:
        plotter._set_in_memory_mode(False)

    plotter.display()
    return plotter
