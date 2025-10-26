"""
Property manager to track and apply changes to figure elements
"""

class PropertyManager:
    """
    Manages property changes and applies them to matplotlib objects.
    Keeps track of all modifications for code generation.
    """
    
    def __init__(self, fig, ax):
        """
        Initialize property manager.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to manage
        ax : matplotlib.axes.Axes
            Primary axes to manage
        """
        self.fig = fig
        self.ax = ax
        
        # Track all property changes
        self.changes = {
            'figure': {},
            'axes': {},
            'lines': {},
            'text': {},
            'legend': {},
            'colorbar': {},
        }
        
    def update(self, category, property_name, value):
        """
        Update a property and apply it to the figure.
        
        Parameters
        ----------
        category : str
            Category: 'figure', 'axes', 'lines', 'text', 'legend', 'colorbar'
        property_name : str
            Property to update
        value : any
            New value
        """
        # Store the change
        if category not in self.changes:
            self.changes[category] = {}
        self.changes[category][property_name] = value
        
        # Apply the change
        self._apply_change(category, property_name, value)
        
    def _apply_change(self, category, property_name, value):
        """Apply a single property change"""
        
        if category == 'figure':
            self._apply_figure_property(property_name, value)
        elif category == 'axes':
            self._apply_axes_property(property_name, value)
        elif category == 'lines':
            self._apply_line_property(property_name, value)
        elif category == 'text':
            self._apply_text_property(property_name, value)
        elif category == 'legend':
            self._apply_legend_property(property_name, value)
        elif category == 'colorbar':
            self._apply_colorbar_property(property_name, value)
            
    def _apply_figure_property(self, prop, value):
        """Apply figure-level properties"""
        if prop == 'figsize':
            self.fig.set_size_inches(value)
        elif prop == 'dpi':
            self.fig.set_dpi(value)
        elif prop == 'facecolor':
            self.fig.patch.set_facecolor(value)
        elif prop == 'tight_layout':
            if value:
                self.fig.tight_layout()
                
    def _apply_axes_property(self, prop, value):
        """Apply axes-level properties"""
        if not self.ax:
            return
            
        if prop == 'xlabel':
            self.ax.set_xlabel(value)
        elif prop == 'ylabel':
            self.ax.set_ylabel(value)
        elif prop == 'title':
            self.ax.set_title(value)
        elif prop == 'xlim':
            self.ax.set_xlim(value)
        elif prop == 'ylim':
            self.ax.set_ylim(value)
        elif prop == 'xscale':
            self.ax.set_xscale(value)
        elif prop == 'yscale':
            self.ax.set_yscale(value)
        elif prop == 'grid':
            self.ax.grid(value)
        elif prop == 'grid_alpha':
            self.ax.grid(alpha=value)
        elif prop == 'grid_linestyle':
            self.ax.grid(linestyle=value)
        elif prop == 'aspect':
            self.ax.set_aspect(value)
        elif prop == 'facecolor':
            self.ax.set_facecolor(value)
            
    def _apply_line_property(self, prop, value):
        """Apply line properties to all lines"""
        if not self.ax:
            return
            
        lines = self.ax.get_lines()
        for line in lines:
            if prop == 'linewidth':
                line.set_linewidth(value)
            elif prop == 'linestyle':
                line.set_linestyle(value)
            elif prop == 'marker':
                line.set_marker(value)
            elif prop == 'markersize':
                line.set_markersize(value)
            elif prop == 'alpha':
                line.set_alpha(value)
                
    def _apply_text_property(self, prop, value):
        """Apply text properties (labels, title, ticks)"""
        if not self.ax:
            return
            
        if prop == 'fontsize':
            # Apply to all text elements
            self.ax.title.set_fontsize(value)
            self.ax.xaxis.label.set_fontsize(value)
            self.ax.yaxis.label.set_fontsize(value)
            for tick in self.ax.get_xticklabels():
                tick.set_fontsize(value)
            for tick in self.ax.get_yticklabels():
                tick.set_fontsize(value)
        elif prop == 'fontfamily':
            self.ax.title.set_fontfamily(value)
            self.ax.xaxis.label.set_fontfamily(value)
            self.ax.yaxis.label.set_fontfamily(value)
        elif prop == 'title_fontsize':
            self.ax.title.set_fontsize(value)
        elif prop == 'label_fontsize':
            self.ax.xaxis.label.set_fontsize(value)
            self.ax.yaxis.label.set_fontsize(value)
        elif prop == 'tick_fontsize':
            for tick in self.ax.get_xticklabels():
                tick.set_fontsize(value)
            for tick in self.ax.get_yticklabels():
                tick.set_fontsize(value)
                
    def _apply_legend_property(self, prop, value):
        """Apply legend properties"""
        if not self.ax:
            return
            
        legend = self.ax.get_legend()
        
        if prop == 'visible':
            if value and legend is None:
                self.ax.legend()
            elif not value and legend is not None:
                legend.remove()
        elif legend is not None:
            if prop == 'location':
                legend.set_loc(value)
            elif prop == 'fontsize':
                for text in legend.get_texts():
                    text.set_fontsize(value)
            elif prop == 'frameon':
                legend.set_frame_on(value)
            elif prop == 'framealpha':
                legend.get_frame().set_alpha(value)
            elif prop == 'ncol':
                # Need to recreate legend with new ncol
                handles, labels = self.ax.get_legend_handles_labels()
                self.ax.legend(handles, labels, ncol=value, loc=legend.get_loc())
                
    def _apply_colorbar_property(self, prop, value):
        """Apply colorbar properties"""
        # Placeholder for colorbar adjustments as needed by UI controls
        return
        
    def get_changes(self):
        """Get all tracked changes"""
        return self.changes

