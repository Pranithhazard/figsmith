"""
Core functionality for Figsmith
"""

from .property_manager import PropertyManager
from .interactive_plotter import InteractivePlotter, load_and_plot

__all__ = [
    'PropertyManager',
    'InteractivePlotter',
    'load_and_plot',
]
