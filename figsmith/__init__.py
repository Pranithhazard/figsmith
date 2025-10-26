"""
Figsmith - Interactive Figure Editor for Scientific Plotting
Designed for scientific computing workflows (e.g., fluid dynamics) in Jupyter.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from .core.interactive_plotter import InteractivePlotter, load_and_plot
from .io import DataLoader

__all__ = [
    'InteractivePlotter',
    'DataLoader',
    'load_and_plot',
]
