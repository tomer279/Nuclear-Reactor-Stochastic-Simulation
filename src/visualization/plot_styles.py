"""Plotting style configuration module.

This module contains the PlotStyle class for configuring consistent
plotting styles across all visualization functions.

Classes:
    PlotStyle: Default plotting style configuration
"""

import matplotlib.pyplot as plt


class PlotStyle:
    """ Default plotting style configuration """

    @staticmethod
    def setup_default_style():
        """Setup consistent plotting style."""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

    @staticmethod
    def setup_publication_style():
        """Setup publication-quality plotting style."""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 10,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'lines.linewidth': 1.5,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'stix'
        })
