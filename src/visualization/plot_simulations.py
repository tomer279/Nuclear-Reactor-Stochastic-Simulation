""" Written by Tomer279 with the assistance of Cursor.ai

Plotting and visualization functions for nuclear reactor simulation results.

This module provides backward compatibility by re-exporting all plotting
functions from the refactored modules.

This module contains all plotting functions for visualizing simulation results,
including count rates, population dynamics,
dead time effects, and comparative analysis.

ORGANIZATION:
============
1. PLOT STYLES - Configuration and styling classes
2. POPULATION DYNAMICS PLOTS - Population vs time plots
3. COUNT RATE PLOTS - CPS analysis plots
4. COMPARISON PLOTS - Side-by-side comparisons
"""

# Import all functions and classes for backward compatibility
from src.visualization.plot_styles import PlotStyle
from src.visualization.plot_population import (
    plot_stochastic_population_dynamics,
    _plot_single_trajectory
)
from src.visualization.plot_cps_comparison import (
    CPSPlotter,
    MethodsTheoreticalPlotter,
    plot_cps_comparison,
    plot_methods_vs_theoretical
)
from src.visualization.plot_laplace_analysis import (
    plot_cps_vs_std,
    plot_noise_amplitude_vs_inverse_alpha
)

# Re-export everything for backward compatibility
__all__ = [
    'PlotStyle',
    'plot_stochastic_population_dynamics',
    '_plot_single_trajectory',
    'CPSPlotter',
    'MethodsTheoreticalPlotter',
    'plot_cps_comparison',
    'plot_methods_vs_theoretical',
    'plot_cps_vs_std',
    'plot_noise_amplitude_vs_inverse_alpha',
]
