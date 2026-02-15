"""Visualization package for nuclear reactor simulation results.

This package provides plotting and visualization functions for simulation
results, including count rates, population dynamics, dead time effects,
and comparative analysis.

Modules:
    plot_styles: Plotting style configuration
    plot_population: Population dynamics plotting
    plot_cps_comparison: CPS comparison plotting
    plot_laplace_analysis: Laplace transform analysis plotting
    plot_simulations: Backward compatibility module (deprecated)
"""

from src.visualization.plot_styles import PlotStyle
from src.visualization.plot_population import plot_stochastic_population_dynamics
from src.visualization.plot_cps_comparison import (
    plot_cps_comparison,
    plot_methods_vs_theoretical
)
from src.visualization.plot_laplace_analysis import (
    plot_cps_vs_std,
    plot_noise_amplitude_vs_inverse_alpha
)

__all__ = [
    'PlotStyle',
    'plot_stochastic_population_dynamics',
    'plot_cps_comparison',
    'plot_methods_vs_theoretical',
    'plot_cps_vs_std',
    'plot_noise_amplitude_vs_inverse_alpha',
]