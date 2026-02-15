"""Population dynamics plotting module.

This module contains functions for visualizing population dynamics
from stochastic simulations.

Functions:
    plot_stochastic_population_dynamics: Plot population evolution from
        stochastic simulation
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.plot_styles import PlotStyle

def plot_stochastic_population_dynamics(
        population_matrix: np.ndarray,
        time_matrix: np.ndarray,
        fission_value: float,
        downsample_factor: Optional[int] = None,
        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot population dynamics from a single stochastic simulation.

    This function plots all trajectories (rows) from a population matrix,
    following the project's plotting standards.

    Parameters
    ----------
    population_matrix : np.ndarray
        Population matrix with shape (num_trajectories, num_steps)
    time_matrix : np.ndarray
        Time matrix with shape (num_trajectories, num_steps)
    fission_value : float
        Fission value for labeling the plot
    downsample_factor : int, optional
        Factor to downsample data for plotting. If None, plot all points.
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    plt.Figure
        The created figure
    """
    # Setup professional plotting style (following streamlit_app standards)
    PlotStyle.setup_default_style()

    # Validate input
    if population_matrix.ndim != 2 or time_matrix.ndim != 2:
        raise ValueError(
            "Both matrices must be 2D with shape (trajectories, steps)")

    if population_matrix.shape != time_matrix.shape:
        raise ValueError(
            "population_matrix and time_matrix must have the same shape")

    num_trajectories = population_matrix.shape[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate colors for different trajectories
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_trajectories))

    # Plot each trajectory
    for i in range(num_trajectories):
        time_data = time_matrix[i, :]
        pop_data = population_matrix[i, :]
        _plot_single_trajectory(ax, time_data, pop_data,
                                colors[i], downsample_factor)

    # Set labels and title (following streamlit_app style)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Population')
    ax.set_title(f'Population Evolution (Fission Rate: {fission_value})',
                 fontsize=16, fontweight='bold')

    # Professional grid and legend
    ax.grid(True, alpha=0.3)
    if 1 < num_trajectories <= 10:
        ax.legend(fontsize=12, loc='best')

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def _plot_single_trajectory(ax, time_data, pop_data,
                            color, downsample_factor=None):
    """
    Plot a single trajectory from the population matrix.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    time_data : np.ndarray
        Time data for this trajectory
    pop_data: np.ndarray
        Population data for this trajectory
    color : str or tuple
        Color for the trajectory line
    downsample_factor : int, optional
        Factor to downsample data for plotting
    """
    # Apply downsampling if specified
    if downsample_factor is not None:
        time_data = time_data[::downsample_factor]
        pop_data = pop_data[::downsample_factor]

    # Plot the trajectory
    ax.plot(time_data, pop_data,
            color=color, linewidth=2, alpha=0.7)
