""" Written by Tomer279 with the assistance of Cursor.ai """
"""
Plotting and visualization functions for nuclear reactor simulation results.

This module contains all plotting functions for visualizing simulation results,
including count rates, population dynamics, dead time effects, and comparative analysis.

ORGANIZATION:
============
1. PLOT STYLES - Configuration and styling classes
2. POPULATION DYNAMICS PLOTS - Population vs time plots
3. COUNT RATE PLOTS - CPS analysis plots
4. COMPARISON PLOTS - Side-by-side comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# =============================================================================
# 1. PLOT STYLES
# =============================================================================

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


# =============================================================================
# 2. POPULATION DYNAMICS PLOTS
# =============================================================================

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
        raise ValueError("Both matrices must be 2D with shape (trajectories, steps)")
    
    if population_matrix.shape != time_matrix.shape:
        raise ValueError("population_matrix and time_matrix must have the same shape")
    
    num_trajectories, num_steps = population_matrix.shape
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate colors for different trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
    
    # Plot each trajectory
    for i in range(num_trajectories):
        _plot_single_trajectory(ax, time_matrix, population_matrix,
                                i, colors[i], downsample_factor)
    
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


def _plot_single_trajectory(ax, time_matrix, population_matrix,
                            trajectory_idx, color, downsample_factor = None,
                            max_trajectories_for_legend = 10):
    
    """
    Plot a single trajectory from the population matrix.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    time_matrix : np.ndarray
        Time matrix with shape (num_trajectories, num_steps)
    population_matrix : np.ndarray
        Population matrix with shape (num_trajectories, num_steps)
    trajectory_idx : int
        Index of the trajectory to plot
    color : str or tuple
        Color for the trajectory line
    downsample_factor : int, optional
        Factor to downsample data for plotting
    max_trajectories_for_legend : int
        Maximum number of trajectories to show in legend
    """
    
    time_data = time_matrix[trajectory_idx, :]
    pop_data = population_matrix[trajectory_idx, :]
        
    # Apply downsampling if specified
    if downsample_factor is not None:
        time_data = time_data[::downsample_factor]
        pop_data = pop_data[::downsample_factor]
        
    # Plot the trajectory
    ax.plot(time_data, pop_data, 
            color=color, linewidth=2, alpha=0.7,
            label= (f'Trajectory {trajectory_idx + 1}' 
                    if trajectory_idx < max_trajectories_for_legend else None))


# =============================================================================
# CPS COMPARISON PLOT
# =============================================================================

def plot_cps_comparison(stochastic_cps: list, 
                       em_cps: Optional[list] = None, 
                       taylor_cps: Optional[list] = None,
                       config=None,
                       dead_time_type: str = 'without',
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comparison plot of CPS values from stochastic,
    Euler-Maruyama, and/or Taylor simulations.
    
    Parameters
    ----------
    stochastic_cps : list
        List of CPS values from stochastic simulations
    em_cps : Optional[list], default=None
        List of CPS values from Euler-Maruyama simulations. 
        If None, EM plots will be skipped.
    taylor_cps : Optional[list], default=None
        List of CPS values from Taylor method simulations. 
        If None, Taylor plots will be skipped.
    config : SimulationConfig
        Configuration object
    dead_time_type : str
        Type of dead time ('without', 'const','uniform', 'normal', 'gamma')
    save_path : Optional[str]
        Path to save the plot. If None, the plot will not be saved automatically.
        
    Returns
    -------
    plt.Figure
        The generated comparison plot
    """
    
    # Input validation
    if not stochastic_cps or len(stochastic_cps) == 0:
        raise ValueError("stochastic_cps cannot be empty")
    
    if config is None:
        raise ValueError("config parameter is required")
    
    # Validate data lengths match config
    expected_length = len(config.alpha_inv_vec)
    if len(stochastic_cps) != expected_length:
        raise ValueError(f"stochastic_cps length ({len(stochastic_cps)})"
                         f"must match config.alpha_inv_vec length ({expected_length})")
    
    if em_cps is not None and len(em_cps) != expected_length:
        raise ValueError(f"em_cps length ({len(em_cps)}) "
                         f"must match config.alpha_inv_vec length ({expected_length})")
    
    if taylor_cps is not None and len(taylor_cps) != expected_length:
        raise ValueError(f"taylor_cps length ({len(taylor_cps)})"
                         f"must match config.alpha_inv_vec length ({expected_length})")
        
    PlotStyle.setup_default_style()
    
    # Determine which methods are available
    has_em = em_cps is not None
    has_taylor = taylor_cps is not None
    
    # Determine subplot layout based on available methods
    if has_em and has_taylor:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        axs = [ax1, ax2, ax3, ax4]
        main_title = "Count Rate Comparison: Stochastic vs Euler-Maruyama vs Taylor"
    elif has_em or has_taylor:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        axs = [ax1, ax2]
        main_title = (f"Count Rate Comparison: "
                      f"Stochastic vs {'Euler-Maruyama' if has_em else 'Taylor'}")
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        axs = [ax1]
        main_title = "Count Rate: Stochastic Simulation"
    
    # Add the main title
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98)
    
    # Prepare data for plotting
    if isinstance(stochastic_cps[0], np.ndarray):
        stochastic_means = np.array([np.mean(cps) for cps in stochastic_cps])
        stochastic_stds = np.array([np.std(cps) for cps in stochastic_cps])
    else:
        stochastic_means = stochastic_cps
        stochastic_stds = None
    
    title_suffix = _get_dead_time_title(dead_time_type)
    
    # Plot 1: Direct comparison
    _plot_cps_comparison_subplot(ax1, x_data = config.alpha_inv_vec,
                                 y_data = stochastic_means,
                                 label = 'Stochastic Simulation',
                                 color = 'blue', marker = 'o-',
                                 title = f'CPS Comparison - {title_suffix}')
    
    if has_em:
        ax1.plot(config.alpha_inv_vec, em_cps, 's-', color='red', 
                 linewidth=2, markersize=6, label='Euler-Maruyama')
    
    if has_taylor:
        ax1.plot(config.alpha_inv_vec, taylor_cps, '^-', color='green', 
                 linewidth=2, markersize=6, label='Taylor Method')
    
    # Add error bars for stochastic if available
    if stochastic_stds is not None:
        ax1.fill_between(config.alpha_inv_vec, 
                        stochastic_means - stochastic_stds,
                        stochastic_means + stochastic_stds, 
                        alpha=0.3, color='blue', label='Stochastic ±1σ')
    
    # Plot additional comparisons
    plot_idx = 1
    
    if has_em and len(axs) > 1:
        em_diff = _calculate_relative_difference(stochastic_means, em_cps)
        _plot_cps_comparison_subplot(axs[plot_idx], config.alpha_inv_vec, em_diff,
                                   'EM vs Stochastic', 'purple', 'o-',
                                   f'EM vs Stochastic Difference - {title_suffix}',
                                   ylabel='Relative Difference (%)')
        plot_idx += 1
    
    if has_taylor and len(axs) > plot_idx:
        taylor_diff = _calculate_relative_difference(stochastic_means, taylor_cps)
        _plot_cps_comparison_subplot(axs[plot_idx], config.alpha_inv_vec, taylor_diff,
                                   'Taylor vs Stochastic', 'orange', 'o-',
                                   f'Taylor vs Stochastic Difference - {title_suffix}',
                                   ylabel='Relative Difference (%)')
        plot_idx += 1
    
    if has_em and has_taylor and len(axs) > plot_idx:
        taylor_em_diff = _calculate_relative_difference(taylor_cps, em_cps)
        _plot_cps_comparison_subplot(axs[plot_idx], config.alpha_inv_vec, taylor_em_diff,
                                   'Taylor vs EM', 'brown', 'o-',
                                   f'Taylor vs EM Difference - {title_suffix}',
                                   ylabel='Relative Difference (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig

def _plot_cps_comparison_subplot(ax, x_data, y_data, label, color, marker='o-', 
                                title="", xlabel="Inverse Rossi-alpha (1/α)", 
                                ylabel="Counts Per Second (CPS)"):
    """
    Helper function to plot a single CPS comparison subplot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    x_data : np.ndarray
        X-axis data (alpha_inv_vec)
    y_data : np.ndarray
        Y-axis data (CPS values)
    label : str
        Label for the plot line
    color : str
        Color for the plot line
    marker : str
        Marker style for the plot
    title : str
        Title for the subplot
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    """
    ax.plot(x_data, y_data, marker, color=color, linewidth=2, markersize=6,
            label=label)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_data)
    ax.set_xticklabels([f'{val:.3f}' for val in x_data], rotation=45)
    
def _calculate_relative_difference(values1, values2):
    """
    Calculate relative difference between two arrays, handling division by zero.
    
    Parameters
    ----------
    values1, values2 : np.ndarray
        Arrays to compare
        
    Returns
    -------
    np.ndarray
        Relative difference as percentage
    """
    values1 = np.array(values1)
    values2 = np.array(values2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_diff = np.abs(values1 - values2) / np.where(values2 != 0, values2, np.nan)
    return relative_diff * 100

def _get_dead_time_title(dead_time_type):
    """Get formatted title for dead time type."""
    dead_time_titles = {
        'without': 'Without Dead Time',
        'const': 'Constant Dead Time', 
        'uniform': 'Uniform Dead Time',
        'normal' : 'Normal Dead Time',
        'gamma' : 'Gamma Dead Time', 
    }
    return dead_time_titles.get(dead_time_type, dead_time_type.title())
    

    