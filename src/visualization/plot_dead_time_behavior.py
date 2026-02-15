"""Dead time behavior comparison plotting module.

This module contains functions for visualizing the theoretical relationship
between observed CPS (m) and theoretical CPS (n) for different dead time models.

Functions:
    plot_dead_time_behavior: Plot m vs n for non-paralyzable and paralyzable systems
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.plot_styles import PlotStyle


def plot_dead_time_behavior(
        tau: float,
        n_max: Optional[float] = None,
        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot observed CPS (m) vs theoretical CPS (n) for dead time models.

    This function creates a comparison plot showing the relationship between
    observed count rate (m) and theoretical count rate (n) for both
    non-paralyzable and paralyzable dead time systems.

    Parameters
    ----------
    tau : float
        Dead time in seconds
    n_max : float, optional
        Maximum n value to plot. If None, defaults to 3/tau
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The created figure

    Notes
    -----
    The plot includes:
      - Non-paralyzable curve: n = m/(1 - m*tau)
      - Paralyzable curve: m = n * exp(-n*tau)
      - Diagonal reference line: m = n
      - Key reference lines at 1/tau, 1/(tau*e), and peak values
    """
    # Setup plotting style
    PlotStyle.setup_default_style()

    # Set default n_max if not provided
    if n_max is None:
        n_max = 3.0 / tau

    # Create n array (theoretical CPS)
    n = np.linspace(0, n_max, 1000)

    # Calculate m for non-paralyzable system
    # From n = m/(1 - m*tau), solving for m: m = n/(1 + n*tau)
    m_nonparalyzable = n / (1 + n * tau)

    # Calculate m for paralyzable system
    # m = n * exp(-n*tau)
    m_paralyzable = n * np.exp(-n * tau)

    # Find key points
    # For paralyzable: maximum occurs when dm/dn = 0
    # dm/dn = exp(-n*tau) - n*tau*exp(-n*tau) = exp(-n*tau)*(1 - n*tau) = 0
    # So maximum at n = 1/tau
    n1 = 1.0 / tau
    m1 = n1 * np.exp(-1.0)  # m1 = (1/tau) * exp(-1) = 1/(tau*e)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot diagonal reference line m = n
    ax.plot(n, n, 'k--', linewidth=1.5, label='m = n', alpha=0.7)

    # Plot non-paralyzable curve
    ax.plot(n, m_nonparalyzable, 'b-', linewidth=2,
            label='Nonparalyzable', alpha=0.8)

    # Plot paralyzable curve
    ax.plot(n, m_paralyzable, 'r-', linewidth=2,
            label='Paralyzable', alpha=0.8)

    # Add horizontal dashed line at (τe)⁻¹
    ax.axhline(y=m1, color='gray', linestyle='--',
               linewidth=1, alpha=0.5)

    # Add vertical dashed line at τ⁻¹
    ax.axvline(x=n1, color='gray', linestyle='--',
               linewidth=1, alpha=0.5)

    # Set axis limits
    ax.set_xlim(0, n_max)
    ax.set_ylim(0, max(np.max(m_nonparalyzable), np.max(m_paralyzable)) * 1.1)

    # Create custom tick locations and labels (only symbolic)
    # X-axis ticks
    x_tick_locations = []
    x_tick_labels = []
    
    # Add τ⁻¹
    if n1 <= n_max:
        x_tick_locations.append(n1)
        x_tick_labels.append(r'$\tau^{-1}$')
    
    # Y-axis ticks
    y_tick_locations = []
    y_tick_labels = []
    
    # Add (τe)⁻¹
    if m1 <= ax.get_ylim()[1]:
        y_tick_locations.append(m1)
        y_tick_labels.append(r'$(\tau e)^{-1}$')
    
    # Add τ⁻¹ on y-axis (maximum for non-paralyzable)
    tau_inv = 1.0 / tau
    if tau_inv <= ax.get_ylim()[1]:
        y_tick_locations.append(tau_inv)
        y_tick_labels.append(r'$\tau^{-1}$')
    
    # Set custom ticks (only symbolic values)
    ax.set_xticks(x_tick_locations)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_locations)
    ax.set_yticklabels(y_tick_labels)

    # Add labels and formatting
    ax.set_xlabel('n (Theoretical CPS)', fontsize=14)
    ax.set_ylabel('m (Observed CPS)', fontsize=14)
    ax.set_title('Dead Time Behavior: Observed vs Theoretical CPS', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

if __name__ == "__main__":
    fig = plot_dead_time_behavior(tau = 0.001, n_max = 5_000)
    plt.show()