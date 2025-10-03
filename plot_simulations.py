""" Written by Tomer279 with the assistance of Cursor.ai

Plotting and visualization functions for nuclear reactor simulation results.

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

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


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


# =============================================================================
# CPS COMPARISON PLOT
# =============================================================================


class CPSPlotter:
    """
    A class for creating CPS comparison plots with dynamic method management.

    This class handles the creation of comprehensive comparison plots
    with support for any number of numerical methods without hard-coding.
    """

    def __init__(self, default_styles: Optional[dict] = None):
        """
        Initialize the CPS plotter.

        Parameters
        ----------
        default_styles : dict, optional
            Default styling for known methods:
                {method_name: {'color': 'red',
                               'marker': 's',
                               'label': 'Name'}}
        """
        self.default_styles = default_styles or self._get_default_styles()
        self.method_counter = 0  # For tracking method order

    def _get_default_styles(self):
        """Get default styles for common methods."""
        return {
            'em': {'color': 'red',
                   'marker': 's',
                   'label': 'Euler-Maruyama'},
            'taylor': {'color': 'green',
                       'marker': '^',
                       'label': 'Taylor Method'},
            'rk': {'color': 'purple',
                   'marker': 'd',
                   'label': 'Runge-Kutta'},
        }

    def add_method_style(
            self, method_name: str, color: str,
            marker: str, label: str):
        """Add or update styling for a specific method."""
        self.default_styles[method_name] = {
            'color': color,
            'marker': marker,
            'label': label
        }

    def plot_comparison(
            self,
            data: dict,
            meta: Optional[dict] = None) -> plt.Figure:
        """
        Create comprehensive CPS comparison plot.

        Parameters
        ----------
        data : dict
            Required: 'stochastic', 'fission_params' (or 'alpha_inv_vec')
            Optional: 'methods' dict with method_name -> cps_data pairs
        meta : dict, optional
            Keys: 'dead_time_type', 'mean_tau',
            't_end', 'grid_points', 'save_path'
        """
        # Extract and validate data
        plot_data = self._extract_plot_data(data)
        plot_meta = self._extract_plot_meta(meta)

        # Create figure
        fig = self._create_figure(plot_data, plot_meta)

        return fig

    def _extract_plot_data(self, data):
        """Extract and validate all plotting data."""
        # Get stochastic data
        stochastic_cps = data['stochastic']

        # Get alpha_inv_vec
        if 'alpha_inv_vec' in data:
            alpha_inv_vec = data['alpha_inv_vec']
        elif 'fission_params' in data:
            alpha_inv_vec = data['fission_params'].alpha_inv_vec
        else:
            print("Warning: No alpha_inv_vec or fission_params provided")
            alpha_inv_vec = np.linspace(0.1, 2.0, len(stochastic_cps))

        # Get methods data
        methods = data.get('methods', {})

        # Validate all data lengths
        expected_length = len(alpha_inv_vec)
        self._validate_data_lengths(stochastic_cps, methods, expected_length)

        # Compute stochastic means
        stochastic_means = self._compute_stochastic_means(stochastic_cps)

        return {
            'alpha_inv_vec': alpha_inv_vec,
            'stochastic_means': stochastic_means,
            'methods': methods
        }

    def _extract_plot_meta(self, meta):
        """Extract plotting metadata with defaults."""
        return _extract_plot_meta(meta, 'without')

    def _validate_data_lengths(self, stochastic_cps, methods, expected_length):
        """Validate that all data arrays have the expected length."""
        _validate_data_lengths(stochastic_cps, methods, expected_length,
                               "Stochastic CPS")

    def _compute_stochastic_means(self, stochastic_cps):
        """Compute mean values from stochastic CPS data."""
        stochastic_means = []
        for cps in stochastic_cps:
            if hasattr(cps, '__len__') and len(cps) > 0:
                stochastic_means.append(np.mean(cps))
            else:
                stochastic_means.append(cps)
        return np.array(stochastic_means)

    def _create_figure(self, plot_data, plot_meta):
        """Create the complete CPS comparison figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Build title
        title = self._build_title(plot_meta)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Plot each subplot
        self._plot_methods_overlay(ax1, plot_data, plot_meta)

        # Plot relative differences for each method (dynamic)
        subplot_idx = 2
        for method_name, method_data in plot_data['methods'].items():
            if subplot_idx <= 4:  # We have 4 subplots total
                ax = [ax1, ax2, ax3, ax4][subplot_idx - 1]
                self._plot_relative_difference(
                    ax, plot_data, method_name, method_data)
                subplot_idx += 1

        plt.tight_layout()

        if plot_meta['save_path']:
            plt.savefig(plot_meta['save_path'], dpi=300, bbox_inches='tight')

        return fig

    def _build_title(self, plot_meta):
        """Build title for the plot."""
        base_title = (
            'Count Rate Comparison'
            f' - {plot_meta["dead_time_type"].title()} Dead Time'
        )

        return _build_plot_title(base_title, plot_meta)

    def _plot_methods_overlay(self, ax, plot_data, plot_meta):
        """Plot all methods overlaid using dynamic styling."""
        alpha_inv_vec = plot_data['alpha_inv_vec']
        stochastic_means = plot_data['stochastic_means']
        methods = plot_data['methods']

        # Plot stochastic baseline
        ax.plot(alpha_inv_vec, stochastic_means, 'o-', color='blue',
                label=f'Stochastic ({plot_meta["dead_time_type"]} dead time)',
                linewidth=2, markersize=6)

        # Plot each method with dynamic styling
        for method_name, method_data in methods.items():
            style = self._get_method_style(method_name)
            ax.plot(alpha_inv_vec, method_data, style['marker'],
                    color=style['color'], label=style['label'],
                    linewidth=2, markersize=6)

        ax.set_xlabel('Inverse Rossi-alpha (1/α)')
        ax.set_ylabel('Counts Per Second (CPS)')
        ax.set_title('CPS Comparison - Constant Dead Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _get_method_style(self, method_name):
        """Get styling for a method, generating defaults if needed."""
        return _get_method_style(
            method_name, self.default_styles, [self.method_counter])

    def _plot_relative_difference(
            self, ax, plot_data, method_name, method_data):
        """Plot relative difference for a specific method."""
        alpha_inv_vec = plot_data['alpha_inv_vec']
        stochastic_means = plot_data['stochastic_means']

        rel_diff = (method_data - stochastic_means) / stochastic_means * 100

        style = self._get_method_style(method_name)

        ax.plot(alpha_inv_vec, rel_diff, 'o-', color=style['color'],
                label=f'{style["label"]} vs Stochastic',
                linewidth=2, markersize=6)
        ax.set_ylabel('Relative Difference (%)')
        ax.set_title(f'{style["label"]} vs Stochastic Difference')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add x-label only for bottom plots
        if ax.get_subplotspec().rowspan.start == 1:
            ax.set_xlabel('Inverse Rossi-alpha (1/α)')


class MethodsTheoreticalPlotter:
    """
    A class for creating methods vs theoretical comparison plots
    with dynamic method management.

    This class handles the creation of comprehensive comparison plots
    showing numerical methods against theoretical values,
    with support for any number of numerical methods.
    """

    def __init__(self, default_styles: Optional[dict] = None):
        """
        Initialize the methods theoretical plotter.

        Parameters
        ----------
        default_styles : dict, optional
            Default styling for known methods:
                {method_name: {'color': 'red',
                               'marker': 's',
                               'label': 'Name'}}
        """
        self.default_styles = default_styles or self._get_default_styles()
        self.method_counter = 0  # For tracking method order

    def _get_default_styles(self):
        """Get default styles for common methods."""
        return {
            'em': {'color': 'blue',
                   'marker': 'o',
                   'label': 'Euler-Maruyama'},
            'taylor': {'color': 'green',
                       'marker': '^',
                       'label': 'Taylor'},
            'rk': {'color': 'orange',
                   'marker': 's',
                   'label': 'Runge-Kutta'},
        }

    def add_method_style(
            self, method_name: str, color: str,
            marker: str, label: str):
        """Add or update styling for a specific method."""
        self.default_styles[method_name] = {
            'color': color,
            'marker': marker,
            'label': label
        }

    def plot_comparison(
            self, data: dict, meta: Optional[dict] = None) -> plt.Figure:
        """
        Create comprehensive methods vs theoretical comparison plot.

        Parameters
        ----------
        data : dict
            Required: 'theoretical', 'fission_params' (or 'alpha_inv_vec')
            Optional: 'methods' dict with method_name -> cps_data pairs
        meta : dict, optional
            Keys: 'dead_time_type', 'mean_tau',
                  't_end', 'grid_points', 'save_path'
        """
        # Extract and validate data
        plot_data = self._extract_plot_data(data)
        plot_meta = self._extract_plot_meta(meta)

        # Create figure
        fig = self._create_figure(plot_data, plot_meta)

        return fig

    def _extract_plot_data(self, data):
        """Extract and validate all plotting data."""
        # Get theoretical data
        theoretical_cps = data['theoretical']

        # Get alpha_inv_vec
        if 'alpha_inv_vec' in data:
            alpha_inv_vec = data['alpha_inv_vec']
        elif 'fission_params' in data:
            alpha_inv_vec = data['fission_params'].alpha_inv_vec
        else:
            print("Warning: No alpha_inv_vec or fission_params provided")
            fallback_length = len(
                theoretical_cps) if theoretical_cps is not None else 10
            alpha_inv_vec = np.linspace(0.1, 2.0, fallback_length)

        # Get methods data
        methods = data.get('methods', {})

        # Validate all data lengths
        expected_length = len(alpha_inv_vec)
        self._validate_data_lengths(theoretical_cps, methods, expected_length)

        return {
            'alpha_inv_vec': alpha_inv_vec,
            'theoretical_cps': theoretical_cps,
            'methods': methods
        }

    def _extract_plot_meta(self, meta):
        """Extract plotting metadata with defaults."""
        return _extract_plot_meta(meta, 'constant')

    def _validate_data_lengths(
            self, theoretical_cps, methods, expected_length):
        """Validate that all data arrays have the expected length."""
        _validate_data_lengths(
            theoretical_cps, methods, expected_length, "Theoretical CPS")

    def _create_figure(self, plot_data, plot_meta):
        """Create the complete methods vs theoretical comparison figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Build title
        title = self._build_title(plot_meta)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Plot each subplot
        self._plot_methods_scatter(ax1, plot_data)
        self._plot_absolute_errors(ax2, plot_data)
        self._plot_relative_differences(ax3, plot_data)
        self._plot_method_comparison(ax4, plot_data)

        plt.tight_layout()

        if plot_meta['save_path']:
            plt.savefig(plot_meta['save_path'], dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_meta['save_path']}")
        else:
            plt.show()

        return fig

    def _build_title(self, plot_meta):
        """Build title for the plot."""
        subtitle_parts = [f'{plot_meta["dead_time_type"].title()} Dead Time']
        base_title = 'Method Comparison: Numerical vs Theoretical'

        if subtitle_parts:
            base_title = f'{base_title} | {subtitle_parts[0]}'

        return _build_plot_title(base_title, plot_meta)

    def _plot_methods_scatter(self, ax, plot_data):
        """Plot all methods vs theoretical as scatter plot."""
        alpha_inv_vec = plot_data['alpha_inv_vec']
        theoretical_cps = plot_data['theoretical_cps']
        methods = plot_data['methods']

        # Plot each method
        for method_name, method_data in methods.items():
            style = self._get_method_style(method_name)
            ax.scatter(alpha_inv_vec, method_data, c=style['color'],
                       marker=style['marker'], s=60, alpha=0.7,
                       label=style['label'])

        # Plot theoretical line
        if theoretical_cps is not None:
            ax.plot(alpha_inv_vec, theoretical_cps, 'r--',
                    linewidth=2, alpha=0.8, label='Theoretical')

        ax.set_xlabel('Alpha Inverse (s)')
        ax.set_ylabel('Count Rate (CPS)')
        ax.set_title('Count Rates vs Alpha Inverse Values')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_absolute_errors(self, ax, plot_data):
        """Plot absolute errors from theoretical values."""
        alpha_inv_vec = plot_data['alpha_inv_vec']
        theoretical_cps = plot_data['theoretical_cps']
        methods = plot_data['methods']

        if theoretical_cps is None:
            ax.text(0.5, 0.5, 'No theoretical data',
                    transform=ax.transAxes, ha='center', va='center')
            return

        # Plot absolute errors for each method
        for method_name, method_data in methods.items():
            style = self._get_method_style(method_name)
            abs_error = np.abs(method_data - theoretical_cps)
            ax.plot(alpha_inv_vec, abs_error, marker=style['marker'],
                    color=style['color'], linestyle='-', linewidth=2,
                    markersize=6, alpha=0.7,
                    label=f'{style["label"]} Error')

        ax.set_xlabel('Alpha Inverse (s)')
        ax.set_ylabel('Absolute Error from Theoretical (CPS)')
        ax.set_title('Error Comparison from Theoretical Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Use log scale for better visualization

    def _plot_relative_differences(self, ax, plot_data):
        """Plot relative differences from theoretical values."""
        alpha_inv_vec = plot_data['alpha_inv_vec']
        theoretical_cps = plot_data['theoretical_cps']
        methods = plot_data['methods']

        if theoretical_cps is None:
            ax.text(0.5, 0.5, 'No theoretical data',
                    transform=ax.transAxes, ha='center', va='center')
            return

        # Plot relative differences for each method
        for method_name, method_data in methods.items():
            style = self._get_method_style(method_name)
            rel_diff = (method_data - theoretical_cps) / theoretical_cps * 100
            ax.plot(alpha_inv_vec, rel_diff, marker=style['marker'],
                    color=style['color'], linestyle='-', linewidth=2,
                    markersize=6, alpha=0.7, label=style['label'])

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Alpha Inverse (s)')
        ax.set_ylabel('Relative Difference from Theoretical (%)')
        ax.set_title('Relative Differences from Theoretical Values')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_method_comparison(self, ax, plot_data):
        """Plot method comparison scatter plot."""
        methods = plot_data['methods']
        method_list = list(methods.items())

        if len(method_list) < 2:
            ax.text(0.5, 0.5, 'Need at least 2 methods for comparison',
                    transform=ax.transAxes, ha='center', va='center')
            return

        # Plot pairwise comparisons
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (method1_name, method1_data) in enumerate(method_list[:-1]):
            for _, (method2_name, method2_data) in (
                    enumerate(method_list[i+1:], i+1)):
                color = colors[i % len(colors)]
                ax.scatter(
                    method1_data, method2_data, c=color, alpha=0.7, s=60,
                    label=f'{method1_name} vs {method2_name}')

        # Add diagonal line for perfect correlation
        all_data = [data for _, data in method_list]
        if all_data:
            min_val = min(np.nanmin(data) for data in all_data)
            max_val = max(np.nanmax(data) for data in all_data)
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

        ax.set_xlabel('Method 1 CPS')
        ax.set_ylabel('Method 2 CPS')
        ax.set_title('Method Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _get_method_style(self, method_name):
        """Get styling for a method, generating defaults if needed."""
        return _get_method_style(
            method_name, self.default_styles, [self.method_counter])


def plot_cps_comparison(data: dict,
                        meta: Optional[dict] = None) -> plt.Figure:
    """
    Create comprehensive comparison plot of CPS values.

    Parameters
    ----------
    data : dict
        Required: 'stochastic', 'fission_params' (or 'alpha_inv_vec')
        Optional: 'methods' dict with method_name -> cps_data pairs
    meta : Optional[dict], optional
        Keys: 'dead_time_type', 'mean_tau',
        't_end', 'grid_points', 'save_path'
    """
    plotter = CPSPlotter()
    return plotter.plot_comparison(data, meta)


def plot_methods_vs_theoretical(
        data: dict, meta: Optional[dict] = None) -> plt.Figure:
    """
    Create comprehensive comparison plots of numerical methods vs theoretical.

    Parameters
    ----------
    data : dict
        Required: 'theoretical', 'fission_params' (or 'alpha_inv_vec')
        Optional: 'methods' dict with method_name -> cps_data pairs
    meta : dict, optional
        Keys: 'dead_time_type', 'mean_tau', 't_end',
              'grid_points', 'save_path'

    Returns
    -------
    plt.Figure
        The generated figure object
    """
    plotter = MethodsTheoreticalPlotter()
    return plotter.plot_comparison(data, meta)


def _build_plot_title(base_title: str, plot_meta: dict,
                      subtitle_prefix: str = "Simulation Parameters") -> str:
    """
    Build a formatted plot title with parameters.
    """
    title_parts = [base_title]

    # Add simulation parameters
    param_parts = []
    if plot_meta.get('mean_tau') is not None:
        param_parts.append(f'τ = {plot_meta["mean_tau"]:.2e} s')
    if plot_meta.get('t_end') is not None:
        param_parts.append(f'Δt = {plot_meta["t_end"]:.3f} s')
    if plot_meta.get('grid_points') is not None:
        param_parts.append(f'N = {plot_meta["grid_points"]:,}')

    if param_parts:
        subtitle = f'{subtitle_prefix}: {", ".join(param_parts)}'
        return f'{title_parts[0]}\n{subtitle}'

    return title_parts[0]


def _get_method_style(
        method_name: str,
        default_styles: dict,
        method_counter: list) -> dict:
    """
    Get styling for a method, generating defaults if needed.

    Parameters
    ----------
    method_name : str
        Name of the method
    default_styles : dict
        Dictionary of existing styles
    method_counter : list
        List with single integer for counter (mutable reference)

    Returns
    -------
    dict
        Style dictionary with 'color', 'marker', 'label' keys
    """
    if method_name in default_styles:
        return default_styles[method_name]

    # Generate new style for unknown method
    style = _generate_new_style(method_name, method_counter)
    default_styles[method_name] = style
    return style


def _generate_new_style(method_name: str, method_counter: list) -> dict:
    """
    Generate a new style for an unknown method.

    Parameters
    ----------
    method_name : str
        Name of the method
    method_counter : list
        List with single integer for counter (mutable reference)

    Returns
    -------
    dict
        Style dictionary with 'color', 'marker', 'label' keys
    """
    colors = ['red', 'green', 'purple', 'orange', 'brown',
              'pink', 'gray', 'olive', 'cyan', 'magenta']
    markers = ['s', '^', 'd', 'v', 'p', '*', 'h', 'H', '+', 'x']

    color_idx = method_counter[0] % len(colors)
    marker_idx = method_counter[0] % len(markers)

    method_counter[0] += 1

    return {
        'color': colors[color_idx],
        'marker': markers[marker_idx],
        'label': method_name.replace('_', ' ').title()
    }


def _extract_plot_meta(meta: dict, default_dead_time: str = 'without') -> dict:
    """
    Extract plotting metadata with defaults.

    Parameters
    ----------
    meta : dict
        Input metadata dictionary
    default_dead_time : str
        Default dead time type

    Returns
    -------
    dict
        Extracted metadata with defaults
    """
    meta = meta or {}
    return {
        'dead_time_type': meta.get('dead_time_type', default_dead_time),
        'mean_tau': meta.get('mean_tau'),
        't_end': meta.get('t_end'),
        'grid_points': meta.get('grid_points'),
        'save_path': meta.get('save_path')
    }


def _validate_data_lengths(baseline_data, methods: dict, expected_length: int,
                           baseline_name: str = "Data"):
    """
    Validate that all data arrays have the expected length.

    Parameters
    ----------
    baseline_data : array-like
        Baseline data to validate
    methods : dict
        Dictionary of method data to validate
    expected_length : int
        Expected length for all arrays
    baseline_name : str
        Name of baseline data for error messages
    """
    if baseline_data is not None and len(baseline_data) != expected_length:
        raise ValueError(
            f"{baseline_name} length ({len(baseline_data)}) "
            f"must match alpha_inv_vec length ({expected_length})")

    for method_name, method_data in methods.items():
        if len(method_data) != expected_length:
            raise ValueError(
                f"{method_name} CPS length ({len(method_data)}) "
                f"must match alpha_inv_vec length ({expected_length})")


def _calculate_relative_difference(values1, values2):
    """
    Calculate relative difference between two arrays,
    handling division by zero.

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
        relative_diff = np.abs(values1 - values2) / \
            np.where(values2 != 0, values2, np.nan)
    return relative_diff * 100


def _get_dead_time_title(dead_time_type):
    """Get formatted title for dead time type."""
    dead_time_titles = {
        'without': 'Without Dead Time',
        'const': 'Constant Dead Time',
        'uniform': 'Uniform Dead Time',
        'normal': 'Normal Dead Time',
        'gamma': 'Gamma Dead Time',
    }
    return dead_time_titles.get(dead_time_type, dead_time_type.title())
