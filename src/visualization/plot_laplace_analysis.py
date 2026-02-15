"""Laplace transform analysis plotting module.

This module contains functions for visualizing count rates and noise
amplitudes calculated using Laplace transforms of dead time distributions.

Functions:
    plot_cps_vs_std: Plot CPS as a function of standard deviation
    plot_noise_amplitude_vs_inverse_alpha: Plot noise amplitude from
        Laplace as a function of inverse alpha
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from src.utils.laplace_transforms import (DeadTimeLaplaceCalculator,
                                          calculate_noise_amplitude_from_laplace)
from src.models.core_parameters import FissionParameters, PhysicalParameters
from src.visualization.plot_styles import PlotStyle

def plot_cps_vs_std(
        config: Optional[dict] = None,
        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot CPS as a function of standard deviation for different inverse-alpha values.

    This function creates a plot showing how CPS varies with dead time standard
    deviation for different inverse-alpha values and dead time distribution models.
    The CPS is calculated using the theoretical model: CPS = a * L_tau(a), where
    a is the detection rate at equilibrium and L_tau is the Laplace transform.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with optional keys:
            - 'inverse_alpha_values': list of inverse-alpha values
              (default: [33.333, 111.111, 250])
            - 'std_values': list of standard deviation values
              (default: [1e-7, 2e-7, 3e-7])
            - 'mean_tau_s': mean dead time in seconds (default: 1e-6)
            - 'detect_rate': detection rate constant (default: 10.0)
            - 'source_rate': source rate constant (default: 1000.0)
            - 'dead_time_models': list of models (default: ['uniform', 'normal', 'gamma'])
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.

    Returns
    -------
    plt.Figure
        The created figure

    Examples
    --------
    >>> fig = plot_cps_vs_std()
    >>> plt.show()
    """
    if config is None:
        config = {}

    alpha_vals = config.get('inverse_alpha_values', [166.667, 200, 250])
    std_vals = config.get('std_values',
                          [0.01e-7, 0.5e-7, 1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7])
    plot_config = {
        'mean_tau_s': config.get('mean_tau_s', 1e-6),
        'detect_rate': config.get('detect_rate', 10.0),
        'source_rate': config.get('source_rate', 1000.0),
        'dead_time_models': config.get('dead_time_models', ['uniform', 'normal', 'gamma'])
    }
    percentage_vals = [(std / plot_config['mean_tau_s'])
                       * 100 for std in std_vals]

    PlotStyle.setup_default_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    _plot_cps_lines(ax, alpha_vals, std_vals, percentage_vals, plot_config)

    ax.set_xlabel('Standard Deviation (% of Mean)', fontsize=14)
    ax.set_ylabel('Counts Per Second (CPS)', fontsize=14)
    ax.set_title(
        'CPS vs Standard Deviation for Different Inverse-Alpha Values',
        fontsize=16
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    # ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_noise_amplitude_vs_inverse_alpha(
        config: Optional[dict] = None,
        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot noise amplitude from Laplace as a function of inverse alpha.

    This function creates a plot showing how noise amplitude varies with
    inverse Rossi-alpha for normal dead time distribution with different
    standard deviation values. The noise amplitude is calculated using the
    Laplace transform of the dead time distribution.

    The inverse alpha values correspond to the default fission values
    (33.94-33.992) used in the configuration.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with optional keys:
            - 'physical_params': PhysicalParameters object
              (default: PhysicalParameters() with defaults)
            - 'mean_tau_s': mean dead time in seconds (default: 1e-6)
            - 'std_percentages': list of standard deviation percentages
              (default: [10, 20, 30])
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.

    Returns
    -------
    plt.Figure
        The created figure

    Examples
    --------
    >>> fig = plot_noise_amplitude_vs_inverse_alpha()
    >>> plt.show()
    """
    if config is None:
        config = {}

    plot_config = _extract_noise_plot_config(config)

    # Setup plotting style
    PlotStyle.setup_default_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    _plot_noise_amplitude_lines(ax, plot_config)

    # Set labels and title
    ax.set_xlabel('Inverse Rossi-alpha (1/α)', fontsize=14)
    ax.set_ylabel('Noise Amplitude σ₃', fontsize=14)
    ax.set_title(
        'Noise Amplitude from Laplace vs Inverse Alpha\n'
        f'Normal Dead Time, τ = {plot_config["mean_tau_s"]:.1e} s',
        fontsize=16, fontweight='bold'
    )
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def _extract_noise_plot_config(config: dict) -> dict:
    """
    Extract and validate configuration for noise amplitude plot.
    """
    physical_params = config.get('physical_params', PhysicalParameters())
    mean_tau_s = config.get('mean_tau_s', 1e-6)
    std_percentages = config.get('std_percentages', [10, 20, 30])

    # Get inverse alpha values from default fission configuration
    fission_params = FissionParameters(physical_params=physical_params)
    alpha_inv_vec = fission_params.alpha_inv_vec

    return {
        'physical_params' : physical_params,
        'mean_tau_s' : mean_tau_s,
        'std_percentages': std_percentages,
        'alpha_inv_vec': alpha_inv_vec,
        'detect_rate': physical_params.detect,
        'source_rate': physical_params.source
        }

def _plot_noise_amplitude_lines(ax, plot_config: dict):
    """
    Plot noise amplitude lines for different standard deviation percentages.
    """
    calculator = DeadTimeLaplaceCalculator()
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']

    for idx, std_percent in enumerate(plot_config['std_percentages']):
        noise_amplitudes = _calculate_noise_amplitudes_for_std(
            calculator, plot_config, std_percent
        )

        # Plot the line for this std percentage
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(
            plot_config['alpha_inv_vec'], noise_amplitudes,
            marker=marker, linestyle='-', color=color,
            linewidth=2, markersize=6,
            label=f'Normal Dead Time, std = {std_percent}%'
        )

def _calculate_noise_amplitudes_for_std(
        calculator: DeadTimeLaplaceCalculator,
        plot_config: dict,
        std_percent: float
        ) -> list:
    """
    Calculate noise amplitudes for a given standard deviation percentage
    """
    mean_tau_s = plot_config['mean_tau_s']
    std_tau_s = mean_tau_s * (std_percent / 100.0)
    detect_rate = plot_config['detect_rate']
    source_rate = plot_config['source_rate']
    alpha_inv_vec = plot_config['alpha_inv_vec']

    noise_amplitudes = []
    for alpha_inv in alpha_inv_vec:
        # Calculate detection rate: a = detect_rate * source_rate * alpha_inv
        detection_rate = detect_rate * source_rate * alpha_inv

        # Calculate Laplace transform for normal dead time
        laplace_result = calculator.calculate_laplace_transform(
            x=detection_rate,
            tau=mean_tau_s,
            distribution='normal',
            std=std_tau_s
        )

        # Calculate noise amplitude from Laplace
        noise_amplitude = calculate_noise_amplitude_from_laplace(
            laplace_result, detection_rate
        )
        noise_amplitudes.append(noise_amplitude)

    return noise_amplitudes



def _plot_cps_lines(ax, alpha_vals, std_vals, percentage_vals, plot_config):
    """
    Plot CPS lines for different inverse-alpha values and dead time models.
    """
    styles = [
        ('blue', 'o', '-'), ('red', 's', '--'), ('green', '^', '-.'),
        ('purple', 'd', '-'), ('orange', 'v', '--'), ('brown', 'p', '-.'),
        ('pink', '*', '-'), ('gray', 'h', '--'), ('olive', '+', '-.')
    ]

    line_idx = 0
    for alpha_inv in alpha_vals:
        detection_rate = (
            plot_config['detect_rate'] * plot_config['source_rate'] * alpha_inv
        )
        for model in plot_config['dead_time_models']:
            cps_vals = _calculate_cps_for_std_range(
                detection_rate, plot_config['mean_tau_s'], model, std_vals
            )
            color, marker, linestyle = styles[line_idx % len(styles)]
            ax.plot(percentage_vals, cps_vals, color=color, marker=marker,
                    linestyle=linestyle, label=f'1/α = {
                        alpha_inv:.3f}, {model.capitalize()}',
                    linewidth=2, markersize=6)
            # Plot approximation line
            cps_approx_vals = _calculate_cps_approx_for_std_range(
                detection_rate, plot_config['mean_tau_s'], std_vals
            )
            ax.plot(percentage_vals, cps_approx_vals,
                    color=color, marker=marker,
                    linestyle=':',  # Use dotted line to distinguish
                    label=f'1/α = {alpha_inv:.3f}, {model.capitalize()
                                                    } (approx)',
                    linewidth=2, markersize=4, alpha=0.7)
            line_idx += 1


def _calculate_cps_second_order_approx(detection_rate, mean_tau, std):
    """
    Calculate CPS using second-order Taylor approximation of Laplace transform.

    Uses: L_τ(s) ≈ 1 - s*E[τ] + (1/2)*s^2*E[τ^2]
    where s = detection_rate, E[τ] = mean_tau, E[τ^2] = mean_tau^2 + std^2
    """
    s = detection_rate
    exponent = - mean_tau * s + (s*std)**2 / (2 * (1 + (std**2 * s)/mean_tau))

    laplace_approx = np.exp(exponent)
    return s * laplace_approx


def _calculate_cps_for_std_range(detection_rate, mean_tau, model, std_vals):
    """
    Calculate CPS values for a range of standard deviation values.
    """
    calculator = DeadTimeLaplaceCalculator()
    return [
        detection_rate * calculator.calculate_laplace_transform(
            x=detection_rate, tau=mean_tau, distribution=model, std=std
        ).value
        for std in std_vals
    ]


def _calculate_cps_approx_for_std_range(detection_rate, mean_tau, std_vals):
    """
    Calculate CPS values using second-order approximation for a range of std values.

    Parameters
    ----------
    detection_rate : float
        Detection rate at equilibrium
    mean_tau : float
        Mean dead time in seconds
    std_vals : list
        List of standard deviation values

    Returns
    -------
    list
        List of CPS values using approximation
    """
    return [
        _calculate_cps_second_order_approx(detection_rate, mean_tau, std)
        for std in std_vals
    ]
