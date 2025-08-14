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
import utils as utl
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
        population_matrices: list,
        fission_values: list,
        selected_fission_values: Optional[list] = None,
        downsample_factor: Optional[int] = None,
        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot population dynamics from stochastic simulations for specific fission values.
    
    Parameters
    ----------
    population_matrices : list
        List of population matrices for each fission value
    fission_values : list or np.ndarray
        List of all fission values corresponding to the matrices
    selected_fission_values : list, optional
        List of specific fission values to plot. If None, plot all.
    downsample_factor : int, optional
        Factor to downsample data for plotting. If None, plot all points.
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    PlotStyle.setup_default_style()
    
    # Convert fission_values to list if it's a numpy array
    fission_values_list = list(fission_values)
    
    # Filter matrices and fission values based on selection
    if selected_fission_values is not None:
        # Find indices of selected fission values
        selected_indices = []
        selected_fission_filtered = []
        selected_pop_matrices = []
        
        for fission in selected_fission_values:
            if fission in fission_values_list:
                idx = fission_values_list.index(fission)
                selected_indices.append(idx)
                selected_fission_filtered.append(fission)
                selected_pop_matrices.append(population_matrices[idx])
            else:
                print(f"Warning: Fission value {fission} not found in available values")
        
        if not selected_pop_matrices:
            raise ValueError("No valid fission values selected")
        
        fission_values_to_plot = selected_fission_filtered
        population_matrices_to_plot = selected_pop_matrices
    else:
        fission_values_to_plot = fission_values_list
        population_matrices_to_plot = population_matrices
    
    # Create subplots for selected fission values
    num_fission = len(fission_values_to_plot)
    
    # Handle single subplot case properly
    if num_fission == 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, (num_fission + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_fission))
    
    for i, (pop_mat, fission) in enumerate(zip(population_matrices_to_plot, fission_values_to_plot)):
        if i < len(axes):
            ax = axes[i]
            
            # Get the single trajectory
            pop_trajectory = pop_mat[0, :]  # Only one trajectory
            
            # Apply downsampling if specified
            if downsample_factor is not None:
                pop_data = pop_trajectory[::downsample_factor]
                step_indices = np.arange(0, len(pop_trajectory), downsample_factor)
            else:
                pop_data = pop_trajectory
                step_indices = np.arange(len(pop_trajectory))
            
            print(f"Plotting fission {fission}: {len(pop_data)} steps")
            
            # Plot the population against step index
            ax.plot(step_indices, pop_data, color=colors[i], linewidth=1, 
                   label=f'Fission = {fission}')
            
            ax.set_xlabel('Simulation Step', fontsize=12)
            ax.set_ylabel('Population', fontsize=12)
            ax.set_title(f'Fission = {fission}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
    
    # Hide unused subplots
    for i in range(num_fission, len(axes)):
        axes[i].set_visible(False)
    
    # Update title to reflect selection
    if selected_fission_values is not None:
        title = f'Stochastic Population Dynamics - Selected Fission Values: {selected_fission_values}'
    else:
        title = 'Stochastic Population Dynamics - All Fission Values'
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_euler_maruyama_population_dynamics(
        population_matrices: list,                                  
        fission_values: list,                                
        selected_fission_values: Optional[list] = None,                              
        dead_time_type: str = 'basic',  
        downsample_factor: Optional[int] = None,                 
        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot population dynamics from Euler-Maruyama simulations for specific fission values.
    
    Parameters
    ----------
    population_matrices : list
        List of population matrices for each fission value
    fission_values : list or np.ndarray
        List of all fission values corresponding to the matrices
    selected_fission_values : list, optional
        List of specific fission values to plot. If None, plot all.
    dead_time_type : str, optional
        Type of dead time ('basic', 'const', 'exp') for title
    downsample_factor : int, optional
        Factor to downsample data for plotting. If None, plot all points.
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    PlotStyle.setup_default_style()
    
    # Convert fission_values to list if it's a numpy array
    fission_values_list = list(fission_values)
    
    # Filter matrices and fission values based on selection
    if selected_fission_values is not None:
        # Find indices of selected fission values
        selected_indices = []
        selected_fission_filtered = []
        selected_pop_matrices = []
        
        for fission in selected_fission_values:
            if fission in fission_values_list:
                idx = fission_values_list.index(fission)
                selected_indices.append(idx)
                selected_fission_filtered.append(fission)
                selected_pop_matrices.append(population_matrices[idx])
            else:
                print(f"Warning: Fission value {fission} not found in available values")
        
        if not selected_pop_matrices:
            raise ValueError("No valid fission values selected")
        
        fission_values_to_plot = selected_fission_filtered
        population_matrices_to_plot = selected_pop_matrices
    else:
        fission_values_to_plot = fission_values_list
        population_matrices_to_plot = population_matrices
    
    # Create subplots for selected fission values
    num_fission = len(fission_values_to_plot)
    
    # Handle single subplot case properly
    if num_fission == 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, (num_fission + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
    
    colors = plt.cm.plasma(np.linspace(0, 1, num_fission))
    
    for i, (pop_mat, fission) in enumerate(zip(population_matrices_to_plot, fission_values_to_plot)):
        if i < len(axes):
            ax = axes[i]
            
            # Get the population data (Euler-Maruyama has 1D arrays)
            pop_data = pop_mat
            
            # Apply downsampling if specified
            if downsample_factor is not None:
                pop_data = pop_data[::downsample_factor]
                step_indices = np.arange(0, len(pop_mat), downsample_factor)
            else:
                step_indices = np.arange(len(pop_mat))
            
            print(f"Plotting Euler-Maruyama fission {fission}: {len(pop_data)} steps")
            
            # Plot the population against step index
            ax.plot(step_indices, pop_data, color=colors[i], linewidth=2, 
                   label=f'Fission = {fission}')
            
            ax.set_xlabel('Simulation Step', fontsize=12)
            ax.set_ylabel('Population', fontsize=12)
            ax.set_title(f'Euler-Maruyama Fission = {fission}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
    
    # Hide unused subplots
    for i in range(num_fission, len(axes)):
        axes[i].set_visible(False)
    
    # Update title to reflect selection and dead time type
    dead_time_label = {
        'basic': 'Without Dead Time',
        'const': 'With Constant Dead Time',
        'exp': 'With Exponential Dead Time'
    }.get(dead_time_type, dead_time_type)
    
    if selected_fission_values is not None:
        title = f'Euler-Maruyama Population Dynamics - {dead_time_label}\nSelected Fission Values: {selected_fission_values}'
    else:
        title = f'Euler-Maruyama Population Dynamics - {dead_time_label}\nAll Fission Values'
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# 4. COUNT RATE PLOTS
# =============================================================================

def plot_stochastic_basic_cps(alpha_inv_vec : np.ndarray, cps : np.ndarray,
                          save_path : Optional[str] = None) -> plt.Figure:
    """
    Plot stochastic simulation count rates without dead time.
    
    Parameters
    ----------
    alpha_inv_vec : np.ndarray
        Array of inverse Rossi-alpha values
    cps : np.ndarray
        Array of count rates (counts per second)
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    PlotStyle.setup_default_style()
    
    plt.plot(alpha_inv_vec, cps, label = "Stochastic simulation",
             color = 'blue', markersize = 6, linewidth = 2)
    plt.xticks(alpha_inv_vec, rotation = 45, fontsize = 10)
    plt.title("Stochastic Count Rates without Dead Time", fontsize = 16,
              fontweight = 'bold')
    plt.xlabel("1/alpha (inverse Rossi-alpha)", fontsize = 14)
    plt.ylabel("Count per Second (CPS)", fontsize = 14)
    plt.grid(True, alpha = 0.3)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    
    return plt.gcf() 
 
   
def plot_stochastic_const_dead_time_cps(alpha_inv_vec : np.ndarray,
                                    cps_tau : np.ndarray,
                                    tau: float,
                                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot stochastic simulation count rates with constant dead time.
    
    Parameters
    ----------
    alpha_inv_vec : np.ndarray
        Array of inverse Rossi-alpha values
    cps_tau : np.ndarray
        Array of count rates with dead time effects
    tau : float
        Dead time value
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    
    PlotStyle.setup_default_style()
    plt.plot(alpha_inv_vec, cps_tau, 's-', label =
             "Stochastic simulation with constant dead time", color = 'red')
    plt.xticks(alpha_inv_vec, rotation = 45, fontsize = 10)
    plt.title(f"Stochastic Count Rates with Constant Dead Time tau = {tau:.1e}")
    plt.xlabel("1/alpha (inverse Rossi-alpha)", fontsize = 14)
    plt.ylabel("Count per Second (CPS)", fontsize = 14)
    plt.grid(True, alpha = 0.3)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    
    return plt.gcf() 

# =============================================================================
# EULER-MARUYAMA PLOTTING FUNCTIONS
# =============================================================================

def plot_euler_maruyama_basic_cps(alpha_inv_vec : np.ndarray, cps_em : np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Euler-Maruyama count rates without dead time.
    
    Parameters
    ----------
    alpha_inv_vec : np.ndarray
        Array of inverse Rossi-alpha values
    cps_em : np.ndarray
        Array of Euler-Maruyama count rates
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    PlotStyle.setup_default_style()
    
    plt.plot(alpha_inv_vec, cps_em, 'o-',
             label = "Euler-Maruyama without Dead Time",
             color = 'green', markersize = 8, linewidth = 2)
    plt.xticks(alpha_inv_vec, rotation = 45, fontsize = 10)
    plt.title("Euler-Maruyama Count Rates without Dead Time",
              fontsize = 16, fontweight = 'bold')
    plt.xlabel("1/alpha (inverse Rossi-alpha)", fontsize = 14)
    plt.ylabel("Count per Second (CPS)", fontsize = 14)
    plt.grid(True, alpha = 0.3)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    
    return plt.gcf() 


def plot_euler_maruyama_const_dead_time_cps(alpha_inv_vec: np.ndarray,
                                        cps_em_tau: np.ndarray, 
                                       tau: float,
                                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Euler-Maruyama count rates with constant dead time.
    
    Parameters
    ----------
    alpha_inv_vec : np.ndarray
        Array of inverse Rossi-alpha values
    cps_em_tau : np.ndarray
        Array of Euler-Maruyama count rates with dead time
    tau : float
        Dead time value
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    PlotStyle.setup_default_style()
    
    plt.plot(alpha_inv_vec, cps_em_tau, 's-', 
             label="Euler-Maruyama with constant dead time",
             color='purple', markersize=8, linewidth=2)
    plt.xticks(alpha_inv_vec, rotation=45, fontsize=10)
    plt.title(f"Euler-Maruyama Count Rates with Constant Dead Time τ = {tau:.1e}", 
              fontsize=16, fontweight='bold')
    plt.xlabel("1/alpha (inverse Rossi-alpha)", fontsize=14)
    plt.ylabel("Count per Second (CPS)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf() 


def plot_euler_maruyama_exp_dead_time_cps(alpha_inv_vec: np.ndarray,
                                      cps_em_tau: np.ndarray, 
                                     tau: float,
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Euler-Maruyama count rates with exponential dead time.
    
    Parameters
    ----------
    alpha_inv_vec : np.ndarray
        Array of inverse Rossi-alpha values
    cps_em_tau : np.ndarray
        Array of Euler-Maruyama count rates with exponential dead time
    tau : float
        Mean dead time value
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    PlotStyle.setup_default_style()
    
    plt.plot(alpha_inv_vec, cps_em_tau, '^-', 
             label="Euler-Maruyama with Exponential dead time",
             color='orange', markersize=8, linewidth=2)
    plt.xticks(alpha_inv_vec, rotation=45, fontsize=10)
    plt.title(f"Euler-Maruyama Count Rates with Exponential Dead Time τ = {tau:.1e}", 
              fontsize=16, fontweight='bold')
    plt.xlabel("1/alpha (inverse Rossi-alpha)", fontsize=14)
    plt.ylabel("Count per Second (CPS)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf() 

# =============================================================================
# 5. COMPARISON PLOTS
# =============================================================================

def plot_cps_comparison(stochastic_cps: list, 
                       em_cps: list, 
                       taylor_cps: list,
                       config,
                       dead_time_type: str = 'basic',
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comparison plot of CPS values from stochastic, Euler-Maruyama, and Taylor simulations.
    
    Parameters
    ----------
    stochastic_cps : list
        List of CPS values from stochastic simulations
    em_cps : list
        List of CPS values from Euler-Maruyama simulations
    taylor_cps : list
        List of CPS values from Taylor method simulations
    config : SimulationConfig
        Configuration object
    dead_time_type : str
        Type of dead time ('basic', 'const', 'exp', 'without')
    save_path : Optional[str]
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The generated comparison plot
    """
    
    PlotStyle.setup_default_style()
    
    # Change to 2x2 subplot layout to accommodate three methods
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Add the main title
    fig.suptitle("Count Rate Comparison: Stochastic vs Euler-Maruyama vs Taylor",
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Prepare data for plotting
    # For stochastic: use mean values if arrays
    if isinstance(stochastic_cps[0], np.ndarray):
        stochastic_means = np.array([np.mean(cps) for cps in stochastic_cps])
        stochastic_stds = np.array([np.std(cps) for cps in stochastic_cps])
    else:
        stochastic_means = stochastic_cps
        stochastic_stds = None
    
    # For Euler-Maruyama: should be single values
    em_means = em_cps
    
    # For Taylor: should be single values
    taylor_means = taylor_cps
    
    # Set title based on dead time type
    dead_time_titles = {
        'basic': 'Basic Dead Time',
        'const': 'Constant Dead Time', 
        'uniform' : 'Uniform Dead Time',
        'exp': 'Exponential Dead Time',
        'without': 'Without Dead Time'
    }
    
    title_suffix = dead_time_titles.get(dead_time_type, dead_time_type.title())
    
    # Plot 1: Direct comparison of all three methods
    ax1.plot(config.alpha_inv_vec, stochastic_means, 'o-', color='blue', 
             linewidth=2, markersize=6, label='Stochastic Simulation')
    ax1.plot(config.alpha_inv_vec, em_means, 's-', color='red', 
             linewidth=2, markersize=6, label='Euler-Maruyama')
    ax1.plot(config.alpha_inv_vec, taylor_means, '^-', color='green', 
             linewidth=2, markersize=6, label='Taylor Method')  # ← ADD THIS
    
    # Add error bars for stochastic if available
    if stochastic_stds is not None:
        ax1.fill_between(config.alpha_inv_vec, 
                        stochastic_means - stochastic_stds,
                        stochastic_means + stochastic_stds, 
                        alpha=0.3, color='blue', label='Stochastic ±1σ')
    
    ax1.set_xlabel('Inverse Rossi-alpha (1/α)', fontsize=12)
    ax1.set_ylabel('Counts Per Second (CPS)', fontsize=12)
    ax1.set_title(f'CPS Comparison - {title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(config.alpha_inv_vec)
    ax1.set_xticklabels([f'{val:.3f}' for val in config.alpha_inv_vec], rotation=45)
    
    # Plot 2: EM vs Stochastic Relative Difference
    with np.errstate(divide='ignore', invalid='ignore'):
        em_relative_diff = (np.abs(stochastic_means - em_means) / 
                           np.where(em_means != 0, em_means, np.nan))
    
    em_relative_diff_percent = em_relative_diff * 100
    
    ax2.plot(config.alpha_inv_vec, em_relative_diff_percent, 'o-', color='purple', 
             linewidth=2, markersize=6, label='EM vs Stochastic')
    ax2.set_xlabel('Inverse Rossi-alpha (1/α)', fontsize=12)
    ax2.set_ylabel('Relative Difference (%)', fontsize=12)
    ax2.set_title(f'EM vs Stochastic Difference - {title_suffix}', fontsize=14,
                  fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(config.alpha_inv_vec)
    ax2.set_xticklabels([f'{val:.3f}' for val in config.alpha_inv_vec], rotation=45)
    
    # Plot 3: Taylor vs Stochastic Relative Difference
    with np.errstate(divide='ignore', invalid='ignore'):
        taylor_relative_diff = (np.abs(stochastic_means - taylor_means) / 
                               np.where(taylor_means != 0, taylor_means, np.nan))
    
    taylor_relative_diff_percent = taylor_relative_diff * 100
    
    ax3.plot(config.alpha_inv_vec, taylor_relative_diff_percent, 'o-', color='orange', 
             linewidth=2, markersize=6, label='Taylor vs Stochastic')
    ax3.set_xlabel('Inverse Rossi-alpha (1/α)', fontsize=12)
    ax3.set_ylabel('Relative Difference (%)', fontsize=12)
    ax3.set_title(f'Taylor vs Stochastic Difference - {title_suffix}', fontsize=14,
                  fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(config.alpha_inv_vec)
    ax3.set_xticklabels([f'{val:.3f}' for val in config.alpha_inv_vec], rotation=45)
    
    # Plot 4: Taylor vs EM Relative Difference
    with np.errstate(divide='ignore', invalid='ignore'):
        taylor_em_diff = (np.abs(taylor_means - em_means) / 
                          np.where(em_means != 0, em_means, np.nan))
    
    taylor_em_diff_percent = taylor_em_diff * 100
    
    ax4.plot(config.alpha_inv_vec, taylor_em_diff_percent, 'o-', color='brown', 
             linewidth=2, markersize=6, label='Taylor vs EM')
    ax4.set_xlabel('Inverse Rossi-alpha (1/α)', fontsize=12)
    ax4.set_ylabel('Relative Difference (%)', fontsize=12)
    ax4.set_title(f'Taylor vs EM Difference - {title_suffix}', fontsize=14,
                  fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(config.alpha_inv_vec)
    ax4.set_xticklabels([f'{val:.3f}' for val in config.alpha_inv_vec], rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Auto-generate filename if none provided
        auto_filename = utl.generate_filename('plot', 'CPS_Comparison_3Methods',
                                              dead_time_type, config.fission_vec[0], '.png')
        default_save_path = f'plots/{auto_filename}'
        plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot auto-saved to: {default_save_path}")
        
    return fig

