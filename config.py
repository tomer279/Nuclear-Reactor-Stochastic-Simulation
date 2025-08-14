""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Configuration and parameter management for nuclear reactor simulations.

This module contains the SimulationConfig class that manages all simulation
parameters, derived parameters, and configuration settings.
"""

import numpy as np
from utils import mean

class SimulationConfig:
    """
    Configuration class for simulation parameters.
    
    This class manages all parameters needed for nuclear reactor simulations,
    including physical parameters, time parameters, dead time settings,
    and derived parameters.
    """
    
    def __init__(self):
        
        # =============================================================================
        # PHYSICAL PARAMETERS
        # =============================================================================
        
        # Fission probability distribution
        self.p_v = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        
        # Rate constants
        self.absorb = 7.0       # Absorption rate
        self.source = 1000.0    # Source rate
        self.detect = 10.0      # Detection rate
        
        # =============================================================================
        # TIME PARAMETERS
        # =============================================================================
        
        self.t_0 = 0.0                      # Initial time
        self.t_end = 0.1                    # End time
        self.steps = 100_000_000            # Number of simulation steps
        self.grid_points = 10_000_000      # Number of grid points for EM
        
        # =============================================================================
        # DEAD TIME PARAMETERS
        # =============================================================================
        
        # Dead time parameters
        self.mean_tau = 1e-6    # Mean dead time
        self.tau_distribution = 'uniform'  # Distribution type for random dead time
        
        # Dead time distribution parameters
        self.tau_params = {
            'normal': {
                'loc': 1e-6,
                'scale': 0.5 * 1e-7
            },
            'uniform': {
                'low': 1e-6 - 2 * np.sqrt(3) * 1e-7,
                'high': 1e-6 + 2 * np.sqrt(3) * 1e-7
            },
            'exponential': {
                'scale': 1e-6
            },
            'gamma': {
                'shape': 25,
                'scale': 0.04e-06
            }
        }
        
        # =============================================================================
        # FISSION PARAMETERS
        # =============================================================================
        
        # Fission values to simulate
        self.fission_vec = np.array([33.94, 33.95, 33.96,
                                     33.97, 33.98, 33.982, 
                                     33.984, 33.986, 33.988, 33.99, 33.992])
        self.fission_vec = np.round(self.fission_vec , decimals = 4)
        
        # =============================================================================
        # DERIVED PARAMETERS (calculated automatically)
        # =============================================================================
        
        self._calculate_derived_parameters()
        
        # =============================================================================
        # MAPPING DICTIONARIES
        # =============================================================================
        
        self.fission_to_alpha_inv = dict(zip(
            self.fission_vec.tolist(), 
            self.alpha_inv_vec.tolist()))
        
        # =============================================================================
        # SIMULATION CONTROL PARAMETERS
        # =============================================================================
        
        # File naming and saving
        self.save_results = True
        self.output_prefix = 'f'
        self.output_directory = './'
        
        # Progress tracking
        self.show_progress = True
        self.progress_interval = 10  # Show progress every 10%
    
    
    def _calculate_derived_parameters(self):
        """ Calculate derived parameters from base parameters """
        
        # Expected value of fission probability distribution
        self.vbar = mean(self.p_v)
        
        # Total reaction rate for each fission value
        self.lam_vec = self.fission_vec + self.absorb + self.detect
        
        # Rossi-alpha coefficient for each fission value
        self.alpha_vec = self.lam_vec - self.fission_vec * self.vbar
        
        # Inverse Rossi-alpha
        self.alpha_inv_vec = 1 / self.alpha_vec
        
        # Equilibrium population for each fission value
        self.equil = self.source * self.alpha_inv_vec
        
        
    def validate_parameters(self):
        """
        Validate that all parameters are within acceptable ranges.
        
        Returns
        -------
        bool
            True if all parameters are valid
            
        Raises
        ------
        ValueError
            If any parameters are invalid
        """
        # Check physical parameters
        if any(x < 0 for x in [self.absorb, self.source, self.detect]):
            raise ValueError("Rate constants must be non-negative")
        
        if np.any(self.p_v < 0) or not np.isclose(np.sum(self.p_v), 1.0):
            raise ValueError("p_v must be a valid probability distribution")
        
        # Check time parameters
        if self.t_end <= self.t_0:
            raise ValueError("t_end must be greater than t_0")
        
        if self.steps < 1 or self.grid_points < 1:
            raise ValueError("steps and grid_points must be positive")
        
        # Check dead time parameters
        if self.mean_tau <= 0:
            raise ValueError("mean_tau must be positive")
        
        if self.tau_distribution not in self.tau_params:
            raise ValueError(f"Invalid tau_distribution: {self.tau_distribution}")
        
        # Check fission parameters
        if np.any(self.fission_vec <= 0):
            raise ValueError("Fission values must be positive")
        
        return True
    
    
    def get_simulation_summary(self):
        """
        Get a summary of current simulation configuration.
        
        Returns
        -------
        str
            Formatted summary of configuration
        """
        summary = f"""
Simulation Configuration Summary:
================================

Physical Parameters:
- Fission probability distribution: {self.p_v}
- Absorption rate: {self.absorb}
- Source rate: {self.source}
- Detection rate: {self.detect}
- Expected fission yield (vbar): {self.vbar:.4f}

Time Parameters:
- Initial time: {self.t_0}
- End time: {self.t_end}
- Simulation steps: {self.steps:,}
- Grid points: {self.grid_points:,}

Dead Time Parameters:
- Mean dead time: {self.mean_tau:.2e}
- Distribution: {self.tau_distribution}

Fission Values:
- Number of values: {len(self.fission_vec)}
- Range: {self.fission_vec.min():.3f} - {self.fission_vec.max():.3f}
- Values: {self.fission_vec}

Derived Parameters:
- Alpha range: {self.alpha_vec.min():.4f} - {self.alpha_vec.max():.4f}
- Equilibrium population range: {self.equil.min():.2f} - {self.equil.max():.2f}
        """
        return summary
        
    
    def save_config(self, filename):
        """
        Save configuration to file.
        
        Parameters
        ----------
        filename : str
            Path to save configuration
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        config_dict = {
            'p_v': self.p_v.tolist(),
            'absorb': self.absorb,
            'source': self.source,
            'detect': self.detect,
            't_0': self.t_0,
            't_end': self.t_end,
            'steps': self.steps,
            'grid_points': self.grid_points,
            'mean_tau': self.mean_tau,
            'tau_distribution': self.tau_distribution,
            'tau_params': self.tau_params,
            'fission_vec': self.fission_vec.tolist(),
            'save_results': self.save_results,
            'output_prefix': self.output_prefix,
            'output_directory': self.output_directory,
            'show_progress': self.show_progress,
            'progress_interval': self.progress_interval
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
    
    @classmethod
    def load_config(cls, filename):
        """
        Load configuration from file.
        
        Parameters
        ----------
        filename : str
            Path to configuration file
            
        Returns
        -------
        SimulationConfig
            Loaded configuration object
        """
        import json
        
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        # Create new instance
        config = cls()
        
        # Update parameters
        config.p_v = np.array(config_dict['p_v'])
        config.absorb = config_dict['absorb']
        config.source = config_dict['source']
        config.detect = config_dict['detect']
        config.t_0 = config_dict['t_0']
        config.t_end = config_dict['t_end']
        config.steps = config_dict['steps']
        config.grid_points = config_dict['grid_points']
        config.mean_tau = config_dict['mean_tau']
        config.tau_distribution = config_dict['tau_distribution']
        config.tau_params = config_dict['tau_params']
        config.fission_vec = np.array(config_dict['fission_vec'])
        config.save_results = config_dict['save_results']
        config.output_prefix = config_dict['output_prefix']
        config.output_directory = config_dict['output_directory']
        config.show_progress = config_dict['show_progress']
        config.progress_interval = config_dict['progress_interval']
        
        # Recalculate derived parameters
        config._calculate_derived_parameters()
        
        return config
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        