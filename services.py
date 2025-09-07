"""
Service Classes for Nuclear Reactor Stochastic Simulation Dashboard

This file contains service classes that separate business logic from UI concerns,
making the code more maintainable, testable, and organized.

PROBLEMS ADDRESSED:
==================
1. Mixed Responsibilities: Functions mixing business logic, data access, and UI concerns
2. Complex Business Logic Scattered: Core logic embedded in UI functions
3. Direct External Module Dependencies: UI functions directly importing external modules
4. Repetitive Configuration Creation: Multiple places creating SimulationConfig objects
5. Complex Error Handling Mixed with Business Logic: Error handling scattered throughout
6. Hard-to-Test Functions: Functions tightly coupled to Streamlit and external modules

SERVICE CLASSES:
===============
- SimulationService: Handles simulation orchestration and execution
- DeadTimeAnalysisService: Manages dead time analysis calculations
"""

import numpy as np
from typing import Dict, Optional, Tuple

import stochastic_simulation as ss
from config import SimulationConfig
from data_management import DataManager
import count_rates as cr
from models import (SimulationResults, SimulationParameters, DeadTimeConfig, DeadTimeAnalysis)

# =============================================================================
# SIMULATION SERVICE
# =============================================================================

class SimulationService:
    """Service for simulation orchestration and execution"""
    
    def __init__(self, data_manager: DataManager = None):
        self.data_manager = data_manager or DataManager()
    
    def run_single_simulation(self, config: SimulationConfig, steps: int = 10000, 
                            initial_population: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run a single stochastic simulation"""
        # Extract parameters
        p_v = config.p_v
        fission = config.fission_vec[0]  # Use first fission rate
        absorb = config.absorb
        source = config.source
        detect = config.detect
        t_0 = config.t_0
        
        if initial_population is None:
            # Use equilibrium value
            n_0 = np.array([config.equil[0]])
        else:
            n_0 = np.array([initial_population])
        
        # Run simulation
        time_matrix, population_matrix, detection_matrix = ss.pop_dyn_mat(
            p_v=p_v,
            fission=fission,
            absorb=absorb,
            source=source,
            detect=detect,
            n_0=n_0,
            t_0=t_0,
            steps=steps,
            prefix="streamlit_demo",
            data_manager=self.data_manager
        )
        
        return time_matrix, population_matrix, detection_matrix
    
    def run_multiple_simulations(self, params: SimulationParameters) -> Dict[float, SimulationResults]:
        """Run simulations for all selected fission rates"""
        results = {}
        
        for fission_rate in params.fission_rates:
            # Create config for this fission rate
            config = self._create_config(params, fission_rate)
            
            # Run simulation
            time_matrix, population_matrix, detection_matrix = self.run_single_simulation(
                config, params.simulation_steps, params.initial_population
            )
            
            results[fission_rate] = SimulationResults(
                time_matrix = time_matrix,
                population_matrix = population_matrix,
                detection_matrix = detection_matrix,
                fission_rate = fission_rate
            )
        
        return results
    
    def _create_config(self, params: SimulationParameters, fission_rate: float) -> SimulationConfig:
        """Create SimulationConfig for a specific fission rate"""
        config = SimulationConfig()
        config.detect = params.detection_rate
        config.absorb = params.absorption_rate
        config.source = params.source_rate
        config.fission_vec = np.array([fission_rate])
        config._calculate_derived_parameters()
        return config
    
# =============================================================================
# DEAD TIME ANALYSIS SERVICE
# =============================================================================

class DeadTimeAnalysisService:
    """Service for dead time analysis calculations"""
    
    def run_analysis(self, results_dict: Dict[float, SimulationResults], 
                    params: SimulationParameters, mean_dead_time: float,
                    dead_time_std_percent: float, dead_time_type: str, 
                    analysis_name: str) -> DeadTimeAnalysis:
        """Run complete dead time analysis and return DeadTimeAnalysis object"""
        # Calculate alpha inverse values for each fission rate
        alpha_inv_results = {}
        
        for fission_rate in sorted(params.fission_rates):
            alpha_inv_results[fission_rate] = self._calculate_cps_for_fission_rate(
                results_dict, params, fission_rate, mean_dead_time,
                dead_time_std_percent, dead_time_type
            )
        
        # Create DeadTimeAnalysis object
        dead_time_config = DeadTimeConfig(
            mean_dead_time=mean_dead_time,
            std_percent=dead_time_std_percent,
            distribution_type=dead_time_type,
            analysis_name=analysis_name
        )
        
        return DeadTimeAnalysis(
            config=dead_time_config,
            cps_results=alpha_inv_results
        )
    
    def add_theoretical_cps(self, analysis: DeadTimeAnalysis, 
                           params: SimulationParameters) -> DeadTimeAnalysis:
        """Add theoretical CPS to an existing analysis"""
        theoretical_cps_results = {}
        
        for fission_rate in sorted(analysis.cps_results.keys()):
            # Get parameters for this fission rate
            config = self._create_config(params, fission_rate)
            
            # Calculate theoretical CPS using count_rates module
            mean_tau_s = analysis.config.get_tau_seconds()
            std_tau_s = analysis.config.get_std_seconds()
            
            theoretical_cps = cr.calculate_theoretical_cps(
                analysis.config.distribution_type, mean_tau_s, std_tau_s,
                config.detect, config.equil[0]
            )
            
            theoretical_cps_results[fission_rate] = theoretical_cps
        
        # Update the analysis with theoretical CPS
        analysis.theoretical_cps = theoretical_cps_results
        return analysis
    
    def _calculate_cps_for_fission_rate(self, results_dict: Dict[float, SimulationResults], 
                                      params: SimulationParameters, fission_rate: float,
                                      mean_dead_time: float, dead_time_std_percent: float,
                                      dead_time_type: str) -> Dict[str, float]:
        """Calculate CPS and alpha inverse for a single fission rate"""
        result = results_dict[fission_rate]
        time_matrix = result.time_matrix
        detection_matrix = result.detection_matrix
        
        # Calculate alpha inverse using config
        config = self._create_config(params, fission_rate)
        alpha_inv = config.alpha_inv_vec[0]
        
        # Calculate distribution parameters based on mean and std
        mean_tau_s = mean_dead_time * 1e-6  # Convert μs to seconds
        std_tau_s = (dead_time_std_percent / 100.0) * mean_tau_s  # Convert percentage to std
        
        cps_values = self._calculate_cps_values(time_matrix, detection_matrix, 
                                              dead_time_type, mean_tau_s, std_tau_s)
        
        return {
            'cps': np.mean(cps_values),
            'alpha_inv': alpha_inv
        }
    
    def _calculate_cps_values(self, time_matrix: np.ndarray, detection_matrix: np.ndarray,
                            dead_time_type: str, mean_tau_s: float, std_tau_s: float) -> np.ndarray:
        """Calculate CPS values based on dead time type"""
        if dead_time_type.lower() == 'constant':
            return cr.count_per_second_const_dead_time(
                time_matrix, detection_matrix, tau=mean_tau_s
            )
        
        elif dead_time_type.lower() == 'normal':
            return cr.count_per_second_rand_dead_time(
                time_matrix, detection_matrix,
                distribution='normal',
                loc=mean_tau_s,
                scale=std_tau_s
            )
        
        elif dead_time_type.lower() == 'uniform':
            # Uniform distribution: low and high (maintaining std)
            low = mean_tau_s - np.sqrt(3) * std_tau_s
            high = mean_tau_s + np.sqrt(3) * std_tau_s
            return cr.count_per_second_rand_dead_time(
                time_matrix, detection_matrix,
                distribution='uniform',
                low=low,
                high=high
            )
        
        elif dead_time_type.lower() == 'gamma':
            # Gamma distribution: shape = (mean/std)^2, scale = std^2/mean
            shape = (mean_tau_s / std_tau_s) ** 2
            scale = (std_tau_s ** 2) / mean_tau_s
            return cr.count_per_second_rand_dead_time(
                time_matrix, detection_matrix,
                distribution='gamma',
                shape=shape,
                scale=scale
            )
        
        else:
            raise ValueError(f"Unknown dead time type: {dead_time_type}")
    
    def _create_config(self, params: SimulationParameters, fission_rate: float) -> SimulationConfig:
        """Create SimulationConfig for a specific fission rate"""
        config = SimulationConfig()
        config.detect = params.detection_rate
        config.absorb = params.absorption_rate
        config.source = params.source_rate
        config.fission_vec = np.array([fission_rate])
        config._calculate_derived_parameters()
        return config