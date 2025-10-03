"""
Service Classes for Nuclear Reactor Stochastic Simulation Dashboard

This file contains service classes that separate business logic
from UI concerns, making the code more maintainable, testable, and organized.

SERVICE CLASSES:
===============
- SimulationService: Handles simulation orchestration and execution
- DeadTimeAnalysisService: Manages dead time analysis calculations
"""

from typing import Dict, Optional, Tuple
import numpy as np
from stochastic_simulation import StochasticSimulator, SimulationParameters
from core_parameters import (
    PhysicalParameters, TimeParameters,
    RateConstants, FissionDistribution,
    FissionParameters)
from simulation_setting import SimulationControl
from data_management import DataManager
import count_rates as cr
from models import (
    SimulationResults,
    SimulationParameters as ModelsSimulationParameters,
    DeadTimeConfig,
    DeadTimeAnalysis)

# =============================================================================
# SIMULATION SERVICE
# =============================================================================


class SimulationService:
    """
    Service for simulation orchestration and execution.

    This class handles the complete simulation workflow including parameter
    setup, simulation execution, and result management. It provides a clean
    interface between the UI layer and the core simulation engine.

    Attributes
    ----------
    data_manager : DataManager
        Data manager for saving and loading simulation results
    _fission_params_cache : dict
        Cache for FissionParameters objects to avoid recalculation

    Public Methods
    --------------
    run_single_simulation(params, fission_rate, steps, initial_population)
        Run a single stochastic simulation for a specific fission rate
    run_multiple_simulations(params)
        Run simulations for all selected fission rates

    Private Methods
    --------------
    _get_fission_parameters(fission_rates, params)
        Get or create FissionParameters with caching
    _get_initial_population(initial_population, fission_rate, params)
        Determine initial population for simulation
    _create_simulation_parameters(params, fission_rate, steps, n_0)
        Create simulation parameters for StochasticSimulator
    """

    def __init__(self, data_manager: DataManager = None):
        """
        Initialize the simulation service.

        Parameters
        ----------
        data_manager : DataManager, optional
            Data manager for file operations.
            If None, creates a new instance.
        """
        self.data_manager = data_manager or DataManager()
        self._fission_params_cache = {}

    def run_single_simulation(
            self,
            params: ModelsSimulationParameters,
            fission_rate: float,
            steps: int = 10000,
            initial_population: Optional[int] = None) -> (
                Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """
        Run a single stochastic simulation for a specific fission rate.

        This method orchestrates the complete simulation process including
        parameter setup, simulation execution, and result return.

        Parameters
        ----------
        params : ModelsSimulationParameters
            Simulation parameters containing rate constants and settings
        fission_rate : float
            Fission rate constant for this simulation (s⁻¹)
        steps : int, optional
            Number of simulation steps. Default is 10000.
        initial_population : int, optional
            Initial neutron population. If None, uses equilibrium value.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Returns (time_matrix, population_matrix, detection_matrix)
            where each matrix has shape (num_trajectories, num_steps)

        Raises
        ------
        ValueError
            If simulation parameters are invalid
        RuntimeError
            If simulation execution fails
        """

        # Determine initial population
        n_0 = self._get_initial_population(
            initial_population, fission_rate, params)

        # Create simulation parameters
        sim_params = self._create_simulation_parameters(
            params, fission_rate, steps, n_0)

        # Create and run simulator
        simulator = StochasticSimulator(sim_params)
        time_matrix, population_matrix, detection_matrix = (
            simulator.run_simulation(self.data_manager)
        )

        return time_matrix, population_matrix, detection_matrix

    def run_multiple_simulations(
            self,
            params: ModelsSimulationParameters) -> (
            Dict[float, SimulationResults]):
        """
        Run simulations for all selected fission rates.

        This method executes simulations for each fission rate in the
        parameters and returns a dictionary of results keyed by fission rate.

        Parameters
        ----------
        params : ModelsSimulationParameters
            Simulation parameters containing fission rates and settings

        Returns
        -------
        Dict[float, SimulationResults]
            Dictionary mapping fission rates to simulation results

        Raises
        ------
        ValueError
            If no fission rates are specified or parameters are invalid
        RuntimeError
            If simulation execution fails for any fission rate
        """
        results = {}

        for fission_rate in params.fission_rates:
            # Determine initial population for this fission rate
            if params.use_equilibrium:
                fission_params = self._get_fission_parameters(
                    [fission_rate], params)
                equil_value = fission_params.equil[0]
                initial_pop = equil_value
            else:
                initial_pop = params.initial_population

            # Run simulation
            time_matrix, population_matrix, detection_matrix = (
                self.run_single_simulation(
                    params, fission_rate, params.simulation_steps, initial_pop)
            )

            results[fission_rate] = SimulationResults(
                time_matrix=time_matrix,
                population_matrix=population_matrix,
                detection_matrix=detection_matrix,
                fission_rate=fission_rate
            )

        return results

    def _get_initial_population(
            self,
            initial_population: Optional[int],
            fission_rate: float,
            params: ModelsSimulationParameters) -> np.ndarray:
        """Get initial population array for simulation."""
        if initial_population is None:
            # Calculate equilibrium value for this fission rate
            fission_params = self._get_fission_parameters(
                [fission_rate], params)
            equil_value = fission_params.equil[0]
            return np.array([equil_value])

        return np.array([initial_population])

    def _create_simulation_parameters(
            self,
            params: ModelsSimulationParameters,
            fission_rate: float,
            steps: int,
            n_0: np.ndarray) -> SimulationParameters:
        """Create simulation parameters for StochasticSimulator."""
        # Create parameter objects for StochasticSimulator
        rate_constants = RateConstants(
            absorb=params.absorption_rate,
            source=params.source_rate,
            detect=params.detection_rate,
            fission=fission_rate
        )

        # Create fission distribution (using default p_v for now)
        fission_distribution = FissionDistribution()

        physical_params = PhysicalParameters(
            rate_constants=rate_constants,
            fission_distribution=fission_distribution
        )

        time_params = TimeParameters(
            t_0=0.0,
            t_end=0.1,  # Default end time
            steps=steps
        )

        control_params = SimulationControl()

        # Create simulation parameters
        return SimulationParameters(
            physical_params=physical_params,
            time_params=time_params,
            n_0=n_0,
            control_params=control_params
        )

    def _get_fission_parameters(
            self,
            fission_rates: list,
            params: ModelsSimulationParameters) -> FissionParameters:
        """
        Get or create FissionParameters for the given rates and parameters.
        """
        # Create cache key
        cache_key = (tuple(fission_rates), params.detection_rate,
                     params.absorption_rate, params.source_rate)

        if cache_key not in self._fission_params_cache:
            # Create PhysicalParameters with the rate constants
            rate_constants = RateConstants(
                absorb=params.absorption_rate,
                source=params.source_rate,
                detect=params.detection_rate
            )
            physical_params = PhysicalParameters(
                rate_constants=rate_constants,
                fission_distribution=FissionDistribution()
            )

            # Create FissionParameters with the physical parameters
            fission_params = FissionParameters(
                fission_vec=np.array(fission_rates),
                physical_params=physical_params
            )

            self._fission_params_cache[cache_key] = fission_params

        return self._fission_params_cache[cache_key]

# =============================================================================
# DEAD TIME ANALYSIS SERVICE
# =============================================================================


class DeadTimeAnalysisService:
    """
    Service for dead time analysis calculations.

    This class handles dead time analysis including CPS calculations,
    theoretical CPS computation, and result management. It provides
    comprehensive analysis capabilities for different dead time distributions.

    Attributes
    ----------
    _fission_params_cache : dict
        Cache for FissionParameters objects to avoid recalculation

    Public Methods
    --------------
    run_analysis(results_dict, params, dead_time_config)
        Run complete dead time analysis for all fission rates
    add_theoretical_cps(analysis, params)
        Add theoretical CPS calculations to existing analysis

    Private Methods
    --------------
    _calculate_all_cps_results(results_dict, params, dead_time_config)
        Calculate CPS results for all fission rates
    _calculate_cps_for_fission_rate(results_dict, params, fission_rate,
                                    dead_time_config)
        Calculate CPS and alpha inverse for a single fission rate
    _calculate_cps_values(time_matrix, detection_matrix, dead_time_config)
        Calculate CPS values based on dead time configuration
    _get_fission_parameters(fission_rates, params)
        Get or create FissionParameters with caching
    """

    def __init__(self):
        """
        Initialize the dead time analysis service.

        Sets up the fission parameters cache for efficient computation.
        """
        self._fission_params_cache = {}

    def run_analysis(
            self,
            results_dict: Dict[float, SimulationResults],
            params: SimulationParameters,
            dead_time_config: DeadTimeConfig) -> DeadTimeAnalysis:
        """
        Run complete dead time analysis and return DeadTimeAnalysis object.

        This method performs dead time analysis for all fission rates,
        calculating CPS values and alpha inverse relationships.

        Parameters
        ----------
        results_dict : Dict[float, SimulationResults]
            Dictionary of simulation results by fission rate
        params : SimulationParameters
            Simulation parameters containing fission rates and settings
        dead_time_config : DeadTimeConfig
            Dead time configuration parameters including distribution type,
            mean dead time, and standard deviation

        Returns
        -------
        DeadTimeAnalysis
            Complete dead time analysis results
            with CPS and alpha inverse data

        Raises
        ------
        ValueError
            If dead time configuration is invalid
        RuntimeError
            If analysis calculation fails
        """
        # Calculate alpha inverse values for each fission rate
        alpha_inv_results = self._calculate_all_cps_results(
            results_dict, params, dead_time_config)

        return DeadTimeAnalysis(
            config=dead_time_config,
            cps_results=alpha_inv_results
        )

    def _calculate_all_cps_results(
            self,
            results_dict: Dict[float, SimulationResults],
            params: SimulationParameters,
            dead_time_config: DeadTimeConfig) -> Dict[float, Dict[str, float]]:
        """Calculate CPS results for all fission rates."""
        alpha_inv_results = {}

        for fission_rate in sorted(params.fission_rates):
            alpha_inv_results[fission_rate] = (
                self._calculate_cps_for_fission_rate(
                    results_dict, params, fission_rate, dead_time_config)
            )

        return alpha_inv_results

    def add_theoretical_cps(
            self,
            analysis: DeadTimeAnalysis,
            params: ModelsSimulationParameters) -> DeadTimeAnalysis:
        """
        Add theoretical CPS calculations to an existing analysis.

        This method calculates theoretical count rates using analytical
        formulas and adds them to the existing dead time analysis.

        Parameters
        ----------
        analysis : DeadTimeAnalysis
            Existing dead time analysis to enhance
        params : ModelsSimulationParameters
            Simulation parameters for theoretical calculations

        Returns
        -------
        DeadTimeAnalysis
            Updated analysis with theoretical CPS data

        Raises
        ------
        ValueError
            If analysis or parameters are invalid
        RuntimeError
            If theoretical calculation fails
        """
        theoretical_cps_results = {}

        for fission_rate in sorted(analysis.cps_results.keys()):
            fission_params = self._get_fission_parameters(
                [fission_rate], params)
            equil_value = fission_params.equil[0]

            # Calculate theoretical CPS using count_rates module
            mean_tau_s = analysis.config.get_tau_seconds()
            std_tau_s = analysis.config.get_std_seconds()

            theoretical_cps = cr.calculate_theoretical_cps(
                analysis.config.distribution_type, mean_tau_s, std_tau_s,
                params.detection_rate, equil_value
            )

            theoretical_cps_results[fission_rate] = theoretical_cps

        # Update the analysis with theoretical CPS
        analysis.theoretical_cps = theoretical_cps_results
        return analysis

    def _calculate_cps_for_fission_rate(
            self,
            results_dict: Dict[float, SimulationResults],
            params: ModelsSimulationParameters,
            fission_rate: float,
            dead_time_config: DeadTimeConfig) -> Dict[str, float]:
        """Calculate CPS and alpha inverse for a single fission rate"""
        result = results_dict[fission_rate]

        # Get alpha inverse from FissionParameters
        fission_params = self._get_fission_parameters([fission_rate], params)
        alpha_inv = 1.0 / fission_params.alpha_vec[0]

        # Calculate CPS values
        cps_values = self._calculate_cps_values(
            result.time_matrix, result.detection_matrix, dead_time_config)

        return {
            'cps': np.mean(cps_values),
            'alpha_inv': alpha_inv
        }

    def _calculate_cps_values(
            self,
            time_matrix: np.ndarray,
            detection_matrix: np.ndarray,
            dead_time_config: DeadTimeConfig) -> np.ndarray:
        """Calculate CPS values based on dead time type"""
        dead_time_type = dead_time_config.distribution_type.lower()
        mean_tau_s = dead_time_config.get_tau_seconds()
        std_tau_s = dead_time_config.get_std_seconds()

        if dead_time_type.lower() == 'constant':
            return cr.count_per_second_const_dead_time(
                time_matrix, detection_matrix, tau=mean_tau_s
            )

        if dead_time_type.lower() == 'normal':
            return cr.count_per_second_rand_dead_time(
                time_matrix, detection_matrix,
                distribution='normal',
                loc=mean_tau_s,
                scale=std_tau_s
            )

        if dead_time_type.lower() == 'uniform':
            # Uniform distribution: low and high (maintaining std)
            low = mean_tau_s - np.sqrt(3) * std_tau_s
            high = mean_tau_s + np.sqrt(3) * std_tau_s
            return cr.count_per_second_rand_dead_time(
                time_matrix, detection_matrix,
                distribution='uniform',
                low=low,
                high=high
            )

        if dead_time_type.lower() == 'gamma':
            # Gamma distribution: shape = (mean/std)^2, scale = std^2/mean
            shape = (mean_tau_s / std_tau_s) ** 2
            scale = (std_tau_s ** 2) / mean_tau_s
            return cr.count_per_second_rand_dead_time(
                time_matrix, detection_matrix,
                distribution='gamma',
                shape=shape,
                scale=scale
            )

        raise ValueError(f"Unknown dead time type: {dead_time_type}")

    def _get_fission_parameters(
            self,
            fission_rates: list,
            params: ModelsSimulationParameters) -> FissionParameters:
        """
        Get or create FissionParameters
        for the given rates and parameters.
        """
        # Create cache key
        cache_key = (tuple(fission_rates), params.detection_rate,
                     params.absorption_rate, params.source_rate)

        if cache_key not in self._fission_params_cache:
            # Create PhysicalParameters with the rate constants
            rate_constants = RateConstants(
                absorb=params.absorption_rate,
                source=params.source_rate,
                detect=params.detection_rate
            )
            physical_params = PhysicalParameters(
                rate_constants=rate_constants,
                fission_distribution=FissionDistribution()
            )

            # Create FissionParameters with the physical parameters
            fission_params = FissionParameters(
                fission_vec=np.array(fission_rates),
                physical_params=physical_params
            )

            self._fission_params_cache[cache_key] = fission_params

        return self._fission_params_cache[cache_key]
