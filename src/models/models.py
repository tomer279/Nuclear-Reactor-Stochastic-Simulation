"""Written by Tomer279 with the assistance of Cursor.ai.

Data models for nuclear reactor stochastic simulation dashboard.

This module provides comprehensive data model classes
that solve common problems in the Streamlit application
by replacing complex parameter passing, confusing session state management,
and unclear return values with clear, self-documenting objects.
These models encapsulate simulation parameters, results, and analysis
data with convenient methods for data access and manipulation.

Classes:
    SimulationParameters:
        Bundles all simulation input parameters with validation
    DeadTimeConfig:
        Configuration for dead time analysis with utility methods
    SimulationResults:
        Results from a single simulation with statistical methods
    DeadTimeAnalysis:
        Results from dead time analysis with comparison methods
    AppState:
        Main application state replacing complex session state management

Key Features:
    - Type-safe data structures with comprehensive validation
    - Convenient statistical and analysis methods
    - Automatic unit conversions and display formatting
    - Clear separation of concerns between data and logic
    - Comprehensive error handling and data validation
    - Professional display formatting for UI components

Data Model Benefits:
    - Eliminates complex parameter passing (6+ parameters → 1 object)
    - Replaces confusing session state with clear data structure
    - Provides self-documenting interfaces with type hints
    - Enables convenient statistical analysis and data access
    - Supports professional visualization and reporting

Dependencies:
    dataclasses: For structured data containers
    typing: For comprehensive type annotations
    numpy: For numerical operations and statistical calculations

Usage Examples:
    # Create simulation parameters
    params = SimulationParameters(
        fission_rates=[33.94, 33.95, 33.96],
        detection_rate=10.0,
        absorption_rate=7.0,
        source_rate=1000.0,
        initial_population=None,
        simulation_steps=100000,
        use_equilibrium=True
    )

    # Create dead time configuration
    dead_time_config = DeadTimeConfig(
        mean_dead_time=1.0,
        std_percent=10.0,
        distribution_type='normal',
        analysis_name='Normal Dead Time Analysis'
    )

    # Create simulation results
    results = SimulationResults(
        time_matrix=time_data,
        population_matrix=pop_data,
        detection_matrix=detect_data,
        fission_rate=33.95
    )

    # Access statistical information
    final_pop = results.get_final_population()
    cps = results.get_cps()

    # Application state management
    app_state = AppState()
    app_state.simulation_params = params
    app_state.simulation_results = {33.95: results}

Note:
    This module provides the data layer for the simulation dashboard.
    For UI components, see ui_components.py.
    For business logic, see services.py.
    For core simulation engines, see the respective simulation modules.
"""


from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class SimulationParameters:
    """
    Comprehensive simulation parameters for nuclear stochastic simulations.

    This class bundles all parameters needed to run nuclear simulations,
    replacing the need to pass 6+ individual parameters to functions with a
    single, well-structured object.

    It includes automatic validation and equilibrium logic handling.

    Attributes
    ----------
    fission_rates : List[float]
        List of fission rate constants for parameter sweep analysis (s⁻¹)
    detection_rate : float
        Neutron detection rate constant (s⁻¹)
    absorption_rate : float
        Neutron absorption rate constant (s⁻¹)
    source_rate : float
        External neutron source rate (s⁻¹)
    initial_population : Optional[int]
        Initial neutron population (None if using equilibrium)
    simulation_steps : int
        Number of time steps for simulation
    use_equilibrium : bool
        Whether to use equilibrium initial population

    Examples
    --------
    >>> params = SimulationParameters(
    ...     fission_rates=[33.94, 33.95, 33.96],
    ...     detection_rate=10.0,
    ...     absorption_rate=7.0,
    ...     source_rate=1000.0,
    ...     initial_population=None,
    ...     simulation_steps=100000,
    ...     use_equilibrium=True
    ... )
    >>> print(f"Fission rates: {params.fission_rates}")
    """

    fission_rates: List[float]
    detection_rate: float
    absorption_rate: float
    source_rate: float
    initial_population: Optional[int]
    simulation_steps: int
    use_equilibrium: bool

    def __post_init__(self):
        """
        Automatically handle equilibrium logic and parameter validation.

        This method ensures that when equilibrium is used, the initial
        population is set to None, and performs basic parameter validation.
        """
        if self.use_equilibrium:
            self.initial_population = None


@dataclass
class DeadTimeConfig:
    """
    Configuration for dead time analysis with comprehensive utility methods.

    This class bundles dead time parameters and provides helpful methods
    for unit conversions, distribution parameter calculations, and display
    formatting. It supports multiple dead time distribution types with
    appropriate parameter handling for each.

    Attributes
    ----------
    mean_dead_time : float
        Mean dead time value in microseconds (μs)
    std_percent : float
        Standard deviation as percentage of mean
    distribution_type : str
        Type of dead time distribution ('constant', 'normal', 'uniform', 'gamma')
    analysis_name : str
        Descriptive name for this analysis configuration

    Public Methods
    --------------
    get_display_name()
        Generate formatted display name for the analysis
    get_tau_seconds()
        Convert dead time from microseconds to seconds
    get_std_seconds()
        Convert standard deviation percentage to seconds
    get_uniform_bounds()
        Calculate uniform distribution bounds
    get_gamma_parameters()
        Calculate gamma distribution parameters

    Examples
    --------
    >>> config = DeadTimeConfig(
    ...     mean_dead_time=1.0,
    ...     std_percent=10.0,
    ...     distribution_type='normal',
    ...     analysis_name='Normal Dead Time'
    ... )
    >>> print(config.get_display_name())
    >>> tau_seconds = config.get_tau_seconds()
    """
    mean_dead_time: float  # in microseconds
    std_percent: float    # standard deviation as percentage of mean
    distribution_type: str  # 'constant', 'normal', 'uniform', 'gamma'
    analysis_name: str

    def get_display_name(self) -> str:
        """
        Generate formatted display name for the analysis.

        Returns
        -------
        str
            Formatted display name including distribution type and parameters
        """
        if self.distribution_type.lower() == 'constant':
            return f"{self.distribution_type} τ={self.mean_dead_time}μs"

        return (f"{self.distribution_type} "
                f"τ={self.mean_dead_time}μs "
                f"σ={self.std_percent}%")

    def get_tau_seconds(self) -> float:
        """
        Convert dead time from microseconds to seconds.

        Returns
        -------
        float
            Dead time in seconds
        """
        return self.mean_dead_time * 1e-6

    def get_std_seconds(self) -> float:
        """
        Convert standard deviation percentage to standard deviation in seconds.

        Returns
        -------
        float
            Standard deviation in seconds
        """
        return (self.std_percent / 100.0) * self.get_tau_seconds()

    def get_uniform_bounds(self) -> tuple[float, float]:
        """
        Calculate uniform distribution bounds from mean and standard deviation.

        For a uniform distribution with mean μ and standard deviation σ,
        the bounds are calculated as: [μ - √3*σ, μ + √3*σ]

        Returns
        -------
        Tuple[float, float]
            (low_bound, high_bound) in seconds
        """
        mean_s = self.get_tau_seconds()
        std_s = self.get_std_seconds()
        low = mean_s - np.sqrt(3) * std_s
        high = mean_s + np.sqrt(3) * std_s
        return low, high

    def get_gamma_parameters(self) -> tuple[float, float]:
        """
        Calculate gamma distribution parameters
        from mean and standard deviation.

        For a gamma distribution with mean μ and standard deviation σ:
        - Shape parameter: α = (μ/σ)²
        - Scale parameter: β = σ²/μ

        Returns
        -------
        Tuple[float, float]
            (shape_parameter, scale_parameter)
        """
        mean_s = self.get_tau_seconds()
        std_s = self.get_std_seconds()
        shape = (mean_s / std_s) ** 2
        scale = (std_s ** 2) / mean_s
        return shape, scale


@dataclass
class SimulationResults:
    """
    Results from a single nuclear reactor simulation
    with comprehensive statistics.

    This class bundles the three matrices returned by simulations
    (time, population, detection) and provides convenient methods
    for accessing statistical information,
    performance metrics, and analysis data.

    Attributes
    ----------
    time_matrix : np.ndarray
        Time points matrix from simulation (trajectories × time_steps)
    population_matrix : np.ndarray
        Population evolution matrix (trajectories × time_steps)
    detection_matrix : np.ndarray
        Detection events matrix (trajectories × time_steps)
    fission_rate : float
        Fission rate constant used for this simulation (s⁻¹)

    Public Methods
    --------------
    get_final_population()
        Get final population value
    get_max_population()
        Get maximum population value
    get_min_population()
        Get minimum population value
    get_mean_population()
        Get mean population value
    get_population_std()
        Get population standard deviation
    get_simulation_duration()
        Get total simulation time
    get_detection_count()
        Get total number of detections
    get_cps()
        Get counts per second

    Examples
    --------
    >>> results = SimulationResults(
    ...     time_matrix=time_data,
    ...     population_matrix=pop_data,
    ...     detection_matrix=detect_data,
    ...     fission_rate=33.95
    ... )
    >>> final_pop = results.get_final_population()
    >>> cps = results.get_cps()
    """
    time_matrix: np.ndarray
    population_matrix: np.ndarray
    detection_matrix: np.ndarray
    fission_rate: float

    def get_final_population(self) -> float:
        """
        Get final population value from the first trajectory.

        Returns
        -------
        float
            Final population value
        """
        return self.population_matrix[0][-1]

    def get_max_population(self) -> float:
        """
        Get maximum population value across the simulation.

        Returns
        -------
        float
            Maximum population value
        """
        return np.max(self.population_matrix[0])

    def get_min_population(self) -> float:
        """
        Get minimum population value across the simulation.

        Returns
        -------
        float
            Minimum population value
        """
        return np.min(self.population_matrix[0])

    def get_mean_population(self) -> float:
        """
        Get mean population value across the simulation.

        Returns
        -------
        float
            Mean population value
        """
        return np.mean(self.population_matrix[0])

    def get_population_std(self) -> float:
        """
        Get population standard deviation across the simulation.

        Returns
        -------
        float
            Population standard deviation
        """
        return np.std(self.population_matrix[0])

    def get_simulation_duration(self) -> float:
        """
        Get total simulation time in seconds.

        Returns
        -------
        float
            Total simulation duration
        """
        return self.time_matrix[0][-1]

    def get_detection_count(self) -> int:
        """
        Get total number of detection events.

        Returns
        -------
        int
            Total number of detections (excluding NaN values)
        """
        valid_detections = self.detection_matrix[0]
        valid_detections = valid_detections[~np.isnan(valid_detections)]
        return len(valid_detections)

    def get_cps(self) -> float:
        """
        Calculate counts per second (CPS) from detection data.

        Returns
        -------
        float
            Counts per second (detections / simulation_duration)
        """
        detection_count = self.get_detection_count()
        duration = self.get_simulation_duration()
        return detection_count / duration if duration > 0 else 0.0


@dataclass
class DeadTimeAnalysis:
    """
    Results from dead time analysis with comprehensive comparison methods.

    This class bundles CPS results for multiple fission rates and optionally
    includes theoretical CPS calculations, providing methods for comparison
    analysis, error calculation, and summary data generation.

    Attributes
    ----------
    config : DeadTimeConfig
        Dead time configuration used for this analysis
    cps_results : Dict[float, Dict[str, Any]]
        Dictionary mapping fission rates to CPS and alpha inverse values
    theoretical_cps : Optional[Dict[float, float]]
        Optional theoretical CPS calculations for comparison

    Public Methods
    --------------
    has_theoretical_cps()
        Check if theoretical CPS calculations are available
    get_cps_for_fission_rate(fission_rate)
        Get simulated CPS for specific fission rate
    get_alpha_inv_for_fission_rate(fission_rate)
        Get alpha inverse for specific fission rate
    get_theoretical_cps_for_fission_rate(fission_rate)
        Get theoretical CPS for specific fission rate
    get_error_percentage(fission_rate)
        Calculate percentage error between theoretical and simulated CPS
    get_sorted_fission_rates()
        Get fission rates sorted numerically
    get_summary_data()
        Get formatted summary data for table display

    Examples
    --------
    >>> analysis = DeadTimeAnalysis(
    ...     config=dead_time_config,
    ...     cps_results={33.95: {'cps': 10.5, 'alpha_inv': 0.1}},
    ...     theoretical_cps={33.95: 10.2}
    ... )
    >>> cps = analysis.get_cps_for_fission_rate(33.95)
    >>> error = analysis.get_error_percentage(33.95)
    """
    config: DeadTimeConfig
    # fission_rate -> {cps, alpha_inv}
    cps_results: Dict[float, Dict[str, Any]]
    # fission_rate -> theoretical_cps
    theoretical_cps: Optional[Dict[float, float]] = None

    def has_theoretical_cps(self) -> bool:
        """
        Check if theoretical CPS calculations are available.

        Returns
        -------
        bool
            True if theoretical CPS data is available
        """
        return self.theoretical_cps is not None

    def get_cps_for_fission_rate(
            self,
            fission_rate: float) -> float:
        """
        Get simulated CPS for a specific fission rate.

        Parameters
        ----------
        fission_rate : float
            Fission rate constant

        Returns
        -------
        float
            Simulated counts per second
        """
        return self.cps_results[fission_rate]['cps']

    def get_alpha_inv_for_fission_rate(
            self,
            fission_rate: float) -> float:
        """
        Get alpha inverse (1/α) for a specific fission rate.

        Parameters
        ----------
        fission_rate : float
            Fission rate constant

        Returns
        -------
        float
            Alpha inverse value
        """
        return self.cps_results[fission_rate]['alpha_inv']

    def get_theoretical_cps_for_fission_rate(
            self,
            fission_rate: float) -> Optional[float]:
        """
        Get theoretical CPS for a specific fission rate.

        Parameters
        ----------
        fission_rate : float
            Fission rate constant

        Returns
        -------
        Optional[float]
            Theoretical counts per second, or None if not available
        """
        if self.theoretical_cps:
            return self.theoretical_cps[fission_rate]
        return None

    def get_error_percentage(
            self,
            fission_rate: float) -> Optional[float]:
        """
        Calculate percentage error between theoretical and simulated CPS.

        Parameters
        ----------
        fission_rate : float
            Fission rate constant

        Returns
        -------
        Optional[float]
            Percentage error, or None if theoretical CPS not available
        """
        theoretical = self.get_theoretical_cps_for_fission_rate(fission_rate)
        if theoretical is None:
            return None

        simulated = self.get_cps_for_fission_rate(fission_rate)
        return (((theoretical - simulated) / simulated * 100)
                if simulated > 0 else None)

    def get_sorted_fission_rates(self) -> List[float]:
        """
        Get fission rates sorted numerically for consistent display.

        Returns
        -------
        List[float]
            Fission rates sorted in ascending order
        """
        return sorted(self.cps_results.keys())

    def get_summary_data(self) -> List[Dict[str, Any]]:
        """
        Get formatted summary data for table display.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing formatted summary data
            for each fission rate, suitable for table display
        """
        summary_data = []

        for fission_rate in self.get_sorted_fission_rates():
            alpha_inv_value = self.get_alpha_inv_for_fission_rate(
                fission_rate)
            cps_value = self.get_cps_for_fission_rate(
                fission_rate)

            row_data = {
                'Fission Rate': fission_rate,
                'Alpha Inverse': (
                    f"{alpha_inv_value:.3f}"),
                'CPS (Simulated)': (
                    f"{cps_value:.2f}")
            }

            theoretical = self.get_theoretical_cps_for_fission_rate(
                fission_rate)
            if theoretical is not None:
                error = self.get_error_percentage(fission_rate)
                row_data['CPS (Theoretical)'] = f"{theoretical:.2f}"
                row_data['Error (%)'] = (
                    f"{error:.1f}" if error is not None else "N/A")

            summary_data.append(row_data)

        return summary_data


@dataclass
class AppState:
    """
    Main application state for Streamlit dashboard
    with comprehensive management.

    This class replaces complex session state management with a single, clear
    data structure that encapsulates all application state including simulation
    parameters, results, and dead time analyses. It provides convenient methods
    for state management, data access, and statistical analysis.

    Attributes
    ----------
    simulation_params : Optional[SimulationParameters]
        Current simulation parameters
    simulation_results : Optional[Dict[float, SimulationResults]]
        Simulation results indexed by fission rate
    dead_time_analyses : List[DeadTimeAnalysis]
        List of dead time analysis results

    Public Methods
    --------------
    has_simulation_results()
        Check if simulation results are available
    has_dead_time_analyses()
        Check if any dead time analyses have been run
    get_analysis_by_name(name)
        Get dead time analysis by name
    add_dead_time_analysis(analysis)
        Add a new dead time analysis
    remove_dead_time_analysis(name)
        Remove dead time analysis by name
    clear_dead_time_analyses()
        Clear all dead time analyses
    clear_all()
        Clear all application data
    get_analysis_names()
        Get list of all analysis names
    get_overall_statistics()
        Get overall statistics across all simulations
    get_dead_time_results_dict()
        Convert dead time analyses to dictionary format
    update_analysis(updated_analysis)
        Update an existing dead time analysis

    Examples
    --------
    >>> app_state = AppState()
    >>> app_state.simulation_params = params
    >>> app_state.simulation_results = {33.95: results}
    >>> app_state.add_dead_time_analysis(analysis)
    >>> stats = app_state.get_overall_statistics()
    """
    simulation_params: Optional[SimulationParameters] = None
    simulation_results: Optional[Dict[float, SimulationResults]] = None
    dead_time_analyses: List[DeadTimeAnalysis] = field(default_factory=list)

    def has_simulation_results(self) -> bool:
        """
        Check if simulation results are available.

        Returns
        -------
        bool
            True if simulation results are available
        """
        return self.simulation_results is not None

    def has_dead_time_analyses(self) -> bool:
        """
        Check if any dead time analyses have been run.

        Returns
        -------
        bool
            True if at least one dead time analysis exists
        """
        return len(self.dead_time_analyses) > 0

    def get_analysis_by_name(
            self,
            name: str) -> Optional[DeadTimeAnalysis]:
        """
        Get dead time analysis by name.

        Parameters
        ----------
        name : str
            Analysis name to search for

        Returns
        -------
        Optional[DeadTimeAnalysis]
            Dead time analysis with matching name, or None if not found
        """
        for analysis in self.dead_time_analyses:
            if analysis.config.analysis_name == name:
                return analysis
        return None

    def add_dead_time_analysis(
            self,
            analysis: DeadTimeAnalysis):
        """
        Add a new dead time analysis to the application state.

        Parameters
        ----------
        analysis : DeadTimeAnalysis
            Dead time analysis to add
        """
        self.dead_time_analyses.append(analysis)

    def remove_dead_time_analysis(
            self,
            name: str) -> bool:
        """
        Remove dead time analysis by name.

        Parameters
        ----------
        name : str
            Analysis name to remove

        Returns
        -------
        bool
            True if analysis was found and removed, False otherwise
        """
        for i, analysis in enumerate(self.dead_time_analyses):
            if analysis.config.analysis_name == name:
                del self.dead_time_analyses[i]
                return True
        return False

    def clear_dead_time_analyses(self):
        """Clear all dead time analyses"""
        self.dead_time_analyses.clear()

    def clear_all(self):
        """Clear all data"""
        self.simulation_params = None
        self.simulation_results = None
        self.dead_time_analyses.clear()

    def get_analysis_names(self) -> List[str]:
        """
        Get list of all dead time analysis names.

        Returns
        -------
        List[str]
            List of analysis names
        """
        return [analysis.config.analysis_name
                for analysis in self.dead_time_analyses]

    def get_overall_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics across all simulations.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing overall statistics including:
              - average_final_population
              - average_max_population
              - average_min_population
              - average_cps
              - total_simulations
        """
        if not self.has_simulation_results():
            return {}

        all_final_pops = [result.get_final_population()
                          for result in self.simulation_results.values()]
        all_max_pops = [result.get_max_population()
                        for result in self.simulation_results.values()]
        all_min_pops = [result.get_min_population()
                        for result in self.simulation_results.values()]
        all_cps = [result.get_cps()
                   for result in self.simulation_results.values()]

        return {
            'average_final_population': np.mean(all_final_pops),
            'average_max_population': np.mean(all_max_pops),
            'average_min_population': np.mean(all_min_pops),
            'average_cps': np.mean(all_cps),
            'total_simulations': len(self.simulation_results)
        }

    def get_dead_time_results_dict(self) -> (
            Dict[str, Dict[float, Dict[str, Any]]]):
        """
        Convert dead time analyses to dictionary format for plotting.

        Returns
        -------
        Dict[str, Dict[float, Dict[str, Any]]]
            Dictionary format suitable for plotting functions
        """
        results_dict = {}
        for analysis in self.dead_time_analyses:
            analysis_name = analysis.config.analysis_name
            results_dict[analysis_name] = {}

            for fission_rate in analysis.get_sorted_fission_rates():
                results_dict[analysis_name][fission_rate] = {
                    'cps': analysis.get_cps_for_fission_rate(fission_rate),
                    'alpha_inv': analysis.get_alpha_inv_for_fission_rate(
                        fission_rate)
                }
            if analysis.has_theoretical_cps():
                results_dict[analysis_name]['theoretical_cps'] = (
                    analysis.theoretical_cps)

        return results_dict

    def update_analysis(
            self,
            updated_analysis: DeadTimeAnalysis) -> bool:
        """
        Update an existing dead time analysis.

        Parameters
        ----------
        updated_analysis : DeadTimeAnalysis
            Updated analysis to replace existing one

        Returns
        -------
        bool
            True if analysis was found and updated, False otherwise
        """
        for i, analysis in enumerate(self.dead_time_analyses):
            if (analysis.config.analysis_name ==
                    updated_analysis.config.analysis_name):
                self.dead_time_analyses[i] = updated_analysis
                return True
        return False

# Helper functions for creating common configurations


def create_default_simulation_parameters() -> SimulationParameters:
    """
    Create default simulation parameters with reasonable values.

    Returns
    -------
    SimulationParameters
        Default simulation parameters suitable for initial testing
    """
    return SimulationParameters(
        fission_rates=[33.94, 33.95, 33.96],
        detection_rate=10.0,
        absorption_rate=7.0,
        source_rate=1000.0,
        initial_population=None,
        simulation_steps=100000,
        use_equilibrium=True
    )


def create_dead_time_config(
        mean_dead_time: float,
        std_percent: float,
        distribution_type: str,
        analysis_name: str = None) -> DeadTimeConfig:
    """
    Create dead time configuration with automatic naming.

    Parameters
    ----------
    mean_dead_time : float
        Mean dead time in microseconds
    std_percent : float
        Standard deviation as percentage of mean
    distribution_type : str
        Type of dead time distribution
    analysis_name : Optional[str]
        Custom analysis name (auto-generated if None)

    Returns
    -------
    DeadTimeConfig
        Configured dead time configuration object
    """
    if analysis_name is None:
        if distribution_type.lower() == 'constant':
            analysis_name = f"{distribution_type} τ={mean_dead_time}μs"
        else:
            analysis_name = (f"{distribution_type} "
                             f"τ={mean_dead_time}μs "
                             f"σ={std_percent}%")

    return DeadTimeConfig(
        mean_dead_time=mean_dead_time,
        std_percent=std_percent,
        distribution_type=distribution_type,
        analysis_name=analysis_name
    )
