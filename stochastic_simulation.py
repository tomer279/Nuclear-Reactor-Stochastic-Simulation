"""Written by Tomer279 with the assistance of Cursor.ai.

Stochastic Monte Carlo simulation engine for nuclear reactor dynamics.

This module provides comprehensive stochastic simulation capabilities
for modeling neutron population dynamics using Monte Carlo methods.

It implements event-driven simulations that accurately capture the discrete,
stochastic nature of nuclear reactions including
fission, absorption, detection, and source events with realistic timing
and yield distributions.

Classes:
    SimulationMatrices:
        Container for organizing multi-trajectory simulation data
    SimulationParameters:
        Parameter container for stochastic simulation configuration
    EventCalculator:
        Engine for calculating event probabilities and timing
    PopulationUpdater:
        Handler for population changes based on nuclear events
    StochasticSimulator:
        Main stochastic simulation engine and coordinator

Key Features:
    - Event-driven stochastic simulation with exponential timing
    - Multi-trajectory matrix-based organization for efficiency
    - Realistic fission yield modeling with probabilistic distributions
    - Detection event tracking with dead time modeling capabilities
    - Comprehensive rate-based event probability calculations
    - Automatic data management and result organization
    - Progress tracking and statistical analysis utilities

Mathematical Approach:
    The simulation uses kinetic Monte Carlo methods where:
        - Each trajectory evolves via discrete events
          with exponentially distributed timing
        - Event types (source, fission, absorption, detection)
          have population-dependent rates
        - Fission events produce variable yields
          following specified probability distributions
        - Total event rate determines next event timing
          via exponential random variables
        - Population changes are discrete:
          +1 (source), +yield-1 (fission), -1 (absorption/detection)

Dependencies:
    numpy: For numerical operations and random number generation
    typing: For type annotations
    utils: Custom utility functions for progress tracking and data processing
    core_parameters: Parameter container classes and fission distributions
    simulation_setting: Simulation control and output configuration
    data_management: Data storage and organization system

Usage Examples:
    # Initialize simulation parameters
    sim_params = SimulationParameters(
        physical_params=physical_params,
        time_params=time_params,
        n_0=np.array([1000]),  # Initial population
        control_params=SimulationControl()
    )

    # Create and run simulator
    simulator = StochasticSimulator(sim_params)
    t_matrix, pop_matrix, det_matrix = simulator.run_simulation(data_manager)

    # Analyze results
    summary = simulator.get_simulation_summary()
    event_stats = simulator.get_event_statistics(simulator.event_calc)

Note:
    This module handles the stochastic/monte carlo aspect of simulation.
    For numerical integration methods (Euler-Maruyama, Taylor, Runge-Kutta),
    see their respective modules. For count rate analysis, see count_rates.py.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import utils as utl
from core_parameters import (
    PhysicalParameters,
    TimeParameters,
    FissionDistribution)
from simulation_setting import SimulationControl
from data_management import DataManager, StochasticData

rng = np.random.default_rng()


class SimulationMatrices:
    """
    Container for organizing multi-trajectory stochastic simulation data.

    This class provides efficient storage and access to simulation results
    across multiple independent trajectories, organizing population,
    timing, event, and detection data in matrix format for easy analysis
    and statistical processing.

    Attributes
    ----------
    trajectories : int
        Number of independent simulation trajectories
    steps : int
        Number of time steps per trajectory
    population : np.ndarray
        Population matrix [trajectories × steps]
    time : np.ndarray
        Time matrix [trajectories × steps]
    events : np.ndarray
        Event matrix storing event types as strings [trajectories × steps]
    detections : np.ndarray
        Detection matrix storing detection timestamps [trajectories × steps]

    Public Methods
    --------------
    initialize(initial_population, initial_time)
        Initialize all matrices with starting conditions
    get_clean_detections()
        Return detection matrix with NaN values processed
    get_trajectory_data(trajectory_index)
        Extract complete data for a single trajectory
    get_summary_statistics()
        Calculate statistical summaries across all trajectories

    Examples
    --------
    >>> matrices = SimulationMatrices(trajectories=10, steps=1000)
    >>> matrices.initialize(np.array([1000]), t_0=0.0)
    >>> summary = matrices.get_summary_statistics()
    """

    def __init__(
            self,
            trajectories: int,
            steps: int):
        """
        Initialize simulation matrices for multi-trajectory simulation.

        Parameters
        ----------
        trajectories : int
            Number of independent simulation trajectories to store
        steps : int
            Number of time steps per trajectory

        Raises
        ------
        ValueError
            If trajectories or steps are not positive
        """
        self.trajectories = trajectories
        self.steps = steps

        # Pre-allocate all matrices
        self.population = np.zeros((trajectories, steps))
        self.time = np.zeros((trajectories, steps))
        self.events = np.full((trajectories, steps), '', dtype=object)
        self.detections = np.zeros((trajectories, steps))

    def initialize(
            self,
            initial_population: np.ndarray,
            initial_time: float):
        """
        Initialize all matrices with starting conditions.

        Parameters
        ----------
        initial_population : np.ndarray
            Initial neutron population for each trajectory
        initial_time : float
            Starting simulation time (seconds)

        Raises
        ------
        ValueError
            If initial_population shape doesn't match number of trajectories
        """
        self.population[:, 0] = initial_population
        self.time[:, 0] = initial_time
        self.events[:, 0] = ''
        self.detections[:, 0] = np.nan

    def get_clean_detections(self) -> np.ndarray:
        """
        Get processed detection matrix with NaN values properly handled.

        Returns
        -------
        np.ndarray
            Detection matrix processed through utility cleaning function
        """
        return utl.clean_detection_matrix(self.detections)

    def get_trajectory_data(
            self,
            trajectory_index: int) -> Dict[str, np.ndarray]:
        """
        Extract complete simulation data for a single trajectory.

        Parameters
        ----------
        trajectory_index : int
            Index of trajectory to extract (0 to trajectories-1)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
                'population': Population time series
                'time': Time points
                'events': Event sequence
                'detections': Detection timestamps

        Raises
        ------
        IndexError
            If trajectory_index is out of bounds
        """
        return {
            'population': self.population[trajectory_index, :],
            'time': self.time[trajectory_index, :],
            'events': self.events[trajectory_index, :],
            'detections': self.detections[trajectory_index, :]
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistical summaries across all trajectories.

        Returns
        -------
        Dict[str, Any] \n
        Statistical summaries including:
            - 'num_trajectories': Total number of trajectories
            - 'num_steps': Number of time steps per trajectory
            - 'final_population_mean': Mean final population
            - 'final_population_std': Standard deviation of final population
            - 'total_detections': Total number of detection events
            - 'simulation_duration': Time span of simulation
        """
        return {
            'num_trajectories': self.trajectories,
            'num_steps': self.steps,
            'final_population_mean': np.mean(self.population[:, -1]),
            'final_population_std': np.std(self.population[:, -1]),
            'total_detections': np.sum(~np.isnan(self.detections)),
            'simulation_duration': np.max(self.time) - np.min(self.time)
        }


class SimulationParameters:
    """
    Parameter container for stochastic simulation configuration.

    This class bundles all necessary parameters for stochastic Monte Carlo
    simulations, providing validation and convenient access to simulation
    configuration including physical constants, timing, initial conditions,
    and control parameters.

    Attributes
    ----------
    physical : PhysicalParameters
        Physical parameters containing nuclear reaction rate constants
    time : TimeParameters
        Time discretization and simulation duration parameters
    n_0 : np.ndarray
        Initial nuclear population for each trajectory
    control : SimulationControl
        Control parameters for progress reporting and output

    Properties
    ----------
    fission : float
        Fission rate constant from physical parameters

    Public Methods
    --------------
    get_rate_constants()
        Extract all nuclear reaction rate constants as dictionary
    calculate_total_rate(population)
        Calculate total event rate for a given population

    Examples
    --------
    >>> sim_params = SimulationParameters(
    ...     physical_params=physical_params,
    ...     time_params=time_params,
    ...     n_0=np.array([1000, 950]),
    ...     control_params=SimulationControl()
    ... )
    >>> fission = sim_params.fission
    >>> total_rate = sim_params.calculate_total_rate(population=500)
    """

    def __init__(
            self,
            physical_params: PhysicalParameters,
            time_params: TimeParameters,
            n_0: np.ndarray,
            control_params: Optional[SimulationControl] = None):
        """
        Initialize simulation parameters with validation.

        Parameters
        ----------
        physical_params : PhysicalParameters
            Physical parameters containing nuclear reaction constants
        time_params : TimeParameters
            Time discretization parameters
        n_0 : np.ndarray
            Initial neutron population array (one value per trajectory)
        control_params : Optional[SimulationControl]
            Control parameters for progress reporting and output

        Raises
        ------
        ValueError
            If validation checks fail for fission rate or population values
        """
        self.physical = physical_params
        self.time = time_params
        self.n_0 = n_0
        self.control = control_params or SimulationControl()

        self._validate()

    def _validate(self):
        """
        Validate simulation parameters for consistency and correctness.

        Raises
        ------
        ValueError
            If parameters are invalid or inconsistent
        """
        if self.physical.fission is None:
            raise ValueError("Fission rate must be set in physical parameters")

        if len(self.n_0) == 0:
            raise ValueError("Initial population array cannot be empty")

        if np.any(self.n_0 < 0):
            raise ValueError("Initial population values must be non-negative")

    def calculate_total_rate(self, population: float) -> float:
        """
        Calculate total event rate for a given population.

        Parameters
        ----------
        population : float
            Current neutron population

        Returns
        -------
        float
            Total rate for all possible events (s⁻¹)
        """
        return self.physical.calculate_total_rate(population)

    def get_rate_constants(self) -> Dict[str, float]:
        """
        Get all nuclear reaction rate constants as dictionary.

        Returns
        -------
        Dict[str, float]
            Rate constants dictionary including:
                'fission': Fission rate constant
                'absorb': Absorption rate constant
                'detect': Detection rate constant
                'source': Source rate constant
        """
        return self.physical.get_rate_constants()

    @property
    def fission(self) -> float:
        """
        Get fission rate constant from physical parameters.

        Returns
        -------
        float
            Fission rate constant (s⁻¹)
        """
        return self.physical.fission


class EventCalculator:
    """
    Engine for nuclear event probability calculations
    and random event generation.

    This class handles the stochastic aspect of nuclear reactor dynamics by
    calculating event probabilities based on current population and rate
    constants, then generating random events according to these probabilities
    with exponentially distributed timing intervals.

    Attributes
    ----------
    physical : PhysicalParameters
        Physical parameters for rate constant access
    event_types : List[str]
        List of possible nuclear event types

    Public Methods
    --------------
    calculate_total_rate(current_population)
        Calculate total event rate for current population
    calculate_event_probabilities(current_population, total_rate)
        Calculate normalized probability of each event type
    generate_event(event_probs, total_rate)
        Generate random event and time increment

    Examples
    --------
    >>> calculator = EventCalculator(physical_params)
    >>> total_rate = calculator.calculate_total_rate(population=500)
    >>> event_probs = calculator.calculate_event_probabilities(
    ...     500, total_rate)
    >>> event_type, time_inc = calculator.generate_event(
    ...     event_probs, total_rate)
    """

    def __init__(
            self,
            physical_params: PhysicalParameters):
        """
        Initialize event calculator with physical parameters.

        Parameters
        ----------
        physical_params : PhysicalParameters
            Physical parameters containing nuclear reaction rate constants
        """
        self.physical = physical_params
        self.event_types = ['source', 'fission', 'absorption', 'detection']

    def calculate_total_rate(
            self,
            current_population: float) -> float:
        """
        Calculate total event rate for current population.

        The total rate is the sum of all individual event rates:
        Rate_total = source + population × (fission + absorb + detect)

        Parameters
        ----------
        current_population : float
            Current neutron population

        Returns
        -------
        float
            Total event rate (s⁻¹)
        """
        return self.physical.calculate_total_rate(current_population)

    def calculate_event_probabilities(
            self,
            current_population: float,
            total_rate: float) -> np.ndarray:
        """
        Calculate normalized probabilities for each nuclear event type.

        Parameters
        ----------
        current_population : float
            Current neutron population affecting event rates
        total_rate : float
            Total event rate for normalization

        Returns
        -------
        np.ndarray
            Array of probabilities in order:
                [source, fission, absorption, detection]
            Sum of probabilities equals 1.0
        """
        return np.array([
            self.physical.source,
            current_population * self.physical.fission,
            current_population * self.physical.absorb,
            current_population * self.physical.detect
        ] / total_rate)

    def generate_event(
            self,
            event_probs: np.ndarray,
            total_rate: float) -> Tuple[str, float]:
        """
        Generate random nuclear event
        and exponentially distributed time increment.

        Parameters
        ----------
        event_probs : np.ndarray
            Normalized probabilities for each event type
        total_rate : float
            Total event rate determining time increment distribution

        Returns
        -------
        Tuple[str, float]
            Returns (event_type, time_increment) where:
            - event_type:
                One of ['source', 'fission', 'absorption', 'detection']
            - time_increment:
                Exponential random variable (seconds)
        """
        event_name = rng.choice(self.event_types, p=event_probs)

        time_increment = rng.exponential(scale=1/total_rate)

        return event_name, time_increment


class PopulationUpdater:
    """
    Handler for neutron population changes based on nuclear events.

    This class manages population updates according to nuclear physics:
    source events add neutrons, fission events modify population based on
    probabilistic yield distributions, while absorption and detection
    events remove neutrons from the system.

    Attributes
    ----------
    fission_dist : FissionDistribution
        Distribution for sampling fission neutron yields

    Public Methods
    --------------
    update_population(current_population, event_name)
        Apply population changes based on nuclear event
    get_event_description(event_name)
        Get human-readable description of event effects

    Examples
    --------
    >>> updater = PopulationUpdater(fission_distribution)
    >>> new_pop = updater.update_population(
    ...     current_pop=100, event_name='fission')
    >>> desc = updater.get_event_description('source')
    """

    def __init__(
            self,
            fission_distribution: FissionDistribution):
        """
        Initialize population updater with fission yield distribution.

        Parameters
        ----------
        fission_distribution : FissionDistribution
            Probability distribution for sampling fission neutron yields
        """
        self.fission_dist = fission_distribution

    def update_population(
            self,
            current_population: float,
            event_name: str) -> float:
        """
        Apply population changes based on nuclear event type.

        Population changes follow nuclear physics:

        - Source:
            +1 neutron (external source)
        - Fission:
            +yield-1 neutrons (neutron consumed, yield neutrons produced)
        - Absorption:
            -1 neutron (neutron absorbed)
        - Detection:
            -1 neutron (neutron detected)

        Parameters
        ----------
        current_population : float
            Current neutron population before event
        event_name : str
            Type of nuclear event that occurred

        Returns
        -------
        float
            Updated neutron population after event

        Raises
        ------
        ValueError
            If event_name is not recognized event type
        """
        if event_name == 'source':
            return current_population + 1
        if event_name == 'fission':
            particles_produced = self.fission_dist.sample_fission_yield()
            return current_population + particles_produced - 1
        if event_name in ('absorption', 'detection'):
            return current_population - 1
        raise ValueError(f"Unknown event type: {event_name}")

    def get_event_description(
            self,
            event_name: str) -> str:
        """
        Get human-readable description of nuclear event effects.

        Parameters
        ----------
        event_name : str
            Type of nuclear event

        Returns
        -------
        str
            Human-readable description of event effects on population
        """

        descriptions = {
            "source": "Source particle added (+1)",
            "fission": "Fission occurred (variable yield)",
            "absorption": "Neutron absorbed (-1)",
            "detection": "Neutron detected"
        }
        return descriptions.get(event_name, f"Unknown event ({event_name})")


class StochasticSimulator:
    """
    Main stochastic Monte Carlo simulation engine for nuclear reactor dynamics.

    This class orchestrates the complete stochastic simulation workflow,
    managing multiple independent trajectories, event generation, population
    updates, and result organization. It provides the primary interface
    for running Monte Carlo nuclear reactor simulations with comprehensive
    data management and statistical analysis capabilities.

    The simulator implements kinetic Monte Carlo methods where discrete
    nuclear events (source, fission, absorption, detection) occur with
    exponentially distributed timing intervals determined by
    population-dependent rate constants and fission yield distributions.

    Attributes
    ----------
    params : SimulationParameters
        Complete simulation configuration and parameters
    event_calc : EventCalculator
        Engine for event probability calculations and random generation
    pop_updater : PopulationUpdater
        Handler for population changes based on nuclear events
    matrices : Optional[SimulationMatrices]
        Simulation data storage (initialized after simulation runs)

    Public Methods
    --------------
    run_simulation(data_manager)
        Execute complete stochastic simulation workflow
    get_simulation_summary()
        Generate comprehensive simulation statistics and summaries
    get_event_statistics(event_calc)
        Analyze event type distributions and frequencies

    Examples
    --------
    >>> sim_params = SimulationParameters(
    ...     physical_params, time_params, n_0, control)
    >>> simulator = StochasticSimulator(sim_params)
    >>> time_matrix, pop_matrix, det_matrix = simulator.run_simulation(
    ...     data_manager)
    >>> summary = simulator.get_simulation_summary()
    """

    def __init__(self, sim_params: SimulationParameters):
        """
        Initialize stochastic simulator.

        Parameters
        ----------
        sim_params : SimulationParameters
            Simulation parameters.
        """
        self.params = sim_params
        self.event_calc = EventCalculator(sim_params.physical)
        fission_dist = FissionDistribution(sim_params.physical.p_v)
        self.pop_updater = PopulationUpdater(fission_dist)
        self.matrices = None

    def run_simulation(
            self,
            data_manager: DataManager) -> Tuple[np.ndarray,
                                                np.ndarray,
                                                np.ndarray]:
        """
        Execute the complete stochastic Monte Carlo simulation workflow.

        This method coordinates the entire simulation process including:
        1. Matrix initialization for multiple trajectories
        2. Sequential trajectory simulation with event-driven dynamics
        3. Data cleaning and organization
        4. Automatic result saving through data management system
        5. Progress tracking and reporting

        Parameters
        ----------
        data_manager : DataManager
            Data management system for organized result storage

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Simulation results tuple containing:

            - time_matrix:
                Time points for all trajectories [trajectories × steps]
            - population_matrix:
                Population evolution for all trajectories
                [trajectories × steps]
            - detection_matrix:
                Detection event timestamps [trajectories × steps]

        Notes
        -----
        Progress is automatically reported based on control parameters.

        Simulation results are automatically saved
        to organized directory structure.
        """
        # Initialize simulation matrices
        k = len(self.params.n_0)
        self.matrices = SimulationMatrices(k, self.params.time.steps)
        self.matrices.initialize(self.params.n_0, self.params.time.t_0)

        # Run simulation for each trajectory
        for i in range(k):
            if self.params.control.should_show_progress(i, k):
                message = self.params.control.get_progress_message(i, k)
                print(f"Processing trajectory {i + 1}/{k} - {message}")

            self._simulate_single_trajectory(i)

        print("All trajectories completed!")

        clean_detections = self.matrices.get_clean_detections()

        self._save_results(clean_detections, data_manager)

        return self.matrices.time, self.matrices.population, clean_detections

    def _simulate_single_trajectory(self, trajectory_index: int):
        """
        Simulate a single stochastic trajectory using Monte Carlo methods

        This private method implements the core stochastic simulation algorithm
        for one trajectory, involving:
        1. Sequential event generation with exponential timing
        2. Population updates based on nuclear reaction types
        3. Event tracking and detection timestamp recording

        """
        i = trajectory_index
        num_steps = self.params.time.steps - 1

        for j in utl.progress_tracker(num_steps):
            current_pop = self.matrices.population[i, j]
            current_time = self.matrices.time[i, j]

            # Calculate total rate and probabilities
            total_rate = self.event_calc.calculate_total_rate(current_pop)

            if total_rate <= 0:
                # System is static
                self.matrices.time[i, j + 1] = current_time
                self.matrices.population[i, j + 1] = current_pop
                self.matrices.events[i, j + 1] = ''
                self.matrices.detections[i, j + 1] = np.nan
                continue

            # Calculate event probabilities and generate event
            event_probs = self.event_calc.calculate_event_probabilities(
                current_pop, total_rate)
            event_name, time_increment = self.event_calc.generate_event(
                event_probs, total_rate)

            # Update time and record event (store as string in events matrix)
            self.matrices.time[i, j + 1] = current_time + time_increment
            self.matrices.events[i, j + 1] = event_name

            # Update population based on event type
            new_pop = self.pop_updater.update_population(current_pop,
                                                         event_name)
            self.matrices.population[i, j + 1] = new_pop

            # Handle detection events
            if event_name == 'detection':
                self.matrices.detections[i, j + 1] = (current_time
                                                      + time_increment)
            else:
                self.matrices.detections[i, j + 1] = np.nan

    def _save_results(
            self,
            clean_detections: np.ndarray,
            data_manager: DataManager):
        """
        Save simulation results using data manager.

        results to organized directory structures with appropriate naming
        conventions, metadata tagging, and file format standardization.
        Results are saved using the data management system for consistency
        with other simulation methods.
        """
        filename = self.params.control.generate_result_filename(
            'Stochastic', 'basic', self.params.fission
        )
        print(f"Saving stochastic simulation results to: {filename}")

        data = StochasticData(
            population_matrix=self.matrices.population,
            time_matrix=self.matrices.time,
            detection_matrix=clean_detections,
            fission_value=self.params.fission,
            prefix=self.params.control.output.output_prefix
        )

        data_manager.save_stochastic_data(data)

    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for completed simulation.

        This method provides detailed statistical analysis of simulation
        results including population statistics, timing information,
        simulation configuration parameters, and overall performance metrics.
        Useful for validation, comparison analysis, and result verification.

        Returns
        -------
        Dict[str, Any]

        Comprehensive summary dictionary containing:

        - 'status': Simulation run status indicator
        - 'num_trajectories': Number of independent trajectories
        - 'num_steps': Time steps per trajectory
        - 'final_population_mean': Mean final population across trajectories
        - 'final_population_std': Standard deviation of final populations
        - 'total_detections': Total detection events across all trajectories
        - 'simulation_duration': Total time span of simulation
        - 'fission_rate': Fission rate constant used
        - 'time_step': Average time step size
        - 'rate_constants': All nuclear reaction rate constants

        Examples
        --------
        >>> summary = simulator.get_simulation_summary()
        >>> print(f"Final population: {summary['final_population_mean']:.1f} ±"
        ...        f" {summary['final_population_std']:.1f}")
        >>> print(f"Total detections: {summary['total_detections']}")
        """
        if self.matrices is None:
            return {"status": "Simulation not yet run"}

        summary = self.matrices.get_summary_statistics()
        summary.update({
            'fission_rate': self.params.fission,
            'simulation_duration': self.params.time.get_duration(),
            'time_step': self.params.time.get_time_step(),
            'rate_constants': self.params.get_rate_constants()
        })
        return summary

    def get_event_statistics(self,
                             event_calc: EventCalculator) -> Dict[str, int]:
        """
        Analyze event type distribution and frequencies across simulation.

        This method provides detailed statistics about nuclear events that
        occurred during the simulation, counting frequency of each event type
        (source, fission, absorption, detection) across all trajectories
        and time steps. Useful for understanding simulation dynamics
        and validating against expected physics behavior.

        Parameters
        ----------
        event_calc : EventCalculator
            Event calculator instance used during simulation

        Returns
        -------
        Dict[str, int]
            Event statistics dictionary containing:
            - 'source': Number of source events
            - 'fission': Number of fission events
            - 'absorption': Number of absorption events
            - 'detection': Number of detection events
            - Additional status information if simulation not run

        Examples
        --------
        >>> event_stats = simulator.get_event_statistics(simulator.event_calc)
        >>> fission_count = event_stats['fission']
        >>> detection_count = event_stats['detection']
        >>> print(f"Fission events: {fission_count}, "
                  f"Detections: {detection_count}")

        Notes
        -----
        Event counts are summed across all trajectories and time steps.
        Values represent total occurrences during the entire simulation run.
        """
        if self.matrices is None:
            return {'status': "Simulation not yet run"}
        # Count events by type
        event_counts = {}
        for event_name in event_calc.event_types:
            count = np.sum(self.matrices.events == event_name)
            event_counts[event_name] = int(count)

        return event_counts
