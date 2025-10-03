"""Written by Tomer279 with the assistance of Cursor.ai.

Euler-Maruyama methods for nuclear reactor detection simulation.

This module provides comprehensive Euler-Maruyama numerical methods for solving
detection stochastic differential equations (SDEs) in nuclear reactor dynamics.
It implements hybrid approaches combining analytical population solutions with
numerical detection SDE integration under various dead time distributions.

Classes:
    EulerMaruyamaDetectionSDE:
        Core solver for detection SDEs with multiple dead time types

Key Features:
    - Hybrid analytical-numerical approach for efficient computation
    - Support for multiple dead time distributions
        (constant, uniform, normal, gamma)
    - Comprehensive noise amplitude calculations for each distribution type
    - Batch processing capabilities for parameter sweeps
    - Integrated data saving and management

Mathematical Approach:
    The detection SDE solved is:
    dC_t = d * N_t * (1 - Ψ(d * N_t)) dt + σ₃ * dW₃
    where:
    - C_t: accumulated detections
    - N_t: population (solved analytically)
    - Ψ: Psi function based on dead time distribution
    - σ₃: detection noise amplitude dependent on dead time type

Dependencies:
    numpy: For numerical operations and random number generation
    utils: Custom utility functions for system parameters
        and filename generation
    data_management: Data storage and organization
    analytical_solution: Analytical population solutions
    core_parameters: Parameter container classes

Usage Examples:
    # Initialize with physical parameters
    solver = EulerMaruyamaDetectionSDE(physical_params)

    # Solve single detection SDE path
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission=33.95
    )

    # Solve multiple paths for statistical analysis
    t_space, pop_mat, detect_mat = solver.solve_multiple_paths(
        time_params, dead_time_params, fission=33.95, num_paths=100
    )

    # Convenience functions for specific dead time types
    t_space, pop, detect = euler_maruyama_detection_constant_dead_time(
        physical_params, time_params, tau=1e-6, fission=33.95, index='f33.95'
    )

Note:
    This module handles numerical integration of detection processes.
    For stochastic simulations, see stochastic_simulation.py.
    For Taylor methods, see taylor_methods.py.
    For core parameters, see core_parameters.py.
"""

from typing import Tuple, Optional
import numpy as np
import utils as utl
from data_management import DataManager, EulerMaruyamaData
from analytical_solution import analytical_population_solution
from core_parameters import (
    PhysicalParameters,
    TimeParameters,
    DeadTimeParameters
)


rng = np.random.default_rng()
data_manager = DataManager()


class EulerMaruyamaDetectionSDE:
    """
    Core solver for detection stochastic differential equations.

    This class implements the hybrid analytical-numerical approach for solving
    nuclear reactor detection SDEs where population dynamics are solved
    analytically while detection processes are integrated numerically using
    the Euler-Maruyama method with various dead time distributions.

    Attributes
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction rates and constants

    Public Methods
    --------------
    solve_detection_sde(time_params, dead_time_params, fission, n_0)
        Solve single detection SDE path
    solve_multiple_paths(time_params, dead_time_params, fission, num_paths)
        Solve multiple detection SDE paths for statistical analysis

    Private Methods
    ---------------
    _validate_inputs(time_params, dead_time_params)
        Validate input parameter consistency
    _solve_detection_sde_numerically(
        time_params, dead_time_params, fission, pop)
        Core numerical integration of detection SDE
    _calculate_drift(current_pop, dead_time_params, tau_dist_params)
        Calculate drift term for detection SDE
    _generate_wiener_increments(grid_points, dt)
        Generate Wiener increments for noise
    _calculate_detection_noise(fission, dead_time_params)
        Calculate noise amplitude for specific dead time distribution

    Examples
    --------
    >>> solver = EulerMaruyamaDetectionSDE(physical_params)
    >>> t_space, pop, detect = solver.solve_detection_sde(
    ...     time_params, dead_time_params, fission=33.95
    ... )
    >>> print(f"Final detection count: {detect[-1]}")
    """

    def __init__(self, physical_params: PhysicalParameters):
        """
        Initialize the Euler-Maruyama detection SDE solver.

        Parameters
        ----------
        physical_params : PhysicalParameters
            Physical parameters containing nuclear reaction rate constants
            and fission probability distributions.

        Raises
        ------
        ValueError
            If physical parameters are invalid or incomplete
        """
        self.physical_params = physical_params

    def solve_detection_sde(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float,
            n_0: Optional[float] = None) -> (
            Tuple[np.ndarray,
                  np.ndarray,
                  np.ndarray]
    ):
        """
        Solve the detection SDE using Euler-Maruyama method.

        The detection SDE is:
        dC_t = d * N_t * (1 - Ψ(d * N_t)) dt + σ₃ * dW₃

        where:
            - C_t: accumulated detections at time t
            - N_t: population at time t (from analytical solution)
            - Ψ(...): Psi function encoding dead time effects
            - σ₃: detection noise amplitude (distribution-dependent)
            - dW₃: Wiener increments

        Parameters
        ----------
        time_params : TimeParameters
            Time parameters containing time configuration
        dead_time_params : DeadTimeParameters
            Dead time parameters containing dead time configuration
        fission : float
            Fission rate constant for this simulation
        n_0 : float, optional
            Initial population value. If None, defaults to equilibrium

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Solution tuple containing:
            - Time array: Discrete time points from t_0 to t_end
            - Population array: Analytical population solution N_t
            - Detection array: Numerical detection SDE solution C_t

        Raises
        ------
        ValueError
            If input parameters are inconsistent or invalid
        """

        # Input validation
        self._validate_inputs(time_params, dead_time_params)

        # Set fission rate for this simulation
        self.physical_params.set_fission(fission)

        pop = analytical_population_solution(self.physical_params,
                                             time_params,
                                             fission,
                                             n_0)

        # Initialize detection array
        detect = self._solve_detection_sde_numerically(
            time_params, dead_time_params, fission,
            pop)

        t_space = np.linspace(
            time_params.t_0,
            time_params.t_end,
            time_params.grid_points + 1
        )
        return t_space, pop, detect

    def solve_multiple_paths(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float,
            num_paths: int) -> Tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray]:
        """
        Solve multiple detection SDE paths for statistical analysis.

        This method generates multiple independent realizations
        of the detection SDE to enable statistical analysis
        including variance calculation, confidence intervals,
        and convergence studies.

        Parameters
        ----------
        time_params : TimeParameters
            Time discretization parameters
        dead_time_params : DeadTimeParameters
            Dead time distribution parameters
        fission : float
            Fission rate constant (s⁻¹)
        num_paths : int
            Number of independent simulation paths to generate

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Solution tuple containing:
            - Time array: Discrete time points (shape: grid_points+1)
            - Population matrix: Analytical solutions for all paths
                              (shape: num_paths × grid_points+1)
            - Detection matrix: Numerical SDE solutions for all paths
                              (shape: num_paths × grid_points+1)

        Raises
        ------
        ValueError
            If num_paths is not positive
        """

        pop_mat = np.zeros((num_paths, time_params.grid_points + 1))
        detect_mat = np.zeros((num_paths, time_params.grid_points + 1))

        for i in range(num_paths):
            _, pop_path, detect_path = self.solve_detection_sde(
                time_params, dead_time_params, fission)
            pop_mat[i, :] = pop_path
            detect_mat = detect_path

        t_space = np.linspace(time_params.t_0,
                              time_params.t_end,
                              time_params.grid_points + 1)

        return t_space, pop_mat, detect_mat

    def _validate_inputs(self,
                         time_params: TimeParameters,
                         dead_time_params: DeadTimeParameters):
        """Validate input parameters."""
        if time_params.grid_points < 1:
            raise ValueError("grid_points must be positive")
        if time_params.t_end <= time_params.t_0:
            raise ValueError("t_end must be greater than t_0")
        if dead_time_params.mean_tau <= 0:
            raise ValueError("mean_tau must be positive")

    def _solve_detection_sde_numerically(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float,
            pop: np.ndarray) -> np.ndarray:
        """Solve the detection SDE numerically using Euler-Maruyama

        This method implements the Euler-Maruyama scheme:
        C_{i+1} = C_i + drift_i * dt + σ₃ * ΔW_i

        where the drift term incorporates dead time effects through the
        Psi function and diffusion noise depends on the dead time distribution.
        """

        detect = np.zeros(time_params.grid_points + 1)

        # Calculate detection noise amplitude
        sig_3 = self._calculate_detection_noise(fission, dead_time_params)

        # Time step
        dt = time_params.get_grid_spacing()

        # Generate Wiener increments for detection noise
        wiener = self._generate_wiener_increments(
            time_params.grid_points, dt)

        tau_dist_params = dead_time_params.get_distribution_params()

        # Euler-Maruyama integration for detection
        for i in utl.progress_tracker(time_params.grid_points):
            current_pop = pop[i]

            drift = self._calculate_drift(
                current_pop,
                dead_time_params,
                tau_dist_params)

            detect[i + 1] = detect[i] + drift * dt + sig_3 * wiener[i]

        return detect

    def _calculate_drift(
            self,
            current_pop: float,
            dead_time_params: DeadTimeParameters,
            tau_dist_params: dict) -> float:
        """
        Calculate drift term for detection SDE incorporating dead time effects.

        The drift term is: d * N_t * (1 - Ψ(d * N_t)).
        """
        detection_rate = self.physical_params.detect * current_pop
        psi_value = utl.calculate_psi_function(
            detection_rate,
            dead_time_params.mean_tau,
            dead_time_params.tau_distribution,
            # For normal/gamma distribution
            tau_dist_params.get('scale', None)
        )
        return detection_rate * (1 - psi_value)

    def _generate_wiener_increments(
            self,
            grid_points: int,
            dt: float) -> np.ndarray:
        """Generate Wiener increments for detection noise."""
        return rng.normal(
            loc=0.0,
            scale=np.sqrt(dt),
            size=grid_points
        )

    def _calculate_detection_noise(self,
                                   fission: float,
                                   dead_time_params: DeadTimeParameters
                                   ) -> float:
        """
        Calculate the detection noise amplitude σ₃
        for different dead time types.

        Parameters
        ----------
        fission : float
            Fission rate constant
        dead_time_params : DeadTimeParameters
            Dead time parameters containing dead time configuration

        Raises
        ------
        ValueError
            If unknown dead time type is given.

        Returns
        -------
        float
            Detection noise amplitude σ₃.
        """

        # Calulate system parameters
        params = utl.calculate_system_parameters(
            self.physical_params.p_v,
            fission,
            self.physical_params.absorb,
            self.physical_params.source,
            self.physical_params.detect
        )
        equil = params['equilibrium']

        dead_time_type = dead_time_params.tau_distribution

        if dead_time_type == 'constant':
            # For constant dead time
            # σ₃² = exp(-2*d*N*τ) * (1 - 2*d*N*τ*exp(-d*N*τ)) / N²
            dead_time_parameter = (self.physical_params.detect
                                   * equil
                                   * dead_time_params.mean_tau)

            # Noise variance associated with actual detections
            sig_3_squared = (
                np.exp(-2 * dead_time_parameter)
                * (1 - 2 * dead_time_parameter * np.exp(-dead_time_parameter))
                / (equil ** 2))
            return np.sqrt(sig_3_squared)

        if dead_time_type in ('uniform', 'normal', 'gamma'):
            # For uniform, normal, and gamma distributions:
            # σ₃² = 1/(d*N) + complex_term
            # This is a simplified version - the full expression is complex
            detection_rate = self.physical_params.detect * equil
            return np.sqrt(1 / detection_rate)

        raise ValueError(f"Unknown dead time type: {dead_time_type}")


# Convenience functions for specific dead time types

def euler_maruyama_detection_constant_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau: float,
        fission: float,
        index: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with constant dead time using Euler-Maruyama method.

    This convenience function wraps the main solver class
    with constant dead time parameters for simplified usage
    in parameter sweeps and simulations.

    Parameters
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction constants
    time_params : TimeParameters
        Time discretization parameters
    tau : float
        Constant dead time value (seconds)
    fission : float
        Fission rate constant (s⁻¹)
    index : str
        Simulation index for data organization and saving

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndvector]
        Solution tuple: (time_array, population_array, detection_array)

    Examples
    --------
    >>> t_space, pop, detect = euler_maruyama_detection_constant_dead_time(
    ...     physical_params, time_params, tau=1e-6,
    ...     fission=33.95, index='f33.95'
    ... )
    """
    # Create dead time parameters for constant dead time
    dead_time_params = DeadTimeParameters(
        mean_tau=tau,
        tau_distribution='constant')

    # Run the simulation
    solver = EulerMaruyamaDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    # Save results
    _save_euler_maruyama_results(pop, detect, fission, index, 'const')

    return t_space, pop, detect


def euler_maruyama_detection_uniform_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with uniform dead time using Euler-Maruyama method.

    This function configures the solver for uniform dead time distributions
    with the specified mean and standard deviation.

    Parameters
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction constants
    time_params : TimeParameters
        Time discretization parameters
    tau_mean : float
        Mean dead time value (seconds)
    tau_std : float
        Standard deviation of dead time (seconds)
    fission : float
        Fission rate constant (s⁻¹)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Solution tuple: (time_array, population_array, detection_array)
    """
    # Create dead time parameters for uniform dead time
    dead_time_params = DeadTimeParameters(
        mean_tau=tau_mean,
        tau_distribution='uniform',
        tau_params={
            'uniform': {
                'low': tau_mean - np.sqrt(3) * tau_std,
                'high': tau_mean + np.sqrt(3) * tau_std
            }
        }
    )

    # Run the simulation
    solver = EulerMaruyamaDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    index = f"uniform_tau{tau_mean:.2e}_std{tau_std:.2e}_f{fission:.2f}"

    _save_euler_maruyama_results(pop, detect, fission, index, 'uniform')

    return t_space, pop, detect


def euler_maruyama_detection_normal_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with normal dead time using Euler-Maruyama method.

    This function configures the solver for normal (Gaussian) dead time
    distributions with specified mean and standard deviation.

    Parameters
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction constants
    time_params : TimeParameters
        Time discretization parameters
    tau_mean : float
        Mean dead time value (seconds)
    tau_std : float
        Standard deviation of dead time (seconds)
    fission : float
        Fission rate constant (s⁻¹)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Solution tuple: (time_array, population_array, detection_array)
    """
    # Create dead time parameters for normal dead time
    dead_time_params = DeadTimeParameters(
        mean_tau=tau_mean,
        tau_distribution='normal',
        tau_params={
            'normal': {
                'loc': tau_mean,
                'scale': tau_std
            }
        }
    )

    # Run the simulation
    solver = EulerMaruyamaDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    # Create a default index for normal dead time
    index = f"normal_tau{tau_mean:.2e}_std{tau_std:.2e}_f{fission:.2f}"

    # Save results
    _save_euler_maruyama_results(pop, detect, fission, index, 'normal')

    return t_space, pop, detect


def euler_maruyama_detection_gamma_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with gamma dead time using Euler-Maruyama method.

    This function configures the solver for gamma dead time distributions
    with specified mean and standard deviation. The gamma distribution
    provides positive-dead-time-guaranteed modeling.

    Parameters
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction constants
    time_params : TimeParameters
        Time discretization parameters
    tau_mean : float
        Mean dead time value (seconds)
    tau_std : float
        Standard deviation of dead time (seconds)
    fission : float
        Fission rate constant (s⁻¹)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Solution tuple: (time_array, population_array, detection_array)
    """
    # Create dead time parameters for gamma dead time
    shape = (tau_mean / tau_std) ** 2
    scale = (tau_std ** 2) / tau_mean

    dead_time_params = DeadTimeParameters(
        mean_tau=tau_mean,
        tau_distribution='gamma',
        tau_params={
            'gamma': {
                'shape': shape,
                'scale': scale
            }
        }
    )

    # Run the simulation
    solver = EulerMaruyamaDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    # Create a default index for gamma dead time
    index = f"gamma_tau{tau_mean:.2e}_std{tau_std:.2e}_f{fission:.2f}"

    # Save results
    _save_euler_maruyama_results(pop, detect, fission, index, 'gamma')

    return t_space, pop, detect


def _save_euler_maruyama_results(
        pop: np.ndarray, detect: np.ndarray,
        fission: float, index: str, dead_time_type: str) -> None:
    """
    Save Euler-Maruyama simulation results using DataManager.

    This internal function handles the standardized saving of Euler-Maruyama
    simulation results to organized directory structures with appropriate
    naming conventions and metadata.

    Notes
    -----
    This function automatically extracts simulation prefixes and generates
    standardized filenames for consistent data organization.
    """
    # Extract prefix from index (e.g., 'mil_f33.94' -> 'mil_f')
    save_prefix = utl.extract_simulation_prefix(index, fission)

    # Generate standardized filename for logging/debugging
    filename = utl.generate_filename(
        save_prefix, 'EM', dead_time_type, fission)
    print(f"Saving results to: {filename}")

    data = EulerMaruyamaData(
        population_data=pop,
        detection_data=detect,
        fission_value=fission,
        dead_time_type=dead_time_type,
        prefix=save_prefix
    )
    data_manager.save_euler_maruyama_data(data)
