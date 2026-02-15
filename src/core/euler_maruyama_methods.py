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

import numpy as np
from src.utils import utils as utl
from src.services.data_management import DataManager, EulerMaruyamaData
from src.models.core_parameters import (
    PhysicalParameters,
    TimeParameters,
    DeadTimeParameters
)
from src.detection.detection_sde_model import (
    BaseDetectionSolver,
    DetectionSDEModel,
    IntegrationContext,
    SimulationResults
)


rng = np.random.default_rng()
data_manager = DataManager()


class EulerMaruyamaDetectionSDE(BaseDetectionSolver):
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

    def __init__(
            self,
            physical_params,
            sde_model,
            output_prefix: str = None,
            auto_save: bool = True):
        """
        Initialize Euler-Maruyama solver.
        """
        super().__init__(
            physical_params, sde_model, output_prefix, auto_save
        )

    def integrate_step(
            self,
            context: IntegrationContext) -> float:

        drift = self.sde_model.calculate_drift(
            context.state.pop,
            context.params.dead_time_params,
            self.physical_params,
        )

        wiener = context.random_vars['wiener'][context.state.index]

        return (context.state.detect
                + drift * context.params.dt
                + context.params.sig_3 * wiener)

    def setup_random_variables(
            self,
            grid_points: int,
            dt: float) -> dict[str, np.ndarray]:

        return {
            'wiener': rng.normal(
                loc=0.0,
                scale=np.sqrt(dt),
                size=grid_points
            )
        }

    def _save_results(
            self,
            results: SimulationResults) -> None:
        """Save Euler-Maruyama results."""
        _save_euler_maruyama_results(
            results.pop,
            results.detect,
            results.fission,
            results.index,
            results.dead_time_type)


# Convenience functions for specific dead time types

def euler_maruyama_detection_constant_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau: float,
        fission: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    dead_time_params = DeadTimeParameters(
        mean_tau=tau,
        tau_distribution='constant')

    sde_model = DetectionSDEModel()
    solver = EulerMaruyamaDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission)

    return t_space, pop, detect


def euler_maruyama_detection_uniform_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        std_tau=tau_std
    )

    sde_model = DetectionSDEModel()
    solver = EulerMaruyamaDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission)

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
        std_tau=tau_std
    )

    sde_model = DetectionSDEModel()
    solver = EulerMaruyamaDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission)

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
    dead_time_params = DeadTimeParameters(
        mean_tau=tau_mean,
        tau_distribution='gamma',
        std_tau=tau_std
    )

    sde_model = DetectionSDEModel()
    solver = EulerMaruyamaDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission)

    return t_space, pop, detect


def _save_euler_maruyama_results(
        pop: np.ndarray,
        detect: np.ndarray,
        fission: float,
        index: str,
        dead_time_type: str) -> None:
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
