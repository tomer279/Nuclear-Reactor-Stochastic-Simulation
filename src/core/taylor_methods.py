""" Written by Tomer279 with the assistance of Cursor.ai

Taylor method implementations for solving stochastic differential equations.

This module implements Strong Taylor 1.5 and Weak Taylor 2.0 methods
for solving SDEs in the context of nuclear reactor population dynamics.
"""

from typing import Tuple
import numpy as np
from src.utils import utils as utl
from src.models.core_parameters import (
    PhysicalParameters,
    TimeParameters,
    DeadTimeParameters)
from src.detection.detection_sde_model import (
    BaseDetectionSolver,
    DetectionSDEModel,
    IntegrationContext,
    SimulationResults
)
from src.utils.laplace_transforms import DeadTimeLaplaceCalculator
from src.services.data_management import DataManager, TaylorData


rng = np.random.default_rng()
data_manager = DataManager()


def _calculate_drift_derivative(
        pop: float,
        dead_time_params: DeadTimeParameters,
        physical_params: PhysicalParameters,
        laplace_calculator: DeadTimeLaplaceCalculator,
        derivative_order: int = 1
):

    lambda_d = physical_params.detect
    x = lambda_d * pop

    tau = dead_time_params.mean_tau

    std = dead_time_params.get_std()

    laplace_result = laplace_calculator.calculate_laplace_transform(
        x=x,
        tau=tau,
        distribution=dead_time_params.tau_distribution,
        std=std
    )

    if derivative_order == 1:
        df_dx = (
            laplace_result.value
            + x * laplace_result.derivative
        )
        return lambda_d * df_dx

    if derivative_order == 2:
        if laplace_result.second_derivative is None:
            raise ValueError(
                "Second derivative not available for "
                f"{dead_time_params.tau_distribution} distribution"
            )
        d2f_dx2 = (
            2 * laplace_result.derivative
            + x * laplace_result.second_derivative
        )

        return (lambda_d ** 2) * d2f_dx2

    raise ValueError(
        f"Derivative order {derivative_order} not supported. "
    )


class TaylorDetectionSDE(BaseDetectionSolver):
    """
    High-precision Taylor method solver for
    detection stochastic differential equations.

    This class implements Taylor numerical methods for solving
    detection SDEs in nuclear dynamics, providing superior accuracy
    compared to lower-order methods through higher-order terms and
    drift derivative calculations.

    It uses hybrid analytical-numerical approaches
    where population dynamics are solved analytically while detection processes
    are integrated numerically.

    The implementation uses weak 2.0 Taylor schemes which are equivalent to
    strong 1.5 Taylor methods for scalar diffusion systems, offering excellent
    balance between computational efficiency and numerical accuracy.

    Attributes
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction rate constants

    Public Methods
    --------------
    solve_detection_sde(time_params, dead_time_params, fission, n_0)
        Solve single detection SDE path using Taylor method
    solve_multiple_paths(time_params, dead_time_params, fission, num_paths)
        Solve multiple detection SDE paths for statistical analysis

    Private Methods
    ---------------
    _validate_inputs(time_params, dead_time_params)
        Validate input parameter consistency
    _solve_detection_sde_taylor(time_params, dead_time_params, fission, pop)
        Core Taylor numerical integration of detection SDE
    _calculate_detection_noise(fission, dead_time_params)
        Calculate noise amplitude for specific dead time distribution

    Examples
    --------
    >>> solver = TaylorDetectionSDE(physical_params)
    >>> t_space, pop, detect = solver.solve_detection_sde(
    ...     time_params, dead_time_params, fission=33.95
    ... )
    >>> print(f"Final detection count: {detect[-1]}")
    """

    def __init__(
            self,
            physical_params: PhysicalParameters,
            sde_model: DetectionSDEModel,
            output_prefix: str = None,
            auto_save: bool = True):
        """
        Initialize the Taylor detection SDE solver.

        Parameters
        ----------
        physical_params : PhysicalParameters
            Physical parameters containing nuclear reaction rate constants
            and fission probability distributions

        Raises
        ------
        ValueError
            If physical parameters are invalid or incomplete
        """
        super().__init__(
            physical_params, sde_model, output_prefix, auto_save)
        self.laplace_calculator = DeadTimeLaplaceCalculator()

    def integrate_step(self, context: IntegrationContext) -> float:
        """
        Taylor 2.0 integration step.

        Implements the Weak Taylor 2.0 scheme with higher-order terms:
            C_{i+1} = C_i + drift * dt + σ₃ * (dW + drift' * dZ)
                      + 0.5 * (drift * drift' + 0.5 * σ₃² * drift'') * dt²

        Parameters
        ----------
        context : IntegrationContext
            Integration context containing state, parameters, and random variables

        Returns
        -------
        float
            Next detection value C_{i+1}
        """
        # Calculate drift using unified model
        drift = self.sde_model.calculate_drift(
            context.state.pop,
            context.params.dead_time_params,
            self.physical_params
        )

        # Calculate drift derivatives using Laplace transforms
        drift_derivative = _calculate_drift_derivative(
            context.state.pop,
            context.params.dead_time_params,
            self.physical_params,
            self.laplace_calculator,
            derivative_order=1
        )

        drift_second_derivative = _calculate_drift_derivative(
            context.state.pop,
            context.params.dead_time_params,
            self.physical_params,
            self.laplace_calculator,
            derivative_order=2
        )

        # Get random variables for this step
        wiener = context.random_vars['wiener'][context.state.index]
        mixed_wiener = context.random_vars['mixed_wiener'][context.state.index]

        # Weak Taylor 2.0 formula
        next_detect = (
            context.state.detect
            + drift * context.params.dt
            + context.params.sig_3 * (wiener + drift_derivative * mixed_wiener)
            + 0.5 * (context.params.dt ** 2) * (
                drift * drift_derivative
                + 0.5 * (context.params.sig_3 ** 2) * drift_second_derivative
            )
        )

        return next_detect

    def setup_random_variables(
            self,
            grid_points: int,
            dt: float) -> dict[str, np.ndarray]:
        """
        Generate Wiener and mixed Wiener increments for Taylor 2.0 method.

        The weak 2.0 Taylor method requires both standard Wiener increments
        and mixed Wiener processes (integrals of Wiener processes) for
        higher-order accuracy.

        Parameters
        ----------
        grid_points : int
            Number of grid points
        dt : float
            Time step size

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing:
            - 'wiener': Standard Wiener increments with variance dt
            - 'mixed_wiener': Mixed Wiener processes for higher-order terms
        """
        wiener = rng.normal(
            loc=0.0,
            scale=np.sqrt(dt),
            size=grid_points
        )

        mixed_wiener = (
            0.5 * dt * wiener +
            rng.normal(
                loc=0.0,
                scale=np.sqrt(dt/3),
                size=grid_points
            )
        )

        return {
            'wiener': wiener,
            'mixed_wiener': mixed_wiener
        }

    def _save_results(
            self,
            results: SimulationResults) -> None:
        """Save Taylor results."""
        _save_taylor_results(
            results.pop,
            results.detect,
            results.fission,
            results.index,
            results.dead_time_type)


# Convenience functions for specific dead time types


def taylor_detection_constant_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau: float,
        fission: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with constant dead time using Taylor method.

    This convenience function wraps the main solver class with constant
    dead time parameters for simplified usage in parameter sweeps
    and simulations.

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
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Solution tuple: (time_array, population_array, detection_array)

    Examples
    --------
    >>> t_space, pop, detect = taylor_detection_constant_dead_time(
    ...     physical_params,
    ...     time_params,
    ...     tau=1e-6,
    ...     fission=33.95,
    ...     index='f33.95')
    """
    # Create dead time parameters for constant dead time
    dead_time_params = DeadTimeParameters(
        mean_tau=tau,
        tau_distribution='constant')

    sde_model = DetectionSDEModel()
    solver = TaylorDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    return t_space, pop, detect


def taylor_detection_uniform_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with uniform dead time using Taylor method.

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
    solver = TaylorDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    return t_space, pop, detect


def taylor_detection_normal_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with normal dead time using Taylor method.

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

    # Run the simulation
    sde_model = DetectionSDEModel()
    solver = TaylorDetectionSDE(physical_params, sde_model)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    return t_space, pop, detect


def taylor_detection_gamma_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with gamma dead time using Taylor method.

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

    # Run the simulation
    sde_model = DetectionSDEModel()
    solver = TaylorDetectionSDE(physical_params, sde_model)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    return t_space, pop, detect


def _save_taylor_results(
        pop: np.ndarray,
        detect: np.ndarray,
        fission: float,
        index: str,
        dead_time_type: str) -> None:
    """
    Save Taylor method simulation results using DataManager.

    This internal function handles the standardized saving of Taylor method
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
        save_prefix, 'Taylor', dead_time_type, fission)
    print(f"Saving results to: {filename}")

    data = TaylorData(
        population_data=pop,
        detection_data=detect,
        fission_value=fission,
        dead_time_type=dead_time_type,
        prefix=save_prefix
    )
    data_manager.save_taylor_data(data)
