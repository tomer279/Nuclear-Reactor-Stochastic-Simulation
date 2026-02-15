"""Written by Tomer279 with the assistance of Cursor.ai.

Runge-Kutta methods for nuclear reactor detection simulation.

This module provides Runge-Kutta numerical methods for solving
detection stochastic differential equations (SDEs) using the unified
detection SDE model framework.

Classes
-------
RungeKuttaDetectionSDE
    Runge-Kutta 3.0 solver using unified detection SDE model
SupportValueParameters
    Container for support value calculation parameters

Functions
---------
calculate_support_values(pop, params)
    Calculate support values for Runge-Kutta 3.0 method
runge_kutta_detection_constant_dead_time(...)
    Convenience function for constant dead time
runge_kutta_detection_uniform_dead_time(...)
    Convenience function for uniform dead time
runge_kutta_detection_normal_dead_time(...)
    Convenience function for normal dead time
runge_kutta_detection_gamma_dead_time(...)
    Convenience function for gamma dead time
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from src.utils import utils as utl
from src.services.data_management import DataManager, RungeKuttaData
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


@dataclass
class SupportValueParameters:
    """
    Container for support value calculation parameters.

    Support values are used in weak Runge-Kutta 3.0 methods to evaluate
    the drift function at strategically chosen points around the current state
    for higher-order accuracy.

    Attributes
    ----------
    drift_func : Callable
        Drift function to evaluate at support points
    dt : float
        Time step size
    sig_3 : float
        Detection noise amplitude σ₃
    xi : int
        Rademacher random variable (-1 or +1)
    rho : int
        Additional Rademacher random variable (-1 or +1)

    Examples
    --------
    >>> params = SupportValueParameters(
    ...     drift_func=lambda p: calculate_drift(p, ...),
    ...     dt=1e-6, sig_3=0.1, xi=1, rho=-1
    ... )
    """
    drift_func: Callable
    dt: float
    sig_3: float
    xi: int
    rho: int


def calculate_support_values(
        pop: float,
        params: SupportValueParameters) -> dict[str, float]:
    """
    Calculate support values for weak Runge-Kutta 3.0 method.

    Support values are drift function evaluations at strategically
    chosen points around the current population, providing the
    foundation for the multi-term Runge-Kutta 3.0 integration formula.

    Parameters
    ----------
    pop : float
        Current population value N_i
    params : SupportValueParameters
        Container with calculation parameters

    Returns
    -------
    dict[str, float]
        Dictionary containing support values at different evaluation points:
          - 'plus_xi': 
              :math:`f_\\xi^+ = f\\left(N + f(N)dt + \\sigma_3 \\sqrt{dt} \\xi\\right)`
          - 'minus_xi': 
              :math:`f_\\xi^- = f\\left(N + f(N)dt - \\sigma_3 \\sqrt{dt} \\xi\\right)`
          - 'plus_rho': 
              :math:`f_\\rho^+ = f\\left(N + f(N)dt + \\sigma_3 \\sqrt{dt} \\rho\\right)`
          - 'tilde_plus': 
              :math:`\\tilde{f}_\\xi^+ = f\\left(N + 2f(N)dt + \\sigma_3 \\sqrt{2dt} \\xi\\right)`
          - 'tilde_minus': 
              :math:`\\tilde{f}_\\xi^- = f\\left(N + 2f(N)dt - \\sigma_3 \\sqrt{2dt} \\xi\\right)`

    Examples
    --------
    >>> params = SupportValueParameters(
    ...     drift_func=drift_function, dt=1e-6, sig_3=0.1, xi=1, rho=-1
    ... )
    >>> support_vals = calculate_support_values(pop=100.0, params=params)
    """
    sqrt_dt = np.sqrt(params.dt)
    sqrt_2dt = np.sqrt(2 * params.dt)
    drift_pop = params.drift_func(pop)

    return {
        'plus_xi': params.drift_func(
            pop
            + drift_pop * params.dt
            + params.sig_3 * sqrt_dt * params.xi),
        'minus_xi': params.drift_func(
            pop
            + drift_pop * params.dt
            - params.sig_3 * sqrt_dt * params.xi),
        'plus_rho': params.drift_func(
            pop
            + drift_pop * params.dt
            + params.sig_3 * sqrt_dt * params.rho),
        'tilde_plus': params.drift_func(
            pop
            + 2 * drift_pop * params.dt
            + params.sig_3 * sqrt_2dt * params.xi),
        'tilde_minus': params.drift_func(
            pop
            + 2 * drift_pop * params.dt
            - params.sig_3 * sqrt_2dt * params.xi)
    }


class RungeKuttaDetectionSDE(BaseDetectionSolver):
    """
    Runge-Kutta 3.0 method solver for detection SDE.

    This class implements the weak Runge-Kutta 3.0 numerical integration scheme
    for the detection SDE. The method uses support value calculations
    and four-term integration formula for a theoretically improved
    numerical stability and accuracy.

    The Runge-Kutta 3.0 formula is:
        C_{i+1} = first_term + second_term + third_term + fourth_term

    where each term incorporates support values and Rademacher random variables.

    Attributes
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction rate constants
    sde_model : DetectionSDEModel
        Unified detection SDE model for common calculations

    Methods
    -------
    integrate_step(context)
        Runge-Kutta 3.0 integration step implementation
    setup_random_variables(grid_points, dt)
        Generate Wiener, mixed Wiener, and Rademacher random variables

    Examples
    --------
    >>> sde_model = DetectionSDEModel()
    >>> solver = RungeKuttaDetectionSDE(physical_params, sde_model)
    >>> t, pop, detect = solver.solve_detection_sde(
    ...     time_params, dead_time_params, fission=33.95
    ... )

    Notes
    -----
    The Runge-Kutta 3.0 method achieves third-order weak convergence using
    Rademacher random variables and support value calculations, providing
    excellent balance between accuracy and computational efficiency.
    """

    def __init__(
            self,
            physical_params,
            sde_model,
            output_prefix: str = None,
            auto_save: bool = True):

        super().__init__(
            physical_params, sde_model, output_prefix, auto_save
        )

    def integrate_step(
            self,
            context: IntegrationContext
    ) -> float:
        """
        Runge-Kutta 3.0 integration step.

        Implements the weak Runge-Kutta 3.0 scheme with four terms:
            C_{i+1} = first_term + second_term + third_term + fourth_term

        Parameters
        ----------
        context : IntegrationContext
            Integration context containing state, parameters, and random variables

        Returns
        -------
        float
            Next detection value C_{i+1}
        """
        # Create drift function using unified model
        def drift_func(population):
            return self.sde_model.calculate_drift(
                population,
                context.params.dead_time_params,
                self.physical_params
            )

        # Get random variables for this step
        idx = context.state.index
        xi = context.random_vars['xi'][idx]
        rho = context.random_vars['rho'][idx]
        wiener = context.random_vars['wiener'][idx]
        mixed_wiener = context.random_vars['mixed_wiener'][idx]

        # Calculate support values
        support_params = SupportValueParameters(
            drift_func=drift_func,
            dt=context.params.dt,
            sig_3=context.params.sig_3,
            xi=xi,
            rho=rho
        )
        support_values = calculate_support_values(
            context.state.pop,
            support_params
        )

        # Calculate drift at current population
        drift_current = drift_func(context.state.pop)

        # Weak Runge-Kutta 3.0 four-term formula
        first_term = (
            context.state.detect
            + drift_current * context.params.dt
            + context.params.sig_3 * wiener
        )

        second_term = 0.5 * (
            support_values['plus_xi'] + support_values['minus_xi']
            - 1.5 * drift_current
            - 0.25 * (support_values['tilde_plus'] +
                      support_values['tilde_minus'])
        ) * context.params.dt

        third_term = np.sqrt(2 / context.params.dt) * (
            np.sqrt(0.5) *
            (support_values['plus_xi'] - support_values['minus_xi'])
            - 0.25 * (support_values['tilde_plus'] -
                      support_values['tilde_minus'])
        ) * xi * mixed_wiener

        fourth_term = (1/6) * (
            drift_func(
                context.state.pop
                + (drift_current +
                   support_values['plus_xi']) * context.params.dt
                + (xi + rho) * context.params.sig_3 *
                np.sqrt(context.params.dt)
            )
            - support_values['plus_xi']
            - support_values['plus_rho']
            + drift_current
        ) * (
            (xi + rho) * wiener * np.sqrt(context.params.dt)
            + context.params.dt
            + xi * rho * (wiener ** 2 - context.params.dt)
        )

        return first_term + second_term + third_term + fourth_term

    def setup_random_variables(
            self,
            grid_points: int,
            dt: float) -> dict[str, np.ndarray]:
        """
        Generate random variables for Runge-Kutta 3.0 method.

        The weak Runge-Kutta 3.0 method requires Wiener increments,
        mixed Wiener processes, and two independent Rademacher random variables.

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
            - 'wiener': Standard Wiener increments
            - 'mixed_wiener': Mixed Wiener processes
            - 'xi': Rademacher random variables (-1 or +1)
            - 'rho': Additional Rademacher random variables (-1 or +1)
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

        xi = rng.choice([-1, 1], size=grid_points)
        rho = rng.choice([-1, 1], size=grid_points)

        return {
            'wiener': wiener,
            'mixed_wiener': mixed_wiener,
            'xi': xi,
            'rho': rho
        }

    def _save_results(
            self,
            results: SimulationResults) -> None:
        """Save Runge-Kutta results."""
        _save_runge_kutta_results(
            results.pop,
            results.detect,
            results.fission,
            results.index,
            results.dead_time_type)


# Convenience functions for specific dead time types

def runge_kutta_detection_constant_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau: float,
        fission: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with constant dead time using Runge-Kutta method.

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
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Solution tuple: (time_array, population_array, detection_array)

    Examples
    --------
    >>> t_space, pop, detect = runge_kutta_detection_constant_dead_time(
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

    # Run the simulation
    sde_model = DetectionSDEModel()
    solver = RungeKuttaDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    return t_space, pop, detect


def runge_kutta_detection_uniform_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with uniform dead time using robust Runge-Kutta method.

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
    solver = RungeKuttaDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    return t_space, pop, detect


def runge_kutta_detection_normal_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with normal dead time using Runge-Kutta method.

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
    solver = RungeKuttaDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    return t_space, pop, detect


def runge_kutta_detection_gamma_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve detection SDE with gamma dead time using Runge-Kutta method.

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
    solver = RungeKuttaDetectionSDE(physical_params, sde_model)

    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    return t_space, pop, detect


def _save_runge_kutta_results(
        pop: np.ndarray,
        detect: np.ndarray,
        fission: float,
        index: str,
        dead_time_type: str) -> None:
    """
    Save Runge-Kutta simulation results using DataManager.

    This internal function handles the standardized saving of Runge-Kutta
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
        save_prefix, 'RK', dead_time_type, fission)
    print(f"Saving results to: {filename}")

    data = RungeKuttaData(
        population_data=pop,
        detection_data=detect,
        fission_value=fission,
        dead_time_type=dead_time_type,
        prefix=save_prefix
    )
    data_manager.save_runge_kutta_data(data)
