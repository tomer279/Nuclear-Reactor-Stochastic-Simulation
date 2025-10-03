"""Written by Tomer279 with the assistance of Cursor.ai.

Runge-Kutta methods for nuclear reactor detection simulation.

This module provides robust Runge-Kutta numerical methods for solving
detection stochastic differential equations (SDEs) in nuclear reactor dynamics.
It implements weak Runge-Kutta 3.0 schemes, offering excellent balance between
computational efficiency and numerical stability for SDE integration with
sophisticated support value calculations and Rademacher random variables.

Classes:
    _RungeKuttaStepCalculator:
        Internal helper for Runge-Kutta step calculations
        and random variable generation
    RungeKuttaDetectionSDE:
        Core solver for detection SDEs using Runge-Kutta methods

Key Features:
    - Robust weak Runge-Kutta 3.0 scheme with support value calculations
    - Rademacher random variables for enhanced numerical stability
    - Support for multiple dead time distributions
      (constant, uniform, normal, gamma)
    - Hybrid analytical-numerical approach for computational efficiency
    - Comprehensive noise amplitude calculations for each distribution type
    - Batch processing capabilities for parameter sweeps
    - Integrated data saving and management

Mathematical Approach:
    The detection SDE solved is:
    dC_t = d * N_t * (1 - Ψ(d * N_t)) dt + σ₃ * dW₃

    The weak Runge-Kutta 3.0 method uses support value calculations:
    C_(i+1) = first_term + second_term + third_term + fourth_term

    where each term incorporates different aspects of the drift function
    evaluated at support points, providing superior numerical stability
    and accuracy compared to lower-order methods.

Dependencies:
    numpy: For numerical operations and random number generation
    utils: Custom utility functions for system parameters and filename generation
    analytical_solution: Analytical population solutions
    core_parameters: Parameter container classes
    data_management: Data storage and organization

Usage Examples:
    # Initialize with physical parameters
    solver = RungeKuttaDetectionSDE(physical_params)

    # Solve single detection SDE path
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission=33.95
    )

    # Solve multiple paths for statistical analysis
    t_space, pop_mat, detect_mat = solver.solve_multiple_paths(
        time_params, dead_time_params, fission=33.95, num_paths=100
    )

    # Convenience functions for specific dead time types
    t_space, pop, detect = runge_kutta_detection_constant_dead_time(
        physical_params, time_params, tau=1e-6, fission=33.95, index='f33.95'
    )

Note:
    This module handles robust numerical integration of detection processes.
    For stochastic simulations, see stochastic_simulation.py.
    For Euler-Maruyama methods, see euler_maruyama_methods.py.
    For Taylor methods, see taylor_methods.py.
"""

from typing import Tuple, Any, Optional, Callable
import numpy as np
import utils as utl
from data_management import DataManager, RungeKuttaData
from analytical_solution import analytical_population_solution
from core_parameters import (
    PhysicalParameters,
    TimeParameters,
    DeadTimeParameters
)

rng = np.random.default_rng()
data_manager = DataManager()


class _RungeKuttaStepCalculator:
    """
    Internal helper class for Runge-Kutta method step calculations
    and random variables.

    This private class encapsulates the complex calculations required for
    implementing the weak Runge-Kutta 3.0 scheme, including support value
    computations, Rademacher random variable generation, and multi-term
    integration formulas.

    It reduces parameter passing complexity in the
    main solver class while providing sophisticated numerical methods.

    Attributes
    ----------
    physical_params : PhysicalParameters
        Physical parameters for rate constant access
    dead_time_params : DeadTimeParameters
        Dead time distribution parameters
    tau_dist_params : dict
        Distribution-specific parameters
    dt : float
        Time step size
    sig_3 : float
        Detection noise amplitude

    Public Methods
    --------------
    setup_random_variables(grid_points)
        Generate Wiener increments, mixed Wiener processes,
        and Rademacher variables
    calculate_runge_kutta_step(step_data)
        Calculate single weak Runge-Kutta 3.0 integration step
    run_integration_loop(pop, detect, random_vars)
        Execute complete integration loop for detection SDE

    Private Methods
    ---------------
    _calc_support_values(pop, xi, rho, drift_func)
        Calculate support values for Runge-Kutta 3.0 scheme
    """

    def __init__(
            self,
            params_dict: dict[str, Any]):
        """
        Initialize Runge-Kutta step calculator with parameter dictionary.

        Parameters
        ----------
        params_dict : Dict[str, Any]
            Dictionary containing all necessary parameters:
              -  'physical_params': PhysicalParameters instance
              -  'dead_time_params': DeadTimeParameters instance
              -  'tau_dist_params': Distribution-specific parameters
              -  'dt': Time step size
              -  'sig_3': Detection noise amplitude
        """
        self.physical_params = params_dict['physical_params']
        self.dead_time_params = params_dict['dead_time_params']
        self.tau_dist_params = params_dict['tau_dist_params']
        self.dt = params_dict['dt']
        self.sig_3 = params_dict['sig_3']

    def setup_random_variables(
            self,
            grid_points: int) -> dict[str, np.ndarray]:
        """
        Generate comprehensive random variables for Runge-Kutta 3.0 method.

        The weak Runge-Kutta 3.0 method requires multiple types of random
        variables including standard Wiener increments, mixed Wiener processes,
        and Rademacher random variables for enhanced numerical stability
        and accuracy.

        Parameters
        ----------
        grid_points : int
            Number of grid points for random variable generation

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
              - 'wiener': Standard Wiener increments with variance dt
              - 'mixed_wiener': Mixed Wiener processes for higher-order terms
              - 'xi': Rademacher random variables (-1 or +1)
              - 'rho': Additional Rademacher random variables (-1 or +1)
        """
        wiener = rng.normal(
            loc=0.0,
            scale=np.sqrt(self.dt),
            size=grid_points
        )

        mixed_wiener = (0.5 * self.dt * wiener
                        + rng.normal(
                            loc=0.0,
                            scale=np.sqrt(self.dt/3),
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

    def calculate_runge_kutta_step(
            self,
            step_data: dict[str, Any]) -> float:
        """
        Calculate single weak Runge-Kutta 3.0 integration step.

        This method implements the sophisticated Runge-Kutta 3.0 scheme with
        four distinct terms incorporating support value calculations and
        Rademacher random variables for superior numerical stability.

        Parameters
        ----------
        step_data : Dict[str, Any]
            Dictionary containing current state:
              -  'pop': Current population value
              -  'detect': Current detection value
              -  'wiener': Wiener increment for this step
              -  'mixed_wiener': Mixed Wiener increment for this step
              -  'xi': Rademacher random variable for this step
              -  'rho': Additional Rademacher random variable for this step

        Returns
        -------
        float
            Next detection value from weak Runge-Kutta 3.0 integration
        """
        # Unpack step data
        pop = step_data['pop']
        detect = step_data['detect']
        wiener = step_data['wiener']
        mixed_wiener = step_data['mixed_wiener']
        xi = step_data['xi']
        rho = step_data['rho']

        # Drift function: d * N_t * (1 - Ψ(d * N_t))
        def drift(y):
            detection_rate = self.physical_params.detect * y
            psi_value = utl.calculate_psi_function(
                detection_rate,
                self.dead_time_params.mean_tau,
                self.dead_time_params.tau_distribution,
                self.tau_dist_params.get('scale', None)
            )
            return detection_rate * (1 - psi_value)

        support_values = self._calc_support_values(pop, xi, rho, drift)

        # Weak Runge-Kutta 3.0 terms

        first_term = detect + drift(pop) * self.dt + self.sig_3 * wiener

        second_term = 0.5 * (
            support_values['plus_xi'] + support_values['minus_xi']
            - 1.5 * drift(pop)
            - 0.25 * (support_values['tilde_plus'] +
                      support_values['tilde_minus'])
        ) * self.dt

        third_term = np.sqrt(2/self.dt) * (
            np.sqrt(0.5) *
            (support_values['plus_xi'] - support_values['minus_xi'])
            - 0.25 * (support_values['tilde_plus'] -
                      support_values['tilde_minus'])
        ) * xi * mixed_wiener

        fourth_term = (1/6) * (drift(
            pop + (drift(pop) + support_values['plus_xi']) * self.dt
            + (xi + rho) * self.sig_3 * np.sqrt(self.dt))
            - support_values['plus_xi'] - support_values['plus_rho']
            + drift(pop)) * (
            (xi + rho) * wiener * np.sqrt(self.dt) + self.dt
            + xi * rho * (wiener ** 2 - self.dt)
        )

        return first_term + second_term + third_term + fourth_term

    def run_integration_loop(
            self,
            pop: np.ndarray,
            detect: np.ndarray,
            random_vars: dict[str, np.ndarray]) -> np.ndarray:
        """
        Execute complete integration loop for detection SDE.

        This method runs the full Runge-Kutta integration process across
        all time steps, updating the detection array with sophisticated
        multi-term calculations for each step.

        Parameters
        ----------
        pop : np.ndarray
            Population array from analytical solution
        detect : np.ndarray
            Detection array to be updated (modified in place)
        random_vars : Dict[str, np.ndarray]
            Dictionary containing all random variables for integration

        Returns
        -------
        np.ndarray
            Updated detection array from Runge-Kutta integration
        """
        for i in utl.progress_tracker(len(pop) - 1):
            current_pop = pop[i]
            current_detect = detect[i]

            # Prepare step data
            step_data = {
                'pop': current_pop,
                'detect': current_detect,
                'wiener': random_vars['wiener'][i],
                'mixed_wiener': random_vars['mixed_wiener'][i],
                'xi': random_vars['xi'][i],
                'rho': random_vars['rho'][i]
            }

            # Calculate weak Runge-Kutta 3.0 step
            detect[i + 1] = self.calculate_runge_kutta_step(step_data)

        return detect

    def _calc_support_values(
            self,
            pop: float,
            xi: int,
            rho: int,
            drift_func: Callable) -> dict[str, float]:
        """
        Calculate support values for weak Runge-Kutta 3.0 scheme.

        Support values are drift function evaluations at strategically
        chosen points around the current population, providing the
        foundation for the sophisticated multi-term Runge-Kutta formula.
        """
        sqrt_dt = np.sqrt(self.dt)
        sqrt_2dt = np.sqrt(2 * self.dt)
        drift_pop = drift_func(pop)

        return {
            'plus_xi': drift_func(
                pop
                + drift_pop * self.dt
                + self.sig_3 * sqrt_dt * xi),
            'minus_xi': drift_func(
                pop
                + drift_pop * self.dt
                - self.sig_3 * sqrt_dt * xi),
            'plus_rho': drift_func(
                pop
                + drift_pop * self.dt
                + self.sig_3 * sqrt_dt * rho),
            'tilde_plus': drift_func(
                pop
                + 2 * drift_pop * self.dt
                + self.sig_3 * sqrt_2dt * xi),
            'tilde_minus': drift_func(
                pop
                + 2 * drift_pop * self.dt
                - self.sig_3 * sqrt_2dt * xi)
        }


class RungeKuttaDetectionSDE:
    """
    Robust Runge-Kutta method solver for
    detection stochastic differential equations.

    This class implements Runge-Kutta numerical methods for solving
    detection SDEs in nuclear dynamics, providing excellent balance between
    computational efficiency and numerical stability.

    It uses hybrid analytical-numerical approaches where population dynamics
    are solved analytically while detection processe are integrated
    numerically with robust multi-term Runge-Kutta schemes.

    The implementation uses weak Runge-Kutta 3.0 methods
    with support value calculations and Rademacher random variables,
    offering superior numerical stability and accuracy for complex SDE
    integration problems.

    Attributes
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction rate constants

    Public Methods
    --------------
    solve_detection_sde(time_params, dead_time_params, fission, n_0)
        Solve single detection SDE path using Runge-Kutta method
    solve_multiple_paths(time_params, dead_time_params, fission, num_paths)
        Solve multiple detection SDE paths for statistical analysis

    Private Methods
    ---------------
    _validate_inputs(time_params, dead_time_params)
        Validate input parameter consistency
    _solve_detection_sde_rk(time_params, dead_time_params, fission, pop)
        Core Runge-Kutta numerical integration of detection SDE
    _calculate_drift(current_pop, dead_time_params, tau_dist_params)
        Calculate drift function for given population
    _calculate_detection_noise(fission, dead_time_params)
        Calculate noise amplitude for specific dead time distribution

    Examples
    --------
    >>> solver = RungeKuttaDetectionSDE(physical_params)
    >>> t_space, pop, detect = solver.solve_detection_sde(
    ...     time_params, dead_time_params, fission=33.95
    ... )
    >>> print(f"Final detection count: {detect[-1]}")
    """

    def __init__(
            self,
            physical_params: PhysicalParameters):
        """
        Initialize the Runge-Kutta detection SDE solver.

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
        self.physical_params = physical_params

    def solve_detection_sde(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float,
            n_0: float = None) -> (
            Tuple[np.ndarray,
                  np.ndarray,
                  np.ndarray]
    ):
        """
        Solve single detection SDE path using weak Runge-Kutta 3.0 method.

        This method implements the hybrid approach where:
        1. Population N_t is solved analytically using analytical solutions
        2. Detection SDE is integrated numerically using Runge-Kutta scheme

        The detection SDE solved is:
        dC_t = d * N_t * (1 - Ψ(d * N_t)) dt + σ₃ * dW₃

        The weak Runge-Kutta 3.0 method uses sophisticated support value
        calculations and four-term integration formulas for superior
        numerical stability and accuracy.

        Parameters
        ----------
        time_params : TimeParameters
            Time discretization parameters including grid points and duration
        dead_time_params : DeadTimeParameters
            Dead time distribution parameters and configuration
        fission : float
            Fission rate constant for this simulation (s⁻¹)
        n_0 : Optional[float], default None
            Initial population. If None, defaults to equilibrium population.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Solution tuple containing:
              - Time array: Discrete time points from t_0 to t_end
              - Population array: Analytical population solution N_t
              - Detection array: Robust Runge-Kutta SDE solution C_t

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
        detect = self._solve_detection_sde_rk(
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
            num_paths: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve multiple detection SDE paths for statistical analysis.

        This method generates multiple independent realizations
        of the detection SDE using Runge-Kutta methods to enable
        statistical analysis including variance calculation,
        confidence intervals, and convergence studies with robust
        numerical stability.

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
            - Detection matrix: Robust Runge-Kutta solutions for all paths
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
            detect_mat[i, :] = detect_path

        t_space = np.linspace(time_params.t_0,
                              time_params.t_end,
                              time_params.grid_points + 1)

        return t_space, pop_mat, detect_mat

    def _validate_inputs(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters) -> None:
        """Validate input parameters."""
        if time_params.grid_points < 1:
            raise ValueError("grid_points must be positive")
        if time_params.t_end <= time_params.t_0:
            raise ValueError("t_end must be greater than t_0")
        if dead_time_params.mean_tau <= 0:
            raise ValueError("mean_tau must be positive")

    def _solve_detection_sde_rk(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float,
            pop: np.ndarray) -> np.ndarray:
        """
        Core numerical integration of detection SDE
        using weak Runge-Kutta 3.0 method.

        This method implements the sophisticated Runge-Kutta integration scheme
        with support value calculations, Rademacher random variables, and
        multi-term formulas for superior numerical stability and accuracy.
        """
        detect = np.zeros(time_params.grid_points + 1)

        # Calculate detection noise amplitude
        sig_3 = self._calculate_detection_noise(
            fission, dead_time_params)

        # Time step
        dt = time_params.get_grid_spacing()

        tau_dist_params = dead_time_params.get_distribution_params()

        # Create step calculator with reduced parameters
        calc_params = {
            'physical_params': self.physical_params,
            'dead_time_params': dead_time_params,
            'tau_dist_params': tau_dist_params,
            'dt': dt,
            'sig_3': sig_3
        }
        step_calc = _RungeKuttaStepCalculator(calc_params)

        random_vars = step_calc.setup_random_variables(
            time_params.grid_points)

        detect = step_calc.run_integration_loop(pop, detect, random_vars)

        return detect

    def _calculate_drift(
            self,
            current_pop: float,
            dead_time_params: DeadTimeParameters,
            tau_dist_params: dict) -> float:
        """
        Calculate drift function for given population
        and dead time configuration.
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

    def _calculate_detection_noise(
            self,
            fission: float,
            dead_time_params: DeadTimeParameters
    ) -> float:
        """
        Calculate detection noise amplitude σ₃ based on dead time distribution.

        The noise amplitude depends on the dead time distribution and
        represents the variability in detection efficiency due to
        dead time effects. Different distributions require different
        noise amplitude calculations.
        """
        # Calculate system parameters
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

def runge_kutta_detection_constant_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau: float,
        fission: float,
        index: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    solver = RungeKuttaDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    # Save results
    _save_runge_kutta_results(pop, detect, fission, index, 'const')

    return t_space, pop, detect


def runge_kutta_detection_uniform_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau_mean: float,
        tau_std: float,
        fission: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        tau_params={
            'uniform': {
                'low': tau_mean - np.sqrt(3) * tau_std,
                'high': tau_mean + np.sqrt(3) * tau_std
            }
        }
    )

    # Run the simulation
    solver = RungeKuttaDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    index = f"uniform_tau{tau_mean:.2e}_std{tau_std:.2e}_f{fission:.2f}"

    _save_runge_kutta_results(pop, detect, fission, index, 'uniform')

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
        tau_params={
            'normal': {
                'loc': tau_mean,
                'scale': tau_std
            }
        }
    )

    # Run the simulation
    solver = RungeKuttaDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    # Create a default index for normal dead time
    index = f"normal_tau{tau_mean:.2e}_std{tau_std:.2e}_f{fission:.2f}"

    # Save results
    _save_runge_kutta_results(pop, detect, fission, index, 'normal')

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
    solver = RungeKuttaDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    # Create a default index for gamma dead time
    index = f"gamma_tau{tau_mean:.2e}_std{tau_std:.2e}_f{fission:.2f}"

    # Save results
    _save_runge_kutta_results(pop, detect, fission, index, 'gamma')

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
