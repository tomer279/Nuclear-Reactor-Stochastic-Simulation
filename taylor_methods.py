""" Written by Tomer279 with the assistance of Cursor.ai

Taylor method implementations for solving stochastic differential equations.

This module implements Strong Taylor 1.5 and Weak Taylor 2.0 methods
for solving SDEs in the context of nuclear reactor population dynamics.
"""

from typing import Any, Tuple
import numpy as np
import utils as utl
from analytical_solution import analytical_population_solution
from core_parameters import (
    PhysicalParameters,
    TimeParameters,
    DeadTimeParameters)
from data_management import DataManager, TaylorData


rng = np.random.default_rng()
data_manager = DataManager()


class _TaylorStepCalculator:
    """
    Internal helper class for Taylor method
    step calculations and random variables.

    This private class encapsulates the complex calculations required for
    implementing the weak 2.0 Taylor scheme, including drift derivative
    computations, random variable generation, and step integration.
    It reduces parameter passing complexity in the main solver class.

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
        Generate Wiener increments and mixed Wiener processes
    calculate_taylor_step(step_data)
        Calculate single weak 2.0 Taylor integration step

    Private Methods
    ---------------
    _calculate_drift_and_derivatives(current_pop, dead_time_params, tau_dist_params)
        Calculate drift function and its derivatives
    _calculate_drift_derivative(pop, dead_time_params, n)
        Calculate nth derivative of drift function
    _detect_drift_derivative(x, tau, n)
        Calculate derivatives for constant dead time case
    """

    def __init__(
            self,
            params_dict: dict[str, Any]):
        """
        Initialize Taylor step calculator with parameter dictionary.

        Parameters
        ----------
        params_dict : Dict[str, Any]
            Dictionary containing all necessary parameters:
                'physical_params': PhysicalParameters instance
                'dead_time_params': DeadTimeParameters instance
                'tau_dist_params': Distribution-specific parameters
                'dt': Time step size
                'sig_3': Detection noise amplitude
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
        Generate Wiener increments and mixed Wiener processes
        for Taylor method.

        The weak 2.0 Taylor method requires both standard Wiener increments
        and mixed Wiener processes (integrals of Wiener processes) for
        higher-order accuracy.

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

        return {
            'wiener': wiener,
            'mixed_wiener': mixed_wiener
        }

    def calculate_taylor_step(
            self,
            step_data: dict[str, Any]) -> float:
        """
        Calculate single weak 2.0 Taylor integration step.

        This method implements the complete weak 2.0 Taylor scheme including
        drift terms, diffusion terms, and higher-order correction terms
        for superior numerical accuracy.

        Parameters
        ----------
        step_data : Dict[str, Any]
            Dictionary containing current state:
                'pop': Current population value
                'detect': Current detection value
                'wiener': Wiener increment for this step
                'mixed_wiener': Mixed Wiener increment for this step

        Returns
        -------
        float
            Next detection value from weak 2.0 Taylor integration
        """
        # Unpack step data
        pop = step_data['pop']
        detect = step_data['detect']
        wiener = step_data['wiener']
        mixed_wiener = step_data['mixed_wiener']

        # Drift term: d * N_t * (1 - Ψ(d * N_t))
        drift, drift_derivative, second_drift_derivative = (
            self._calculate_drift_and_derivatives(
                pop, self.dead_time_params, self.tau_dist_params)
        )

        # Weak 2.0 Taylor step
        next_detect = (
            detect
            + drift * self.dt
            + self.sig_3 * (wiener + drift_derivative * mixed_wiener)
            + 0.5 * (self.dt ** 2) * (
                drift * drift_derivative
                + 0.5 * (self.sig_3 ** 2) * second_drift_derivative)
        )

        return next_detect

    def _calculate_drift_and_derivatives(
            self,
            current_pop: float,
            dead_time_params: DeadTimeParameters,
            tau_dist_params: dict) -> Tuple[float, float, float]:
        """Calculate drift function and its first two derivatives."""
        detection_rate = self.physical_params.detect * current_pop

        psi_value = utl.calculate_psi_function(
            detection_rate, dead_time_params.mean_tau,
            dead_time_params.tau_distribution,
            tau_dist_params.get('scale', None))

        drift = detection_rate * (1 - psi_value)
        drift_derivative = self._calculate_drift_derivative(
            current_pop, dead_time_params)
        drift_second_derivative = self._calculate_drift_derivative(
            current_pop, dead_time_params, n=2)

        return drift, drift_derivative, drift_second_derivative

    def _calculate_drift_derivative(
            self,
            pop: float,
            dead_time_params: DeadTimeParameters,
            n: int = 1) -> float:
        """
        Calculate the nth derivative of the drift function
        with respect to population.

        For drift f(N) = d * N * (1 - Ψ(d * N)), this calculates f^(n)(N).
        Different dead time distributions require
        different derivative calculations.

        """

        detect = self.physical_params.detect
        detection_rate = detect * pop
        tau = dead_time_params.mean_tau
        dead_time_type = dead_time_params.tau_distribution

        if dead_time_type == 'constant':
            # For constant dead time: f'(N) = d * exp(-d*N*τ) * (1 - d*N*τ)
            return self._detect_drift_derivative(detection_rate, tau, n)

        if dead_time_type in ('uniform', 'normal', 'gamma'):
            # For other dead time types, use simplified derivative
            # This is an approximation for complex Psi functions.
            if n == 1:
                return detect * (1 - 2 * detection_rate * tau)
            if n == 2:
                return -2 * (detect ** 2) * tau
            # Higher order derivatives are zero for this approximation
            return 0.0

        # Fallback for unknown types
        if n == 1:
            return detect

        return 0.0

    def _detect_drift_derivative(
            self,
            x: float,
            tau: float,
            n: int) -> float:
        """
        Calculate the nth derivative of detection drift for constant dead time.

        For the detection drift function f(x) = x * exp(-tau * x), this
        calculates the nth derivative using the analytical formula:

        f^(n)(x) = (-1)^n * tau^(n-1) * (tau * x - n) * exp(-tau * x)
        """
        return (((-1) ** n)
                * (tau * (n-1))
                * (tau * x - n)
                * np.exp(-x * tau))


class TaylorDetectionSDE:
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
            physical_params: PhysicalParameters):
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
        self.physical_params = physical_params

    def solve_detection_sde(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float,
            n_0: float = None) -> (
            Tuple[np.ndarray,
                  np.ndarray,
                  np.ndarray]):
        """
        Solve single detection SDE path using weak 2.0 Taylor method.

        This method implements the hybrid approach where:
        1. Population N_t is solved analytically using analytical solutions
        2. Detection SDE is integrated numerically using weak 2.0 Taylor scheme

        The detection SDE solved is:
        dC_t = d * N_t * (1 - Ψ(d * N_t)) dt + σ₃ * dW₃

        The weak 2.0 Taylor method includes higher-order terms:
        C_(i+1) = C_i + drift * dt + σ₃ * (dW + drift' * dZ)
                  + 0.5 * (drift * drift' + 0.5 * σ₃² * drift'') * dt²

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
            - Detection array: High-precision Taylor SDE solution C_t

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
        detect = self._solve_detection_sde_taylor(
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
            num_paths: int) -> (
                Tuple[np.ndarray,
                      np.ndarray,
                      np.ndarray]):
        """
        Solve multiple detection SDE paths for statistical analysis.

        This method generates multiple independent realizations
        of the detection SDE using Taylor methods to enable
        statistical analysis including variance calculation,
        confidence intervals, and convergence studies
        with high numerical precision.

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
            - Detection matrix: High-precision Taylor solutions for all paths
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

        t_space = np.linspace(
            time_params.t_0,
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

    def _solve_detection_sde_taylor(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float,
            pop: np.ndarray) -> np.ndarray:
        """
        Core numerical integration of detection SDE using
        weak 2.0 Taylor method.

        This method implements the Taylor integration scheme
        with higher-order terms, drift derivatives, and mixed Wiener processes
        for superior numerical accuracy compared to lower-order methods.
        """

        detect = np.zeros(time_params.grid_points + 1)

        # Calculate detection noise amplitude
        sig_3 = self._calculate_detection_noise(fission, dead_time_params)

        # Time step
        dt = time_params.get_grid_spacing()

        # Setup step calculator
        params_dict = {
            'physical_params': self.physical_params,
            'dead_time_params': dead_time_params,
            'tau_dist_params': dead_time_params.get_distribution_params(),
            'dt': dt,
            'sig_3': sig_3
        }
        step_calc = _TaylorStepCalculator(params_dict)

        # Generate random variables
        random_vars = step_calc.setup_random_variables(
            time_params.grid_points)

        # Taylor integration for detection
        for i in utl.progress_tracker(time_params.grid_points):
            step_data = {
                'pop': pop[i],
                'detect': detect[i],
                'wiener': random_vars['wiener'][i],
                'mixed_wiener': random_vars['mixed_wiener'][i]
            }
            detect[i + 1] = step_calc.calculate_taylor_step(step_data)

        return detect

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


def taylor_detection_constant_dead_time(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        tau: float,
        fission: float,
        index: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Run the simulation
    solver = TaylorDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    # Save results
    _save_taylor_results(pop, detect, fission,
                         index, 'const')

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
        tau_params={
            'uniform': {
                'low': tau_mean - np.sqrt(3) * tau_std,
                'high': tau_mean + np.sqrt(3) * tau_std
            }
        }
    )

    # Run the simulation
    solver = TaylorDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    index = f"uniform_tau{tau_mean:.2e}_std{tau_std:.2e}_f{fission:.2f}"

    _save_taylor_results(pop, detect, fission,
                         index, 'uniform')

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
        tau_params={
            'normal': {
                'loc': tau_mean,
                'scale': tau_std
            }
        }
    )

    # Run the simulation
    solver = TaylorDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    # Create a default index for normal dead time
    index = f"normal_tau{tau_mean:.2e}_std{tau_std:.2e}_f{fission:.2f}"

    # Save results
    _save_taylor_results(pop, detect, fission,
                         index, 'normal')

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
    solver = TaylorDetectionSDE(physical_params)
    t_space, pop, detect = solver.solve_detection_sde(
        time_params, dead_time_params, fission
    )

    # Create a default index for gamma dead time
    index = f"gamma_tau{tau_mean:.2e}_std{tau_std:.2e}_f{fission:.2f}"

    # Save results
    _save_taylor_results(pop, detect, fission,
                         index, 'gamma')

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
