"""Written by Tomer279 with the assistance of Cursor.ai.

Unified Detection SDE Model for Nuclear Reactor Simulations.

This module provides a unified model for the detection stochastic 
differential equation, by centralizing the common equation, drift function,
noise amplitude calculations, and integration framework.

The unified model encapsulates the detection SDE:

    .. math::
        dC_t = \\lambda_d N_t(1 - \\Psi(\\lambda_d N_t)) dt + \\sigma_3dW₃

where:
    - :math:`C_t`: accumulated detections at time t
    - :math:`N_t`: population at time t (solved analytically)
    - :math:`\\Psi`: Psi function encoding dead time effects
    - :math:`\\sigma_3`: detection noise amplitude (distribution-dependent)
    - :math:`dW_3`: Wiener increments

Classes
-------
StepState
    Container for current state values at integration step (detect, pop, index).

IntegrationParameters
    Container for integration parameters that remain constant during simulation
    (dt, sig_3, dead_time_params).

IntegrationContext
    Context container bundling state, parameters, and random variables for
    integration steps.

DetectionSDEModel
    Unified model containing common equation components for drift and noise
    amplitude calculations using rigorous Laplace transform approach.

BaseDetectionSolver
    Abstract base class for numerical method implementations providing common
    framework for hybrid analytical-numerical approach.

Dependencies
------------
- numpy : For numerical operations and random number generation
- abc : For abstract base class functionality
- dataclasses : For structured parameter containers
- utils : Custom utility functions for system parameters and Psi function
- analytical_solution : Analytical population solutions
- core_parameters : Parameter container classes
- laplace_transforms : Laplace transform calculations for noise amplitude

Usage Example
-------------
Create a unified model and solve detection SDE using a numerical method:

    >>> import numpy as np
    >>> from src.detection.detection_sde_model import DetectionSDEModel, BaseDetectionSolver
    >>> from src.models.core_parameters import PhysicalParameters, DeadTimeParameters
    >>> 
    >>> # Initialize model
    >>> sde_model = DetectionSDEModel()
    >>> 
    >>> # Setup parameters
    >>> p_v = np.array([0.0, 0.1, 0.3, 0.4, 0.2])
    >>> physical_params = PhysicalParameters(
    ...     p_v=p_v, absorb=7.0, detect=10.0, source=1000.0
    ... )
    >>> dead_time_params = DeadTimeParameters(
    ...     mean_tau=1e-6, tau_distribution='constant'
    ... )
    >>> 
    >>> # Calculate drift and noise amplitude
    >>> drift = sde_model.calculate_drift(
    ...     population=100.0,
    ...     dead_time_params=dead_time_params,
    ...     physical_params=physical_params
    ... )
    >>> sigma_3 = sde_model.calculate_noise_amplitude(
    ...     fission=33.95,
    ...     dead_time_params=dead_time_params,
    ...     physical_params=physical_params
    ... )

Note:
    This module provides the unified foundation for all detection SDE solvers.
    For specific numerical method implementations, see the refactored method modules.
    For analytical solutions, see analytical_solution.py.
    
The equation for the noise amplitude is:

.. math::
    \\sigma_3^2 = \\frac{1 + 2aL'(a)}{aL(a)}

where :math:`L(a)` is the Laplace transform of the dead time distribution PDF
and :math:`L'(a)` is its derivative, evaluated at :math:`a = \\lambda_d N_t`

Drift derivatives are calculated using the formula:
    
.. math::
    f'(x) = L(x) + xL'(x)

where :math:`f(x) = xL(x)` is the drift function.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.utils import utils as utl
from src.utils.analytical_solution import analytical_population_solution
from src.models.core_parameters import (
    TimeParameters,
    DeadTimeParameters,
    PhysicalParameters
)
from src.utils.laplace_transforms import (
    DeadTimeLaplaceCalculator,
    calculate_noise_amplitude_from_laplace
)


@dataclass
class StepState:
    """
    Container for current state values at integration step.

    This dataclass bundles the current state information for a single
    integration step.

    Attributes
    ----------
    detect : float
        Current detection value :math:`C_i`
    pop : float
        Current population value :math:`N_i`
    index : int
        Current step index i (for accessing random variable arrays)
    """
    detect: float
    pop: float
    index: float


@dataclass
class IntegrationParameters:
    """
    Container for integration parameters that remain constant 
    during simulation.

    This dataclass groups parameters that don't change throughout the
    integration loop.

    Attributes
    ----------
    dt : float
        Time step size
    sig_3 : float
        Detection noise amplitude :math:`\\sigma_3`
    dead_time_params : DeadTimeParameters
        Dead time distribution parameters
    """
    dt: float
    sig_3: float
    dead_time_params: DeadTimeParameters


@dataclass
class IntegrationContext:
    """
    Context container for integration step parameters.

    This class bundles all parameters needed for a single integration step,
    By nesting related parameters in StepState and IntegrationParameters, 
    this class maintains only 3 attributes.

    Attributes
    ----------
    state : StepState
        Current state values (detection, population, step index)
    params : IntegrationParameters
        Integration parameters (dt, noise amplitude, dead time params)
    random_vars : dict[str, np.ndarray]
        Method-specific random variables (e.g., Wiener increments, Rademacher)

    Examples
    --------
    Create an integration context for a single step:

    >>> state = StepState(detect=0.0, pop=100.0, index=0)
    >>> params = IntegrationParameters(
    ...     dt=1e-6, sig_3=0.1, 
    ...     dead_time_params=dead_time_params
    ... )
    >>> context = IntegrationContext(
    ...     state=state, params=params, random_vars=random_vars
    ... )
    >>> # Access values
    >>> current_detect = context.state.detect
    >>> dt = context.params.dt
    >>> wiener = context.random_vars['wiener'][context.state.index]
    """
    state: StepState
    params: IntegrationParameters
    random_vars: dict[str, np.ndarray]


@dataclass
class SimulationResults:
    """
    Container for simulation results to be saved.

    This reduces parameter count in save functions by bundling related data.

    Attributes
    ----------
    pop : np.ndarray
        Population array
    detect : np.ndarray
        Detection array
    fission : float
        Fission rate value
    index : str
        Descriptive index for file naming
    dead_time_type : str
        Dead time distribution type
    """
    pop: np.ndarray
    detect: np.ndarray
    fission: float
    index: str
    dead_time_type: str


class DetectionSDEModel:
    """
    Unified model for the detection stochastic differential equation.

    This class encapsulates the common detection SDE equation and provides
    unified implementations of drift function, noise amplitude calculations,
    and analytical population solutions that are shared across all numerical
    methods. It serves as the single source of truth for the detection equation.

    The detection SDE solved is:

    .. math::
        dC_t = \\lambda_d N_t(1 - \\Psi(\\lambda_d * N_t)) dt + \\sigma_3dW₃

    where the drift term incorporates dead time effects through the Psi function
    and the diffusion noise depends on the dead time distribution type.
    The noise amplitude is calculated using the Laplace transform of the
    dead time distribution.

    Attributes
    ----------
    laplace_calculator : DeadTimeLaplaceCalculator
        Calculator for Laplace transforms and derivatives of dead time
        distributions, used for rigorous noise amplitude calculations.

    Methods
    --------------
    calculate_drift(population, dead_time_params)
        Calculate drift term for detection SDE.
    calculate_noise_amplitude(fission, dead_time_params, physical_params)
        Calculate detection noise amplitude σ₃ using Laplace transform.
    solve_population_analytically(physical_params, time_params, fission, n_0)
        Solve population dynamics analytically.
    validate_inputs(time_params, dead_time_params)
        Validate input parameter consistency.

    Examples
    --------
    Calculate drift and noise amplitude for constant dead time:

    >>> import numpy as np
    >>> from src.detection.detection_sde_model import DetectionSDEModel
    >>> from src.models.core_parameters import PhysicalParameters, DeadTimeParameters
    >>> 
    >>> sde_model = DetectionSDEModel()
    >>> 
    >>> # Setup parameters
    >>> p_v = np.array([0.0, 0.1, 0.3, 0.4, 0.2])
    >>> physical_params = PhysicalParameters(
    ...     p_v=p_v, absorb=7.0, detect=10.0, source=1000.0
    ... )
    >>> dead_time_params = DeadTimeParameters(
    ...     mean_tau=1e-6, tau_distribution='constant'
    ... )
    >>> 
    >>> # Calculate drift term
    >>> drift = sde_model.calculate_drift(
    ...     population=100.0,
    ...     dead_time_params=dead_time_params,
    ...     physical_params=physical_params
    ... )
    >>> print(f"Drift: {drift:.6f}")
    >>> 
    >>> # Calculate noise amplitude
    >>> sigma_3 = sde_model.calculate_noise_amplitude(
    ...     fission=33.95,
    ...     dead_time_params=dead_time_params,
    ...     physical_params=physical_params
    ... )
    >>> print(f"Noise amplitude: {sigma_3:.6e}")
    """

    def __init__(self):
        """
        Initialize the detection SDE model with a Laplace calculator.
        """
        self.laplace_calculator = DeadTimeLaplaceCalculator()

    def calculate_drift(
            self,
            population: float,
            dead_time_params: DeadTimeParameters,
            physical_params: PhysicalParameters) -> float:
        """
        Calculate the drift term for the detection SDE.

        The drift term is:

        .. math::
            N_t \\lambda_d(1 - \\Psi(\\lambda_d N_t))

        Where :math:`\\Psi` is the corresponding Psi function
        for the distribution of :math:`\\tau`.

        Parameters
        ----------
        population : float
            Current population value :math:`N_t`
        dead_time_params : DeadTimeParameters
            Dead time distribution parameters
        physical_params : PhysicalParameters
            Physical parameters containing detection rate

        Returns
        -------
        float
            Drift term value for the detection SDE
        """
        detection_rate = physical_params.detect * population

        std = dead_time_params.get_std()

        psi_value = utl.calculate_psi_function(
            detection_rate,
            dead_time_params.mean_tau,
            dead_time_params.tau_distribution,
            std
        )

        return detection_rate * (1 - psi_value)

    def calculate_noise_amplitude(
            self,
            fission: float,
            dead_time_params: DeadTimeParameters,
            physical_params: PhysicalParameters
    ) -> float:
        """
        Calculate the detection noise amplitude :math:`\\sigma_3` of the 
        detection SDE using the Laplace transform
        of the dead time distribution.

        The method uses the equation:

        .. math::
            \\sigma^2 = \\frac{1 + 2 a L'(a)}{a L(a)}

        Where :math:`L` is the Laplace transform of the dead time distribution,
        :math:`L'` is its derivative,  and :math:`a = N_t \\lambda_d`
        is the detection rate, evaluated at the equilibrium

        Parameters
        ----------
        fission : float
            Fission rate constant
        dead_time_params : DeadTimeParameters
            Dead time parameters containing dead time configuration
        physical_params : PhysicalParameters
            Physical parameters containing rate constants.

        Returns
        -------
        noise_amplitude : float
            Detection noise amplitude :math:`\\sigma_3`

        """
        params = utl.calculate_system_parameters(
            physical_params.p_v,
            fission,
            physical_params.absorb,
            physical_params.source,
            physical_params.detect)

        equil = params['equilibrium']

        detection_rate = physical_params.detect * equil

        std = dead_time_params.get_std()

        laplace_result = self.laplace_calculator.calculate_laplace_transform(
            x=detection_rate,
            tau=dead_time_params.mean_tau,
            distribution=dead_time_params.tau_distribution,
            std=std)

        noise_amplitude = calculate_noise_amplitude_from_laplace(
            laplace_result,
            detection_rate)

        return noise_amplitude

    def solve_population_analytically(
            self,
            physical_params: PhysicalParameters,
            time_params: TimeParameters,
            fission: float,
            n_0: Optional[float] = None) -> np.ndarray:
        """
        Solve the population :math:`N_t` analytically.

        This method provides the analytical solution for the population SDE,
        which is an Ornstein-Uhlenbeck process.

        Parameters
        ----------
        physical_params : PhysicalParameters
            Physical parameters containing rate constants
        time_params : TimeParameters
            Time discretization parameters
        fission : float
            Fission rate constant for this simulation
        n_0 : Optional[float], optional
            Initial population value. If None, defaults to equilibrium.

        Returns
        -------
        np.ndarray
            Array of population values at each time point, shape (grid_points+1,)
        """

        return analytical_population_solution(
            physical_params, time_params, fission, n_0)

    def validate_inputs(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters) -> None:
        """
        Validate input parameter consistency.

        Parameters
        ----------
        time_params : TimeParameters
            Time discretization parameters
        dead_time_params : DeadTimeParameters
            Dead time distribution parameters

        Raises
        ------
        ValueError
            If input parameters are inconsistent or invalid

        """

        if time_params.grid_points < 1:
            raise ValueError("grid_points must be positive.")
        if time_params.t_end <= time_params.t_0:
            raise ValueError("t_end must be greater than t_0")
        if dead_time_params.mean_tau <= 0:
            raise ValueError("mean_tau must be positive")


class BaseDetectionSolver(ABC):
    """
    Abstract base class for detection SDE numerical method implementations.

    This class provides the common framework and interface for all numerical
    methods that solve the detection SDE. It encapsulates the hybrid approach
    where population dynamics are solved analytically while detection processes
    are integrated numerically using method-specific schemes.

    The common workflow is:
    1. Validate inputs using unified model
    2. Solve population analytically using unified model
    3. Integrate detection SDE numerically using method-specific implementation
    4. Return time, population, and detection arrays

    Attributes
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction rate constants
    sde_model : DetectionSDEModel
        Unified detection SDE model for common calculations

    Methods
    -------
    solve_detection_sde(time_params, dead_time_params, fission, n_0)
        Solve single detection SDE path using hybrid approach
    integrate_step(context)
        Method-specific integration step (abstract, must be implemented)
    setup_random_variables(grid_points, dt)
        Method-specific random variable generation (abstract, must be implemented)

    Notes
    -----
    Subclasses must implement the abstract methods `integrate_step` and
    `setup_random_variables` to provide the specific numerical integration
    scheme (e.g., Euler-Maruyama, Runge-Kutta 3.0, Taylor 2.0).

    Examples
    --------
    Implementing a new numerical method:

    >>> class EulerMaruyamaDetectionSDE(BaseDetectionSolver):
    ...     def integrate_step(self, context):
    ...         # Implement Euler-Maruyama step
    ...         drift = self.sde_model.calculate_drift(...)
    ...         return context.state.detect + drift * context.params.dt + ...
    ...     
    ...     def setup_random_variables(self, grid_points, dt):
    ...         return {'wiener': np.random.normal(0, np.sqrt(dt), grid_points)}
    """

    def __init__(
            self,
            physical_params: PhysicalParameters,
            sde_model: DetectionSDEModel,
            output_prefix: str = None,
            auto_save: bool = True):
        """
        Initialize the base detection SDE numerical method implementations

        Parameters
        ----------
        physical_params : PhysicalParameters
            Physical parameters containing nuclear reaction rate constants
        sde_model : DetectionSDEModel
            Unified detection SDE model for common calculations
        auto_save : bool
            Whether to automatically save results after solving
        """
        self.physical_params = physical_params
        self.sde_model = sde_model
        self.output_prefix = output_prefix
        self.auto_save = auto_save

    def solve_detection_sde(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float,
            n_0: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the detection SDE using the hybrid analytical-numerical approach.

        This method implements the common framework used by all numerical methods:
        1. Population N_t is solved analytically
        2. Detection SDE is integrated numerically using method-specific scheme

        Parameters
        ----------
        time_params : TimeParameters
            Time discretization parameters
        dead_time_params : DeadTimeParameters
            Dead time distribution parameters
        fission : float
            Fission rate constant for this simulation (s⁻¹)
        n_0 : Optional[float], optional
            Initial population value. If None, defaults to equilibrium,
            by default None

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Solution tuple containing:
            - Time array: Discrete time points from t_0 to t_end
            - Population array: Analytical population solution N_t
            - Detection array: Numerical detection SDE solution C_t

        Examples
        --------
        >>> solver = EulerMaruyamaDetectionSDE(physical_params, sde_model)
        >>> t, pop, detect = solver.solve_detection_sde(
        ...     time_params, dead_time_params, fission=33.95
        ... )
        >>> print(f"Final detection count: {detect[-1]:.2f}")
        """
        self.sde_model.validate_inputs(time_params, dead_time_params)

        self.physical_params.set_fission(fission)

        pop = self.sde_model.solve_population_analytically(
            self.physical_params, time_params, fission, n_0)

        detect = self._solve_detection_sde_numerically(
            time_params, dead_time_params, fission, pop)

        t_space = np.linspace(
            time_params.t_0,
            time_params.t_end,
            time_params.grid_points+1
        )

        if self.auto_save:
            index = self._generate_descriptive_index(
                time_params, dead_time_params, fission)

            results = SimulationResults(
                pop=pop,
                detect=detect,
                fission=fission,
                index=index,
                dead_time_type=dead_time_params.tau_distribution
            )

            self._save_results(results)

        return t_space, pop, detect

    def _generate_descriptive_index(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float
    ) -> str:
        """
        Generate descriptive index with grid points and dead time information.

        Automatically generates prefix from grid_points, 
        then adds dead time info.

        Examples
        --------
        - Constant: '1e6gp_mean1e-06_f33.94'
        - Uniform: '1e6gp_mean1e-06_std1e-07_f33.94'
        """

        grid_prefix = utl.generate_prefix_from_steps(
            time_params.grid_points,
            simulation_type='numerical'
        )

        mean_str = f"mean{dead_time_params.mean_tau:.0e}"

        if dead_time_params.tau_distribution == 'constant':
            return f"{grid_prefix}_{mean_str}_f{fission}"

        std_str = f"std{dead_time_params.get_std():.0e}"
        return f"{grid_prefix}_{mean_str}_{std_str}_f{fission}"

    def _solve_detection_sde_numerically(
            self,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission: float,
            pop: np.ndarray) -> np.ndarray:
        """
        Core numerical integration of detection SDE using method-specific scheme.

        This method implements the common numerical integration framework that
        delegates the actual integration step to method-specific implementations
        via the abstract `integrate_step` method.
        """

        detect = np.zeros(time_params.grid_points + 1)

        sig_3 = self.sde_model.calculate_noise_amplitude(
            fission, dead_time_params, self.physical_params)

        dt = time_params.get_grid_spacing()

        random_vars = self.setup_random_variables(
            time_params.grid_points, dt)

        int_params = IntegrationParameters(
            dt=dt,
            sig_3=sig_3,
            dead_time_params=dead_time_params)

        for i in utl.progress_tracker(time_params.grid_points):

            state = StepState(
                detect=detect[i],
                pop=pop[i],
                index=i)

            context = IntegrationContext(
                state=state,
                params=int_params,
                random_vars=random_vars)

            detect[i+1] = self.integrate_step(context)

        return detect

    @abstractmethod
    def integrate_step(
            self,
            context: IntegrationContext) -> float:
        """
        Method-specific integration step implementation.

        This abstract method must be implemented by each numerical method
        to provide the specific integration scheme (Euler-Maruyama,
        Runge-Kutta, Taylor, etc.).

        Parameters
        ----------
        context : IntegrationContext
            Integration context containing:
                - state: Current state (detect, pop, index)
                - params: Integration parameters (dt, sig_3, dead_time_params)
                - random_vars: Method-specific random variables

        Returns
        -------
        float
            Next detection value C_{i+1} from method-specific integration

        Examples
        --------
        Euler-Maruyama implementation:

        >>> def integrate_step(self, context):
        ...     drift = self.sde_model.calculate_drift(
        ...         context.state.pop,
        ...         context.params.dead_time_params,
        ...         self.physical_params
        ...     )
        ...     wiener = context.random_vars['wiener'][context.state.index]
        ...     return (context.state.detect + 
        ...             drift * context.params.dt + 
        ...             context.params.sig_3 * wiener)
        """

    @abstractmethod
    def setup_random_variables(
            self,
            grid_points: int,
            dt: float) -> dict[str, np.ndarray]:
        """
        Method-specific random variable generation.

        This abstract method must be implemented by each numerical method
        to generate the specific random variables required for the
        integration scheme.

        Parameters
        ----------
        grid_points : int
            Number of grid points for random variable generation
        dt : float
            Time step size

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing method-specific random variables.
            Common keys include:
            - 'wiener': Standard Wiener increments
            - 'mixed_wiener': Mixed Wiener processes

        Examples
        --------
        Euler-Maruyama implementation:

        >>> def setup_random_variables(self, grid_points, dt):
        ...     return {
        ...         'wiener': np.random.normal(
        ...             loc=0.0, scale=np.sqrt(dt), size=grid_points
        ...         )
        ...     }
        """

    @abstractmethod
    def _save_results(
            self,
            results: SimulationResults
    ) -> None:
        """Save simulation results (method-specific implementation)."""
