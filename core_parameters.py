"""Written by Tomer279 with the assistance of Cursor.ai.

Core parameter classes for nuclear reactor simulations.

This module provides the fundamental building blocks for configuring nuclear
reactor simulations. It contains classes that define physical constants,
derived calculations, and temporal parameters essential for reactor modeling.

Classes:
    RateConstants:
        Nuclear reaction rate constants (absorption, detection,
                   fission, source)
    FissionDistribution:
        Probability distribution for fission neutron yield
    PhysicalParameters:
        Unified interface for physical constants and properties
    TimeParameters:
        Simulation timing and discretization parameters
    DeadTimeParameters:
        Detector dead time effects and statistical sampling
    FissionParameters:
        Fission-dependent calculations (Rossi-alpha, equilibrium)

Dependencies:
    numpy: For array operations and mathematical functions
    utils: Custom statistical functions (mean, variance)
    typing: For type annotations and optional parameters

Usage Examples:
    # Basic rate constants
    rates = RateConstants(absorb=7.0, detect=10.0, fission=33.95)

    # Physical parameters container
    physics = PhysicalParameters(
        rate_constants=rates,
        fission_distribution=FissionDistribution()
    )

    # Time discretization
    time_params = TimeParameters(t_end=0.1, steps=1000000)

    # Fission-dependent calculations
    fission_params = FissionParameters(
        fission_vec=np.array([33.94, 33.95]),
        physical_params=physics
    )

Note:
    This module focuses on the core physics and mathematical parameters
    required for reactor simulations. For simulation control and output
    management, see simulation_settings module.
"""

from typing import List, Optional, Dict
import numpy as np
from utils import mean, variance
rng = np.random.default_rng()


class RateConstants:
    """
    Rate constants for reactor simulations.

    This class encapsulates all rate constants used in nuclear reactor
    simulations including absorption, source, detection, and fission rates.
    It provides validation and utility methods for rate constant management.

    Attributes
    ----------
    absorb : float
        Absorption rate constant (1/s)
    source : float
        Source rate constant (1/s)
    detect : float
        Detection rate constant (1/s)
    fission : Optional(float):
        Fission rate constant (1/s)

    Public Methods
    --------------
    get_all_rates()
        Get all rate constants as a dictionary
    set_fission(fission)
        Set the fission rate constant
    to_dict()
        Convert to dictionary for serialization

    Private Methods
    ---------------
    _validate()
        Validate that all rate constants are non-negative
    """

    def __init__(self,
                 absorb: float = 7.0,
                 source: float = 1000.0,
                 detect: float = 10.0,
                 fission: Optional[float] = None):
        """
        Initialize rate constants.

        Parameters
        ----------
        absorb : float, optional
            Absorption rate constant (s⁻¹). The default is 7.0.
        source : float, optional
            Source rate constant (s⁻¹). The default is 1000.0.
        detect : float, optional
            Detection rate constant (s⁻¹). The default is 10.0.
        fission : float, optional
            Fission rate constant (s⁻¹). If None, will be set later.

        Raises
        ------
        ValueError
            If any rate constant is negative
        """

        self.absorb = absorb
        self.source = source
        self.detect = detect
        self.fission = fission

        self._validate()

    def get_all_rates(self) -> Dict[str, float]:
        """
        Get all rate constants as a dictionary.

        Returns
        -------
        Dict[str, float]
            Dictionary containing all rate constants with keys:
            'fission', 'absorb', 'source', 'detect'

        Raises
        ------
        ValueError
            If fission rate has not been set
        """
        if self.fission is None:
            raise ValueError("Fission rate must be set "
                             "before getting all rates")
        return {
            'fission': self.fission,
            'absorb': self.absorb,
            'source': self.source,
            'detect': self.detect
        }

    def set_fission(self, fission: float):
        """
        Set the fission rate constant.

        Parameters
        ----------
        fission : float
            Fission rate constant (s⁻¹)

        Raises
        ------
        ValueError
            If fission rate is negative
        """
        if fission < 0:
            raise ValueError("Fission rate must be non-negative")
        self.fission = fission

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict
            Dictionary representation of rate constants
        """
        return {
            'fission': self.fission,
            'absorb': self.absorb,
            'source': self.source,
            'detect': self.detect
        }

    def _validate(self):
        """Validate that all rate constants are non-negative."""
        if any(x < 0 for x in [self.absorb, self.source, self.detect]):
            raise ValueError("Rate constants must be non-negative")

        if self.fission is not None and self.fission < 0:
            raise ValueError("Fission rate must be non-negative")


class FissionDistribution:
    """
    Fission probability distribution for neutron yield modeling.

    This class encapsulates the probability distribution for the number of
    particles produced in fission events. It provides methods for calculating
    statistical properties and sampling from the distribution.

    Attributes
    ----------
    p_v : np.ndarray
        Fission probability distribution array

    Public Methods
    --------------
    get_variance()
        Calculate variance of the distribution
    sample_fission_yield():
        Sample a random fission yield
    to_dict()
        Convert to dictionary for serialization
    vbar()
        Calculate expected value of the distribution

    Private Methods
    ---------------
    _validate()
        Validate that p_v is a valid probability distribution
    """

    def __init__(self, p_v: Optional[List[float]] = None):
        """
        Initialize fission probability distribution.

        Parameters
        ----------
        p_v : Optional[List[float]], optional
            Fission probability distribution.
            Defaults to [1/6, 1/3, 1/3, 1/6]
        """
        self.p_v = (np.array(p_v) if p_v is not None
                    else np.array([1/6, 1/3, 1/3, 1/6]))

        self._validate()

    def get_variance(self) -> float:
        """
        Calculate variance of fission probability distribution.

        Returns
        -------
        float
            Variance of the fission yield distribution
        """
        return variance(self.p_v)

    def sample_fission_yield(self) -> int:
        """
        Sample a random fission yield from the distribution.

        Returns
        -------
        int
            Random fission yield sampled from the distribution
        """
        return rng.choice(len(self.p_v), p=self.p_v)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict
            Dictionary representation of the fission distribution
        """
        return {
            'p_v': self.p_v.tolist()
        }

    def vbar(self) -> float:
        """
        Calculate expected value of fission probability distribution.

        Returns
        -------
        float
            Expected value (mean) of the fission yield distribution
        """
        return mean(self.p_v)

    def _validate(self):
        """Validate fission probability distribution."""
        if np.any(self.p_v < 0) or not np.isclose(np.sum(self.p_v), 1):
            raise ValueError("p_v must be a valid probability distribution")


class PhysicalParameters:
    """
    Physical parameters for nuclear reactor simulations.

    This class groups together all physical constants and derived parameters
    related to nuclear reactions and particle dynamics. It provides a unified
    interface for accessing rate constants and fission distribution properties.

    Attributes
    ----------
    rates : RateConstants
        Nuclear reactor rate constants
    fission_dist : FissionDistribution
        Fission probability distribution

    Public Methods
    --------------
    calculate_total_rate(population)
        Calculate total event rate
    get_rate_constants()
        Get all rate constants as dictionary
    set_fission(fission)
        Set the fission rate constant
    to_dict()
        Convert to dictionary for serialization
    vbar()
        Get expected value of fission distribution

    Properties
    ----------
    absorb
        Absorption rate constant
    detect
        Detection rate constant
    fission
        Fission rate constant
    p_v
        Fission probability distribution
    source
        Source rate constant
    """

    def __init__(self,
                 rate_constants: Optional[RateConstants] = None,
                 fission_distribution: Optional[FissionDistribution] = None):
        """
        Initialize physical parameters.

        Parameters
        ----------
        rate_constants : RateConstants, optional
            Nuclear reactor rate constants. Uses defaults if None.
        fission_distribution : FissionDistribution, optional
            Fission probabiliity distribution. Uses defaults if None.
        """
        self.rates = rate_constants or RateConstants()
        self.fission_dist = fission_distribution or FissionDistribution()

    def calculate_total_rate(self, population: float) -> float:
        """
        Calculate total event rate for given population.

        Parameters
        ----------
        population : float
            Current neutron population

        Returns
        -------
        float
            Total event rate: (fission + absorb + detect) * population + source
        Raises
        ------
        ValueError
            If fission rate has not been set
        """
        if self.fission is None:
            raise ValueError(
                "Fission rate must be set before calculating total rate")

        return ((self.fission + self.absorb + self.detect) * population
                + self.source)

    def get_rate_constants(self) -> Dict[str, float]:
        """
        Get all rate constants as a dictionary.

        Returns
        -------
        Dict[str, float]
            Dictionary containing all rate constants
        """
        return self.rates.get_all_rates()

    @property
    def p_v(self) -> np.ndarray:
        """Get fission probability distribution"""
        return self.fission_dist.p_v

    def set_fission(self, fission: float):
        """
        Set the fission rate constant.

        Parameters
        ----------
        fission : float
            Fission rate constant (s⁻¹)
        """
        return self.rates.set_fission(fission)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict
            Dictionary representation of physical parameters
        """
        return {
            'rate_constants': self.rates.to_dict(),
            'fission_distribution': self.fission_dist.to_dict()
        }

    def vbar(self) -> float:
        """
        Get expected value of fission probability distribution.

        Returns
        -------
        float
            Expected value of the fission yield distribution
        """
        return self.fission_dist.vbar()

    @property
    def absorb(self) -> float:
        """
        Get absorption rate constant.

        Returns
        -------
        float
            Absorption rate constant (s⁻¹)
        """
        return self.rates.absorb

    @property
    def detect(self) -> float:
        """
        Get detection rate constant.

        Returns
        -------
        float
            Detection rate constant (s⁻¹)
        """
        return self.rates.detect

    @property
    def fission(self) -> Optional[float]:
        """
        Get fission rate constant.

        Returns
        -------
        Optional[float]
            Fission rate constant (s⁻¹) or None if not set
        """
        return self.rates.fission

    @property
    def source(self) -> float:
        """
        Get fission probability distribution.

        Returns
        -------
        np.ndarray
            Fission probability distribution array
        """
        return self.rates.source


class TimeParameters:
    """
    Time-related parameters for simulations.

    This class groups together all parameters related to simulation timing,
    discretization, and temporal resolution. It provides methods for
    calculating derived time parameters.

    Attributes
    ----------
    t_0 : float
        Initial time
    t_end :float
        End time
    steps : int
        Number of simulation steps
    grid_points : int
        Number of grid points for numerical methods

    Public Methods
    --------------
    get_duration()
        Get simulation duration
    get_grid_spacing()
        Get grid spacing for numerical methods
    get_time_step()
        Get time step size
    to_dict()
        Convert to dictionary for serialization

    Private Methods
    ---------------
    _validate()
        Validate time parameters
    """

    def __init__(self,
                 t_0: float = 0.0,
                 t_end: float = 0.1,
                 steps: int = 100_000_000,
                 grid_points: int = 10_000_000):
        """
        Initialize time parameters

        Parameters
        ----------
        t_0 : float, optional
            Initial time. The default is 0.0.
        t_end : float, optional
            End time. The default is 0.1.
        steps : int, optional
            Number of simulation steps. The default is 100_000_000.
        grid_points : 10_000_000
            Number of grid points for numerical methods.
        """

        self.t_0 = t_0
        self.t_end = t_end
        self.steps = steps
        self.grid_points = grid_points

        self._validate()

        if self.steps < 1 or self.grid_points < 1:
            raise ValueError("steps and grid_points must be positive")

    def get_duration(self) -> float:
        """
        Get simulation duration.

        Returns
        -------
        float
            Duration of the simulation (t_end - t_0)
        """
        return self.t_end - self.t_0

    def get_grid_spacing(self) -> float:
        """
        Get grid spacing for numerical methods.

        Returns
        -------
        float
            Grid spacing: duration / (grid_points - 1)
        """
        return self.get_duration() / (self.grid_points - 1)

    def get_time_step(self) -> float:
        """
        Get time step size.

        Returns
        -------
        float
            Time step size: duration / (steps - 1)
        """
        return self.get_duration() / (self.steps - 1)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict
            Dictionary representation of time parameters
        """
        return {
            't_0': self.t_0,
            't_end': self.t_end,
            'steps': self.steps,
            'grid_points': self.grid_points
        }

    def _validate(self):
        """Validate time parameters."""
        if self.t_end <= self.t_0:
            raise ValueError("t_end must be greater than t_0")


class DeadTimeParameters:
    """
    Dead time parameters for detector modeling.

    This class groups together all parameters related to dead time effects
    and their statistical distributions. It supports multiple distribution
    types including constant, normal, uniform, exponential, and gamma.

    Attributes
    ----------
    mean_tau : float
        Mean dead time
    tau_distribution : str
        Distribution type for random dead time
    tau_params : Dict
        Parameters for each distribution type

    Public Methods
    --------------
    get_distribution_params()
        Get parameters for current distribution
    sample_dead_time()
        Sample random dead time from distribution
    set_distribution(distribution)
        Set the dead time distribution type
    to_dict()
        Convert to dictionary for serialization

    Private Methods
    ---------------
    _validate()
        Validate dead time parameters
    """

    def __init__(self,
                 mean_tau: float = 1e-6,
                 tau_distribution: str = 'uniform',
                 tau_params: Optional[Dict] = None):

        self.mean_tau = mean_tau
        self.tau_distribution = tau_distribution

        if tau_params is None:

            self.tau_params = {
                'constant': {
                    'tau': 1e-6
                },
                'normal': {
                    'loc': 1e-6,
                    'scale': 0.5 * 1e-7
                },
                'uniform': {
                    'low': 1e-6 - 2 * np.sqrt(3) * 1e-7,
                    'high': 1e-6 + 2 * np.sqrt(3) * 1e-7
                },
                'exponential': {
                    'scale': 1e-6
                },
                'gamma': {
                    'shape': 25,
                    'scale': 0.04e-06
                }
            }
        else:
            self.tau_params = tau_params

    def get_distribution_params(self) -> Dict:
        """
        Get parameters for the current dead time distribution.

        Returns
        -------
        Dict
            Parameters for the current distribution type

        Raises
        ------
        ValueError
            If current distribution is not in tau_params
        """
        return self.tau_params[self.tau_distribution]

    def sample_dead_time(self) -> float:
        """
        Sample a random dead time from the distribution.

        Returns
        -------
        float
            Random dead time sampled from the specified distribution

        Raises
        ------
        ValueError
            If distribution type is unknown
        """
        params = self.get_distribution_params()

        if self.tau_distribution == 'normal':
            return rng.normal(params['loc'], params['scale'])
        if self.tau_distribution == 'uniform':
            return rng.uniform(params['low'], params['high'])
        if self.tau_distribution == 'exponential':
            return rng.exponential(params['scale'])
        if self.tau_distribution == 'gamma':
            return rng.gamma(params['shape'], params['scale'])
        raise ValueError(f"Unknown distribution: {self.tau_distribution}")

    def set_distribution(self, distribution: str):
        """
        Set the dead time distribution type.

        Parameters
        ----------
        distribution : str
            Distribution type
            ('constant', 'normal', 'uniform', 'exponential', 'gamma')

        Raises
        ------
        ValueError
            If distribution is not supported
        """
        if distribution not in self.tau_params:
            raise ValueError(f"Invalid tau_distribution: {distribution}")
        self.tau_distribution = distribution

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict
            Dictionary representation of dead time parameters
        """
        return {
            'mean_tau': self.mean_tau,
            'tau_distribution': self.tau_distribution,
            'tau_params': self.tau_params
        }

    def _validate(self):
        """Validate dead time parameters."""
        if self.mean_tau <= 0:
            raise ValueError("mean_tau must be positive.")

        if self.tau_distribution not in self.tau_params:
            raise ValueError("Initial tau_distribution: "
                             f"{self.tau_distribution}")


class FissionParameters:
    """
    Fission-related parameters and derived calculations.

    This class groups together fission values and all derived parameters
    that depend on fission values. It calculates Rossi-alpha coefficients,
    equilibrium populations, and other derived quantities.

    Attributes
    ----------
    fission_vec : np.ndarray
        Array of fission values to simulate
    lam_vec : np.ndarray
        Total reaction rate for each fission value
    alpha_vec : np.ndarray
        Rossi-alpha coefficient for each fission value
    alpha_inv_vec : np.ndarray
        Inverse Rossi-alpha
    equil : np.ndarray
        Equilibrium population for each fission value
    fission_to_alpha_inv : Dict
        Mapping from fission to alpha inverse

    Public Methods
    --------------
    to_dict()
        Convert to dictionary for serialization
    update_physical_parameters(physical_params)
        Update physical parameters

    Private Methods
    ---------------
    _calculate_derived_parameters()
        Calculate derived parameters
    _validate()
        Validate fission parameters
    """

    def __init__(self,
                 fission_vec: Optional[np.ndarray] = None,
                 physical_params: Optional[PhysicalParameters] = None):
        """
        Initialize fission parameters.

        Parameters
        ----------
        fission_vec : Optional[np.ndarray], optional
            Array of fission values to simulate.
        physical_params : Optional[PhysicalParameters], optional
            Physical parameters needed for derived calculations
        """
        if fission_vec is None:
            self.fission_vec = np.array([33.94, 33.95, 33.96, 33.97, 33.98,
                                         33.982, 33.984, 33.986, 33.988,
                                         33.99, 33.992])
            self.fission_vec = np.round(self.fission_vec, decimals=4)
        else:
            self.fission_vec = np.array(fission_vec)

        self._physical_params = physical_params

        self._calculate_derived_parameters()

        self._validate()

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict
            Dictionary representation of fission parameters
        """
        return {
            'fission_vec': self.fission_vec.tolist(),
            'lam_vec': self.lam_vec.tolist(),
            'alpha_vec': self.alpha_vec.tolist(),
            'alpha_inv_vec': self.alpha_inv_vec.tolist(),
            'equil': self.equil.tolist(),
            'fission_to_alpha_inv': self.fission_to_alpha_inv
        }

    def update_physical_parameters(self,
                                   physical_params: PhysicalParameters()):
        """
        Update physical parameters and recalculate derived parameters.

        Parameters
        ----------
        physical_params : PhysicalParameters
            New physical parameters to use for calculations
        """
        self._physical_params = physical_params
        self._calculate_derived_parameters()

    def _calculate_derived_parameters(self):
        """
        Calculate derived parameters
        from fission values and physical parameters

        This method calculates:
            - Total reaction rate (lam_vec)
            - Rossi-alpha coefficient (alpha_vec)
            - Inverse Rossi-alpha (alpha_inv_vec)
            - Equilibrium population (equil)
            - Fission to alpha inverse mapping
        """
        if self._physical_params is None:
            temp_physical = PhysicalParameters()
            absorb = temp_physical.absorb
            source = temp_physical.source
            detect = temp_physical.detect
            vbar = temp_physical.vbar()
        else:
            absorb = self._physical_params.absorb
            source = self._physical_params.source
            detect = self._physical_params.detect
            vbar = self._physical_params.vbar()

        # Total reaction rate for each fission value
        self.lam_vec = self.fission_vec + absorb + detect

        # Rossi-alpha coefficient for each fission value
        self.alpha_vec = self.lam_vec - self.fission_vec * vbar

        # Inverse Rossi-alpha
        self.alpha_inv_vec = 1 / self.alpha_vec

        # Equilibrium population for each fission value
        self.equil = source * self.alpha_inv_vec

        self.fission_to_alpha_inv = dict(zip(
            self.fission_vec.tolist(),
            self.alpha_inv_vec.tolist()))

    def _validate(self):
        """Validate fission parameters"""
        if np.any(self.fission_vec <= 0):
            raise ValueError("Fission values must be positive")
