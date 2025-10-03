"""Written by Tomer279 with the assistance of Cursor.ai.

Count rate calculation module for nuclear reactor simulations.

This module provides comprehensive count rate calculation capabilities for
nuclear reactor simulations, supporting both stochastic and numerical
methods with various dead time distributions. It offers both class-based
and functional interfaces for flexible count rate analysis.

Classes:
    CountRateCalculator:
        Unified interface for count rate calculations
        across simulation methods

Key Features:
    - Support for stochastic simulations (basic and constant dead time)
    - Numerical method integration (Euler-Maruyama, Taylor, Runge-Kutta)
    - Multiple dead time distributions (constant, uniform, normal, gamma)
    - Theoretical count rate calculations
    - Comprehensive dead time modeling

Dependencies:
    numpy: For array operations and mathematical functions
    scipy.stats: For statistical functions (normal distribution CDF)
    utils: Custom utility functions for system parameter calculations

Usage Examples:
    # Using the class-based interface
    calculator = CountRateCalculator(
        simul_time_vec=time_data,
        simul_detect_vec=detection_data,
        mean_tau=1e-6,
        em_detect_vec=em_data
    )
    results = calculator.calculate_all()

    # Direct function calls for specific calculations
    basic_cps = count_per_second(time_matrix, detection_matrix)
    const_cps = count_per_second_const_dead_time(
        time_matrix, detection_matrix, tau=1e-6
    )

    # Theoretical calculations
    theoretical_cps = calculate_theoretical_cps(
        'constant', mean_tau_s=1e-6, std_tau_s=0,
        detect=10.0, equil=1000.0
    )

Note:
    This module focuses on count rate calculations and dead time effects.
    For core physics parameters, see core_parameters.py. For simulation
    control, see simulation_setting.py.
"""

from typing import Union, List
import numpy as np
import utils as utl
from scipy.stats import norm

rng = np.random.default_rng()


class CountRateCalculator:
    """
    Unified count rate calculator for nuclear reactor simulations.

    This class provides a comprehensive interface for calculating count rates
    from various simulation methods (stochastic, Euler-Maruyama, Taylor,
    Runge-Kutta) with different dead time distributions. It handles the
    complexity of managing multiple simulation data sources and provides
    a clean interface for count rate analysis.

    Attributes
    ----------
    simul_time_vec : List
        List of time matrices from stochastic simulations
    simul_detect_vec : List
        List of detection matrices from stochastic simulations
    kwargs : Dict
        Optional parameters including numerical method data and configuration
    results : Dict
        Dictionary storing calculated count rates
    methods : List[str]
        List of supported numerical methods: ['em', 'taylor', 'rk']
    distributions : List[str]
        List of supported dead time distributions: ['basic', 'const',
        'uniform', 'normal', 'gamma']

    Public Methods
    --------------
    calculate_all()
        Calculate all available count rates
    calculate_theoretical_cps_for_fission_rates()
        Calculate theoretical CPS for multiple fission rates

    Private Methods
    ---------------
    _calculate_stochastic_cps()
        Calculate stochastic simulation count rates
    _calculate_numerical_cps()
        Calculate numerical method count rates
    _calculate_theoretical_approx()
        Calculate theoretical approximation
    _get_simulation_duration()
        Get simulation duration from parameters
    _process_method_distribution()
        Process specific method-distribution combinations
    """

    def __init__(self, simul_time_vec, simul_detect_vec, **kwargs):
        """
        Initialize the count rate calculator

        Parameters
        ----------
        simul_time_vec : list
            List of time matrices
        simul_detect_vec : list
            List of detection matrices
        **kwargs : dict
            Optional parameters including:
            - em_detect_vec, em_const_detect_vec, etc.
            - taylor_const_detect_vec, etc.
            - rk_const_detect_vec, etc.
            - mean_tau: Mean dead time
            - time_params: Time parameters object
        """
        self.simul_time_vec = simul_time_vec
        self.simul_detect_vec = simul_detect_vec
        self.kwargs = kwargs
        self.results = {}

        self.methods = ['em', 'taylor', 'rk']
        self.distributions = ['basic',
                              'const',
                              'uniform',
                              'normal',
                              'gamma']

    def calculate_all(self):
        """
        Calculate all types of count rates.

        Returns
        -------
        dict
            Dictionary containing all calculated count rates
        """
        self._calculate_stochastic_cps()
        self._calculate_numerical_cps()
        self._calculate_theoretical_approx()
        return self.results

    def _calculate_numerical_cps(self):
        """Calculate count rates for numerical methods."""
        duration = self._get_simulation_duration()

        for method in self.methods:
            for distribution in self.distributions:
                self._process_method_distribution(
                    method, distribution, duration)

    def _calculate_stochastic_cps(self):
        """Calculate stochastic count rates."""
        print("Calculating stochastic count rates")
        self.results['stochastic_basic'] = np.array([
            count_per_second(time_mat, detect_mat)
            for time_mat, detect_mat in
            zip(self.simul_time_vec, self.simul_detect_vec)]).flatten()

        mean_tau = self.kwargs.get('mean_tau')
        if mean_tau is not None:
            self.results['stochastic_const_tau'] = np.array([
                count_per_second_const_dead_time(
                    time_mat, detect_mat, mean_tau)
                for time_mat, detect_mat in
                zip(self.simul_time_vec, self.simul_detect_vec)]).flatten()

    def _calculate_theoretical_approx(self):
        """Calculate theoretical approximation."""
        mean_tau = self.kwargs.get('mean_tau')
        if mean_tau is not None and 'stochastic_basic' in self.results:
            print("Calculating theoretical approximation...")
            self.results['theoretical_approx'] = (
                self.results['stochastic_basic']
                * np.exp(-self.results['stochastic_basic'] * mean_tau)
            )

    def _get_simulation_duration(self):
        """
        Get simulation duration from time_params,
        or fallback to stochastic.
        """
        time_params = self.kwargs.get('time_params')
        if time_params is not None:
            duration = time_params.t_end - time_params.t_0
            print("Using configured time duration: "
                  f"{duration:.6f} seconds")
        else:
            duration = self.simul_time_vec[0][:, -1].item()
            print(f"Using stochastic time duration: "
                  f"{duration:.6f} seconds")
        return duration

    def _process_method_distribution(self,
                                     method,
                                     distribution,
                                     duration):
        """Process a specific method-distribution combination"""
        if distribution == 'basic':
            kwarg_key = f'{method}_detect_vec'
            result_key = f'{method}_basic'
            description = f'{method.upper()} without dead time'
        else:
            kwarg_key = f'{method}_{distribution}_detect_vec'
            result_key = f'{method}_{distribution}_tau'
            description = f'{method.upper()} with {distribution} dead time'

        detect_vec = self.kwargs.get(kwarg_key)
        if detect_vec is not None:
            print(f"Calculating {description}")
            self.results[result_key] = np.array([
                detect_vec[j][-1] / duration
                for j in range(len(detect_vec))
            ])

    def calculate_theoretical_cps_for_fission_rates(
            self,
            fission_rates,
            dead_time_type,
            mean_tau_s,
            std_tau_s,
            **physical_kwargs):
        """
        Calculate theoretical CPS for multiple fission rates

        Parameters
        ----------
        fission_rates : array-like
            Array of fission rates
        dead_time_type : str
            Type of dead time distribution
        mean_tau_s : float
            Mean dead time in seconds
        std_tau_s : float
            Standard deviation of dead time in seconds
        **physical_kwargs : dict
            Physical parameters: detect, absorb, source, p_v

        Returns
        -------
        theoretical_cps : np.ndarray
            Array of theoretical count rates for each fission rate
        """
        detect = physical_kwargs.get('detect')
        absorb = physical_kwargs.get('absorb')
        source = physical_kwargs.get('source')
        p_v = physical_kwargs.get('p_v')

        theoretical_cps = np.zeros(len(fission_rates))

        for i, fission_rate in enumerate(fission_rates):
            # Calculate equilibrium for this fission rate
            params = utl.calculate_system_parameters(
                p_v, fission_rate, absorb, source, detect)
            equil = params['equilibrium']

            # Calculate theoretical cps
            theoretical_cps[i] = calculate_theoretical_cps(
                dead_time_type, mean_tau_s, std_tau_s, detect, equil
            )

        return theoretical_cps


def count_per_second(
        time_mat: np.ndarray,
        detect_mat: np.ndarray) -> np.array:
    """
    Calculate counts per second without dead time.

    This function computes the count rate for each row in the detection matrix.

    Parameters
    ----------
    time_mat : np.ndarray
        Matrix containing time information,
        where time_mat[i,-1] gives the total time for row i
    detect_mat : np.ndarray
        Matrix containing detection times,
        where NaN values indicate no detection

    Returns
    -------
    np.ndarray
        Array of count rates (counts per second) for each row

    Raises
    ------
    ValueError
        If time_mat and detect_mat have incompatible shapes
        or if tau is not positive
    """
    # Validate input shapes are  compatbile
    if time_mat.shape[0] != detect_mat.shape[0]:
        raise ValueError(
            "time_mat and detect_mat must have same number of rows")

    num_rows = time_mat.shape[0]
    cps = np.zeros(num_rows)  # initialize count rates array

    # Process each trajectory/row separately
    for i in range(num_rows):
        # Get detection times for this trajectory
        detect_vec = detect_mat[i, :]
        duration = time_mat[i, -1]    # Get total duration for this trajectory

        # Filter out NaN values to get only valid detections
        valid_detections = detect_vec[~np.isnan(detect_vec)]

        # Calculate count rate: number of detections / total time
        cps[i] = len(valid_detections) / duration

    return cps


def count_per_second_const_dead_time(
        time_mat: np.ndarray,
        detect_mat: np.ndarray,
        tau: float) -> np.array:
    """
    Calculate counts per second considering constant dead time effects.

    This function computes the count rate for each row in the detection matrix,
    only counting events that are separated by more than the dead time (tau).

    Parameters
    ----------
    time_mat : np.ndarray
        Matrix containing time information,
        where time_mat[i,-1] gives the total time for row i
    detect_mat : np.ndarray
        Matrix containing detection times,
        where NaN values indicate no detection
    tau : float
        Dead time threshold - minimum time between consecutive detections

    Returns
    -------
    np.ndarray
        Array of count rates (counts per second) for each row
    """

    # Initialize output array
    num_rows = len(time_mat)
    cps = np.zeros(num_rows)
    valid_detections_list = []  # store valid detections for each row

    # Process each row vectorized where possible
    for i in range(num_rows):
        duration = time_mat[i, -1]

        # Get valid detections
        valid_detects = detect_mat[i][~np.isnan(detect_mat[i])]
        # Save the valid detections
        valid_detections_list.append(valid_detects)
        if len(valid_detects) <= 1:
            cps[i] = len(valid_detects) / duration
            continue

        # Calculate time differences between consecutive detections
        time_diffs = np.diff(valid_detects)

        # Count events separated by more than tau
        valid_events = np.sum(time_diffs > tau) + 1

        # Calculate rate
        cps[i] = valid_events / duration

    return cps


def count_per_second_rand_dead_time(
        time_mat: np.ndarray,
        detect_mat: np.ndarray,
        distribution='normal',
        **tau_params) -> np.array:
    """
    Calculate counts per second considering random dead time effects.

    This function computes the count rate for each row in the detection matrix,
    only counting events that are separated by more than the dead time (tau).

    Parameters
    ----------
    time_mat : np.ndarray
        Matrix containing time information,
        where time_mat[i,-1] gives the total time for row i
    valid_detects : np.ndarray
        Matrix containing detection times,
        where NaN values indicate no detection
    distribution : string
        Specifing what distribution is tau generated with
        distribution can either be normal, uniform, exponential, or gamma.

    Returns
    -------
    np.ndarray
        Array of count rates (counts per second) for each row
    """

    num_rows = len(time_mat)
    cps = np.zeros(num_rows)

    for i in range(num_rows):
        valid_detects = detect_mat[i, :]
        valid_detects = valid_detects[~np.isnan(valid_detects)]

        num_detects = len(valid_detects)
        duration = time_mat[i, -1]

        if num_detects <= 1:
            cps[i] = num_detects / duration
            continue

        # Generate tau values for this row's detections
        tau = _generate_tau_for_distribution(
            num_detects - 1, distribution, **tau_params)

        # Calculate time differences between consecutive detections
        time_diffs = np.diff(valid_detects)

        # Count events separated by more than tau
        valid_events = np.sum(time_diffs > tau) + 1

        # Calculate rate
        cps[i] = valid_events / duration

    return cps


def _generate_tau_for_distribution(
        num_values: int,
        distribution: str = 'normal',
        **params) -> np.ndarray:
    """
    Generate tau values based on the specified distribution.
    """
    if distribution == 'normal':
        loc = params.get('loc', 1e-6)
        scale = params.get('scale', 0.5 * 1e-7)
        tau = rng.normal(loc=loc, scale=scale, size=num_values)

    elif distribution == 'uniform':
        low = params.get('low', 1e-6 - np.sqrt(3) * 1e-7)
        high = params.get('high', 1e-6 + np.sqrt(3) * 1e-7)
        tau = rng.uniform(low=low, high=high, size=num_values)

    elif distribution == 'exponential':
        scale = params.get('scale', 1e-6)
        tau = rng.exponential(scale=scale, size=num_values)

    elif distribution == 'gamma':
        shape = params.get('shape', 25)
        scale = params.get('scale', 0.04e-06)
        tau = rng.gamma(shape, scale, size=num_values)

    else:
        raise ValueError(f"Invalid distribution: {distribution}." +
                         " Use 'normal', 'uniform', 'exponential', or 'gamma'")

    # Ensure all tau values are positive
    tau = np.abs(tau)

    return tau


def calculate_all_count_rates(
        simul_time_vec: List,
        simul_detect_vec: List,
        **kwargs):
    """
    Calculate all types of count rates using the CountRateCalculator.

    This convenience function provides a functional interface to the
    CountRateCalculator class, allowing users to calculate all available
    count rates with a single function call.

    Parameters
    ----------
    simul_time_vec : List
        List of time matrices from stochastic simulations
    simul_detect_vec : List
        List of detection matrices from stochastic simulations
    **kwargs : Dict
        Additional parameters for the CountRateCalculator

    Returns
    -------
    Dict
        Dictionary containing all calculated count rates (same format
        as CountRateCalculator.calculate_all())
    """

    calculator = CountRateCalculator(
        simul_time_vec, simul_detect_vec, **kwargs)
    return calculator.calculate_all()


def calculate_theoretical_cps(
        dead_time_type: str,
        mean_tau_s: float,
        std_tau_s: float,
        detect: float,
        equil: float) -> float:
    """
    Calculate theoretical CPS based on dead time distribution.

    This function computes the theoretical count rate for different dead time
    distributions using analytical formulas.

    Parameters
    ----------
    dead_time_type : str
        Type of dead time distribution:
            'constant', 'uniform', 'normal', or 'gamma'
    mean_tau_s : float
        Mean dead time in seconds
    std_tau_s : float
        Standard deviation of dead time in seconds
    detect : float
        Detection rate (s⁻¹)
    equil : float
        Equilibrium neutron population

    Returns
    -------
    float
        Theoretical count rate (CPS)
    """
    if dead_time_type.lower() == 'constant':
        return detect * equil * np.exp(- detect * equil * mean_tau_s)

    if dead_time_type.lower() == 'uniform':
        a_param = np.sqrt(3) * std_tau_s
        return (np.exp(-detect * equil * mean_tau_s) *
                np.sinh(detect * equil * a_param) / a_param)

    if dead_time_type.lower() == 'normal':
        return detect * equil * (
            1 - (norm.cdf(mean_tau_s / std_tau_s) -
                 np.exp(0.5 * (detect * equil * std_tau_s) ** 2 -
                        detect * equil * mean_tau_s) *
                 norm.cdf(mean_tau_s / std_tau_s - detect * equil * std_tau_s)
                 )
        )

    if dead_time_type.lower() == 'gamma':
        shape = (mean_tau_s / std_tau_s) ** 2
        scale = (std_tau_s ** 2) / mean_tau_s
        return detect * equil / ((scale * detect * equil + 1) ** shape)

    raise ValueError(f"Invalid dead time type: {dead_time_type}."
                     " Use 'constant', 'uniform', 'normal', or 'gamma'")


def calculate_theoretical_cps_for_fission_rates(
        fission_rates: Union[List, np.ndarray],
        dead_time_type: str,
        mean_tau_s: float,
        std_tau_s: float,
        *args,
        **kwargs):
    """
    Calculate theoretical CPS for multiple fission rates.

    This function computes theoretical count rates for a range of fission rates
    using the specified dead time distribution.

    Parameters
    ----------
    fission_rates : array-like
        Array of fission rates
    dead_time_type : str
        Type of dead time distribution
    mean_tau_s : float
        Mean dead time in seconds
    std_tau_s : float
        Standard deviation of dead time in seconds
    detect : float
        Detection rate (s⁻¹)
    absorb : float
        Absorption rate (s⁻¹)
    source : float
        Source rate (s⁻¹)
    p_v : float
        Prompt neutron generation probability

    Returns
    -------
    np.ndarray
        Array of theoretical count rates for each fission rate
    """
    # Handle both positional and keyword arguments
    if args and len(args) == 4:
        detect, absorb, source, p_v = args

    elif ('detect' in kwargs and 'absorb' in kwargs and
          'source' in kwargs and 'p_v' in kwargs):
        # Keyword style
        detect = kwargs['detect']
        absorb = kwargs['absorb']
        source = kwargs['source']
        p_v = kwargs['p_v']
    else:
        raise ValueError(
            "Must provide detect, absorb, source, p_v "
            "either as positional args or keywords")

    # Create a temporary calculator instance
    calculator = CountRateCalculator([], [])

    return calculator.calculate_theoretical_cps_for_fission_rates(
        fission_rates, dead_time_type, mean_tau_s, std_tau_s,
        detect=detect, absorb=absorb, source=source, p_v=p_v)
