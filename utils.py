"""Written by Tomer279 with the assistance of Cursor.ai.

Utility functions for nuclear physics calculations and simulation support.

This module provides comprehensive utility functions for nuclear physics
calculations, statistical analysis, simulation setup, data processing, and file
management. It serves as the foundational support system for all numerical
methods and simulation engines, providing essential mathematical functions,
parameter calculations, and operational utilities.

Key Features:
    - Statistical functions for probability distribution analysis
    - Dead time modeling with Psi function calculations
    - System parameter calculations for nuclear reactor dynamics
    - Data processing and matrix cleaning utilities
    - File management and naming convention standardization
    - Progress tracking and simulation monitoring
    - Random number generation for stochastic processes

Mathematical Functions:
    - Mean, variance, and moment calculations for probability distributions
    - Psi function calculations for various dead time distributions
    - System parameter derivations (Rossi-alpha, equilibrium, noise amplitudes)
    - Detection matrix cleaning and data validation

Dependencies:
    numpy: For numerical operations and array processing
    scipy.stats: For statistical functions (normal distribution CDF)

Usage Examples:
    # Statistical calculations
    mean_val = mean(probability_distribution)
    variance_val = variance(probability_distribution)

    # Dead time modeling
    psi_val = calculate_psi_function(detection_rate, dead_time, 'constant')

    # System parameter calculations
    params = calculate_system_parameters(p_v, fission, absorb, source, detect)
    equilibrium = params['equilibrium']

    # Data processing
    clean_detections = clean_detection_matrix(detection_matrix)

    # File management
    filename = generate_filename('f', 'EM', 'const', 33.95)

    # Progress tracking
    for i in progress_tracker(1000):
        # Simulation step
        pass

Note:
    This module provides foundational utilities used across all simulation
    methods. For specific numerical methods, see their respective modules:
    stochastic_simulation.py, euler_maruyama_methods.py, etc.
"""

from typing import Generator, Optional
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng()

# =============================================================================
# 1. STATISTICAL FUNCTIONS
# =============================================================================


def mean(p: np.ndarray) -> float:
    """
    Calculate the mean (expected value) of a probability distribution.

    The mean is defined as:
    E[p] = Σ(k=0 to len(p)-1) k * p(k)

    Parameters
    ----------
    p : np.ndarray
        Probability distribution array where p[k] is the probability
        of outcome k

    Returns
    -------
    float
        The mean (expected value) of the probability distribution

    Examples
    --------
    >>> p = np.array([0.1, 0.3, 0.4, 0.2])
    >>> mean(p)
    1.7
    """
    indices = np.arange(len(p))
    return np.dot(indices, p)


def mean_square(p: np.array) -> float:
    """
    Calculate the second moment (mean of squares)
    of a probability distribution.

    The second moment is defined as:
    E[p²] = Σ(k=0 to len(p)-1) k² * p(k)

    Parameters
    ----------
    p : np.ndarray
        Probability distribution array where p[k] is the probability
        of outcome k

    Returns
    -------
    float
        The second moment (mean of squares) of the probability distribution

    Examples
    --------
    >>> p = np.array([0.1, 0.3, 0.4, 0.2])
    >>> mean_square(p)
    3.5
    """
    indices = np.arange(len(p))
    return np.dot(indices ** 2, p)


def variance(p: np.array) -> float:
    """
    Calculate the variance of a probability distribution.

    The variance is defined as:
    Var(p) = E[p²] - (E[p])² = mean_square(p) - mean(p)²

    Parameters
    ----------
    p : np.ndarray
        Probability distribution array where p[k] is the probability
        of outcome k

    Returns
    -------
    float
        The variance of the probability distribution

    Examples
    --------
    >>> p = np.array([0.1, 0.3, 0.4, 0.2])
    >>> variance(p)
    0.61
    """

    return mean_square(p) - mean(p) ** 2


def calculate_psi_function(
        x: float,
        tau: float,
        dead_time_type: str,
        std: Optional[float] = None) -> float:
    """
    Calculate the Psi function for dead time modeling in detection systems.

    The Psi function Ψ(x) represents the average probability for a detection
    to occur within a dead time period of the previous detection. It is
    calculated as:
    Ψ(x) = 1 - ∫₀^∞ e^(-x*s) * f_τ(s) ds

    where f_τ is the probability density function
    of the dead time distribution.

    Parameters
    ----------
    x : float
        Detection rate parameter (typically d * N where d is detection
        rate constant and N is population)
    tau : float
        Mean dead time value (seconds)
    dead_time_type : str
        Type of dead time distribution:
          - 'constant': Fixed dead time
          - 'uniform': Uniform distribution
          - 'normal': Normal (Gaussian) distribution
          - 'gamma': Gamma distribution
    std : Optional[float]
        Standard deviation of dead time distribution (required for
        'uniform', 'normal', and 'gamma' types)

    Returns
    -------
    float
        Psi function value Ψ(x) representing dead time effects

    Raises
    ------
    ValueError
        If dead_time_type is not supported or std is None when required

    Examples
    --------
    >>> # Constant dead time
    >>> psi = calculate_psi_function(10.0, 1e-6, 'constant')
    >>> print(f"Psi value: {psi:.6f}")

    >>> # Normal dead time
    >>> psi = calculate_psi_function(10.0, 1e-6, 'normal', std=0.1e-6)
    """

    if dead_time_type == 'constant':
        return 1 - np.exp(-x * tau)

    if dead_time_type == 'uniform':
        a_param = np.sqrt(3) * std
        return (1 - (np.exp(-x * tau) * np.sinh(x * a_param)) / (x * a_param))

    if dead_time_type == 'normal':

        return (1 - (norm.cdf(tau / std)
                     - np.exp(0.5 * (x * std) ** 2 - x * tau)
                     * norm.cdf(tau/std - x * std)))
    if dead_time_type == 'gamma':
        shape = (tau / std) ** 2
        scale = (std ** 2) / tau
        return 1 - 1 / ((scale * x + 1) ** shape)

    return None


# =============================================================================
# 2. SIMULATION SETUP FUNCTIONS
# =============================================================================


def calculate_system_parameters(
        p_v: np.ndarray,
        fission_rate: float,
        absorb_rate: float,
        source_rate: float,
        detect_rate: float) -> dict[str, float]:
    """
    Calculate comprehensive system parameters for nuclear reactor dynamics.

    This function computes all essential parameters used across different
    numerical methods for nuclear reactor simulation, including reaction
    rates, equilibrium values, noise amplitudes, and statistical properties.

    Parameters
    ----------
    p_v : np.ndarray
        Probability distribution array for neutron yield in fission events
    fission_rate : float
        Fission rate constant (source_rate⁻¹)
    absorb_rate : float
        Absorption rate constant (source_rate⁻¹)
    source_rate : float
        Source term constant (source_rate⁻¹)
    detect_rate : float
        Detection rate constant (source_rate⁻¹)

    Returns
    -------
    Dict[str, float]
        Dictionary containing calculated system parameters:
          - 'lam': Total reaction rate (fission_rate + absorb_rate + detect_rate)
          - 'vbar': Expected neutron yield per fission
          - 'vbar_square': Second moment of neutron yield distribution
          - 'alpha': Rossi-alpha coefficient (lam - fission_rate * vbar)
          - 'equilibrium': Equilibrium population (source_rate / alpha)
          - 'sig_1_squared': Noise variance for population dynamics
          - 'sig_2_squared': Noise variance for detection processes
          - 'sig_1': Noise amplitude for population dynamics
          - 'sig_2': Noise amplitude for detection processes

    Examples
    --------
    >>> p_v = np.array([0.1, 0.3, 0.4, 0.2])  # Neutron yield distribution
    >>> params = calculate_system_parameters(p_v, 33.95, 0.1, 10.0, 0.01)
    >>> print(f"Equilibrium population: {params['equilibrium']:.2f}")
    >>> print(f"Rossi-alpha: {params['alpha']:.4f}")
    """
    lam = fission_rate + absorb_rate + detect_rate
    vbar = mean(p_v)
    vbar_square = mean_square(p_v)

    alpha = lam - fission_rate * vbar
    equil = source_rate / alpha

    # Noise variance associated with the power
    sig_1_squared = equil * (
        fission_rate + absorb_rate
        + fission_rate * (vbar_square - 2 * vbar)) + source_rate

    # Noise variance associated with the physical detections
    sig_2_squared = equil * detect_rate

    sig_1 = np.sqrt(sig_1_squared)
    sig_2 = np.sqrt(sig_2_squared)

    return {
        'lam': lam,
        'vbar': vbar,
        'vbar_square': vbar_square,
        'alpha': alpha,
        'equilibrium': equil,
        'sig_1_squared': sig_1_squared,
        'sig_2_squared': sig_2_squared,
        'sig_1': sig_1,
        'sig_2': sig_2
    }


# =============================================================================
# 3. DATA PROCESSING FUNCTIONS
# =============================================================================

def clean_detection_matrix(detect_mat: np.ndarray) -> np.ndarray:
    """
    Clean detection matrix by removing NaN values and standardizing format.

    This function processes detection matrices from stochastic simulations
    by removing NaN values (representing non-detection events) and creating
    a standardized matrix format with consistent dimensions across all
    trajectories.

    Parameters
    ----------
    detect_mat : np.ndarray
        Original detection matrix with shape (trajectories, time_steps)
        containing detection timestamps and NaN values for non-detections

    Returns
    -------
    np.ndarray
        Cleaned detection matrix with shape (trajectories, max_detections)
        containing only valid detection timestamps, padded with zeros
        for trajectories with fewer detections

    Examples
    --------
    >>> # Original matrix with NaN values
    >>> detect_mat = np.array([[1.0, 2.0, np.nan, 4.0],
    ...                        [0.5, np.nan, np.nan, 3.5]])
    >>> clean_mat = clean_detection_matrix(detect_mat)
    >>> print(clean_mat.shape)  # (2, 3) - max 3 detections across trajectories
    """
    k = detect_mat.shape[0]
    cleaned_matrices = []

    for i in range(k):
        # Get valid detections for this trajectory (remove np.nan)
        valid_detections = detect_mat[i, :]
        valid_detections = valid_detections[~np.isnan(valid_detections)]
        cleaned_matrices.append(valid_detections)

    # Find the maximum number of detections across all trajectories
    max_detections = max(len(det) for det in cleaned_matrices)

    # Create final matrix with consistent shape
    clean_detect_mat = np.zeros((k, max_detections))
    for i in range(k):
        valid_detections = cleaned_matrices[i]
        clean_detect_mat[i, :len(valid_detections)] = valid_detections

    return clean_detect_mat

# =============================================================================
# 5. FILE MANAGEMENT FUNCTIONS
# =============================================================================


def extract_simulation_prefix(index: str, fission_value: float) -> str:
    """
    Extract simulation prefix from index string for file organization.

    This function parses simulation index strings to extract the prefix
    portion, which is used for consistent file naming and directory
    organization across different simulation runs.

    Parameters
    ----------
    index : str
        Simulation index string (e.g., 'mil_f33.94', 'short_f33.98')
    fission_value : float
        Fission rate constant used in the simulation

    Returns
    -------
    str
        Extracted prefix string (e.g., 'mil_f', 'short_f')

    Examples
    --------
    >>> prefix = extract_simulation_prefix('mil_f33.94', 33.94)
    >>> print(prefix)  # 'mil_f'

    >>> prefix = extract_simulation_prefix('short_f33.98', 33.98)
    >>> print(prefix)  # 'short_f'
    """
    return index.split(str(fission_value))[0]


def generate_filename(
        prefix: str,
        method: str,
        dead_time_type: str,
        fission_value: float,
        extension: str = '.npy') -> str:
    """
    Generate standardized filename for simulation results.

    This function creates consistent, descriptive filenames for simulation
    results following established naming conventions. The generated filenames
    include method type, dead time model, and parameter values for easy
    identification and organization.

    Parameters
    ----------
    prefix : str
        File prefix for simulation identification (e.g., 'mil_f', 'short_f')
    method : str
        Numerical method identifier ('EM', 'Taylor', 'RK', 'Stochastic')
    dead_time_type : str
        Dead time model type ('basic', 'const', 'uniform', 'normal', 'gamma')
    fission_value : float
        Fission rate constant value
    extension : str, optional
        File extension (default: '.npy')

    Returns
    -------
    str
        Generated standardized filename

    Examples
    --------
    >>> filename = generate_filename('f', 'EM', 'const', 33.95)
    >>> print(filename)  # 'EM_const_Dead_Time_f33.950.npy'

    >>> filename = generate_filename(
    ...     'mil_f', 'Taylor', 'uniform', 33.94, '.dat')
    >>> print(filename)  # 'Taylor_uniform_Dead_Time_mil_f33.940.dat'
    """

    return (f"{method}_{dead_time_type}_Dead_Time_"
            f"{prefix}{fission_value:.3f}{extension}")

# =============================================================================
# 6. PROGRESS TRACKING FUNCTIONS
# =============================================================================


def progress_tracker(grid_points: int) -> Generator[int, None, None]:
    """
    Create a progress tracking generator for simulation monitoring.

    This generator provides progress reporting during long-running simulations,
    printing completion percentages at 10% intervals to keep users informed
    of simulation progress without excessive output.

    Parameters
    ----------
    grid_points : int
        Total number of time steps or iterations in the simulation

    Yields
    ------
    int
        Current iteration index (0 to grid_points-1)

    Examples
    --------
    >>> # Use in simulation loops
    >>> for i in progress_tracker(1000):
    ...     # Perform simulation step
    ...     result = simulate_step(i)
    ...     # Progress will be printed at 10%, 20%, 30%, etc.
    """

    next_percent = 10

    for i in range(grid_points):
        progress = (i + 1) * 100 / grid_points
        if progress >= next_percent:
            print(f"{next_percent:.0f}% complete")
            next_percent += 10
        yield i
