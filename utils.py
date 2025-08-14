""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Utility functions for nuclear reactor physics calculations.

This module contains utility functions for calculating statistical functions,
simulation setup, data processing, and file managment.

ORGANIZATION:
============

1. STATISTICAL FUNCTIONS - Probability distribution calculations
2. SIMULATION SETUP - Array initialization and system parameters
3. RANDOM NUMBER GENERATION - Wiener processes and random increments
4. DATA PROCESSING - Matrix cleaning and data manipulation
5. FILE MANAGEMENT - Naming conventions and file operations
6. PROGRESS TRACKING - Simulation progress monitoring
7. VALIDATION FUNCTIONS - Validating parameters and variables
"""


import numpy as np
import math

rng = np.random.default_rng()

# =============================================================================
# 1. STATISTICAL FUNCTIONS
# =============================================================================
def mean(p: np.array) -> float:
    """
    "Calculating the mean (or expected value) of a given probability distribution 'p'
    The mean is defined as:
        E[p] = sum_{k=0}^{len(p)} k * p_(k)
    Parameters
    ----------
    p: np.array
        probability distrubtion array
    
    Returns
    -------
    float
        The mean of p
    """
    indices = np.arange(len(p))
    return np.dot(indices,p)

def mean_square(p: np.array) -> float:
    """
    "Calculating the second moment (or mean of square) of a given probability distribution 'p'
    The second moment is defined as:
        E[p^2] = sum_{k=0}^{len(p)} k^2 * p_(k)
    Parameters
    ----------
    p: np.array
        probability distrubtion array
    
    Returns
    -------
    float
        The second moment of p
    """
    indices = np.arange(len(p))
    return np.dot(indices ** 2 , p)


def variance(p: np.array) -> float:
    """
    "Calculating the variance of a given probability distribution 'p'
    The variance is defined as:
        Var(p) = mean_square(p) - mean(p)^2
    Parameters
    ----------
    p: np.array
        probability distrubtion array
    
    Returns
    -------
    float
        The variance of p
        
    Raises
    ------
    """
        
    return mean_square(p) - mean(p) ** 2


def diven_factor(p: np.array, n: int) -> float:
    """
    Calculating the Diven Factor of order 'n' for a given probability distribution 'p.'
    The Diven Factor is defined as:
        D_n = (1/n!) * sum_{k=n}^{len(p)} k!/(n-k)! * p_(k-1)
        
    TODO: this function may be used in future analysis for higher-order moments
    and variance calculations in neutron count distributions.
    
    The Diven Factor is particularly important in nuclear reactor physics for:
    - Calculating variance in neutron count distributions
    - Characterizing the stochastic nature of fission events
    - Analyzing reactor noise and fluctuations
    
    Parameters
    ----------
    p : np.array
        probability distrubtion array.
    n : int
        order of the Diven Factor

    Returns
    -------
    float
        The Diven Factor of order 'n.'
    Notes
    -----
    
    For n=2, this gives the second factorial moment which is related to the variance
    of the particle production distribution.
    """
    if n == 0:
        return 1.0 # D_0 = 1 by definition
    
    if n == 1:
        # D_1 is the mean of the distribution
        return mean(p)
    
    # Calculate Diven factor using vectorized operations
    k_values = np.arange(n, len(p))
    
    # Calculate k! / (k-n)! = k * (k-1) * ... (k - n + 1) efficiently
    # This is the falling factorial (k)_n
    falling_factorial = np.ones_like(k_values, dtype = float)
    for i in range(n):
        falling_factorial *= (k_values - i)
    
    # Calculate the sum
    result = np.sum(falling_factorial * p[k_values]) / math.factorial(n)
    return result


# =============================================================================
# 2. SIMULATION SETUP FUNCTIONS
# =============================================================================

def initialize_simulation_arrays(grid_points: int,
                                  n_0: float,
                                  t_0: float,
                                  t_end: float
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize arrays for numerical methods simulation.
    
    Parameters
    ----------
    grid_points : int
        Number of time steps
    n_0 : float
        Initial population value
    t_0 : float
        Initial time
    t_end : float
        End time
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dt]
        Time space, population array, detection array, and time difference
    """
    dt = (t_end - t_0) / grid_points
    t_space = np.linspace(t_0, t_end, grid_points + 1)
    
    pop = np.zeros(grid_points + 1)
    pop[0] = n_0
    detect = np.zeros(grid_points + 1)
    
    return t_space, pop, detect, dt


def calculate_system_parameters(p_v : np.ndarray,
                                 f : float,
                                 a : float,
                                 s : float,
                                 d : float
                                 ):
    """
    Calculate common system parameters used across numerical methods.
    
    Parameters
    ----------
    p_v : np.ndarray
        Probability distribution array for particle generation in fission
    f : float
        Fission rate constant
    a : float
        Absorption rate constant
    s : float
        Source term constant
    d : float
        Detection rate constant (default: 0)
    tau : float
        Dead time constant
        
    Returns
    -------
    dict
        Dictionary containing calculated parameters:
        - 'lam': total reaction rate
        - 'vbar': expected value of p_v
        - 'vbar_square': second moment of p_v
        - 'alpha': Rossi-alpha coefficient
        - 'equilibrium': equilibrium population (s/alpha)
        - 'sig_tilde_squared': noise parameter
        - 'sig_hat': noise amplitude
    """
    
    lam = f + a + d  # total reaction rate
    vbar = mean(p_v)  # expected value of p_v
    vbar_square = mean_square(p_v)  # second moment of p_v
    # Compute important parameters
    alpha = lam - f * vbar  # Rossi-alpha coefficient
    equil = s / alpha # equilibrium population
    
    # Noise variance associated with the power
    sig_1_squared = equil * (f + a + f * (vbar_square - 2 * vbar)) + s
    
    # Noise variance associated with the physical detections
    sig_2_squared = equil * d
    
    sig_1 = np.sqrt(sig_1_squared)
    sig_2 = np.sqrt(sig_2_squared)
    
    return {
        'lam' : lam,
        'vbar' : vbar,
        'vbar_square' : vbar_square,
        'alpha' : alpha,
        'equilibrium' : equil,
        'sig_1' : sig_1,
        'sig_2' : sig_2
    }


# =============================================================================
# 3. RANDOM NUMBER GENERATION FUNCTIONS
# =============================================================================

def generate_random_increments(grid_points : int,
                                dt : float,
                                num_processes : int):
    """ 
    Generate random Wiener increments for Euler-Maruyama integration
    
    Parameters
    ----------
    grid_points : int
        Number of time steps
    dt : float
        Time step size
    num_processes : int
        Number of Wiener processes to generate
        
    Returns
    -------
    tuple
        Tuple of num_processes numpy arrays, each containing random increments
    """
    
    if num_processes < 1:
        raise ValueError(f"num_processes must be positive, got {num_processes}")
    
    # Generate all Wiener processes at once
    increments = []
    for i in range(num_processes):
        dW_vec = rng.normal(loc = 0.0, scale = np.sqrt(dt), size = grid_points)
        increments.append(dW_vec)
        
    return tuple(increments)


def generate_iterated_integral(grid_points : int,
                               dt: float,
                               increments) -> tuple:
    """
    Generate mixed Wiener increments (dZ) for Taylor methods.
    
    This function generates the iterated integrals dZ that appear in taylor methods.
    The dZ increments satify:
    - E[dZ] = 0
    - E[dZ^2] = dt^3 / 12
    - dZ = 0.5 * dt *(dW + xi), where xi ~ N(0,sqrt(dt/3))
    
    Parameters
    ----------
    grid_points : int
        Number of time steps
    dt : float
        Time step size
    increments : tuple
        Tuple of Wiener increments (dW) arrays, each of length grid_points

    Returns
    -------
    tuple
        Tuple of dZ arrays, each of length grid_points

    """
    
    num_processes = len(increments)
    dZ_arrays = []
    
    for i in range(num_processes):
        # Generate independent Gaussian noise for each process
        gaussian = rng.normal(loc = 0.0, scale = np.sqrt(dt/3.0), size = grid_points)
        
        # Calculate dZ = 0.5 * dt * (dW + xi)
        dZ = 0.5 * dt * (increments[i] + gaussian)
        dZ_arrays.append(dZ)
        
    return tuple(dZ_arrays)
        

# =============================================================================
# 4. DATA PROCESSING FUNCTIONS
# =============================================================================

def clean_detection_matrix(detect_mat : np.ndarray) -> np.ndarray:
    """
    Clean detection matrix by removing np.nan values and creating a matrix
    with only valid detection times.
    
    Parameters
    ----------
    detect_mat : np.ndarray
        Original detection matrix with np.nan values
    
    Returns
    -------
    Np.ndarray
        Clean detection matrix with only valid detection times
    """   
    k = detect_mat.shape[0]       
    cleaned_matrices = []
    
    for i in range(k):
        # Get valid detections for this trajectory (remove np.nan)
        valid_detections = detect_mat[i,:]
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

def extract_simulation_prefix(index: str, fission_value : float) -> str:
    """
    Extract simulation prefix from index string.

    Parameters
    ----------
    index : str
        Simulation index (e.g., 'mil_f33.94')
    fission_value : float
        Fission rate constant

    Returns
    -------
    str
        Extract prefix (e.g., 'mil_f')
        
    Examples
    --------
    >>> extract_simulation_prefix('mil_f33.94', 33.94)
    'mil_f'
    >>> extract_simulation_prefix('short_f33.98', 33.98)
    'short_f'
    """
    return index.split(str(fission_value))[0]


def generate_filename(prefix : str, method : str, dead_time_type : str,
                      fission_value : float, extension: str = '.npy') -> str:
    """
    Generate standardized filename for simulation results.

    Parameters
    ----------
    prefix : str
        File prefix (e.g., 'mil_f', 'short_f')
    method : str
        Numerical method ('EM', 'Taylor_strong_1_5', 'Taylr_weak_2_0')
    dead_time_type : str
        Dead time model('basic', 'const', 'uniform', 'exp')
    fission_value : float
        Fission rate constant
    extension : str, optional
        File extension (default: '.npy')

    Returns
    -------
    str
        Generated filename

    """
    
    return f"{method}_{dead_time_type}_Dead_Time_{prefix}{fission_value:.3f}{extension}"

# =============================================================================
# 6. PROGRESS TRACKING FUNCTIONS
# =============================================================================

def progress_tracker(grid_points : int):
    """
    Create a progress tracking generator

    Parameters
    ----------
    grid_points : int
        number of time steps
    
    Yields
    -------
    int
        Current iteration index

    """
    
    next_percent = 10

    for i in range(grid_points):
        progress = (i + 1) * 100 / grid_points
        if progress >= next_percent:
            print(f"{next_percent:.0f}% complete")
            next_percent += 10 
        yield i



    

