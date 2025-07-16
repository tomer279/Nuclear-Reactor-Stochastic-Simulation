""" Written by Tomer279 with the assistance of Cursor.ai """

"""
LEGACY SIMULATION FILE - DEPRECATED

This file contains the original implementation of nuclear reactor simulations
with all functions in a single file. This approach has been replaced by a
modular structure with separate files for different functionalities.

CURRENT MODULAR STRUCTURE:
- config.py: Configuration management and parameters
- data_management.py: Data loading, saving, and organization
- count_rates.py: Count rate calculations and dead time effects
- plot_simulations.py: Visualization and plotting functions
- simulation_runner.py: Simulation execution and orchestration
- euler_maruyama_methods.py: Euler-Maruyama simulation methods
- taylor_methods.py: Taylor series expansion methods

DEPRECATED FUNCTIONS IN THIS FILE:
- Stochastic simulation functions → simulation_runner.py
- Euler-Maruyama simulation functions → euler_maruyama_methods.py
- Taylor method simulation functions → taylor_methods.py
- Data loading/saving functions → data_management.py
- Plotting and visualization functions → plot_simulations.py
- Count rate calculations → count_rates.py
- Configuration and parameter management → config.py

SIMULATION METHODS INCLUDED:
- Stochastic simulations
- Euler-Maruyama method (with/without dead time)
- Taylor series expansion methods
- Population dynamics calculations
- Detection and count rate analysis
- Dead time effects (basic, constant, exponential)

This file is kept for reference and historical purposes only.
DO NOT USE FOR NEW DEVELOPMENT.
"""

# =============================================================================
# DEPRECATED: This file is no longer used in the current project structure
# =============================================================================
# 
# The functions in this file have been reorganized into separate modules:
# 
# SIMULATION METHODS:
# - Stochastic simulations → simulation_runner.py
# - Euler-Maruyama methods → euler_maruyama_methods.py
# - Taylor series methods → taylor_methods.py
#
# DATA MANAGEMENT:
# - Data loading/saving → data_management.py
# - Configuration → config.py
#
# ANALYSIS & VISUALIZATION:
# - Count rate calculations → count_rates.py
# - Plotting functions → plot_simulations.py
#
# Please use the new modular structure instead of this legacy file.
# =============================================================================

import math
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# =============================================================================
# UTILITY FUNCTIONS
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
# COUNT RATE CALCULATION FUNCTIONS
# =============================================================================

def count_per_second(time_mat, detect_mat) -> np.array:
    """
    Calculate counts per second without dead time.
    
    This function computes the count rate for each row in the detection matrix.
    
    Parameters
    ----------
    time_mat : np.ndarray
        Matrix containing time information, where time_mat[i,-1] gives the total time for row i
    detect_mat : np.ndarray
        Matrix containing detection times, where NaN values indicate no detection
        
    Returns
    -------
    np.ndarray
        Array of count rates (counts per second) for each row
    
    Raises
    ------
    ValueError
        If time_mat and detect_mat have incompatible shapes or if tau is not positive
    """
    # Validate input shapes are  compatbile
    if time_mat.shape[0] != detect_mat.shape[0]:
        raise ValueError("time_mat and detect_mat must have same number of rows")
        
    num_rows = time_mat.shape[0]
    cps = np.zeros(num_rows) # initialize count rates array
    
    # Process each trajectory/row separately
    for i in range(num_rows):
        detect_vec = detect_mat[i,:] # Get detection times for this trajectory
        duration = time_mat[i,-1]    # Get total duration for this trajectory
        
        # Filter out NaN values to get only valid detections
        valid_detections = detect_vec[~np.isnan(detect_vec)]
        
        # Calculate count rate: number of detections / total time
        cps[i] = len(valid_detections) / duration
        
    return cps
    

def count_per_second_const_dead_time(time_mat, detect_mat, tau) -> np.array:
    """
    Calculate counts per second considering constant dead time effects.
    
    This function computes the count rate for each row in the detection matrix,
    only counting events that are separated by more than the dead time (tau).
    
    Parameters
    ----------
    time_mat : np.ndarray
        Matrix containing time information, where time_mat[i,-1] gives the total time for row i
    detect_mat : np.ndarray
        Matrix containing detection times, where NaN values indicate no detection
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
    valid_detections_list = [] # store valid detections for each row
    
    # Process each row vectorized where possible
    for i in range(num_rows):
        # Get valid detections
        valid_detects = detect_mat[i][~np.isnan(detect_mat[i])]
        valid_detections_list.append(valid_detects) # Save the valid detections
        if len(valid_detects) <= 1:
            cps[i] = len(valid_detects) / time_mat[i,-1]
            continue
            
        # Calculate time differences between consecutive detections
        time_diffs = np.diff(valid_detects)
        
        # Count events separated by more than tau
        valid_events = np.sum(time_diffs > tau) + 1
        
        # Calculate rate
        cps[i] = valid_events / time_mat[i,-1]
        
    return cps


def count_per_second_rand_dead_time(
        time_mat, valid_detects, distribution = 'normal', **tau_params) -> np.array:
    """
    Calculate counts per second considering random dead time effects.
    
    This function computes the count rate for each row in the detection matrix,
    only counting events that are separated by more than the dead time (tau).
    
    Parameters
    ----------
    time_mat : np.ndarray
        Matrix containing time information, where time_mat[i,-1] gives the total time for row i
    detect_mat : np.ndarray
        Matrix containing detection times, where NaN values indicate no detection
    distribution : string
        Specifing what distribution is tau generated with
        distribution can either be normal, exponential, or uniform.
    
    Returns
    -------
    np.ndarray
        Array of count rates (counts per second) for each row
    """
        
    # Initialize output array
    num_rows = len(time_mat)
    num_detects = len(valid_detects)
    cps = np.zeros(num_rows)
    
    tau = _generate_tau_for_distribution(num_detects - 1, distribution,
                                        **tau_params)
    
    # Process each row vectorized where possible
    for i in range(num_rows):
        if num_detects <= 1:
            cps[i] = len(valid_detects) / time_mat[i,-1]
            continue
            
        # Calculate time differences between consecutive detections
        time_diffs = np.diff(valid_detects)
        
        # Count events separated by more than tau
        valid_events = np.sum(time_diffs > tau) + 1
        
        # Calculate rate
        cps[i] = valid_events / time_mat[i,-1]
        
    return cps

def _generate_tau_for_distribution(num_values, distribution='normal', **params):
    """
    Generate tau values based on the specified distribution.
    
    Parameters
    ----------
    num_values : int
        Number of tau values to generate
    distribution : str
        Distribution type: 'normal', 'uniform', or 'exponential'
    **params : dict
        Distribution parameters
        
    Returns
    -------
    np.ndarray
        Array of generated tau values
    """
    if distribution == 'normal':
        loc = params.get('loc', 1e-6)
        scale = params.get('scale', 0.5 * 1e-7)
        tau = rng.normal(loc=loc, scale=scale, size=num_values)
        
    elif distribution == 'uniform':
        low = params.get('low', 1e-6 - 2 * np.sqrt(3) * 1e-7 )
        high = params.get('high', 1e-6 + 2 * np.sqrt(3)  * 1e-7)
        tau = rng.uniform(low=low, high=high, size=num_values)
        
    elif distribution == 'exponential':
        scale = params.get('scale', 1e-6)
        tau = rng.exponential(scale=scale, size=num_values)
    
    elif distribution == 'gamma':
        shape = params.get('shape', 25)
        scale = params.get('scale', 0.04e-06)
        tau = rng.gamma(shape, scale, size = num_values)
    
    else:
        raise ValueError(f"Invalid distribution: {distribution}." +
                         " Use 'normal', 'uniform', 'exponential', or 'gamma'")
    
    # Ensure all tau values are positive
    tau = np.abs(tau)
    
    return tau


def calculate_all_count_rates(simul_time_vec, simul_detect_vec, em_detect_vec = None,
                              em_const_detect_vec = None, em_exp_detect_vec = None,
                              mean_tau = None):
    """
   Calculate all types of count rates in one organized function.
   
   Parameters
   ----------
   simul_time_vec : list
       List of time matrices
   simul_detect_vec : list
       List of detection matrices
   em_detect_vec : list, optional
       List of Euler-Maruyama detection arrays (without dead time)
   em_const_detect_vec : list, optional
       List of Euler-Maruyama detection arrays (with constant dead time)
   mean_tau : float, optional
       Mean dead time for calculations

       
   Returns
   -------
   dict
       Dictionary containing all calculated count rates with the following keys:
           
           - 'stochastic_basic': np.ndarray
           Count rates from stochastic simulation without dead time effects.
           Shape: (num_fission_values * num_trajectories,)
           Values: Counts per second (float)
           
       - 'stochastic_const_tau': np.ndarray (if mean_tau provided)
           Count rates from stochastic simulation with constant dead time.
           Shape: (num_fission_values * num_trajectories,)
           Values: Counts per second (float)
           
       - 'em_basic': np.ndarray (if em_detect_vec provided)
           Count rates from Euler-Maruyama simulation without dead time.
           Shape: (num_fission_values,)
           Values: Counts per second (float)
           
       - 'em_const_tau': np.ndarray (if em_const_detect_vec provided)
           Count rates from Euler-Maruyama simulation with constant dead time.
           Shape: (num_fission_values,)
           Values: Counts per second (float)
           
       - 'theoretical_approx': np.ndarray 
           (if mean_tau provided and stochastic_basic exists)
           Theoretical approximation using exponential dead time correction.
           Shape: (num_fission_values * num_trajectories,)
           Values: Counts per second (float)
   """
    
    results = {}
    
    # Calculate stochastic count rates
    print("Calculating stochastic count rates")
    results['stochastic_basic'] = np.array([
        count_per_second(time_mat, detect_mat)
        for time_mat,detect_mat in 
        zip(simul_time_vec, simul_detect_vec)]).flatten()
    
    if mean_tau is not None:
        results['stochastic_const_tau'] = np.array([
            count_per_second_const_dead_time(time_mat, detect_mat, mean_tau)
            for time_mat, detect_mat in 
            zip(simul_time_vec, simul_detect_vec)]).flatten()
    
    # Calculate Euler-Maruyama count rates
    if em_detect_vec is not None:
        print("Calculating Euler-Maruyama count rates (without dead time)")
        results['em_basic'] = np.array([
            em_detect_vec[j][-1] / simul_time_vec[j][:,-1].item()
            for j in range(len(simul_time_vec))
            ])
    
    if em_const_detect_vec is not None:
        print("Calculating Euler-Maruyama count rates (with constant dead time)")
        results['em_const_tau'] = np.array([
            em_const_detect_vec[j][-1] / simul_time_vec[j][:,-1].item()
            for j in range(len(simul_time_vec))
            ])
    
    if em_exp_detect_vec is not None:
        print("Calculating Euler-Maruyama count rates (with exponential dead time)")
        results['em_exp_tau'] = np.array([
            em_exp_detect_vec[j][-1] / simul_time_vec[j][:,-1].item()
            for j in range(len(simul_time_vec))
            ])
    
    # Calculate theoretical approximation
    if mean_tau is not None and 'stochastic_basis' in results:
        print("Calculating theoretical approximation...")
        results['theoretical_approx'] = results['stochastic_basic'] * \
            np.exp(-results['stochastic_basic'] * mean_tau)
    
    
    return results


# =============================================================================
# STOCHASTIC SIMULATION FUNCTIONS
# =============================================================================

def pop_dyn_mat(p_v: np.ndarray,
                f: float,
                a: float,
                s: float,
                d: float,
                n_0: np.ndarray,
                t_0: float,
                steps: int,
                index: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates population dynamics in matrix form with multiple initial conditions.
    
    This function implements a stochastic simulation of population dynamics using
    a matrix approach to handle multiple trajectories simultaneously. Each trajectory
    represents a different initial condition from n_0.
    
    Parameters
    ----------
    p_v : np.ndarray
        Probability distribution array for the number of particles produced in fission
    f : float
        Fission rate constant
    a : float
        Absorption rate constant
    s : float
        Source term constant
    d : float
        Detection rate constant
    n_0 : np.ndarray
        Array of initial population values
    t_0 : float
        Initial time
    steps : int
        Number of time steps to simulate
    index : str
        Identifier for saving simulation results
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, nd.array]
        Returns (time_matrix, population_matrix) where:
        - time_matrix has shape (k, steps) containing time points
        - population_matrix has shape (k, steps) containing population values
        - detection_matrix has shape (k, valid_detections) 
        containing only valid detection times
        k is the number of initial conditions
        
    Notes
    -----
    The simulation uses numpy's random choice for event selection with
    probabilities determined by the current population and rate constants.
    Results are saved to files with names based on the provided index.
    """
    # Input validation
    if steps < 1:
        raise ValueError("steps must be positive")
    if t_0 < 0:
        raise ValueError("initial time must be non-negative")
    if any(x < 0 for x in [f, a, s, d]):
        raise ValueError("rate constants must be non-negative")
        
        
    k = len(n_0)
    num_steps = steps - 1
    
    # Pre-allocate arrays
    pop_mat = np.zeros((k, steps))
    time_mat = np.zeros((k, steps))
    event_mat = np.zeros((k, steps))
    detect_mat = np.zeros((k, steps))
    
    # Initialize first column
    pop_mat[:, 0] = n_0
    time_mat[:, 0] = t_0
    event_mat[:, 0] = np.nan
    detect_mat[:, 0] = np.nan
    
    
    # Main simulation loop
    for i in range(k):
        if i & 10 == 0:
            print(f"Processing trajectory {i + 1}/{k}")
        _simulate_single_trajectory(i, num_steps, pop_mat,
                                    time_mat, event_mat, detect_mat, p_v,
                                    f, a, s, d)
    
    # Post-process detect_mat to remove np.nan values
    clean_detect_mat = _clean_detection_matrix(detect_mat)
    
    # Save results
    np.save(f'Simul_Pop_Matrix_{index}', pop_mat)
    np.save(f'Simul_Time_Matrix_{index}', time_mat)
    np.save(f'Detection_Matrix_{index}', clean_detect_mat)
    
    return time_mat, pop_mat, clean_detect_mat


def _update_population_for_event(current_population, event_type, p_v):
    "Update population based on the type of event that occurred."
    
    if event_type == 0: # Source
        return current_population + 1
    elif event_type == 1: # Fission
        particles_produced = rng.choice(len(p_v), p = p_v)
        return current_population + particles_produced - 1
    elif event_type == 2 or event_type == 3: # absorption or detection
        return current_population - 1
    else:
        raise ValueError(f"Unknown event type: {event_type}")
   

def _simulate_single_trajectory(
        trajectory_index, num_steps, pop_mat, time_mat, event_mat, detect_mat,
        p_v, f, a, s, d):
    "Simulate a single trajectory"
    next_percent = 10
    
    i = trajectory_index
    for j in range(num_steps): 
        progress = (j + 1) * 100 / num_steps
        if progress >= next_percent:
            print(f"{next_percent:.0f}% complete")
            next_percent += 10   
            
        current_pop = pop_mat[i, j]
        current_time = time_mat[i, j]
            
        # Calculate total rate and probabilities
        total_rate = (f + a + d) * current_pop + s
            
        if total_rate <= 0:
            # System is static
            time_mat[i, j + 1] = time_mat[i, j]
            pop_mat[i, j + 1] = current_pop
            event_mat[i, j + 1] = np.nan
            detect_mat[i, j + 1] = np.nan
            continue
            
            # Calculate event probabilities
        event_probs = np.array([
            s / total_rate,                 # source
            current_pop * f / total_rate,   # fission
            current_pop * a / total_rate,   # absorption
            current_pop * d / total_rate    # detection
        ])
        
        # Generate event and time increment
        event_type = rng.choice(4, p = event_probs)
        time_increment = rng.exponential(scale = 1/total_rate)
        
        # Update time and record event
        time_mat[i, j + 1] = current_time + time_increment
        event_mat[i, j + 1] = event_type
        
        # Update population based on event type
        new_pop = _update_population_for_event(current_pop, event_type, p_v)
        pop_mat[i, j + 1] = new_pop
        
        # Handle detection events
        if event_type == 3: # Detection
            detect_mat[i, j + 1] = current_time + time_increment
        else:
            detect_mat[i, j + 1] = np.nan
                
   
def _clean_detection_matrix(detect_mat : np.ndarray) -> np.ndarray:
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
        
        # Pad with zeros to maintain consistent matrix shape
        # Find the maximum number of detections across all trajectories
        if i == 0:
            max_detections = len(valid_detections)
        else:
            max_detections = max(max_detections, len(valid_detections))
        
        cleaned_matrices.append(valid_detections)
    
    # Create final matrix with consistent shape
    clean_detect_mat = np.zeros((k, max_detections))
    for i in range(k):
        valid_detections = cleaned_matrices[i]
        clean_detect_mat[i, :len(valid_detections)] = valid_detections
    
    return clean_detect_mat
  
# =============================================================================
# EULER-MARUYAMA SIMULATION FUNCTIONS
# =============================================================================  
    
def euler_maruyama(p_v: np.ndarray,
                 f: float,
                 a: float,
                 s: float,
                 d: float,
                 t_0: float,
                 t_end: float,
                 grid_points: int,
                 n_0: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the Euler-Maruyama method for solving stochastic differential equations (SDEs)
    in the context of population dynamics.
    
    This method is used to numerically solve the SDE:
    dN_t = (-αN_t + s)dt + sigma_hat * dW 
    which is eq. 14 from Dubi & Atar (2017)
    "Modeling neutron count distribution in a subcritical core by stochastic
     differential equations"
    where:
    - N_t is the size of the neutron population at time t
    - α is the Rossi-alpha coefficient
    - s is the source term
    - sigma_hat is the noise amplitude
    - W is a Wiener process
    
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
        Detection rate constant
    n_0 : float
        Initial population value
    t_0 : float
        Initial time
    t_end : float
        End time
    grid_points : int
        Number of points in the time grid
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Returns (t_space, ys) where:
        - t_space is the array of time points
        - ys is the array of population values at each time point
        
    Notes
    -----
    The implementation uses the basic Euler-Maruyama discretization:
    N(t + dt) = N(t) + drift*dt + diffusion*dW
    where drift = -αN + s and diffusion = sigma_hat
    """
    
    # Input validation
    if grid_points < 1:
        raise ValueError("grid_points must be positive")
    if t_end <= t_0:
        raise ValueError("t_end must be greater than t_0")
    if any(x < 0 for x in [f, a, s, d]):
        raise ValueError("rate constants must be non-negative")
    if not isinstance(p_v, np.ndarray):
        raise TypeError("p_v must be a numpy array")
        
    # Calculate coefficients
    lam = f + a + d  # total reaction rate
    vbar = mean(p_v)  # expected value of p_v
    vbar_square = mean_square(p_v)  # second moment of p_v
    
    # Compute important parameters
    alpha = lam - f * vbar  # Rossi-alpha coefficient
    sig_tilde_squared = lam + f * (vbar_square - 2 * vbar)  # noise parameter
    sig_hat = np.sqrt(sig_tilde_squared * (s / alpha) + s)  # noise amplitude
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = s / alpha
    
    # Time discretization
    dt = (t_end - t_0) / grid_points
    t_space = np.linspace(t_0, t_end, grid_points + 1)
    
    # Initialize solution array
    ys = np.zeros(grid_points + 1)
    ys[0] = n_0
    
    # Main integration loop with vectorized operations where possible
    drift = lambda y: -alpha * y + s
    
    # Generate all random increments at once
    dW_vec = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    
    # Euler-Maruyama step
    for i in range(grid_points):
        y = ys[i]
        ys[i + 1] = y + drift(y) * dt + sig_hat * dW_vec[i]
        
        # Optional: Add non-negativity constraint if needed
        # ys[i + 1] = max(0, ys[i + 1])
    
    return t_space, ys


def euler_maruyama_system_basic(p_v: np.ndarray,
                 f: float,
                 a: float,
                 s: float,
                 d: float,
                 t_0: float,
                 t_end: float,
                 grid_points: int,
                 index : str,
                 n_0: float = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
   This method is used to numerically solve the coupled SDE system:
   dN_t = (-alpha_1 * N_t + s)dt + sig_1 dW_1 - dD_t
   dD_t = dN_t dt + sig_2 * dW_2
   which is eq. 20 from Dubi & Atar (2017)
   "Modeling neutron count distribution in a subcritical core by stochastic
    differential equations" 
   
   where:
   - N_t is the size of the neutron population at time t
   - D_t is the accumulated detections, i.e., the number of actual detections
         at the interval [0,t]
   - alpha_1 is the modified Rossi-alpha coefficient, defined by:
       alpha_1 = f + a - vbar * f
   - s is the source term
   - sig_1, sig_2 are noise amplitudes, defined by:
       sig_1^2 = s * (f + a + f * (vbar_squared - 2 * vbar)) / alpha
       sig_2^2 = s * d / alpha
   - W₁, W₂ are independent Wiener processes
   
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
       Detection rate constant
   t_0 : float
       Initial time
   t_end : float
       End time
   grid_points : int
       Number of points in the time grid
   n_0 : float, optional
       Initial population value. If None, defaults to equilibrium value s/alpha
       
   Returns
   -------
   tuple[np.ndarray, np.ndarray, np.ndarray]
       Returns (t_space, pop, detect) where:
       - t_space is the array of time points
       - pop is the array of population values at each time point
       - detect is the array of detection values at each time point
   """
    
    # Input validation
    if grid_points < 1:
        raise ValueError("grid_points must be positive")
    if t_end <= t_0:
        raise ValueError("t_end must be greater than t_0")
    if any(x < 0 for x in [f, a, s, d]):
        raise ValueError("rate constants must be non-negative")
        
    # Calculate coefficients
    lam = f + a + d  # total reaction rate
    vbar = mean(p_v)  # expected value of p_v
    vbar_square = mean_square(p_v)  # second moment of p_v
    
    # Compute important parameters
    alpha = lam - f * vbar  # Rossi-alpha coefficient
    alpha_1 = f + a - vbar * f
    sig_1_squared = s * ( f + a + f * (vbar_square - 2 * vbar)) / alpha + s
    sig_2_squared = s * d / alpha
    sig_1 = np.sqrt(sig_1_squared)
    sig_2 = np.sqrt(sig_2_squared)
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = s / alpha
    
    # Time discretization
    dt = (t_end - t_0) / grid_points
    t_space = np.linspace(t_0, t_end, grid_points + 1)
    
    # Initialize solution array
    pop = np.zeros(grid_points + 1)
    pop[0] = n_0
    detect = np.zeros(grid_points + 1)
    
    # Define drift functions
    drift_pop = lambda y: - alpha_1 * y + s
    drift_detect = lambda y: d * y
    
    # Generate all random increments at once
    dW_vec = rng.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points).tolist()
    dV_vec = rng.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points).tolist()
    
    next_percent = 10
    
    # Euler-Maruyama integration
    for i in range(grid_points):
        progress = (i + 1) * 100 / grid_points
        if progress >= next_percent:
            print(f"{next_percent:.0f}% complete")
            next_percent += 10  
        current_pop = float(pop[i])
        current_detect = float(detect[i])
        
        # Update detection first (independend of population update)
        detect_increment = drift_detect(current_pop) * dt + sig_2 * dW_vec[i]
        detect[i + 1] = current_detect + detect_increment
        
        # Update population (subtract only the increment in detection)
        pop_increment = drift_pop(current_pop) * dt + sig_1 * dV_vec[i] - detect_increment
        pop[i + 1] = current_pop + pop_increment
        
        # optional: Ensure non-negativity
        # pop[i + 1] = max(0,pop[i + 1])
        
    np.save(f'EM_Pop_Without_Dead_Time_f{index}', pop)
    np.save(f'EM_Detect_Without_Dead_Time_f{index}', detect)
    return t_space, pop, detect


def euler_maruyama_system_with_const_dead_time(p_v: np.ndarray,
                 f: float,
                 a: float,
                 s: float,
                 d: float,
                 tau: float,
                 t_0: float,
                 t_end: float,
                 grid_points: int,
                 index : str,
                 n_0: float = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
   This method is used to numerically solve the coupled SDE system:
   dN_t = (-alpha * N_t + s)dt + sig_1 dW_1 - sig_2 dW_2
   dC_t = d * N_t * exp(- d * N_t * tau) dt + sig_3 dW_3
   which is eq. 3.22 from Dubi (2022)
   "Modeling neutron count distribution in a subcritical core by stochastic
    differential equations" 
   
   where:
   - N_t is the size of the neutron population at time t
   - C_t is the number of accumulated detections with dead time,
         i.e., the number of actual detections in the interval [0,t]
   - alpha is the modified Rossi-alpha coefficient, defined by:
       alpha = -(f + a + d) + vbar * f
   - sig_1 is the noise variance associated with the power:
       sig_1^2 = s * (f + a + f * (vbar_squared - 2 * vbar)) / alpha + s
   - sig_2 is the noise variance associated with the physical detections
       sig_2^2 = s * d / alpha
   - sig_3 is the noise variance associated with the actual detections.
       sig_3^2 = exp(- 2 * tau * d * s/alpha)/(s/alpha)^2 
           * (1- 2 * d * S/alpha * tau * exp(-d * S/alpha * tau))
       
   - W_1 , W_2 , W_3 are Wiener processes, s. t. W_1, W_2 are independent,
       and the correlation between W_2 and W_3 is given by:
       dW_2 * dW_3 = (sig_2 / sig_3) * exp(-d * S/alpha * tau) dt
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
       Detection rate constant
   tau : float
       Detector dead time
   t_0 : float
       Initial time
   t_end : float
       End time
   grid_points : int
       Number of points in the time grid
   n_0 : float, optional
       Initial population value. If None, defaults to equilibrium value s/alpha
       
   Returns
   -------
   tuple[np.ndarray, np.ndarray, np.ndarray]
       Returns (t_space, pop, detect) where:
       - t_space is the array of time points
       - pop is the array of population values at each time point
       - detect is the array of detection values at each time point
   """
    
    # Input validation
    if grid_points < 1:
        raise ValueError("grid_points must be positive")
    if t_end <= t_0:
        raise ValueError("t_end must be greater than t_0")
    if any(x < 0 for x in [f, a, s, d]):
        raise ValueError("rate constants must be non-negative")
        
    # Calculate coefficients
    lam = f + a + d  # total reaction rate
    vbar = mean(p_v)  # expected value of p_v
    vbar_square = mean_square(p_v)  # second moment of p_v
    
    # Compute important parameters
    alpha = lam - f * vbar  # Rossi-alpha coefficient
    equil = s / alpha
    dead_time_parameter = d * equil * tau
    
    sig_1_squared = equil * (f + a + f * (vbar_square - 2 * vbar)) + s
    sig_2_squared = equil * d
    sig_3_squared = (np.exp(-2 * dead_time_parameter)
       * (1 - 2 * dead_time_parameter * np.exp(-dead_time_parameter))
       / (equil ** 2))
    sig_1 = np.sqrt(sig_1_squared)
    sig_2 = np.sqrt(sig_2_squared)
    sig_3 = np.sqrt(sig_3_squared)
    #correlation = (sig_2 / sig_3) * np.exp(-dead_time_parameter)
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
    # Time discretization
    dt = (t_end - t_0) / grid_points
    t_space = np.linspace(t_0, t_end, grid_points + 1)
    
    # Initialize solution array
    pop = np.zeros(grid_points + 1)
    pop[0] = n_0
    detect = np.zeros(grid_points + 1)
    
    # Define drift functions
    drift_pop = lambda y: - alpha * y + s
    drift_detect = lambda y:  d * y * np.exp(- d * y * tau)
    
    # Generate all random increments at once
    dW1_vec = rng.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    dW2_vec = rng.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    dW3_vec = rng.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    
    # Create dW3_vec so that W_2 and W_3 are correlated by:
    # dW_2 dW_3 = correlation * dt
    # dW3_vec = correlation * dW2_vec + np.sqrt(1 - correlation ** 2) * dW3_indep
    
    next_percent = 10
    
    # Euler-Maruyama integration
    for i in range(grid_points):
        progress = (i + 1) * 100 / grid_points
        if progress >= next_percent:
            print(f"{next_percent:.0f}% complete")
            next_percent += 10  
        current_pop = float(pop[i])
        current_detect = float(detect[i])
        
        # Update detection first (independend of population update)
        detect_increment = (drift_detect(current_pop) * dt 
                            + sig_3 * dW3_vec[i])
        detect[i + 1] = current_detect + detect_increment
        
        # Update population (subtract only the increment in detection)
        pop_increment = (drift_pop(current_pop) * dt 
                         + sig_1 * dW1_vec[i] - sig_2 * dW2_vec[i])
        pop[i + 1] = current_pop + pop_increment
        
        # optional: Ensure non-negativity
        # pop[i + 1] = max(0,pop[i + 1])
        
    np.save(f'EM_Pop_Const_Dead_Time_f{index}', pop)
    np.save(f'EM_Detect_Const_Dead_Time_f{index}', detect)
    return t_space, pop, detect


def euler_maruyama_system_with_exp_dead_time(p_v: np.ndarray,
                 f: float,
                 a: float,
                 s: float,
                 d: float,
                 tau_mean: float,
                 t_0: float,
                 t_end: float,
                 grid_points: int,
                 index : str,
                 n_0: float = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
   This method is used to numerically solve the coupled SDE system:
   dN_t = (-alpha * N_t + s)dt + sig_1 dW_1 - sig_2 dW_2
   dC_t = d * N_t * (1- Ψ(d * N_t)) dt + sig_3 dW_3
   which is eq. 3.18 from Dubi (2022)
   "Modeling neutron count distribution in a subcritical core by stochastic
    differential equations" 
   
   where:
   - N_t is the size of the neutron population at time t
   - C_t is the number of accumulated detections with dead time,
         i.e., the number of actual detections in the interval [0,t]
   - Ψ is the avergace probability for a detection to occur within a dead time
       for the previous detection.
   For our context, the value of Ψ for exponential dead time is given by:
       Ψ(x) = 1 - 1/(1 - d * N_t * tau_mean)
   - alpha is the modified Rossi-alpha coefficient, defined by:
       alpha = -(f + a + d) + vbar * f
   - sig_1 is the noise variance associated with the power:
       sig_1^2 = s * (f + a + f * (vbar_squared - 2 * vbar)) / alpha + s
   - sig_2 is the noise variance associated with the physical detections
       sig_2^2 = s * d / alpha
   - sig_3 is the noise variance associated with the actual detections.    
       sig_3^2 = term_1 + term_2 + term_3, where:
       term_1 = 1 / d * N_t
       term_2 = complex expression involving tau_mean and d * N_t
       term_3 = another complex expression
   - W_1 , W_2 , W_3 are Wiener processes, s. t. W_1, W_2 are independent,
       W_3 can either be independent, or be correlated with W_3, with correlation:
       dW_2 * dW_3 = (sig_2 / sig_3) * exp(-d * S/alpha * tau) dt
       
   TODO: This function is reserved for later used, and further work on it is required
       to check its validity.
       
   FIXME: The function currently suffers from caluclation issues, mainly with 
       detection diffusion. It needs to be determined if the expression for
       the diffusion is correct.
       
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
       Detection rate constant
   tau : float
       Detector dead time
   t_0 : float
       Initial time
   t_end : float
       End time
   grid_points : int
       Number of points in the time grid
   n_0 : float, optional
       Initial population value. If None, defaults to equilibrium value s/alpha
       
   Returns
   -------
   tuple[np.ndarray, np.ndarray, np.ndarray]
       Returns (t_space, pop, detect) where:
       - t_space is the array of time points
       - pop is the array of population values at each time point
       - detect is the array of detection values at each time point
   """
    
    # Input validation
    if grid_points < 1:
        raise ValueError("grid_points must be positive")
    if t_end <= t_0:
        raise ValueError("t_end must be greater than t_0")
    if any(x < 0 for x in [f, a, s, d]):
        raise ValueError("rate constants must be non-negative")
    if tau_mean <= 0:
        raise ValueError("tau_mean must be positive")
        
    # Calculate coefficients
    lam = f + a + d  # total reaction rate
    vbar = mean(p_v)  # expected value of p_v
    vbar_square = mean_square(p_v)  # second moment of p_v
    
    # Compute important parameters
    alpha = lam - f * vbar  # Rossi-alpha coefficient
    equil = s / alpha
    psi = lambda y: 1 - 1/(1 - y * tau_mean)
    
    sig_1_squared = equil * (f + a + f * (vbar_square - 2 * vbar)) + s
    sig_2_squared = equil * d
    sig_1 = np.sqrt(sig_1_squared)
    sig_2 = np.sqrt(sig_2_squared)
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
    # Time discretization
    dt = (t_end - t_0) / grid_points
    t_space = np.linspace(t_0, t_end, grid_points + 1)
    
    # Initialize solution array
    pop = np.zeros(grid_points + 1)
    pop[0] = n_0
    detect = np.zeros(grid_points + 1)
    
    # Define drift functions
    drift_pop = lambda y: - alpha * y + s
    drift_detect = lambda y:  d * y * ( 1 - psi(d * y))
    
    # Define diffusion functions:
    def _compute_diffusion_terms(detection_rate , tau_mean):
        """ Compute the three terms for the diffusion coefficient."""
        
        term_1 = 1 / detection_rate
        
        numerator_2 = 2 * tau_mean ** 4 * (1 - detection_rate * tau_mean) * \
            (4 * detection_rate * tau_mean + 3)
        denominator_2 = detection_rate *((detection_rate * tau_mean + 1) ** 2) * \
            (2 * detection_rate * tau_mean + 1) ** 3
        term_2 = numerator_2 / denominator_2
        
        numerator_3 = ( 1 - detection_rate * tau_mean) * \
            (2 * detection_rate * (5 * detection_rate * tau_mean + 3) + 1)
        denominator_3 = (detection_rate ** 5) * \
            (2 * detection_rate * tau_mean + 1) ** 3
        term_3 = numerator_3 / denominator_3
        
        return term_1 , term_2, term_3
    
    def diffusion_detect(y):
        """ Compute the diffusion coefficient for detection """
        detection_rate = d * y
        term_1 , term_2, term_3 = _compute_diffusion_terms(detection_rate, tau_mean)
        return np.sqrt(term_1 + term_2 + term_3)
    
    # Generate all random increments at once
    dW1_vec = rng.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    dW2_vec = rng.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    dW3_vec = rng.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    
    # Create dW3_vec so that W_2 and W_3 are correlated by:
    # dW_2 dW_3 = correlation * dt
    # dW3_vec = correlation * dW2_vec + np.sqrt(1 - correlation ** 2) * dW3_indep
    
    next_percent = 10
    
    # Euler-Maruyama integration
    for i in range(grid_points):
        progress = (i + 1) * 100 / grid_points
        if progress >= next_percent:
            print(f"{next_percent:.0f}% complete")
            next_percent += 10  
        current_pop = float(pop[i])
        current_detect = float(detect[i])
        
        # Update detection first (independend of population update)
        detect_increment = (drift_detect(current_pop) * dt 
                            + diffusion_detect(current_pop) * dW3_vec[i])
        detect[i + 1] = current_detect + detect_increment
        
        # Update population (subtract only the increment in detection)
        pop_increment = (drift_pop(current_pop) * dt 
                         + sig_1 * dW1_vec[i] - sig_2 * dW2_vec[i])
        pop[i + 1] = current_pop + pop_increment
        
        # optional: Ensure non-negativity
        # pop[i + 1] = max(0,pop[i + 1])
        
    np.save(f'EM_Pop_Exp_Dead_Time_f{index}', pop)
    np.save(f'EM_Detect_Exp_Dead_Time_f{index}', detect)
    return t_space, pop, detect


# =============================================================================
# TAYLOR METHOD SIMULATION FUNCTIONS
# =============================================================================

def strong_taylor(p_v: np.ndarray,
                f: float,
                a: float,
                s: float,
                d: float,
                n_0: float,
                t_0: float,
                t_end: float,
                grid_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the Strong Taylor method (order 1.5) 
    for solving stochastic differential equations (SDEs)
    in the context of population dynamics.
    
    This method provides stronger convergence than Euler-Maruyama 
    by including additional terms in the Taylor expansion. 
    The SDE being solved is:
    dN_t = (-αN_t + s)dt + sigma_hat * dW (Dubi & Atar 2017, eq. 14)
    
    The strong Taylor scheme includes an additional drift correction term:
    N(t + dt) = N(t) + drift*dt + diffusion*dW - 0.5*drift*alpha*dt^2
    
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
        Detection rate constant
    n_0 : float
        Initial population value
    t_0 : float
        Initial time
    t_end : float
        End time
    grid_points : int
        Number of points in the time grid
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Returns (t_space, ys) where:
        - t_space is the array of time points
        - ys is the array of population values at each time point
        
    Notes
    -----
    The strong Taylor method has better convergence properties than Euler-Maruyama
    for paths of the solution, not just for moments or distributions.
    The additional drift correction term improves accuracy for systems with
    significant drift components.
    """
    # Input validation
    if grid_points < 1:
        raise ValueError("grid_points must be positive")
    if t_end <= t_0:
        raise ValueError("t_end must be greater than t_0")
    if any(x < 0 for x in [f, a, s, d]):
        raise ValueError("rate constants must be non-negative")
    if not isinstance(p_v, np.ndarray):
        raise TypeError("p_v must be a numpy array")
        
    # Calculate coefficients
    lam = f + a + d  # total reaction rate
    vbar = mean(p_v)  # expected value of p_v
    vbar_square = mean_square(p_v)  # second moment of p_v
    
    # Compute important parameters
    alpha = lam - f * vbar  # Rossi-alpha coefficient
    sig_tilde = lam + f * (vbar_square - 2 * vbar)  # noise parameter
    sig_hat = np.sqrt(sig_tilde * (s / alpha) + s)  # noise amplitude
    
    # Time discretization
    dt = (t_end - t_0) / grid_points
    t_space = np.linspace(t_0, t_end, grid_points + 1)
    
    # Initialize solution array
    ys = np.zeros(grid_points + 1)
    ys[0] = n_0
    
    # Define drift and its derivative
    drift = lambda y: -alpha * y + s
    drift_derivative = lambda _: -alpha  # derivative of drift with respect to y
    diffusion = sig_hat
    
    # Generate all random increments at once
    dW_vec = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    
    # Strong Taylor integration loop
    for i in range(grid_points):
        y = ys[i]
        current_drift = drift(y)
        
        # Strong Taylor step with drift correction
        ys[i + 1] = (y + 
                     current_drift * dt + 
                     diffusion * dW_vec[i] - 
                     0.5 * current_drift * drift_derivative(y) * dt**2)
        
        # Optional: Add non-negativity constraint if needed
        # ys[i + 1] = max(0, ys[i + 1])
    
    return t_space, ys


def weak_taylor(p_v: np.ndarray,
               f: float,
               a: float,
               s: float,
               d: float,
               t_0: float,
               t_end: float,
               grid_points: int,
               n_0: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the Weak Taylor method (order 2.0) 
    for solving stochastic differential equations (SDEs)
    in the context of population dynamics.
    
    This method provides better convergence for the statistical moments of the solution
    compared to both Euler-Maruyama and Strong Taylor methods. The SDE being solved is:
    dN_t = (-αN_t + s)dt + sigma_hat * dW (Dubi & Atar 2017, eq. 14)
    
    The weak Taylor scheme includes both drift correction and noise-induced drift terms:
    N(t + dt) = N(t) + drift*dt + diffusion*dW - 0.5*drift*alpha*dt^2 - alpha*diffusion*dZ
    where dZ is a mixed Wiener increment satisfying E[dZ] = 0 and E[dZ^2] = dt^3/12
    
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
        Detection rate constant
    n_0 : float
        Initial population value
    t_0 : float
        Initial time
    t_end : float
        End time
    grid_points : int
        Number of points in the time grid
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Returns (t_space, ys) where:
        - t_space is the array of time points
        - ys is the array of population values at each time point
        
    Notes
    -----
    The weak Taylor method is particularly useful when interested in statistical properties
    of the solution (like moments, distributions) rather than individual path accuracy.
    The additional noise-induced drift term accounts for the interaction between
    the drift and diffusion components of the system.
    """
    # Input validation
    if grid_points < 1:
        raise ValueError("grid_points must be positive")
    if t_end <= t_0:
        raise ValueError("t_end must be greater than t_0")
    if any(x < 0 for x in [f, a, s, d]):
        raise ValueError("rate constants must be non-negative")
    if not isinstance(p_v, np.ndarray):
        raise TypeError("p_v must be a numpy array")
        
    # Calculate coefficients
    lam = f + a + d  # total reaction rate
    vbar = mean(p_v)  # expected value of p_v
    vbar_square = mean_square(p_v)  # second moment of p_v
    
    # Compute important parameters
    alpha = lam - f * vbar  # Rossi-alpha coefficient
    
    # noise parameter squared
    sig_tilde_squared = lam + f * (vbar_square - 2 * vbar)  
    sig_hat = np.sqrt(sig_tilde_squared * (s / alpha) + s)  # noise amplitude
    
    # Time discretization
    dt = (t_end - t_0) / grid_points
    t_space = np.linspace(t_0, t_end, grid_points + 1)
    
    # Initialize solution array
    ys = np.zeros(grid_points + 1)
    ys[0] = s / alpha
    
    # Define drift and its derivative
    drift = lambda y: -alpha * y + s
    drift_derivative = lambda _: -alpha  # derivative of drift with respect to y
    diffusion = sig_hat
    
    # Generate all random increments at once
    dW_vec = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    # Generate mixed Wiener increments for the noise-induced drift term
    dZ_vec = 0.5 * dt * (dW_vec + np.random.normal(loc=0.0, scale=np.sqrt(dt/3),
                                                   size=grid_points))
    
    # Weak Taylor integration loop
    for i in range(grid_points):
        y = ys[i]
        current_drift = drift(y)
        
        # Weak Taylor step with both drift correction and noise-induced drift
        ys[i + 1] = (y + 
                     current_drift * dt + 
                     diffusion * dW_vec[i] - 
                     0.5 * current_drift * drift_derivative(y) * dt**2 -
                     alpha * diffusion * dZ_vec[i])
        
        # Optional: Add non-negativity constraint if needed
        # ys[i + 1] = max(0, ys[i + 1])
    
    return t_space, ys


def runge_kutta(p_v : np.ndarray,
                f : float,
                a : float,
                s : float,
                d : float,
                n_0 : float,
                t_0 : float,
                t_end : float,
                grid_points : int) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO: implement Runge-Kutta method for SDE:
        dN_t = (-αN_t + s)dt + sigma_hat * dW
    
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
        Detection rate constant
    n_0 : float
        Initial population value
    t_0 : float
        Initial time
    t_end : float
        End time
    grid_points : int
        Number of points in the time grid
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Returns (t_space, ys) where:
        - t_space is the array of time points
        - ys is the array of population values at each time point
    """
    # Calculate coefficients (same as other methods):
    lam = f + a + d
    vbar = mean(p_v)
    vbar_square = mean_square(p_v)
    alpha = lam - f * vbar
    sig_tilde = lam + f * (vbar_square - 2 * vbar)
    sig_hat = np.sqrt(sig_tilde * (s / alpha) + s)
    
    # Time discretization
    dt = (t_end - t_0) / grid_points
    t_space = np.linspace(t_0 , t_end, grid_points + 1)
    
    # Initialize solution array
    ys = np.zeros(grid_points + 1)
    ys[0] = n_0
    
    # TODO: Implement Runge-Kutta steps
    # y = ys[i - 1]
    # u = y - alpha * ys[i - 1] + s
    # ys[i] = u * dt + sig_hat * dW(dt)
    #         - 0.5 * u * alpha * (dt ** 2)
    #         - alpha * sig_hat * dZ(dt))
    
    raise NotImplementedError("Runge-Kutta method not yet implemented")
    
    return t_space, ys


def euler_maruyama_matrix(p_v, f, a, s, d, n_0, t_0, t_end, grid_points, paths):
    """
    TODO: FUTURE IMPROVEMENTS NEEDED
    
    This function needs significant improvements:
    
    IMPROVEMENTS REQUIRED:
    ====================
    1. Implement vectorized operations for better performance
    2. Add progress tracking and memory management
    3. Implement parallel processing for multiple paths
    4. Add memory-efficient batch processing

    
    CURRENT ISSUES:
    ===============
    - No input validation
    - Inefficient memory usage (growing arrays)
    - No progress tracking
    - Sequential processing (could be parallelized)
    """
    t_init, n_init = euler_maruyama(p_v, f, a, s, d, n_0, t_0, t_end, grid_points - 1)
    n_mat = np.array([n_init])
    t_mat = np.array([t_init])
    for i in range(1, paths):
        ts, ys = euler_maruyama(p_v, f, a, s, d, n_0, t_0, t_end, grid_points - 1)
        n_mat = np.append(n_mat, [ys], axis=0)
        t_mat = np.append(t_mat, [ts], axis = 0)
        print(i)
    np.save('EM_Pop_matrix999', n_mat)
    np.save('EM_Time_Matrix999', t_mat)
    return t_mat, n_mat


def strong_taylor_matrix(p_v, f, a, s, d, n_0, t_0, t_end, grid_points, paths):
    """
    TODO: FUTURE IMPROVEMENTS NEEDED
    
    This function needs significant improvements:
    
    IMPROVEMENTS REQUIRED:
    ====================
    1. Implement vectorized operations for better performance
    2. Add progress tracking and memory management
    3. Implement parallel processing for multiple paths
    4. Add memory-efficient batch processing

    
    CURRENT ISSUES:
    ===============
    - No input validation
    - Inefficient memory usage (growing arrays)
    - No progress tracking
    - Sequential processing (could be parallelized)
    """
    t_init, n_init = strong_taylor(p_v, f, a, s, d, n_0, t_0, t_end, grid_points - 1)
    n_mat = np.array([n_init])
    t_mat = np.array([t_init])
    for i in range(1, paths):
        print(i)
        ts, ys = strong_taylor(p_v, f, a, s, d, n_0, t_0, t_end, grid_points - 1)
        n_mat = np.append(n_mat, [ys], axis=0)
        t_mat = np.append(t_mat, [ts], axis = 0)
    np.save('strong_taylor_matrix', n_mat)
    return t_mat, n_mat


def weak_taylor_matrix(p_v, f, a, s, d, n_0, t_0, t_end, grid_points, paths):
    """
    TODO: FUTURE IMPROVEMENTS NEEDED
    
    This function needs significant improvements:
    
    IMPROVEMENTS REQUIRED:
    ====================
    1. Implement vectorized operations for better performance
    2. Add progress tracking and memory management
    3. Implement parallel processing for multiple paths
    4. Add memory-efficient batch processing

    
    CURRENT ISSUES:
    ===============
    - No input validation
    - Inefficient memory usage (growing arrays)
    - No progress tracking
    - Sequential processing (could be parallelized)
    """
    t_init, n_init = weak_taylor(p_v, f, a, s, d, n_0, t_0, t_end, grid_points - 1)
    n_mat = np.array([n_init])
    t_mat = np.array([t_init])
    for i in range(1, paths):
        print(i)
        ts, ys = weak_taylor(p_v, f, a, s, d, n_0, t_0, t_end, grid_points - 1)
        n_mat = np.append(n_mat, [ys], axis=0)
        t_mat = np.append(t_mat, [ts], axis = 0)
    np.save('weak_taylor_matrix', n_mat)
    return t_mat, n_mat



# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_simulation_data(fission_vec):
    """
    Load simulation data with error handling.
    """
    simul_time = {}
    simul_detect = {}
    #simul_pop = {}
    
    for f in fission_vec:
        print(f"Loading data for fission = {f}")
        simul_time[f] = np.load(f"Simul_Time_Matrix_hunmil_f{f}.npy")
        simul_detect[f] = np.load(f"Detection_Matrix_hunmil_f{f}.npy")
        #simul_pop[f] = np.load(f"Simul_Pop_Matrix_f{f}.npy")
    simul_time_vec = list(simul_time.values())
    simul_detect_vec = list(simul_detect.values())
    #simul_pop_vec = list(simul_pop.values())
    return simul_time_vec, simul_detect_vec


def load_euler_maruyama_basic(fission_vec):
    em_detect = {}
    em_pop = {}
    for f in fission_vec:
        print(f"Loading data for fission = {f}")
        em_detect[f] = np.load(f'EM_Detect_Without_Dead_Time_f{f}.npy')
        em_pop[f] = np.load(f'EM_Pop_Without_Dead_Time_f{f}.npy')
    em_detect_vec = list(em_detect.values())
    em_pop_vec = list(em_pop.values())
    return em_detect_vec, em_pop_vec


def load_euler_maruyama_with_const_dead_time_data(fission_vec):
    em_detect_const = {}
    em_pop_const = {}
    for f in fission_vec:
        print(f"Loading data for fission = {f}")
        em_detect_const[f] = np.load(f'EM_Detect_Const_Dead_Time_f{f}.npy')
        em_pop_const[f] = np.load(f'EM_Pop_Const_Dead_Time_f{f}.npy')
    em_detect_vec = list(em_detect_const.values())
    em_pop_vec = list(em_pop_const.values())
    return em_detect_vec, em_pop_vec


def load_euler_maruyama_with_exp_dead_time_data(fission_vec):
    em_detect_exp = {}
    em_pop_exp = {}
    for f in fission_vec:
        print(f"Loading data for fission = {f}")
        em_detect_exp[f] = np.load(f'EM_Detect_Exp_Dead_Time_f{f}.npy')
        em_pop_exp[f] = np.load(f'EM_Pop_Exp_Dead_Time_f{f}.npy')
    em_detect_vec = list(em_detect_exp.values())
    em_pop_vec = list(em_pop_exp.values())
    return em_detect_vec, em_pop_vec


# =============================================================================
# SIMULATION EXECUTION FUNCTIONS
# =============================================================================

def run_stochastic_simulations(fission_vec, equil, p_v, absorb, source, detect, t_0, steps, prefix='f'):
    """
    Run stochastic simulations for all fission values and save results.
    """
    print("=" * 60)
    print("RUNNING STOCHASTIC SIMULATIONS")
    print("=" * 60)
    
    for i, fission in enumerate(fission_vec):
        print(f"iteration {i + 1}/{len(fission_vec)}, fission = {fission}")
        n_0 = equil[i] * np.ones(1)
        pop_dyn_mat(p_v, fission, absorb, source, detect, n_0, t_0, steps, 
                   prefix + f"{fission}")
    
    print("Stochastic simulations complete!")


def run_euler_maruyama_simulations(fission_vec, simul_time_vec, p_v, absorb, source, 
                                  detect, t_0, grid_points, mean_tau, prefix='',
                                  run_without_dead_time = True, run_with_const_dead_time = True,
                                  run_with_exp_dead_time = True):
    """
    Run Euler-Maruyama simulations for all fission values and save results.
    """
    print("=" * 60)
    print("RUNNING EULER-MARUYAMA SIMULATIONS")
    print("=" * 60)
    
    for i, fission in enumerate(fission_vec):
        print(f"iteration {i + 1}/{len(fission_vec)}, fission = {fission}")
        t_end = simul_time_vec[i][:,-1].item()
        
        # Without dead time
        if run_without_dead_time:
            euler_maruyama_system_basic(
                p_v, fission, absorb, source, detect, t_0, t_end, 
                grid_points, f"{fission}"
                )
        
        # With constant dead time
        if run_with_const_dead_time:
            euler_maruyama_system_with_const_dead_time(
                p_v, fission, absorb, source, detect, mean_tau, t_0, t_end, 
                grid_points, f"{fission}"
                )
        
        # with exponential dead time
        if run_with_exp_dead_time:
            euler_maruyama_system_with_exp_dead_time(
                p_v, fission, absorb, source, detect, mean_tau, t_0, t_end, 
                grid_points, f"{fission}"
                )
            
    print("Euler-Maruyama simulations complete!")


def execute_simulations(run_stochastic=False, run_em=False, 
                        em_without_dead_time = True, em_with_const_dead_time = True,
                        em_with_exp_dead_time = True):
    """
    Execute simulations based on data
    """
    
    if run_stochastic:
        print("Running stochastic simulations...")
        run_stochastic_simulations(fission_vec, equil, p_v, absorb, source, detect, t_0, steps)
    
    # Always load data (either existing or newly created)
    simul_time_vec, simul_detect_vec = load_simulation_data(fission_vec)
    em_detect_vec, em_pop_vec = (
        load_euler_maruyama_basic(fission_vec))
    em_const_detect_vec, em_const_pop_vec = (
        load_euler_maruyama_with_const_dead_time_data(fission_vec))
    em_exp_detect_vec, em_exp_pop_vec = (
        load_euler_maruyama_with_exp_dead_time_data(fission_vec))
    
    if run_em:
        print("Running Euler-Maruyama simulations...")
        run_euler_maruyama_simulations(fission_vec, simul_time_vec, p_v, absorb, source, 
                                     detect, t_0, grid_points, mean_tau,
                                     run_without_dead_time = em_without_dead_time,
                                     run_with_dead_time = em_with_const_dead_time,
                                     run_with_exp_dead_time = em_with_exp_dead_time)
    
    return (simul_time_vec, simul_detect_vec, 
            em_detect_vec, em_pop_vec,
            em_const_detect_vec, em_const_pop_vec)

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_stochastic_basic(alpha_inv_vec, cps):
    """
    Plot stochastic simulation count rates without dead time
    """
    plt.plot(alpha_inv_vec, cps, label = "Stochastic simulation")
    plt.xticks(alpha_inv_vec, rotation = 45, fontsize = 10)
    plt.title("Stochastic Count Rates without Dead Time")
    plt.xlabel("1/alpha")
    plt.ylabel("Count per Second (CPS)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf() 
 
   
def plot_stochastic_const_dead_time(alpha_inv_vec, cps_tau, tau):
    """
    Plot stochastic simulation count rates with constant dead time tau
    """
    plt.plot(alpha_inv_vec, cps_tau, label =
             "Stochastic simulation with constant dead time")
    plt.xticks(alpha_inv_vec, rotation = 45, fontsize = 10)
    plt.title(f"Stochastic Count Rates with Constant Dead Time tau = {tau:.1e}")
    plt.xlabel("1/alpha")
    plt.ylabel("Count per Second (CPS)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf() 


def plot_euler_maruyama_basic(alpha_inv_vec, cps_em):
    """
    Plot Euler-Maruyama count rates without dead time
    """
    plt.plot(alpha_inv_vec, cps_em, 'o', label =
             "Euler-Maruyama without Dead Time")
    plt.xticks(alpha_inv_vec, rotation = 45, fontsize = 10)
    plt.title("Euler-Maruyama Count Rates without Dead Time")
    plt.xlabel("1/alpha")
    plt.ylabel("Count per Second (CPS)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf() 


def plot_euler_maruyama_const_dead_time(alpha_inv_vec, cps_em_tau, tau):
    """
    Plot Euler-Maruyama count rates with constant dead time
    """
    plt.plot(alpha_inv_vec, cps_em_tau, 's-', 
             label = "Euler-Maruyama with constant dead time")
    plt.xticks(alpha_inv_vec, rotation = 45, fontsize = 10)
    plt.title(f"Euler-Maruyama Count Rates with Constant Dead Time tau = {tau}")
    plt.xlabel("1/alpha")
    plt.ylabel("Count per Second (CPS)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf() 

def plot_euler_maruyama_exp_dead_time(alpha_inv_vec, cps_em_tau, tau):
    """
    Plot Euler-Maruyama count rates with constant dead time
    """
    plt.plot(alpha_inv_vec, cps_em_tau, 's-', 
             label = "Euler-Maruyama with Exponential dead time")
    plt.xticks(alpha_inv_vec, rotation = 45, fontsize = 10)
    plt.title(f"Euler-Maruyama Count Rates with Exponential Dead Time tau = {tau}")
    plt.xlabel("1/alpha")
    plt.ylabel("Count per Second (CPS)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf() 
   

#implementation of code

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Physical parameters
p_v = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
absorb = 7
source = 1000
detect = 10

# Time parameters
t_0 = 0
steps = 100_000_000
grid_points = 100_000_0

# Dead time parameters
mean_tau = 1 * (10 ** -6)
dist = 'uniform'

# =============================================================================
# DERIVED PARAMETERS
# =============================================================================

vbar = mean(p_v)  # expected value of p_v
fission_vec = 33.9 + np.array([.04, .05, .06, .07, .08,
                       .082, .084, .086, .088, .09, .092]) # sample points

lam_vec = fission_vec + absorb + detect
alpha_vec = (lam_vec - fission_vec * vbar)
alpha_inv_vec = 1 /(alpha_vec)
equil = source * alpha_inv_vec

fission_to_alpha_inv = dict(zip(fission_vec.tolist(), alpha_inv_vec.tolist()))
fission_vec = np.round(fission_vec , decimals = 4)

# =============================================================================
# SIMULATION EXECUTION
# =============================================================================

# Execute simulations (load existing data by default)

simul_time_vec, simul_detect_vec = load_simulation_data(fission_vec)

#run_euler_maruyama_simulations(
#    fission_vec, simul_time_vec, p_v, absorb, source, detect, t_0, grid_points, mean_tau,
#    run_without_dead_time = False, run_with_const_dead_time = False, 
#    run_with_exp_dead_time = True)

#em_detect_vec, em_pop_vec = load_euler_maruyama_basic(fission_vec)

em_const_detect_vec, em_const_pop_vec = (
    load_euler_maruyama_with_const_dead_time_data(fission_vec))

em_exp_detect_vec, em_exp_pop_vec = (
   load_euler_maruyama_with_exp_dead_time_data(fission_vec))

# Calculate all count rates
count_rates = calculate_all_count_rates(simul_time_vec, simul_detect_vec, mean_tau = mean_tau,
                                        em_const_detect_vec = em_const_detect_vec)

plot_stochastic_const_dead_time(alpha_inv_vec, count_rates['stochastic_const_tau'], mean_tau)
plot_euler_maruyama_const_dead_time(alpha_inv_vec, count_rates['em_const_tau'], mean_tau);
plt.show()


