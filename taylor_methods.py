""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Taylor method implementations for solving stochastic differential equations.

This module implements Strong Taylor 1.5 and Weak Taylor 2.0 methods 
for solving SDEs in the context of nuclear reactor population dynamics.
"""

import numpy as np
import utils as utl
from data_management import DataManager

rng = np.random.default_rng()

data_manager = DataManager()

def strong_taylor(p_v: np.ndarray,
                f: float,
                a: float,
                s: float,
                d: float,
                t_0: float,
                t_end: float,
                grid_points: int,
                n_0: float = None) -> tuple[np.ndarray, np.ndarray]:
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
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    lam = params['lam']
    vbar = params['vbar']
    vbar_square = params['vbar_square']
    alpha = params['alpha']
    equil = params['equilibrium']
    
    sig_tilde = lam + f * (vbar_square - 2 * vbar)  # noise parameter
    sig_hat = np.sqrt(sig_tilde * (s / alpha) + s)  # noise amplitude
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
    t_space, pop, detect, dt = utl.initialize_simulation_arrays(grid_points,
                                                             n_0,
                                                             t_0,
                                                             t_end)
    
    # Define drift and its derivative
    drift = lambda y: -alpha * y + s
    drift_derivative = lambda _: -alpha  # derivative of drift with respect to y
    
    # Generate all random increments at once
    dW_vec = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    
    # Strong Taylor integration loop
    for i in range(grid_points):
        current_pop = pop[i]
        current_drift = drift(current_pop)
        
        # Strong Taylor step with drift correction
        pop[i + 1] = (
            current_pop + 
            current_drift * dt + 
            sig_hat * dW_vec[i] - 
            0.5 * current_drift * drift_derivative(current_pop) * dt**2
            )
    
    return t_space, pop


def strong_taylor_system_with_const_dead_time(p_v: np.ndarray,
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
    using 1.5 Strong Taylor method.
   
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
        
    # Calculate system parameters using helper function    
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    alpha = params['alpha']
    equil = params['equilibrium']
    sig_1 = params['sig_1']
    sig_2 = params['sig_2']
    
    dead_time_parameter = d * equil * tau
    
    sig_3_squared = (np.exp(-2 * dead_time_parameter)
       * (1 - 2 * dead_time_parameter * np.exp(-dead_time_parameter))
       / (equil ** 2))
    
    sig_3 = np.sqrt(sig_3_squared)
    
    # Use when we want to correlate between the Wiener processes
    # correlation = (sig_2 / sig_3) * np.exp(-dead_time_parameter)
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
    # Initialize arrays using helper function
    t_space, pop, detect, dt = utl.initialize_simulation_arrays(grid_points,
                                                             n_0,
                                                             t_0,
                                                             t_end)
    # Define drift functions and their derivatives
    drift_pop = lambda y: - alpha * y + s
    derivative_drift_pop = lambda y: - alpha
    
    drift_detect = lambda y:  d * y * np.exp(-d * y * tau)
    #derivative_drift_detect = lambda y : (d * np.exp(-d * y * tau) * (1 - d * y * tau))
    
    # Generate all random increments at once
    dW1_vec, dW2_vec, dW3_vec = utl.generate_random_increments(grid_points, dt, 3)
    
    dZ1_vec, dZ2_vec, dZ3_vec = utl.generate_iterated_integral(grid_points, dt,
                                                      (dW1_vec, dW2_vec, dW3_vec))
    
    # Create dW3_vec so that W_2 and W_3 are correlated by:
    # dW_2 dW_3 = correlation * dt
    # dW3_vec = correlation * dW2_vec + np.sqrt(1 - correlation ** 2) * dW3_indep
    
    # Strong Taylor 1.5 integration
    for i in utl.progress_tracker(grid_points):
        current_pop = float(pop[i])
        current_detect = float(detect[i])
        
        pop[i + 1] = (
            current_pop + drift_pop(current_pop) * dt
            + 0.5 * derivative_drift_pop(current_pop) 
            * drift_pop(current_pop) * (dt ** 2)
            + sig_1 * dW1_vec[i] - sig_2 * dW2_vec[i]
            + derivative_drift_pop(current_pop) 
            * (sig_1 * dZ1_vec[i] - sig_2 * dZ2_vec[i])
            )
    
        detect[i + 1] = (
            current_detect + (drift_detect(current_pop) * dt 
                                + sig_3 * dW3_vec[i]))
        
        # optional: Ensure non-negativity
        # pop[i + 1] = max(0,pop[i + 1])
    
    # Extract prefix from index (e.g., 'mil_f33.94' -> 'mil_f')
    _save_taylor_results(pop, detect, f, index, 'strong_system', 'const')
    return t_space, pop, detect


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
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    lam = params['lam']
    vbar = params['vbar']
    vbar_square = params['vbar_square']
    alpha = params['alpha']
    equil = params['equilibrium']
    
    # noise parameter squared
    sig_tilde_squared = lam + f * (vbar_square - 2 * vbar)  
    sig_hat = np.sqrt(sig_tilde_squared * (s / alpha) + s)  # noise amplitude
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
    # Initialize arrays using helper function
    t_space, pop, detect, dt = utl.initialize_simulation_arrays(grid_points,
                                                             n_0,
                                                             t_0,
                                                             t_end)
    
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
        current_pop = pop[i]
        current_drift = drift(current_pop)
        
        # Weak Taylor step with both drift correction and noise-induced drift
        pop[i + 1] = (
            current_pop + 
            current_drift * dt + 
            diffusion * dW_vec[i] - 
            0.5 * current_drift * drift_derivative(current_pop) * dt**2 -
            alpha * diffusion * dZ_vec[i]
            )
    
    return t_space, pop

        
def strong_taylor_matrix(p_v : np.ndarray, 
                         f : float,
                         a : float,
                         s : float,
                         d : float,
                         t_0 : float,
                         t_end : float,
                         grid_points : int,
                         n_0 : np.ndarray):
    """
    Generate Strong Taylor matrix for multiple initial populations.
    
    Parameters
    ----------
    p_v : np.ndarray
        Probability distribution for particle generation
    f : float
        Fission rate constant
    a : float
        Absorption rate constant
    s : float
        Source rate constant
    d : float
        Detection rate constant
    t_0 : float
        Initial time
    t_end : float
        End time
    grid_points : int
        Number of time grid points
    n_0 : np.ndarray
        Array of initial populations (length determines number of paths)
       
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time matrix and population matrix
    """
    
    # Paths determined by number of populations given
    paths = len(n_0)
    
    # Pre-allocate arrays 
    n_mat = np.zeros((paths, grid_points + 1))
    
    # Generate shared time vector
    t_space = np.linspace(t_0, t_end, grid_points + 1)
    
    # Assign initial populations to the first column
    n_mat[:,0] = n_0
    
    progress_interval = max(1, paths // 10)
    
    for i in range(paths):
        init_pop_for_path = n_0[i]
        _, pop = strong_taylor(p_v, f, a, s, d, init_pop_for_path
                                     , t_0, t_end, grid_points - 1)
        
        # Assign simulation results starting from column 1
        n_mat[i, 1:] = pop[1:]
        
        if i % progress_interval == 0:
            progress_percent = (i / paths) * 100
            print(f"Strong Taylor Matrix Progress: {progress_percent:.1f}% ({i} / {paths})")
    
    print(f"Strong Taylor Matrix Complete: {paths} paths generated")
    return t_space, n_mat


def _save_taylor_results(pop : np.ndarray, detect : np.ndarray,
                         fission : float, index : str,
                         method : str, dead_time_type : str):
    
    # Extract prefix from index (e.g., 'mil_f33.94' -> 'mil_f')
    save_prefix = utl.extract_simulation_prefix(index, fission)
    
    # Generate standardized filename for logging/debugging
    filename = utl.generate_filename(save_prefix, 'Taylor', dead_time_type, fission)
    print(f"Saving results to: {filename}")
    
    data_manager.save_taylor_data(
        pop_data = pop,
        detect_data = detect,
        fission_value = fission,
        method = method,
        dead_time_type = dead_time_type,
        prefix = save_prefix)
    
    

