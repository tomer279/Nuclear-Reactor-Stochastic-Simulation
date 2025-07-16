""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Taylor method implementations for solving stochastic differential equations.

This module implements Strong Taylor and Weak Taylor methods for solving SDEs
in the context of nuclear reactor population dynamics.
"""

import numpy as np
from utils import mean, mean_square

rng = np.random.default_rng()

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
