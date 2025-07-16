""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Euler-Maruyama methods for solving stochastic differential equations.

This module implements various Euler-Maruyama methods for solving SDEs
in the context of nuclear reactor population dynamics, including
different dead time effects.
"""

import numpy as np
from utils import mean
from utils import mean_square
from data_management import DataManager

rng = np.random.default_rng()

data_manager = DataManager()


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
    
    data_manager.save_euler_maruyama_data(
        pop_data = pop,
        detect_data = detect, 
        fission_value = float(index),
        dead_time_type = 'basic'
    )
    
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
        
    data_manager.save_euler_maruyama_data(
        pop_data = pop,
        detect_data = detect,
        fission_value = float(index),
        dead_time_type = 'const'
    )
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
        
    data_manager.save_euler_maruyama_data(
        pop_data = pop,
        detect_data = detect,
        fission_value = float(index),
        dead_time_type = 'exp'
    )
    return t_space, pop, detect


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
    return t_mat, n_mat
