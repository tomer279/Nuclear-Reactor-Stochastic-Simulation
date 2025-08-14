""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Euler-Maruyama methods for solving stochastic differential equations.

This module implements various Euler-Maruyama methods for solving SDEs
in the context of nuclear reactor population dynamics, including
different dead time effects.
"""

import numpy as np
import utils as utl
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
        
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    lam = params['lam']
    vbar = params['vbar']
    vbar_square = params['vbar_square']
    alpha = params['alpha']
    equil = params['equilibrium']
    
    # Compute important parameters
    sig_tilde_squared = lam + f * (vbar_square - 2 * vbar)  # noise parameter
    sig_hat = np.sqrt(sig_tilde_squared * (s / alpha) + s)  # noise amplitude
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
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
        
    # Calculate system parameters using helper function    
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    alpha = params['alpha']
    equil = params['equilibrium']
    sig_1 = params['sig_1']
    sig_2 = params['sig_2']
    
    # Rossi-alpha coefficient
    alpha_1 = alpha - d
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
    # Initialize arrays using helper function
    t_space, pop, detect, dt = utl.initialize_simulation_arrays(grid_points,
                                                             n_0,
                                                             t_0,
                                                             t_end)
    # Define drift functions
    drift_pop = lambda y: - alpha_1 * y + s
    drift_detect = lambda y: d * y
    
    # Generate all random increments at once
    dW_vec , dV_vec = utl.generate_random_increments(grid_points, dt, 2)
    
    # Euler-Maruyama integration
    for i in utl.progress_tracker(grid_points):
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
    
    # Save results using helper function
    _save_euler_maruyama_results(pop, detect, f, index, 'basic')
    
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
        
    # Calculate system parameters using helper function    
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    alpha = params['alpha']
    equil = params['equilibrium']
    sig_1 = params['sig_1']
    sig_2 = params['sig_2']
    
    dead_time_parameter = d * equil * tau
    
    # Noise variance associated with actual detections
    sig_3_squared = (np.exp(-2 * dead_time_parameter)
       * (1 - 2 * dead_time_parameter * np.exp(-dead_time_parameter))
       / (equil ** 2))
    
    sig_3 = np.sqrt(sig_3_squared)
    # correlation = (sig_2 / sig_3) * np.exp(-dead_time_parameter)
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
    # Initialize arrays using helper function
    t_space, pop, detect, dt = utl.initialize_simulation_arrays(grid_points,
                                                             n_0,
                                                             t_0,
                                                             t_end)
    # Define drift functions
    drift_pop = lambda y: - alpha * y + s
    drift_detect = lambda y:  d * y * np.exp(- d * y * tau)
    
    # Generate all random increments at once
    dW1_vec, dW2_vec, dW3_vec = utl.generate_random_increments(grid_points, dt, 3)
    
    # Create dW3_vec so that W_2 and W_3 are correlated by:
    # dW_2 dW_3 = correlation * dt
    # dW3_vec = correlation * dW2_vec + np.sqrt(1 - correlation ** 2) * dW3_indep
    
    # Euler-Maruyama integration
    for i in utl.progress_tracker(grid_points):
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
    
    # Extract prefix from indes (e.g., 'mil_f33.94' -> 'mil_f')
    _save_euler_maruyama_results(pop, detect, f, index, 'const')
    return t_space, pop, detect


def euler_maruyama_system_with_uniform_dead_time(p_v: np.ndarray,
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
       Ψ(x) = (exp(-2 * d * N_t) - 1) / 2 * d * N_t * mean_tau + 1
   - alpha is the modified Rossi-alpha coefficient, defined by:
       alpha = -(f + a + d) + vbar * f
   - sig_1 is the noise variance associated with the power:
       sig_1^2 = s * (f + a + f * (vbar_squared - 2 * vbar)) / alpha + s
   - sig_2 is the noise variance associated with the physical detections
       sig_2^2 = s * d / alpha
   - sig_3 is the noise variance associated with the actual detections.    
       sig_3^2 = 1/(d * N_t) + complex term
   - W_1 , W_2 , W_3 are Wiener processes, s. t. W_1, W_2 are independent,
       W_3 can either be independent, or be correlated with W_3, with correlation:
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
       
   FIXME: The function currently suffers from caluclation issues, mainly with 
       detection diffusion. It needs to be determined if the expression for
       the diffusion is correct.    
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
        
    # Calculate system parameters using helper function    
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    alpha = params['alpha']
    equil = params['equilibrium']
    sig_1 = params['sig_1']
    sig_2 = params['sig_2']
    
    psi = lambda y: (np.exp(-2 * y * tau_mean) - 1) / (2 * y * tau_mean) + 1
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
    t_space, pop, detect, dt = utl.initialize_simulation_arrays(grid_points,
                                                            n_0,
                                                            t_0,
                                                            t_end)
    # Define drift functions
    drift_pop = lambda y: - alpha * y + s
    drift_detect = lambda y:  d * y * ( 1 - psi(d * y))
    
    def diffusion_detect(y):
        """ Compute the diffusion coefficient for detection """
        D = d * y   # Detection Rate (lam_d * N_t)
        nominator = 4 * (D * tau_mean + 1) * np.exp(-2 * D * tau_mean) + \
            2 * D * tau_mean - 4
        denominator = (D ** 5) * (1 - np.exp(-2 * D * tau_mean))
        return np.sqrt(1/D + nominator / denominator)
    
    # Generate all random increments at once
    dW1_vec, dW2_vec, dW3_vec = utl.generate_random_increments(grid_points, dt, 3)
    
    # Create dW3_vec so that W_2 and W_3 are correlated by:
    # dW_2 dW_3 = correlation * dt
    # dW3_vec = correlation * dW2_vec + np.sqrt(1 - correlation ** 2) * dW3_indep
    
    # Euler-Maruyama integration
    for i in utl.progress_tracker(grid_points):
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
    
    # Save results using helper function
    _save_euler_maruyama_results(pop, detect, f, index, 'uniform')
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
    # Calculate system parameters using helper function    
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    alpha = params['alpha']
    equil = params['equilibrium']
    sig_1 = params['sig_1']
    sig_2 = params['sig_2']
    
    psi = lambda y: 1 - 1/(1 - y * tau_mean)
    
    # set initial population to equilibrium if not provided
    if n_0 is None:
        n_0 = equil
    
    t_space, pop, detect, dt = utl.initialize_simulation_arrays(grid_points,
                                                             n_0,
                                                             t_0,
                                                             t_end)
    
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
    dW1_vec , dW2_vec, dW3_vec = utl.generate_random_increments(grid_points, dt, 3)
    
    # Create dW3_vec so that W_2 and W_3 are correlated by:
    # dW_2 dW_3 = correlation * dt
    # dW3_vec = correlation * dW2_vec + np.sqrt(1 - correlation ** 2) * dW3_indep
    
    # Euler-Maruyama integration
    for i in utl.progress_tracker(grid_points):
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
    
    # Extract prefix from index (e.g., 'mil_f33.94' -> 'mil_f')
    _save_euler_maruyama_results(pop, detect, f , index , 'exp')
    
    return t_space, pop, detect


def euler_maruyama_matrix(p_v: np.ndarray,
                          f: float,
                          a: float, 
                          s: float, 
                          d: float,
                          t_0 : float, 
                          t_end: float, 
                          grid_points: int,
                          n_0: np.ndarray = None,
                          paths: int = None):
    """
    Generate multiple Euler-Maruyama simulation paths using the existing euler_maruyama function.
    
    This function efficiently generates multiple independent realizations by calling
    the existing euler_maruyama function with proper memory management and progress tracking
    using the available helper functions.
    
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
        
    n_0 : np.ndarray, optional
        Vector of initial population values, one for each simulation path.
        If provided, the length determines the number of paths (paths parameter ignored).
        If None, the paths parameter must be specified.
    paths : int, optional
        Number of simulation paths to generate, all starting from equilibrium population.
        Only used when n_0 is None.  
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Returns (t_space, n_mat) where:
        
        - t_space : np.ndarray, shape (grid_points + 1,)
            Time points from t_0 to t_end (inclusive).
            Shared across all simulation paths.
            
        - n_mat : np.ndarray, shape (num_paths, grid_points + 1)
            Population matrix where n_mat[i, j] represents the population
            of path i at time t_space[j].
            Each row corresponds to one simulation path.
            n_mat[i, 0] equals the initial population for path i.  
            
    Raises
    ------
    ValueError
        If neither n_0 nor paths is provided, or if both are None.
        
    Notes
    -----
    The function uses the following approach:
    
    1. If n_0 is provided, uses its length as the number of paths.
       If n_0 is None, generates paths number of trajectories from equilibrium.
       
    2. When paths is used, calculates the equilibrium
       population as s/α where α = (f + a + d) - f * mean(p_v).
       
    3. Each path is generated using the same euler_maruyama function
       used for single-path simulations, ensuring identical numerical methods.
    
    The stochastic differential equation being solved is:
    dN(t) = [(-α)N(t) + s]dt + σ_hat * dW(t)
    
    where α is the Rossi-alpha coefficient and σ_hat is the noise amplitude. 

    Examples
    --------
    Generate 5 paths starting from equilibrium:
    
    >>> t_space, n_mat = euler_maruyama_matrix(p_v, 33.94, 0.1, 1.0, 0.01, 
    ...                                       0, 1, 1000, paths=5)
    >>> print(f"Generated {n_mat.shape[0]} paths with {n_mat.shape[1]} time points")
    
    Generate 3 paths with custom initial populations:
    
    >>> initial_pops = np.array([900, 1000, 1100])
    >>> t_space, n_mat = euler_maruyama_matrix(p_v, 33.94, 0.1, 1.0, 0.01,
    ...                                       0, 1, 1000, n_0=initial_pops)
    >>> print(f"Initial populations: {n_mat[:, 0]}")  # Should be [900, 1000, 1100]       
    """
    
    if n_0 is not None:
        num_paths = len(n_0)
    elif paths is not None:
        params = utl.calculate_system_parameters(p_v, f, a, s, d)
        equil = params['equilibrium']
        n_0 = np.full(paths, equil)
        num_paths = paths
    else:
        raise ValueError("Must provide either 'n_0' array or 'paths' number")
    
    # Pre-allocate population matrix
    n_mat = np.zeros((num_paths,grid_points + 1))
    
    # Get shared time vector
    t_space = utl.initialize_simulation_arrays(grid_points, n_0[0], t_0, t_end)[0]
    
    # Generate each path
    for i in range(num_paths):
        n_mat[i,:] = euler_maruyama(p_v, f, a, s, d, t_0, t_end, grid_points, n_0[i])[1]
    
    return t_space, n_mat
    
        
        
def _save_euler_maruyama_results(pop: np.ndarray, detect: np.ndarray, 
                                fission: float, index: str, dead_time_type: str) -> None:
    """
    Save Euler-Maruyama simulation results to file.
    
    Parameters
    ----------
    pop : np.ndarray
        Population data array
    detect : np.ndarray
        Detection data array
    fission : float
        Fission rate constant
    index : str
        Simulation index (e.g., 'mil_f33.94')
    dead_time_type : str
        Type of dead time model ('basic', 'const', 'uniform', 'exp')
    """
    # Extract prefix from index (e.g., 'mil_f33.94' -> 'mil_f')
    save_prefix = utl.extract_simulation_prefix(index, fission)
    
    # Generate standardized filename for logging/debugging
    filename = utl.generate_filename(save_prefix, 'EM', dead_time_type, fission)
    print(f"Saving results to: {filename}")
    
    data_manager.save_euler_maruyama_data(
        pop_data = pop,
        detect_data = detect,
        fission_value = fission,
        dead_time_type = dead_time_type,
        prefix = save_prefix)


    

    
    
    
