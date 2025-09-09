
import numpy as np
import utils as utl

rng = np.random.default_rng()

def analytical_population_solution(
        p_v : np.ndarray,
        f: float,
        a: float,
        s: float,
        d: float,
        t_0 : float,
        t_end : float,
        grid_points : int,
        n_0: float = None
        ):
    """
    Simulate population dynamics using the analytical solution.
    
    This function uses the exact analytical solution for the SDE:
    dN_t = (-α * N_t + S)dt + σ₁ * dW₁ - σ₂ * dW₂
    
    Instead of numerical integration, it uses the exact Markov transition:
    N_(i+1) = μ(dt) + σ(dt) * Z
    
    where:
    μ(dt) = e^(-α*dt)(N_i - S/α) + S/α
    σ²(dt) = (σ₁² + σ₂²)/(2α) * (1 - e^(-2α*dt))
    Z ~ N(0,1)
    
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
        Number of time steps
    n_0 : float, optional
        Initial population value. If None, defaults to equilibrium value S/α
    num_realizations : int, optional
        Number of stochastic realizations to generate. Default is 1.
        
    Returns
    -------
    pop : np.ndarray
        - The array of population values
    """
    
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    alpha = params['alpha']
    equil = params['equilibrium']
    sig_1_squared = params['sig_1_squared']
    sig_2_squared = params['sig_2_squared']
        
    if n_0 is None:
        n_0 = equil
    
    t_space, pop, _, dt = utl.initialize_simulation_arrays(grid_points, n_0, t_0, t_end)
    
    exp_term = np.exp(- alpha * dt)
    var_term = (
        ((sig_1_squared + sig_2_squared) / (2 * alpha))
        * (1 - np.exp(- 2 * alpha * dt))
        )
    
    gaussian = rng.normal(size = grid_points)
    
    for i in utl.progress_tracker(grid_points):
        current_pop = pop[i]
        mean_term = exp_term * (current_pop - equil) + equil
        pop[i + 1] = mean_term + np.sqrt(var_term) * gaussian[i]
        
    return pop


def analytical_detection_euler_maruyama(
        p_v: np.ndarray,
        f: float,
        a: float,
        s: float,
        d: float,
        tau: float,
        t_0: float,
        t_end: float,
        grid_points: int,
        n_0: float = None
        ):
    """
    Simulate detection dynamics using analytical solution for the SDE:
    dC_t = d * N_t * exp(-d * N_t * tau) dt + σ₃ * dW₃
    With Euler-Maruyama integration.
    
    This function first generates the population using the analytical solution,
    then calculates the detections based on the population values, using Euler-Maruyama.
    
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
        Number of time steps
    n_0 : float, optional
        Initial population value. If None, defaults to equilibrium value S/α
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns (t_space, pop, detect) where:
        - t_space is the array of time points
        - pop is the array of population values
        - detect is the array of detection values
    """
    
    # First, get the population solution
    t_space, pop = analytical_population_solution(p_v, f, a, s, d, t_0, t_end, grid_points, n_0)
    
    # Calculate system parameters for detection noise
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    equil = params['equilibrium']
    
    # Calculate dead time parameter and sigma_3
    dead_time_parameter = d * equil * tau
    sig_3_squared = (np.exp(-2 * dead_time_parameter)
                     * (1 - 2 * dead_time_parameter * np.exp(-dead_time_parameter))
                     / (equil ** 2))
    sig_3 = np.sqrt(sig_3_squared)
    
    # Time step
    dt = (t_end - t_0) / grid_points
    
    # Initialize detection array
    detect = np.zeros(grid_points + 1)
    
    # Generate Wiener increments for detection noise
    dW3 = rng.normal(loc=0.0, scale=np.sqrt(dt), size=grid_points)
    
    # Calculate detections step by step
    for i in range(grid_points):
        current_pop = pop[i]
        
        # Drift term: d * N_t * exp(-d * N_t * tau) * dt
        drift = d * current_pop * np.exp(-d * current_pop * tau) * dt
        
        # Diffusion term: σ₃ * dW₃
        diffusion = sig_3 * dW3[i]
        
        # Update detection: C_(i+1) = C_i + drift + diffusion
        detect[i + 1] = detect[i] + drift + diffusion
    
    return t_space, pop, detect


def analytical_solution_strong_taylor(p_v: np.ndarray,
                                      f: float,
                                      a: float,
                                      s: float,
                                      d: float,
                                      tau: float,
                                      t_0: float,
                                      t_end: float,
                                      grid_points: int,
                                      n_0: float = None):
    """
    Simulate detection dynamics using analytical solution for the SDE:
    dC_t = d * N_t * exp(-d * N_t * tau) dt + σ₃ * dW₃
    With Strong Taylor integration of order 1.5.
    
    This function first generates the population using the analytical solution,
    then calculates the detections based on the population values, using Strong Taylor.
    
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
        Number of time steps
    n_0 : float, optional
        Initial population value. If None, defaults to equilibrium value S/α
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns (t_space, pop, detect) where:
        - t_space is the array of time points
        - pop is the array of population values
        - detect is the array of detection values
    """
    
    t_space, pop = analytical_population_solution(p_v, f, a, s, d,
                                                  t_0, t_end, grid_points, n_0)
    
    # Calculate system parameters for detection noise
    params = utl.calculate_system_parameters(p_v, f, a, s, d)
    equil = params['equilibrium']
    
    # Calculate dead time parameter and sigma_3
    dead_time_parameter = d * equil * tau
    sig_3_squared = (np.exp(-2 * dead_time_parameter)
                     * (1 - 2 * dead_time_parameter * np.exp(-dead_time_parameter))
                     / (equil ** 2))
    sig_3 = np.sqrt(sig_3_squared)
    
    # Time step
    t_space, _, detect, dt = utl.initialize_simulation_arrays(grid_points, n_0, t_0, t_end)
    
    # Generate Wiener increments for detection noise
    dW = utl.generate_random_increments(grid_points, dt, 1)
    
    dZ = utl.generate_iterated_integral(grid_points, dt, (dW))
    
    for i in range(grid_points):
        current_pop = pop[i]
        current_detect = detect[i]
        
        drift = d * current_pop * np.exp(-d * current_pop * tau) * dt
        
        drift_derivative = _detect_drift_derivative(d * current_pop, tau, 1)
        drift_second_derivative = _detect_drift_derivative(d * current_pop, tau, 2)
        
        # Update detectionn
        detect[i + 1] = (current_detect + drift * dt +
                         + sig_3 * (dW[i] + drift_derivative * dZ[i])
                         + 0.5 * (drift * drift_derivative 
                                  + 0.5 * sig_3_squared * drift_second_derivative ) * (dt ** 2))
        
    return t_space, pop, detect

def _detect_drift_derivative(x : float, tau: float, n : int):
    """
    Calculate the nth derivative of the detection drift:
        f(x) = x* exp(-tau * x)
    Using the formula:
        f^(n)(x) = (-1)^n * (tau^(n-1)) * (tau * x - n) * (exp(-tau * x))

    Parameters
    ----------
    x : float
        Input value
    tau : float
        Dead time constant
    n : int
        Order of derivative

    Returns
    -------
    The value of the nth derivative of the detection drift at x.

    """
    return ((-1) ** n) * (tau ** (n-1)) * (tau * x - n) * (np.exp(-x * tau))
    




