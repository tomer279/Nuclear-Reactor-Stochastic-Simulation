""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Stochastic simulation functions for population dynamics.

This module implements stochastic simulation of neutron population dynamics
in nuclear reactors using matrix-based approaches for multiple trajectories.
"""

import numpy as np
from data_management import DataManager

rng = np.random.default_rng()


def pop_dyn_mat(p_v: np.ndarray,
                fission: float,
                absorb: float,
                source: float,
                detect: float,
                n_0: np.ndarray,
                t_0: float,
                steps: int,
                prefix: str,
                data_manager: DataManager) -> tuple[np.ndarray, np.ndarray]:
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
    data_manager : DataManager, optional
        Data manager for organized file storag
        
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
    if any(x < 0 for x in [fission, absorb, source, detect]):
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
                                    fission, absorb, source, detect)
    
    # Post-process detect_mat to remove np.nan values
    clean_detect_mat = _clean_detection_matrix(detect_mat)
    
    # Save results using data manager
    # Extract prefix from index (e.g., 'mil_f33.94' -> 'mil_f')
    save_prefix = prefix.split(str(fission))[0] # Split on fission value and take first port
    data_manager.save_stochastic_data(pop_mat, time_mat, clean_detect_mat,
                                      fission, prefix = save_prefix)
    
    return time_mat, pop_mat, clean_detect_mat


def _update_population_for_event(current_population, event_type, p_v):
    "Update population based on the type of event that occurred."
    
    if event_type == 0:
        return current_population + 1
    elif event_type == 1:
        particles_produced = rng.choice(len(p_v), p = p_v)
        return current_population + particles_produced - 1
    elif event_type == 2 or event_type == 3:
        return current_population - 1
    else:
        raise ValueError(f"Unknown event type: {event_type}")
   

def _simulate_single_trajectory(
        trajectory_index, num_steps, pop_mat, time_mat, event_mat, detect_mat,
        p_v, fission, absorb, source, detect):
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
        total_rate = (fission + absorb + detect) * current_pop + source
            
        if total_rate <= 0:
            # System is static
            time_mat[i, j + 1] = time_mat[i, j]
            pop_mat[i, j + 1] = current_pop
            event_mat[i, j + 1] = np.nan
            detect_mat[i, j + 1] = np.nan
            continue
            
            # Calculate event probabilities
        event_probs = np.array([
            source / total_rate,                 # source
            current_pop * fission / total_rate,   # fission
            current_pop * absorb / total_rate,   # absorption
            current_pop * detect / total_rate    # detection
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
  
