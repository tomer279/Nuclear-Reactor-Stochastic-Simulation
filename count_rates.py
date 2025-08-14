""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Count rate calculation functions with various dead time effects.

This module provides functions to calculate count rates from detection data,
considering different types of dead time effects (constant, exponential, random).
"""

import numpy as np

rng = np.random.default_rng()


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
                              em_const_detect_vec = None,
                              em_uniform_detect_vec = None,
                              em_exp_detect_vec = None,
                              taylor_const_detect_vec = None,
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
           
       - 'em_uniform_tau': np.ndarray (if em_uniform_detect_vec provided)
           Count rates from Euler-Maruyama simulation with uniform dead time.
           Shape: (num_fission_values,)
           Values: Counts per second (float) 
    
       - 'em_exp_tau': np.ndarray (if em_exp_detect_vec provided)
           Count rates from Euler-Maruyama simulation with exponential dead time.
           Shape: (num_fission_values,)
           Values: Counts per second (float)
       
       - 'taylor_const_tau': np.ndarray (if taylor_const_detect_vec provided)
           Count rates from Taylor method simulation with constant dead time.
           
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
    
    def _calculate_cps(detect_vec, description):
        """ Calculate Euler-Maruyama CPS for a given detection vector."""
        print(f"Calculating Euler-Maruyama count rates ({description})")
        return np.array([
            detect_vec[j][-1] / simul_time_vec[j][:,-1].item() 
            for j in range(len(simul_time_vec))
            ])
        
    # Calculate Euler-Maruyama count rates using helper function
    em_data_configs = [
        (em_detect_vec, 'em_basic', 'without dead time'),
        (em_const_detect_vec, 'em_const_tau', 'with constant dead time'),
        (em_uniform_detect_vec, 'em_uniform_tau', 'with uniform dead time'),
        (em_exp_detect_vec, 'em_exp_tau', 'with exponential dead time')
        ]
    
    taylor_data_configs = [
        (taylor_const_detect_vec, 'taylor_const_tau', 'with constant dead time'),
        ]
    
    for detect_vec, result_key, description in em_data_configs:
        if detect_vec is not None:
            results[result_key] = _calculate_cps(detect_vec, description)
    
    for detect_vec, result_key, description in taylor_data_configs:
        if detect_vec is not None:
            results[result_key] = _calculate_cps(detect_vec, description)
    
    # Calculate theoretical approximation
    if mean_tau is not None and 'stochastic_basis' in results:
        print("Calculating theoretical approximation...")
        results['theoretical_approx'] = results['stochastic_basic'] * \
            np.exp(-results['stochastic_basic'] * mean_tau)
    
    return results