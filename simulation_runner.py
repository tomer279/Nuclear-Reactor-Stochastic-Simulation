""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Simulation execution and orchestration functions.

This module contains  functions that coordinate and execute various types of 
simulations, parameter sweeps, and analysis workflows.
"""

import numpy as np
from stochastic_simulation import pop_dyn_mat
from euler_maruyama_methods import (
    euler_maruyama_system_basic,
    euler_maruyama_system_with_const_dead_time,
    euler_maruyama_system_with_uniform_dead_time,
    euler_maruyama_system_with_exp_dead_time
)
from taylor_methods import (
    strong_taylor_system_with_const_dead_time
    )
# from count_rates import calculate_all_count_rates
import data_management as dm

from config import SimulationConfig

# =============================================================================
# NOTE: Make sure to test that run_stochastic_simulations is working correctly!
# - Check that the output files are created in the correct folders.
# - Check that the population, time, and detection matrices are generated and saved as expected.
# - Check that the function runs without errors for all values in fission_vec.
# - You can verify by loading the output with DataManager and plotting the results.
# =============================================================================


def run_stochastic_simulations(fission_vec,
                               equil,
                               p_v,
                               absorb,
                               source,
                               detect,
                               t_0, 
                               steps,
                               prefix='f'):
    """
    Run stochastic simulations for all fission values and save results.
    """
    
    print("=" * 60)
    print("RUNNING STOCHASTIC SIMULATIONS")
    print("=" * 60)
    
    data_manager = dm.DataManager()
    
    for i, fission in enumerate(fission_vec):
        print(f"iteration {i + 1}/{len(fission_vec)}, fission = {fission}")
        n_0 = equil[i] * np.ones(1)
        index = f"{prefix}{fission}"
        pop_dyn_mat(p_v, fission, absorb, source, detect, n_0, t_0, steps, 
                   index, data_manager = data_manager)
    
    print("Stochastic simulations complete!")


def run_euler_maruyama_simulations(fission_vec, simul_time_vec, p_v, absorb, source, 
                                  detect, t_0, grid_points, mean_tau, prefix='',
                                  run_basic = True,
                                  run_with_const_dead_time = True,
                                  run_with_uniform_dead_time = True,
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
        if run_basic:
            euler_maruyama_system_basic(
                p_v, fission, absorb, source, detect, t_0, t_end, 
                grid_points, f"{prefix}{fission}"
                )
        
        # With constant dead time
        if run_with_const_dead_time:
            euler_maruyama_system_with_const_dead_time(
                p_v, fission, absorb, source, detect, mean_tau, t_0, t_end, 
                grid_points, f"{prefix}{fission}"
                )
        
        # With uniform dead time
        if run_with_uniform_dead_time:
            euler_maruyama_system_with_uniform_dead_time(
                p_v, fission, absorb, source, detect, mean_tau, t_0, t_end, 
                grid_points, f"{prefix}{fission}")
        
        # with exponential dead time
        if run_with_exp_dead_time:
            euler_maruyama_system_with_exp_dead_time(
                p_v, fission, absorb, source, detect, mean_tau, t_0, t_end, 
                grid_points, f"{prefix}{fission}"
                )
            
    print("Euler-Maruyama simulations complete!")
    

def run_strong_taylor_simulations(fission_vec, simul_time_vec, p_v, absorb, source, 
                          detect, t_0, grid_points, mean_tau, prefix='',
                          run_basic=True, run_with_const_dead_time=True,
                          run_with_uniform_dead_time=True, run_with_exp_dead_time=True):
    """
    Run strong Taylor method simulations for all fission values and save results.
    """
    print("=" * 60)
    print("RUNNING TAYLOR METHOD SIMULATIONS")
    print("=" * 60)
    
    for i, fission in enumerate(fission_vec):
        print(f"iteration {i + 1}/{len(fission_vec)}, fission = {fission}")
        t_end = simul_time_vec[i][:,-1].item()
        
        '''
        # Basic Taylor (no dead time)
        if run_basic:
            strong_taylor_system_basic(
                p_v, fission, absorb, source, detect, t_0, t_end, 
                grid_points, f"{prefix}{fission}"
                )
        '''
        
        # With constant dead time
        if run_with_const_dead_time:
            strong_taylor_system_with_const_dead_time(
                p_v, fission, absorb, source, detect, mean_tau, t_0, t_end, 
                grid_points, f"{prefix}{fission}" )
        
        '''
        # With uniform dead time
        if run_with_uniform_dead_time:
            strong_taylor_system_with_uniform_dead_time(
                p_v, fission, absorb, source, detect, mean_tau, t_0, t_end, 
                grid_points, f"{prefix}{fission}"
                )
            
        # With exponential dead time
        if run_with_exp_dead_time:
            strong_taylor_system_with_exp_dead_time(
                p_v, fission, absorb, source, detect, mean_tau, t_0, t_end, 
                grid_points, f"{prefix}{fission}"
                )
        '''
            
    print("Taylor method simulations complete!")


def execute_simulations(config: SimulationConfig, run_stochastic = False,
                        run_em = False, em_without_dead_time = True,
                        em_with_const_dead_time = True,
                        em_with_uniform_dead_time = True,
                        em_with_exp_dead_time = True):
    """
    Execute simulations using configuration object.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration object containing all simulation parameters
    run_stochastic : bool
        Whether to run stochastic simulations
    run_em : bool
        Whether to run Euler-Maruyama simulations
    em_without_dead_time : bool
        Whether to run EM without dead time
    em_with_const_dead_time : bool
        Whether to run EM with constant dead time
    em_with_exp_dead_time : bool
        Whether to run EM with exponential dead time
        
    Returns
    -------
    tuple
        Simulation results and data
    """
    fission_vec = config.fission_vec
    equil = config.equil
    p_v = config.p_v
    absorb = config.absorb
    source = config.source
    detect = config.detect
    t_0 = config.t_0
    steps = config.steps
    grid_points = config.grid_points
    mean_tau = config.mean_tau
    
    
    if run_stochastic:
        print("Running stochastic simulations...")
        run_stochastic_simulations(fission_vec,equil, p_v, absorb, 
                                   source, detect, t_0, steps)
    
    # Always load data (either existing or newly created)
    simul_time_vec, simul_detect_vec = dm.load_simulation_data(fission_vec)
    em_detect_vec, em_pop_vec = (
        dm.load_euler_maruyama_basic(fission_vec))
    em_const_detect_vec, em_const_pop_vec = (
        dm.load_euler_maruyama_with_const_dead_time_data(fission_vec))
    em_uniform_detect_vec, em_uniform_pop_vec = (
        dm.load_euler_maruyama_with_uniform_dead_time_data(fission_vec))
    em_exp_detect_vec, em_exp_pop_vec = (
        dm.load_euler_maruyama_with_exp_dead_time_data(fission_vec))
    
    if run_em:
        print("Running Euler-Maruyama simulations...")
        run_euler_maruyama_simulations(fission_vec, simul_time_vec, p_v, absorb, source, 
                                     detect, t_0, grid_points, mean_tau,
                                     run_without_dead_time = em_without_dead_time,
                                     run_with_const_dead_time = em_with_const_dead_time,
                                     run_with_uniform_dead_time = em_with_uniform_dead_time,
                                     run_with_exp_dead_time = em_with_exp_dead_time)
    
    return (simul_time_vec, simul_detect_vec, 
            em_detect_vec, em_pop_vec,
            em_const_detect_vec, em_const_pop_vec,
            em_uniform_detect_vec, em_uniform_pop_vec,
            em_exp_detect_vec, em_exp_pop_vec)