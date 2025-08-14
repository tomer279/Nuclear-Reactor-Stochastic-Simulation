""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Nuclear Reactor Stochastic Simulation - Main Execution Script

This is the main entry point for running comprehensive nuclear reactor simulations
using multiple numerical methods: Stochastic, Euler-Maruyama, and Taylor methods.

OVERVIEW:
=========
This script orchestrates the complete simulation pipeline including:
- Stochastic simulations for baseline results
- Euler-Maruyama method simulations with various dead time models
- Strong Taylor 1.5 method simulations
- Count rate analysis and comparison
- Plot generation and visualization

SIMULATION METHODS:
==================
1. STOCHASTIC: Direct simulation 
2. EULER-MARUYAMA: First-order stochastic integration
3. TAYLOR: Strong Taylor 1.5 method for higher accuracy

DEAD TIME MODELS:
=================
- basic: No dead time effects
- const: Constant dead time
- uniform: Uniformly distributed dead time
- exp: Exponentially distributed dead time

USAGE INSTRUCTIONS:
==================
1. Configure simulation parameters in config.py
2. Set run flags (run_stochastic, run_euler_maruyama, run_taylor)
3. Run: python main.py
4. Results are saved to ./data/ directory
5. Plots are generated automatically

OUTPUT FILES:
=============
- Population matrices: .npy files in respective method directories
- Detection matrices: .npy files for count rate analysis
- Time matrices: .npy files for temporal data
- Comparison plots: PNG/PDF files showing method comparisons

REQUIREMENTS:
============
- Python 3.8+
- NumPy, Matplotlib, Pathlib
- All utility modules (utils.py, data_management.py, etc.)

EXAMPLE USAGE:
=============
# Run only Taylor methods:
run_taylor = True
run_stochastic = False
run_euler_maruyama = False

# Run only constant dead time simulations for Euler-Maruayma:
EM_DEAD_TIME_MODELS['const'] = True
EM_DEAD_TIME_MODELS['uniform'] = False
EM_DEAD_TIME_MODELS['exp'] = False

# Run multiple dead time models for Taylor:
TAYLOR_DEAD_TIME_MODELS['const'] = True
TAYLOR_DEAD_TIME_MODELS['uniform'] = True
TAYLOR_DEAD_TIME_MODELS['exp'] = False

# Generate comparison plots:
generate_plots = True
test_cps = True
"""

import matplotlib.pyplot as plt
import numpy as np
from config import SimulationConfig
import simulation_runner as sim_run
from data_management import DataManager
import count_rates as cr
import plot_simulations as ps


# =============================================================================
# SIMULATION CONTROL FLAGS - MODIFY THESE TO CONTROL SIMULATIONS
# =============================================================================
# Set these flags to control which simulations run
run_stochastic = True      # Run direct stochastic simulations
run_euler_maruyama = True  # Run Euler-Maruyama method simulations  
run_taylor = True         # Run Taylor method simulations

# =============================================================================
# ANALYSIS FLAGS
# =============================================================================
# Set these flags to control post-simulation analysis
run_cps_analysis = True           # Run count rate analysis and plot comparison

# =============================================================================
# SIMULATION PREFIXES - MODIFY THESE FOR DIFFERENT SIMULATION TYPES
# =============================================================================
# Prefixes determine the naming convention and organization of simulation results
stochastic_prefix = 'mil_f'       # Million steps
em_prefix = 'short_f'             # Short Simulation time for Euler-Maruayama (100_000 grid points)
taylor_prefix = 'short_f'         # Short simulation time for Taylor methods (100_000 grid points)

# =============================================================================
# DEAD TIME CONFIGURATION - MODIFY THESE TO CONTROL DEAD TIME MODELS
# =============================================================================
# Control which dead time models are simulated for each method

# Euler-Maruyama Dead Time Models
EM_DEAD_TIME_MODELS = {
    'basic': False,      # No dead time effects
    'const': True,       # Constant dead time
    'uniform': False,    # Uniformly distributed dead time
    'exp': False         # Exponentially distributed dead time
}

# Taylor Method Dead Time Models
TAYLOR_DEAD_TIME_MODELS = {
    'basic': False,      # No dead time effects
    'const': True,       # Constant dead time
    'uniform': False,    # Uniformly distributed dead time
    'exp': False         # Exponentially distributed dead time
}

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================
# Control which dead time type to analyze
ANALYZE_DEAD_TIME_TYPE = 'const'  # Options: 'basic', 'const', 'uniform', 'exp'

def main():
    """
    Main execution function demonstrating the complete simulation workflow.
    """
    print("=" * 80)
    print("NUCLEAR REACTOR STOCHASTIC SIMULATION")
    print("=" * 80)
    
    # Initialize configuration
    config = SimulationConfig()
    data_manager = DataManager()
    
    # Extract commonly used parameters
    fission_vec = config.fission_vec
    equil = config.equil
    p_v = config.p_v
    absorb = config.absorb
    source = config.source
    detect = config.detect
    t_0 = config.t_0
    steps = 1_000_000 # Adjust as needed (default is 100_000_000)
    mean_tau = config.mean_tau
    grid_points = 100_000 # Adjust as needed (default is 10_000_000)
    
    print(f"Configuration loaded: {len(fission_vec)} fission values")
    print(f"Simulation parameters: steps={steps}, grid_points={grid_points}")
    
    # =============================================================================
    # STEP 1: RUN STOCHASTIC SIMULATIONS
    # =============================================================================
    
    if run_stochastic:
        print("\n" + "="*60)
        print("STEP 1: RUNNING STOCHASTIC SIMULATIONS")
        print("="*60)
        
        sim_run.run_stochastic_simulations(
            fission_vec, equil, p_v, absorb, source, detect,
            t_0, steps, prefix = stochastic_prefix
        )
    else:
        print("\n" + "="*60)
        print("STEP 1: SKIPPING STOCHASTIC SIMULATIONS")
        print("="*60)
        print("Set run_stochastic = True to run stochastic simulations")
    
    # =============================================================================
    # STEP 2: RUN EULER-MARUAMA SIMULATIONS
    # =============================================================================
    
    if run_euler_maruyama:
        print("\n" + "="*60)
        print("STEP 2: RUNNING EULER-MARUAMA SIMULATIONS")
        print("="*60)
        
        # Load time data from stochastic simulations
        try:
            _, simul_time_vec, _ = data_manager.load_stochastic_data(
                fission_vec=fission_vec, prefix = stochastic_prefix)
            print(f"Loaded time data for {len(simul_time_vec)} fission values")
        except FileNotFoundError:
            print("ERROR: Stochastic time data not found. Run stochastic simulations first.")
            return
        
        # Run Euler-Maruyama simulations
        sim_run.run_euler_maruyama_simulations(
            fission_vec, simul_time_vec, p_v, 
            absorb, source, detect, t_0,
            grid_points, mean_tau, prefix = em_prefix,
            run_basic = EM_DEAD_TIME_MODELS['basic'],
            run_with_const_dead_time = EM_DEAD_TIME_MODELS['const'],
            run_with_uniform_dead_time = EM_DEAD_TIME_MODELS['uniform'],
            run_with_exp_dead_time = EM_DEAD_TIME_MODELS['exp']
        )
        print("Euler-Maruyama simulations completed!")
    else:
        print("\n" + "="*60)
        print("STEP 2: SKIPPING EULER-MARUAMA SIMULATIONS")
        print("="*60)
        print("Set run_euler_maruyama = True to run EM simulations")
    
    
    # =============================================================================
    # STEP 2.5: RUN TAYLOR METHOD SIMULATIONS
    # =============================================================================
    
    if run_taylor:
        print("\n" + "="*60)
        print("STEP 2.5: RUNNING TAYLOR METHOD SIMULATIONS")
        print("="*60)
        
        # Load time data from stochastic simulations
        try:
            _, simul_time_vec, _ = data_manager.load_stochastic_data(
                fission_vec=fission_vec, prefix = stochastic_prefix)
            print(f"Loaded time data for {len(simul_time_vec)} fission values")
        except FileNotFoundError:
            print("ERROR: Stochastic time data not found. Run stochastic simulations first.")
            return
        
        # Run Taylor simulations
        sim_run.run_strong_taylor_simulations(
            fission_vec, simul_time_vec, p_v, 
            absorb, source, detect, t_0,
            grid_points, mean_tau, prefix = taylor_prefix,
            run_basic = TAYLOR_DEAD_TIME_MODELS['basic'],
            run_with_const_dead_time = TAYLOR_DEAD_TIME_MODELS['const'],
            run_with_uniform_dead_time = TAYLOR_DEAD_TIME_MODELS['uniform'],
            run_with_exp_dead_time = TAYLOR_DEAD_TIME_MODELS['exp']
        )
        print("Taylor method simulations completed!")
    else:
        print("\n" + "="*60)
        print("STEP 2.5: SKIPPING TAYLOR METHOD SIMULATIONS")
        print("="*60)
        print("Set run_taylor = True to run Taylor simulations")
    
    
    # =============================================================================
    # STEP 3: COUNT RATE ANALYSIS AND COMPARISON (OPTIONAL)
    # =============================================================================
    
    if run_cps_analysis:
        print("\n" + "="*60)
        print("STEP 3: COUNT RATE ANALYSIS AND COMPARISON")
        print("="*60)
        
        test_cps_calculations(config, data_manager, fission_vec, mean_tau,
                              dead_time_type = ANALYZE_DEAD_TIME_TYPE)
        
    else:
        print("\n" + "="*60)
        print("STEP 3: SKIPPING COUNT RATE ANALYSIS")
        print("="*60)
        print("Set run_cps_analysis = True to run CPS comparison")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE!")
    print("="*80)
    

def test_cps_calculations(config, data_manager, fission_vec, mean_tau,
                          dead_time_type = 'const'):
    """
    Test and compare count rate calculations between all simulation methods.
    
    This function loads simulation results from all methods and calculates
    counts per second (CPS) for comparison. It's essential for validating
    the accuracy of different numerical methods.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration object containing simulation parameters
    data_manager : DataManager
        Data management object for loading simulation results
    fission_vec : List[float]
        List of fission values used in simulations
    mean_tau : float
        Mean dead time value for analysis
        
    Returns
    -------
    None
        Results are printed to console and plots are generated
        
    Notes
    -----
    - Requires all simulation methods to have completed successfully
    - Generates comparison plots showing method accuracy
    - Calculates relative differences between methods
    """
    
    print(f"Testing CPS calculations with {dead_time_type} dead time...")
    
    # Load stochastic data
    print("Loading stochastic data...")
    pop_matrices, time_matrices, detect_matrices = data_manager.load_stochastic_data(
        fission_vec=fission_vec, prefix = stochastic_prefix)
    
    if len(pop_matrices) == 0:
        print("ERROR: No stochastic data loaded!")
        return
    
    # Load Euler-Maruyama data
    print("Loading Euler-Maruyama data...")
    em_pop_matrices, em_detect_matrices = data_manager.load_euler_maruyama_data(
        fission_vec=fission_vec, dead_time_type = dead_time_type, prefix= em_prefix)
    
    if len(em_pop_matrices) == 0:
        print("ERROR: No Euler-Maruyama data loaded!")
        print("Make sure to run Euler-Maruyama simulations first (set run_euler_maruyama = True)")
        return
    
    # Load Taylor method data
    print("Loading Taylor method data...")
    taylor_pop_matrices, taylor_detect_matrices = data_manager.load_taylor_data(
        fission_vec=fission_vec, method = 'strong',
        dead_time_type = dead_time_type, prefix= taylor_prefix)
    
    if len(taylor_pop_matrices) == 0:
        print("ERROR: No Taylor method data loaded!")
        print("Make sure to run Taylor simulations first (set run_taylor = True)")
        return
    
    # Calculate CPS
    print("Calculating count rates...")
    try:
        all_cps = cr.calculate_all_count_rates(
            simul_time_vec=time_matrices, 
            simul_detect_vec=detect_matrices, 
            em_const_detect_vec=em_detect_matrices,
            taylor_const_detect_vec=taylor_detect_matrices,
            mean_tau=mean_tau)
        
        # Extract results based on dead time type
        if dead_time_type == 'basic':
            em_cps = all_cps['em_basic_tau']
            taylor_cps = all_cps['taylor_basic_tau']
        elif dead_time_type == 'const':
            em_cps = all_cps['em_const_tau']
            taylor_cps = all_cps['taylor_const_tau']
        elif dead_time_type == 'uniform':
            em_cps = all_cps['em_uniform_tau']
            taylor_cps = all_cps['taylor_uniform_tau']
        elif dead_time_type == 'exp':
            em_cps = all_cps['em_exp_tau']
            taylor_cps = all_cps['taylor_exp_tau']
        else:
            print(f"ERROR: Unknown dead time type: {dead_time_type}")
            return
        
        stochastic_cps = all_cps['stochastic_const_tau'] # Stochastic is always 'const'
        
        # Display results
        print(f"\nCPS Results for {dead_time_type} dead time (first 5 values):")
        for i, fission in enumerate(fission_vec[:5]):
            print(f"  Fission {fission}:")
            if isinstance(stochastic_cps[i], np.ndarray):
                print(f"    Stochastic CPS: {np.mean(stochastic_cps[i]):.2f}"
                      f"± {np.std(stochastic_cps[i]):.2f}")
            else:
                print(f"    Stochastic CPS: {stochastic_cps[i]:.2f}")
            print(f"    Euler-Maruyama CPS: {em_cps[i]:.2f}")
            print(f"    Taylor CPS: {taylor_cps[i]:.2f}")
            print(f"    EM vs Stochastic:"
                  f"{abs(np.mean(stochastic_cps[i]) - em_cps[i]):.2f}")
            print(f"    Taylor vs Stochastic:"
                  f"{abs(np.mean(stochastic_cps[i]) - taylor_cps[i]):.2f}")
        
        # Create comparison plot
        print(f"\nCreating comparison plot for {dead_time_type} dead time...")
        ps.plot_cps_comparison(
            stochastic_cps=stochastic_cps,
            em_cps=em_cps,
            taylor_cps=taylor_cps,
            config=config,
            dead_time_type= dead_time_type,
            save_path = False
        )
        
        plt.show()
        print(f"CPS analysis for {dead_time_type} completed successfully!")
        
    except Exception as e:
        print(f"ERROR in CPS calculation: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
     main()
