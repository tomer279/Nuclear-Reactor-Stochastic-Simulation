""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Nuclear Reactor Stochastic Simulation - Main Execution Script

This script demonstrates the complete workflow for nuclear reactor simulations,
including stochastic simulations, Euler-Maruyama methods, and count rate analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from config import SimulationConfig
import simulation_runner as sim_run
from data_management import DataManager
import count_rates as cr
import plot_simulations as ps


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
    steps = 1_000_000  # Adjust as needed
    mean_tau = config.mean_tau
    grid_points = 100_000
    
    print(f"Configuration loaded: {len(fission_vec)} fission values")
    print(f"Simulation parameters: steps={steps}, grid_points={grid_points}")
    
    # =============================================================================
    # STEP 1: RUN STOCHASTIC SIMULATIONS (if needed)
    # =============================================================================
    
    run_stochastic = False  # Set to True to run new stochastic simulations
    if run_stochastic:
        print("\n" + "="*60)
        print("STEP 1: RUNNING STOCHASTIC SIMULATIONS")
        print("="*60)
        
        sim_run.run_stochastic_simulations(
            fission_vec, equil, p_v, absorb, source, detect,
            t_0, steps, prefix='mil'
        )
    
    # =============================================================================
    # STEP 2: RUN EULER-MARUAMA SIMULATIONS (OPTIONAL)
    # =============================================================================
    
    run_euler_maruyama = True  # Set to False to skip EM simulations
    if run_euler_maruyama:
        print("\n" + "="*60)
        print("STEP 2: RUNNING EULER-MARUAMA SIMULATIONS")
        print("="*60)
        
        # Load time data from stochastic simulations
        try:
            _, simul_time_vec, _ = data_manager.load_stochastic_data(
                fission_vec=fission_vec, prefix='mil_f')
            print(f"Loaded time data for {len(simul_time_vec)} fission values")
        except FileNotFoundError:
            print("ERROR: Stochastic time data not found. Run stochastic simulations first.")
            return
        
        # Run Euler-Maruyama simulations with 'short_f' prefix
        sim_run.run_euler_maruyama_simulations(
            fission_vec, simul_time_vec, p_v, 
            absorb, source, detect, t_0,
            grid_points, mean_tau, prefix='short_f',
            run_without_dead_time=True,
            run_with_const_dead_time=True,
            run_with_exp_dead_time=False
        )
        print("Euler-Maruyama simulations completed!")
    else:
        print("\n" + "="*60)
        print("STEP 2: SKIPPING EULER-MARUAMA SIMULATIONS")
        print("="*60)
        print("Set run_euler_maruyama = True to run EM simulations")
    
    # =============================================================================
    # STEP 3: COUNT RATE ANALYSIS AND COMPARISON (OPTIONAL)
    # =============================================================================
    
    run_cps_analysis = True  # Set to False to skip CPS analysis
    if run_cps_analysis:
        print("\n" + "="*60)
        print("STEP 3: COUNT RATE ANALYSIS AND COMPARISON")
        print("="*60)
        
        test_cps_calculations(config, data_manager, fission_vec, mean_tau)
    else:
        print("\n" + "="*60)
        print("STEP 3: SKIPPING COUNT RATE ANALYSIS")
        print("="*60)
        print("Set run_cps_analysis = True to run CPS comparison")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE!")
    print("="*80)
    

def test_cps_calculations(config, data_manager, fission_vec, mean_tau):
    """
    Test and compare count rate calculations between stochastic and Euler-Maruyama methods.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration object
    data_manager : DataManager
        Data manager instance
    fission_vec : list
        List of fission values
    mean_tau : float
        Mean dead time
    """
    print("Testing CPS calculations with constant dead time...")
    
    # Load stochastic data
    print("Loading stochastic data...")
    pop_matrices, time_matrices, detect_matrices = data_manager.load_stochastic_data(
        fission_vec=fission_vec, prefix='mil_f')
    
    if len(pop_matrices) == 0:
        print("ERROR: No stochastic data loaded!")
        return
    
    # Load Euler-Maruyama data with 'short_f' prefix
    print("Loading Euler-Maruyama data...")
    em_pop_matrices, em_detect_matrices = data_manager.load_euler_maruyama_data(
        fission_vec=fission_vec, dead_time_type='const', prefix='short_f')  # Using 'short_f' prefix
    
    if len(em_pop_matrices) == 0:
        print("ERROR: No Euler-Maruyama data loaded!")
        print("Make sure to run Euler-Maruyama simulations first (set run_euler_maruyama = True)")
        return
    
    # Calculate CPS
    print("Calculating count rates...")
    try:
        stochastic_cps_const = cr.calculate_all_count_rates(
            simul_time_vec=time_matrices, 
            simul_detect_vec=detect_matrices, 
            em_const_detect_vec=em_detect_matrices,
            mean_tau=mean_tau)
        
        # Extract results
        stochastic_cps = stochastic_cps_const['stochastic_const_tau']
        em_cps = stochastic_cps_const['em_const_tau']
        
        # Display results
        print("\nCPS Results (first 5 values):")
        for i, fission in enumerate(fission_vec[:5]):
            print(f"  Fission {fission}:")
            if isinstance(stochastic_cps[i], np.ndarray):
                print(f"    Stochastic CPS: {np.mean(stochastic_cps[i]):.2f} ± {np.std(stochastic_cps[i]):.2f}")
            else:
                print(f"    Stochastic CPS: {stochastic_cps[i]:.2f}")
            print(f"    Euler-Maruyama CPS: {em_cps[i]:.2f}")
            print(f"    Difference: {abs(np.mean(stochastic_cps[i]) - em_cps[i]):.2f}")
        
        # Create comparison plot
        print("\nCreating comparison plot...")
        ps.plot_cps_comparison(
            stochastic_cps=stochastic_cps,
            em_cps=em_cps,
            config=config,
            dead_time_type='const',
            save_path='cps_comparison_const_dead_time.png'
        )
        
        plt.show()
        print("CPS analysis completed successfully!")
        
    except Exception as e:
        print(f"ERROR in CPS calculation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
     main()
