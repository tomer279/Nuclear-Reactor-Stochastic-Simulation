""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Main execution script for simulations
"""

import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig
import data_management as dm
import count_rates as cr
import plot_simulations as ps


def test_cps_calculations():
    """Test CPS calculations for both stochastic and Euler-Maruyama methods."""
    
    print("=" * 60)
    print("TESTING CPS CALCULATIONS WITH CONSTANT DEAD TIME")
    print("=" * 60)
    
    # Initialize
    config = SimulationConfig()
    data_manager = dm.DataManager()
    
    # Load data
    print("Loading data...")
    pop_matrices, time_matrices, detect_matrices = data_manager.load_stochastic_data(
        fission_vec=config.fission_vec, prefix='hunmil_f')
    
    em_pop_matrices, em_detect_matrices = data_manager.load_euler_maruyama_data(
        fission_vec=config.fission_vec, dead_time_type='const')
    
    # Calculate CPS for constant dead time
    print("Calculating CPS with constant dead time...")
    stochastic_cps_const = cr.calculate_all_count_rates(
        simul_time_vec=time_matrices, 
        simul_detect_vec=detect_matrices, 
        em_const_detect_vec=em_detect_matrices,
        mean_tau=config.mean_tau)  # Changed from config.dead_time_const to config.mean_tau
    
    # Extract the specific CPS values we want to compare
    stochastic_cps = stochastic_cps_const['stochastic_const_tau']
    em_cps = stochastic_cps_const['em_const_tau']
    
    # Test and print results
    print("\nCPS Results (first 5 values):")
    for i, fission in enumerate(config.fission_vec[:5]):
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
    print("Test completed successfully!")
    

if __name__ == "__main__":
    test_cps_calculations()

