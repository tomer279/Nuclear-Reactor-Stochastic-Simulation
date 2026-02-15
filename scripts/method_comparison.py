""" Written by Tomer279 with the assistance of Cursor.ai

Updated comprehensive comparison of Runge-Kutta, Euler-Maruyama, and Strong-Taylor methods
for count rate calculations with constant dead time.

This script compares the three numerical methods implemented in analytical_solution.py
and compares their count rates corresponding to different fission values.
"""
# Path setup must come before imports
import sys
from pathlib import Path

# Add project root to Python path so src module can be found
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from config.config import SimulationConfig
from src.utils import analytical_solution as ans
from src.analysis import count_rates as cr
from src.visualization import plot_simulations as ps


def run_method_comparison():
    """
    Run comprehensive comparison of Runge-Kutta, Euler-Maruyama, and Strong-Taylor methods
    for count rate calculations with constant dead time.
    """

    # Initialize configuration
    config = SimulationConfig()

    # Extract commonly used parameters
    fission_vec = config.fission_vec
    alpha_inv_vec = config.alpha_inv_vec  # Add alpha inverse vector
    p_v = config.p_v
    absorb = config.absorb
    source = config.source
    detect = config.detect
    t_0 = config.t_0
    t_end = 0.1  # Use 1 second for count rate calculation
    mean_tau = config.mean_tau
    grid_points = 5_000_000  # Adjust as needed for computational efficiency

    print("=" * 80)
    print("METHOD COMPARISON: Runge-Kutta vs Euler-Maruyama vs Strong-Taylor")
    print("=" * 80)
    print(f"Fission values: {fission_vec}")
    print(f"Alpha inverse values: {alpha_inv_vec}")
    print(f"Dead time (tau): {mean_tau:.2e} seconds")
    print(f"Simulation time: {t_end} seconds")
    print(f"Grid points: {grid_points:,}")
    print("=" * 80)

    # Initialize arrays to store results
    num_fission = len(fission_vec)
    rk_cps = np.zeros(num_fission)
    em_cps = np.zeros(num_fission)
    st_cps = np.zeros(num_fission)

    # Run simulations for each fission value
    for i, fission in enumerate(fission_vec):
        print(
            f"\nProcessing fission value {fission:.3f} ({i+1}/{num_fission})...")

        try:
            # Runge-Kutta method
            print("  Running Runge-Kutta method...")
            _, _, rk_detect = ans.analytical_solution_runge_kutta(
                p_v, fission, absorb, source, detect, mean_tau,
                t_0, t_end, grid_points
            )
            rk_cps[i] = rk_detect[-1] / t_end

            # Euler-Maruyama method
            print("  Running Euler-Maruyama method...")
            _, em_detect = ans.analytical_detection_euler_maruyama(
                p_v, fission, absorb, source, detect, mean_tau,
                t_0, t_end, grid_points
            )
            em_cps[i] = em_detect[-1] / t_end

            # Strong-Taylor method
            print("  Running Strong-Taylor method...")
            _, _, st_detect = ans.analytical_solution_strong_taylor(
                p_v, fission, absorb, source, detect, mean_tau,
                t_0, t_end, grid_points
            )
            st_cps[i] = st_detect[-1] / t_end

            print(
                f"  Results: RK={rk_cps[i]:.6f}, EM={em_cps[i]:.6f}, ST={st_cps[i]:.6f}")

        except Exception as e:
            print(f"  Error processing fission {fission}: {e}")
            rk_cps[i] = np.nan
            em_cps[i] = np.nan
            st_cps[i] = np.nan

    # Calculate theoretical count rates for comparison
    print("\nCalculating theoretical count rates...")
    theoretical_cps = cr.calculate_theoretical_cps_for_fission_rates(
        fission_vec, 'constant', mean_tau, 0.0, detect, absorb, source, p_v
    )

    # Create comparison plots
    print("\nCreating comparison plots...")
    create_updated_comparison_plots(
        fission_vec, alpha_inv_vec, rk_cps, em_cps, st_cps, theoretical_cps, config, grid_points)

    # Print summary statistics
    print_summary_statistics(fission_vec, alpha_inv_vec,
                             rk_cps, em_cps, st_cps, theoretical_cps)

    return {
        'fission_vec': fission_vec,
        'alpha_inv_vec': alpha_inv_vec,
        'rk_cps': rk_cps,
        'em_cps': em_cps,
        'st_cps': st_cps,
        'theoretical_cps': theoretical_cps
    }


def create_updated_comparison_plots(fission_vec, alpha_inv_vec, rk_cps, em_cps, st_cps, theoretical_cps, config, grid_points):
    """
    Create updated comparison plots with the requested changes:
    1. Alpha inverse vector as x-axis
    2. Scatter plots for numerical methods in count rates plot
    3. Line plots for error comparison plots
    4. Grid points in title
    """

    # Setup plotting style
    ps.PlotStyle.setup_default_style()

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Method Comparison: Runge-Kutta vs Euler-Maruyama vs Strong-Taylor\n' +
                 f'Constant Dead Time (τ = {config.mean_tau:.2e} s) - Grid Points: {grid_points:,}', fontsize=16)

    # Plot 1: Count rates vs alpha inverse values (scatter plots for numerical methods)
    ax1.scatter(alpha_inv_vec, rk_cps, c='blue', marker='o',
                s=60, alpha=0.7, label='Runge-Kutta')
    ax1.scatter(alpha_inv_vec, em_cps, c='orange', marker='s',
                s=60, alpha=0.7, label='Euler-Maruyama')
    ax1.scatter(alpha_inv_vec, st_cps, c='green', marker='^',
                s=60, alpha=0.7, label='Strong-Taylor')
    ax1.plot(alpha_inv_vec, theoretical_cps, 'r--',
             linewidth=2, alpha=0.8, label='Theoretical')
    ax1.set_xlabel('Alpha Inverse (s)')
    ax1.set_ylabel('Count Rate (CPS)')
    ax1.set_title('Count Rates vs Alpha Inverse Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error comparison plot (absolute errors from theoretical) - LINE PLOTS
    rk_abs_error = np.abs(rk_cps - theoretical_cps)
    em_abs_error = np.abs(em_cps - theoretical_cps)
    st_abs_error = np.abs(st_cps - theoretical_cps)

    ax2.plot(alpha_inv_vec, rk_abs_error, 'bo-', linewidth=2,
             markersize=6, alpha=0.7, label='Runge-Kutta Error')
    ax2.plot(alpha_inv_vec, em_abs_error, marker='s', color='orange', linestyle='-',
             linewidth=2, markersize=6, alpha=0.7, label='Euler-Maruyama Error')
    ax2.plot(alpha_inv_vec, st_abs_error, marker='^', color='green', linestyle='-',
             linewidth=2, markersize=6, alpha=0.7, label='Strong-Taylor Error')
    ax2.set_xlabel('Alpha Inverse (s)')
    ax2.set_ylabel('Absolute Error from Theoretical (CPS)')
    ax2.set_title('Error Comparison from Theoretical Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Use log scale for better visualization of errors

    # Plot 3: Relative differences from theoretical - LINE PLOTS
    rk_rel_diff = (rk_cps - theoretical_cps) / theoretical_cps * 100
    em_rel_diff = (em_cps - theoretical_cps) / theoretical_cps * 100
    st_rel_diff = (st_cps - theoretical_cps) / theoretical_cps * 100

    ax3.plot(alpha_inv_vec, rk_rel_diff, 'bo-', linewidth=2,
             markersize=6, alpha=0.7, label='Runge-Kutta')
    ax3.plot(alpha_inv_vec, em_rel_diff, marker='s', color='orange',
             linestyle='-', linewidth=2, markersize=6, alpha=0.7, label='Euler-Maruyama')
    ax3.plot(alpha_inv_vec, st_rel_diff, marker='^', color='green',
             linestyle='-', linewidth=2, markersize=6, alpha=0.7, label='Strong-Taylor')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Alpha Inverse (s)')
    ax3.set_ylabel('Relative Difference from Theoretical (%)')
    ax3.set_title('Relative Differences from Theoretical Values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Method comparison (RK vs EM) - scatter plot
    ax4.scatter(em_cps, rk_cps, c='blue', alpha=0.7, s=60)
    min_val = min(np.nanmin(em_cps), np.nanmin(rk_cps))
    max_val = max(np.nanmax(em_cps), np.nanmax(rk_cps))
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    ax4.set_xlabel('Euler-Maruyama CPS')
    ax4.set_ylabel('Runge-Kutta CPS')
    ax4.set_title('Runge-Kutta vs Euler-Maruyama')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('updated_method_comparison_constant_dead_time.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_statistics(fission_vec, alpha_inv_vec, rk_cps, em_cps, st_cps, theoretical_cps):
    """
    Print summary statistics comparing the three methods.
    """

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Calculate statistics for each method
    methods = {
        'Runge-Kutta': rk_cps,
        'Euler-Maruyama': em_cps,
        'Strong-Taylor': st_cps
    }

    for method_name, cps in methods.items():
        if not np.all(np.isnan(cps)):
            rel_diff = (cps - theoretical_cps) / theoretical_cps * 100
            abs_error = np.abs(cps - theoretical_cps)
            print(f"\n{method_name}:")
            print(f"  Mean CPS: {np.nanmean(cps):.6f}")
            print(f"  Std CPS: {np.nanstd(cps):.6f}")
            print(f"  Mean absolute error: {np.nanmean(abs_error):.6f}")
            print(f"  Max absolute error: {np.nanmax(abs_error):.6f}")
            print(
                f"  Mean relative difference from theoretical: {np.nanmean(rel_diff):.4f}%")
            print(f"  Std relative difference: {np.nanstd(rel_diff):.4f}%")
            print(
                f"  Max relative difference: {np.nanmax(np.abs(rel_diff)):.4f}%")
        else:
            print(f"\n{method_name}: All values are NaN")

    # Calculate correlation coefficients
    print(f"\nCorrelation Analysis:")
    if not np.all(np.isnan(rk_cps)) and not np.all(np.isnan(em_cps)):
        rk_em_corr = np.corrcoef(
            rk_cps[~np.isnan(rk_cps)], em_cps[~np.isnan(em_cps)])[0, 1]
        print(f"  Runge-Kutta vs Euler-Maruyama: {rk_em_corr:.6f}")

    if not np.all(np.isnan(rk_cps)) and not np.all(np.isnan(st_cps)):
        rk_st_corr = np.corrcoef(
            rk_cps[~np.isnan(rk_cps)], st_cps[~np.isnan(st_cps)])[0, 1]
        print(f"  Runge-Kutta vs Strong-Taylor: {rk_st_corr:.6f}")

    if not np.all(np.isnan(em_cps)) and not np.all(np.isnan(st_cps)):
        em_st_corr = np.corrcoef(
            em_cps[~np.isnan(em_cps)], st_cps[~np.isnan(st_cps)])[0, 1]
        print(f"  Euler-Maruyama vs Strong-Taylor: {em_st_corr:.6f}")

    # Print detailed results table
    print(f"\n{'='*100}")
    print("DETAILED RESULTS TABLE")
    print(f"{'='*100}")
    print(f"{'Fission':<8} {'Alpha_Inv':<12} {'Theoretical':<12} {'Runge-Kutta':<12} {'Euler-Maruyama':<15} {'Strong-Taylor':<15}")
    print(f"{'-'*100}")
    for i, fission in enumerate(fission_vec):
        print(
            f"{fission:<8.3f} {alpha_inv_vec[i]:<12.6f} {theoretical_cps[i]:<12.6f} {rk_cps[i]:<12.6f} {em_cps[i]:<15.6f} {st_cps[i]:<15.6f}")

    print("=" * 80)


if __name__ == "__main__":
    # Run the comparison
    results = run_method_comparison()
    '''
    # Save results to file
    np.savez('updated_method_comparison_results.npz',
             fission_vec=results['fission_vec'],
             alpha_inv_vec=results['alpha_inv_vec'],
             rk_cps=results['rk_cps'],
             em_cps=results['em_cps'],
             st_cps=results['st_cps'],
             theoretical_cps=results['theoretical_cps'])
    '''

    print("\nResults saved to 'updated_method_comparison_results.npz'")
    print("Updated comparison complete!")
