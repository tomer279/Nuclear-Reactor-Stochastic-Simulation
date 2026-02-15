import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import time

from src.models.core_parameters import (
    RateConstants,
    FissionDistribution,
    TimeParameters,
    PhysicalParameters)
from src.models.simulation_setting import ProgressSettings, SimulationControl
from src.core.stochastic_simulation import StochasticSimulator, SimulationParameters
from src.services.data_management import DataManager
from src.utils.utils import calculate_system_parameters

# Set random seed for reproducibility
np.random.seed(42)


class ConvergenceAnalyzer:
    """
    Analyzer for stochastic simulation convergence to theoretical values.

    This class runs multiple stochastic simulations and analyzes how well
    the empirical statistics match the theoretical expected value and variance
    for neutron population dynamics.
    """

    def __init__(self,
                 num_simulations: int = 1000,
                 steps_per_simulation: int = 100000,
                 fission_rate: float = 33.95):
        """
        Initialize convergence analyzer.

        Parameters
        ----------
        num_simulations : int
            Number of independent simulations to run
        steps_per_simulation : int
            Number of time steps per simulation
        fission_rate : float
            Fission rate constant for the analysis
        """
        self.num_simulations = num_simulations
        self.steps_per_simulation = steps_per_simulation
        self.fission_rate = fission_rate

        # Initialize physical parameters
        rate_constants = RateConstants(
            absorb=7.0,
            detect=10.0,
            source=1000.0)
        fission_distribution = FissionDistribution(p_v=[1/6, 1/3, 1/3, 1/6])

        self.physical_params = PhysicalParameters(
            rate_constants=rate_constants,
            fission_distribution=fission_distribution
        )

        self.physical_params.set_fission(fission_rate)

        # Calculate theoretical parameters
        self.theoretical_params = calculate_system_parameters(
            self.physical_params.p_v,
            self.physical_params.fission,
            self.physical_params.absorb,
            self.physical_params.source,
            self.physical_params.detect
        )

        # Time parameters
        self.time_params = TimeParameters(
            t_0=0.0,
            t_end=0.1,  # 0.1 seconds simulation time
            steps=steps_per_simulation
        )

        progress_settings = ProgressSettings(
            show_progress=True, progress_interval=10)
        self.control_params = SimulationControl(
            progress_settings=progress_settings,
            save_results=False
        )

        # Storage for results
        self.population_matrices = []
        self.time_matrices = []

    def run_convergence_analysis(self) -> Dict:
        """
        Run the complete convergence analysis.

        Returns
        -------
        Dict
            Analysis results including empirical and theoretical statistics
        """
        print(f"Running {self.num_simulations} stochastic simulations...")
        print(f"Each simulation has {self.steps_per_simulation:,} time steps")
        print(f"Fission rate: {self.fission_rate}")
        print(f"Expected equilibrium: {
              self.theoretical_params['equilibrium']:.2f}")
        print("-" * 60)

        start_time = time.time()

        # Run simulations
        for i in range(self.num_simulations):
            # Run single simulation
            print(f"Starting simulation {i + 1}/{self.num_simulations}...")
            pop_matrix, time_matrix = self._run_single_simulation()
            self.population_matrices.append(pop_matrix)
            self.time_matrices.append(time_matrix)

            # Print completion message
            elapsed = time.time() - start_time
            print(f"✓ Completed simulation {i + 1}/{self.num_simulations} "
                  f"({elapsed:.1f}s elapsed)")

        total_time = time.time() - start_time
        print(f"\nAll simulations completed in {total_time:.1f} seconds")

        # Analyze results
        return self._analyze_results()

    def _run_single_simulation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single stochastic simulation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Population and time matrices from the simulation
        """
        # Set initial population to equilibrium
        n_0 = np.array([self.theoretical_params['equilibrium']])

        # Create simulation parameters
        sim_params = SimulationParameters(
            physical_params=self.physical_params,
            time_params=self.time_params,
            n_0=n_0,
            control_params=self.control_params
        )

        # Create simulator
        simulator = StochasticSimulator(sim_params)

        # Create dummy data manager (we won't save results)
        data_manager = DataManager("temp_convergence")

        # Run simulation
        time_matrix, pop_matrix, _ = simulator.run_simulation(data_manager)

        return pop_matrix[0], time_matrix[0]  # Single trajectory

    def _analyze_results(self) -> Dict:
        """
        Analyze simulation results and compare to theoretical values.

        Returns
        -------
        Dict
            Analysis results with empirical and theoretical statistics
        """
        print("\nAnalyzing convergence results...")

        # Convert to numpy arrays for easier analysis
        # Shape: (num_sims, steps)
        pop_matrices = np.array(self.population_matrices)
        # Shape: (num_sims, steps)
        time_matrices = np.array(self.time_matrices)

        # Calculate empirical statistics
        # Mean across simulations
        empirical_mean = np.mean(pop_matrices, axis=0)
        # Variance across simulations
        empirical_var = np.var(pop_matrices, axis=0)
        # Std dev across simulations
        empirical_std = np.std(pop_matrices, axis=0)

        # Calculate theoretical values
        time_points = time_matrices[0]  # All simulations have same time points
        theoretical_mean = np.full_like(
            time_points, self.theoretical_params['equilibrium'])

        # Theoretical variance: (σ₁² + σ₂²)/(2α) × (1 - e^(-2αt))
        alpha = self.theoretical_params['alpha']
        sig_1_sq = self.theoretical_params['sig_1_squared']
        sig_2_sq = self.theoretical_params['sig_2_squared']

        theoretical_var = ((sig_1_sq + sig_2_sq) / (2 * alpha)) * \
            (1 - np.exp(-2 * alpha * time_points))
        theoretical_std = np.sqrt(theoretical_var)

        # Calculate convergence metrics
        mean_error = np.abs(empirical_mean - theoretical_mean)
        var_error = np.abs(empirical_var - theoretical_var)
        std_error = np.abs(empirical_std - theoretical_std)

        # Final values (at end of simulation)
        final_empirical_mean = empirical_mean[-1]
        final_theoretical_mean = theoretical_mean[-1]
        final_empirical_var = empirical_var[-1]
        final_theoretical_var = theoretical_var[-1]

        results = {
            'time_points': time_points,
            'empirical_mean': empirical_mean,
            'theoretical_mean': theoretical_mean,
            'empirical_variance': empirical_var,
            'theoretical_variance': theoretical_var,
            'empirical_std': empirical_std,
            'theoretical_std': theoretical_std,
            'mean_error': mean_error,
            'variance_error': var_error,
            'std_error': std_error,
            'final_empirical_mean': final_empirical_mean,
            'final_theoretical_mean': final_theoretical_mean,
            'final_empirical_var': final_empirical_var,
            'final_theoretical_var': final_theoretical_var,
            'final_mean_error': mean_error[-1],
            'final_variance_error': var_error[-1],
            'theoretical_params': self.theoretical_params,
            'num_simulations': self.num_simulations,
            'steps_per_simulation': self.steps_per_simulation
        }

        return results

    def plot_convergence(self, results: Dict, save_plots: bool = True):
        """
        Create comprehensive convergence plots.

        Parameters
        ----------
        results : Dict
            Analysis results from _analyze_results()
        save_plots : bool
            Whether to save plots to files
        """
        print("\nCreating convergence plots...")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Stochastic Simulation Convergence Analysis\n'
                     f'{results["num_simulations"]:,} simulations, '
                     f'{results["steps_per_simulation"]:,} steps each, '
                     f'Fission rate: {self.fission_rate}',
                     fontsize=14, fontweight='bold')

        time_points = results['time_points']

        # Plot 1: Mean convergence
        ax1 = axes[0, 0]
        ax1.plot(time_points, results['empirical_mean'], 'b-',
                 label='Empirical Mean', linewidth=2)
        ax1.plot(time_points, results['theoretical_mean'], 'r--',
                 label='Theoretical Mean (S/α)', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Population')
        ax1.set_title('Mean Population Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add text box with final values
        textstr = (f'Final Empirical: {results["final_empirical_mean"]:.2f}\n'
                   f'Final Theoretical: {
                       results["final_theoretical_mean"]:.2f}\n'
                   f'Error: {results["final_mean_error"]:.4f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # Plot 2: Variance convergence
        ax2 = axes[0, 1]
        ax2.plot(time_points, results['empirical_variance'], 'b-',
                 label='Empirical Variance', linewidth=2)
        ax2.plot(time_points, results['theoretical_variance'], 'r--',
                 label='Theoretical Variance', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Variance')
        ax2.set_title('Variance Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add text box with final values
        textstr = (f'Final Empirical: {results["final_empirical_var"]:.2f}\n'
                   f'Final Theoretical: {
                       results["final_theoretical_var"]:.2f}\n'
                   f'Error: {results["final_variance_error"]:.4f}')
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # Plot 3: Error evolution
        ax3 = axes[1, 0]
        ax3.semilogy(time_points, results['mean_error'], 'b-',
                     label='Mean Error', linewidth=2)
        ax3.semilogy(time_points, results['variance_error'], 'r-',
                     label='Variance Error', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Absolute Error (log scale)')
        ax3.set_title('Error Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Standard deviation comparison
        ax4 = axes[1, 1]
        ax4.plot(time_points, results['empirical_std'], 'b-',
                 label='Empirical Std Dev', linewidth=2)
        ax4.plot(time_points, results['theoretical_std'], 'r--',
                 label='Theoretical Std Dev', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Standard Deviation Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            filename = f'convergence_analysis_{
                self.fission_rate}_{self.num_simulations}sims.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plots saved as: {filename}")

        plt.show()

        # Create additional detailed plot
        self._plot_detailed_convergence(results, save_plots)

    def _plot_detailed_convergence(self, results: Dict, save_plots: bool = True):
        """
        Create detailed convergence plot showing individual trajectories.

        Parameters
        ----------
        results : Dict
            Analysis results from _analyze_results()
        save_plots : bool
            Whether to save plots to files
        """
        # Sample a few trajectories for visualization
        num_sample_trajectories = min(50, self.num_simulations)
        sample_indices = np.random.choice(self.num_simulations,
                                          num_sample_trajectories,
                                          replace=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        time_points = results['time_points']

        # Plot 1: Sample trajectories with mean
        ax1.set_title(
            f'Sample Trajectories ({num_sample_trajectories} of {self.num_simulations})')
        for i, idx in enumerate(sample_indices):
            alpha_val = 0.1 if i < 10 else 0.05  # More transparent for later trajectories
            ax1.plot(time_points, self.population_matrices[idx],
                     'b-', alpha=alpha_val, linewidth=0.5)

        # Plot empirical and theoretical means
        ax1.plot(time_points, results['empirical_mean'], 'r-',
                 label='Empirical Mean', linewidth=3)
        ax1.plot(time_points, results['theoretical_mean'], 'k--',
                 label='Theoretical Mean (S/α)', linewidth=3)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Population')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Confidence intervals
        ax2.set_title('Empirical vs Theoretical Confidence Intervals')

        # Empirical confidence intervals (±2σ)
        empirical_upper = results['empirical_mean'] + \
            2 * results['empirical_std']
        empirical_lower = results['empirical_mean'] - \
            2 * results['empirical_std']

        # Theoretical confidence intervals (±2σ)
        theoretical_upper = results['theoretical_mean'] + \
            2 * results['theoretical_std']
        theoretical_lower = results['theoretical_mean'] - \
            2 * results['theoretical_std']

        ax2.fill_between(time_points, empirical_lower, empirical_upper,
                         alpha=0.3, color='blue', label='Empirical ±2σ')
        ax2.fill_between(time_points, theoretical_lower, theoretical_upper,
                         alpha=0.3, color='red', label='Theoretical ±2σ')

        ax2.plot(time_points, results['empirical_mean'], 'b-',
                 label='Empirical Mean', linewidth=2)
        ax2.plot(time_points, results['theoretical_mean'], 'r--',
                 label='Theoretical Mean', linewidth=2)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Population')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            filename = f'detailed_convergence_{
                self.fission_rate}_{self.num_simulations}sims.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Detailed plots saved as: {filename}")

        plt.show()

    def print_summary(self, results: Dict):
        """
        Print comprehensive summary of convergence analysis.

        Parameters
        ----------
        results : Dict
            Analysis results from _analyze_results()
        """
        print("\n" + "="*80)
        print("CONVERGENCE ANALYSIS SUMMARY")
        print("="*80)

        params = results['theoretical_params']

        print(f"Simulation Parameters:")
        print(f"  Number of simulations: {results['num_simulations']:,}")
        print(f"  Steps per simulation: {results['steps_per_simulation']:,}")
        print(f"  Fission rate: {self.fission_rate}")
        print(f"  Simulation duration: {
              results['time_points'][-1]:.3f} seconds")

        print(f"\nPhysical Parameters:")
        print(f"  Source rate (S): {self.physical_params.source}")
        print(f"  Absorption rate: {self.physical_params.absorb}")
        print(f"  Detection rate: {self.physical_params.detect}")
        print(f"  Fission rate: {self.physical_params.fission}")
        print(f"  Expected neutron yield (vbar): {params['vbar']:.3f}")
        print(f"  Rossi-alpha (α): {params['alpha']:.6f}")

        print(f"\nTheoretical Values:")
        print(f"  Equilibrium population (S/α): {params['equilibrium']:.2f}")
        print(f"  σ₁² (population noise): {params['sig_1_squared']:.2f}")
        print(f"  σ₂² (detection noise): {params['sig_2_squared']:.2f}")
        print(f"  Final theoretical variance: {
              results['final_theoretical_var']:.2f}")

        print(f"\nEmpirical Results:")
        print(f"  Final empirical mean: {results['final_empirical_mean']:.2f}")
        print(f"  Final empirical variance: {
              results['final_empirical_var']:.2f}")

        print(f"\nConvergence Metrics:")
        print(f"  Mean error: {results['final_mean_error']:.4f}")
        print(f"  Variance error: {results['final_variance_error']:.4f}")
        print(f"  Relative mean error: {
              results['final_mean_error']/params['equilibrium']*100:.2f}%")
        print(f"  Relative variance error: {
              results['final_variance_error']/results['final_theoretical_var']*100:.2f}%")

        print(f"\nConvergence Assessment:")
        mean_converged = results['final_mean_error'] < 0.01 * \
            params['equilibrium']
        var_converged = results['final_variance_error'] < 0.01 * \
            results['final_theoretical_var']

        print(f"  Mean convergence: {
              '✓ PASSED' if mean_converged else '✗ FAILED'}")
        print(f"  Variance convergence: {
              '✓ PASSED' if var_converged else '✗ FAILED'}")

        print("="*80)


def main():
    """
    Main function to run the convergence analysis.
    """
    print("Stochastic Neutron Population Convergence Analysis")
    print("=" * 60)

    # Create analyzer
    analyzer = ConvergenceAnalyzer(
        num_simulations=1000,
        steps_per_simulation=100000,
        fission_rate=33.95
    )

    # Run analysis
    results = analyzer.run_convergence_analysis()

    # Create plots
    analyzer.plot_convergence(results, save_plots=True)

    # Print summary
    analyzer.print_summary(results)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
