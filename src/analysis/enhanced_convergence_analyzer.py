"""Written by Tomer279 with the assistance of Cursor.ai.

Enhanced Convergence Analysis for Stochastic Neutron Population and Detection Count.

This script extends the convergence analysis to validate both:
1. Neutron population convergence to S/α and theoretical variance
2. Detection count convergence to λ_d(S/α)t and theoretical detection variance

The detection analysis uses the equations:
- Expected value: E[D_t] = λ_d * (S/α) * t
- Variance: Var(D_t) = ((v̄² - v̄λ_f)/α³) * λ_d²St * (1 - (1-e^(-αt))/(αt)) + (λ_dSt)/α
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import time

from src.models.core_parameters import (
    PhysicalParameters,
    TimeParameters,
    RateConstants,
    FissionDistribution)
from src.models.simulation_setting import SimulationControl, ProgressSettings
from src.core.stochastic_simulation import StochasticSimulator, SimulationParameters
from src.services.data_management import DataManager
from src.utils.utils import calculate_system_parameters
from src.utils import utils as utl

# Set random seed for reproducibility
np.random.seed(42)


class EnhancedConvergenceAnalyzer:
    """
    Enhanced analyzer for both population and detection count convergence.

    This class runs multiple stochastic simulations and analyzes convergence
    for both neutron population and detection count against their respective
    theoretical values.
    """

    def __init__(self,
                 num_simulations: int = 1000,
                 steps_per_simulation: int = 100000,
                 fission_rate: float = 33.95):
        """
        Initialize enhanced convergence analyzer.

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

        # Control parameters (with progress enabled)
        progress_settings = ProgressSettings(
            show_progress=True, progress_interval=10)
        self.control_params = SimulationControl(
            progress_settings=progress_settings,
            save_results=False
        )

        # Storage for results
        self.population_matrices = []
        self.time_matrices = []
        self.detection_matrices = []

    def run_convergence_analysis(self) -> Dict:
        """
        Run the complete convergence analysis for both population and detection.

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
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Completed {i + 1}/{self.num_simulations} simulations "
                      f"({elapsed:.1f}s elapsed)")

            # Run single simulation
            pop_matrix, time_matrix, det_matrix = self._run_single_simulation()
            self.population_matrices.append(pop_matrix)
            self.time_matrices.append(time_matrix)
            self.detection_matrices.append(det_matrix)

        total_time = time.time() - start_time
        print(f"\nAll simulations completed in {total_time:.1f} seconds")

        # Analyze results
        return self._analyze_results()

    def _run_single_simulation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a single stochastic simulation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Population, time, and detection matrices from the simulation
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
        time_matrix, pop_matrix, det_matrix = simulator.run_simulation(
            data_manager)

        # Single trajectory
        return pop_matrix[0], time_matrix[0], det_matrix[0]

    def _analyze_results(self) -> Dict:
        """
        Analyze simulation results for both population and detection count.
        """
        print("\nAnalyzing convergence results...")

        # Convert to numpy arrays for easier analysis
        # Shape: (num_sims, steps)
        pop_matrices = np.array(self.population_matrices)
        # Shape: (num_sims, steps)
        time_matrices = np.array(self.time_matrices)

        # Handle detection matrices - they may have different shapes
        print("Processing detection data...")
        det_matrices = self._create_cumulative_detection_matrices()
        time_points = time_matrices[0]  # All simulations have same time points

        # Calculate empirical statistics for population
        empirical_pop_mean = np.mean(pop_matrices, axis=0)
        empirical_pop_var = np.var(pop_matrices, axis=0)
        empirical_pop_std = np.std(pop_matrices, axis=0)

        # Calculate empirical statistics for detection count
        empirical_det_mean = np.mean(det_matrices, axis=0)
        empirical_det_var = np.var(det_matrices, axis=0)
        empirical_det_std = np.sqrt(empirical_det_var)

        # Population theoretical values
        theoretical_pop_mean = np.full_like(
            time_points, self.theoretical_params['equilibrium'])
        alpha = self.theoretical_params['alpha']
        sig_1_sq = self.theoretical_params['sig_1_squared']
        sig_2_sq = self.theoretical_params['sig_2_squared']
        theoretical_pop_var = (
            (sig_1_sq + sig_2_sq) / (2 * alpha)) * (1 - np.exp(-2 * alpha * time_points))
        theoretical_pop_std = np.sqrt(theoretical_pop_var)

        # Detection theoretical values
        theoretical_det_mean, theoretical_det_var = (
            self._calculate_theoretical_detection(time_points)
        )
        theoretical_det_std = np.sqrt(theoretical_det_var)

        # Calculate convergence metrics
        pop_mean_error = np.abs(empirical_pop_mean - theoretical_pop_mean)
        pop_var_error = np.abs(empirical_pop_var - theoretical_pop_var)
        det_mean_error = np.abs(empirical_det_mean - theoretical_det_mean)
        det_var_error = np.abs(empirical_det_var - theoretical_det_var)

        # Final values (at end of simulation)
        final_pop_empirical_mean = empirical_pop_mean[-1]
        final_pop_theoretical_mean = theoretical_pop_mean[-1]
        final_pop_empirical_var = empirical_pop_var[-1]
        final_pop_theoretical_var = theoretical_pop_var[-1]

        final_det_empirical_mean = empirical_det_mean[-1]
        final_det_theoretical_mean = theoretical_det_mean[-1]
        final_det_empirical_var = empirical_det_var[-1]
        final_det_theoretical_var = theoretical_det_var[-1]

        # Within-simulation variance analysis
        print("\nWithin-simulation variance analysis:")
        print(f"Theoretical instantaneous variance: {
              theoretical_det_var[-1]:.2f}")

        results = {
            'time_points': time_points,

            # Population results
            'empirical_pop_mean': empirical_pop_mean,
            'theoretical_pop_mean': theoretical_pop_mean,
            'empirical_pop_variance': empirical_pop_var,
            'theoretical_pop_variance': theoretical_pop_var,
            'empirical_pop_std': empirical_pop_std,
            'theoretical_pop_std': theoretical_pop_std,
            'pop_mean_error': pop_mean_error,
            'pop_variance_error': pop_var_error,
            'final_pop_empirical_mean': final_pop_empirical_mean,
            'final_pop_theoretical_mean': final_pop_theoretical_mean,
            'final_pop_empirical_var': final_pop_empirical_var,
            'final_pop_theoretical_var': final_pop_theoretical_var,
            'final_pop_mean_error': pop_mean_error[-1],
            'final_pop_variance_error': pop_var_error[-1],
            'empirical_det_mean': empirical_det_mean,
            'theoretical_det_mean': theoretical_det_mean,
            'empirical_det_variance': empirical_det_var,
            'theoretical_det_variance': theoretical_det_var,
            'empirical_det_std': empirical_det_std,
            'theoretical_det_std': theoretical_det_std,
            'det_mean_error': det_mean_error,
            'det_variance_error': det_var_error,
            'final_det_empirical_mean': final_det_empirical_mean,
            'final_det_theoretical_mean': final_det_theoretical_mean,
            'final_det_empirical_var': final_det_empirical_var,  # Within-simulation variance
            'final_det_theoretical_var': final_det_theoretical_var,
            'final_det_mean_error': det_mean_error[-1],
            'final_det_variance_error': det_var_error[-1],
            'theoretical_params': self.theoretical_params,
            'num_simulations': self.num_simulations,
            'steps_per_simulation': self.steps_per_simulation,
            'detection_matrices': det_matrices
        }

        return results

    def _create_cumulative_detection_matrices(self):

        det_matrices = []
        for i, det_matrix in enumerate(self.detection_matrices):
            # Clean the detection matrix and convert to cumulative counts
            clean_det = utl.clean_detection_matrix(
                det_matrix.reshape(1, -1))[0]

            # Create cumulative detection count over time
            cumulative_detections = np.zeros_like(self.time_matrices[i])

            # Count detections up to each time point
            for j, time_point in enumerate(self.time_matrices[i]):
                # Count how many detections occurred up to this time
                detections_up_to_time = np.sum(clean_det <= time_point)
                cumulative_detections[j] = detections_up_to_time

            det_matrices.append(cumulative_detections)

        return np.array(det_matrices)

        # Convert to numpy array

    def _calculate_theoretical_detection(
            self,
            time_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate theoretical detection count mean and variance.

        Parameters
        ----------
        time_points : np.ndarray
            Time points for calculation

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Theoretical mean and variance arrays
        """
        # Extract parameters
        lambda_d = self.physical_params.detect  # Detection rate
        S = self.physical_params.source         # Source rate
        alpha = self.theoretical_params['alpha']  # Rossi-alpha
        vbar = self.theoretical_params['vbar']    # Expected neutron yield
        vbar_square = self.theoretical_params['vbar_square']
        lambda_f = self.physical_params.fission   # Fission rate

        # Expected value: E[D_t] = λ_d * (S/α) * t
        theoretical_mean = lambda_d * (S / alpha) * time_points

        sig_1_sq = self.theoretical_params['sig_1_squared']
        sig_2_sq = self.theoretical_params['sig_2_squared']

        # Debug: Print parameter values
        print("\nDEBUGGING THEORETICAL DETECTION VARIANCE:")
        print(f"lambda_d: {lambda_d}")
        print(f"S: {S}")
        print(f"alpha: {alpha}")
        print(f"vbar: {vbar}")
        print(f"lambda_f: {lambda_f}")
        print(f"sig_1_squared: {sig_1_sq}")
        print(f"sig_2_squared: {sig_2_sq}")
        print(f"Final time: {time_points[-1]}")

        term1 = ((vbar_square - vbar) * lambda_f *
                 (lambda_d ** 2) * S) / (alpha ** 3)

        # Term 2: (1 - (1-e^(-αt))/(αt))
        term2 = (alpha * time_points +
                 np.exp(-alpha * time_points) - 1) / alpha

        # Final variance
        theoretical_var = term1 * term2 + theoretical_mean

        print(f"Final term1: {term1}")
        print(f"Final term2: {term2}")
        print(f"Final theoretical_var: {theoretical_var[-1]}")

        return theoretical_mean, theoretical_var

    def plot_convergence(self, results: Dict, save_plots: bool = True):
        """
        Create separate convergence plots for population and detection.

        Parameters
        ----------
        results : Dict
            Analysis results from _analyze_results()
        save_plots : bool
            Whether to save plots to files
        """
        print("\nCreating convergence plots...")

        # Create separate plots for population and detection
        self._plot_population_convergence_separate(results, save_plots)
        self._plot_detection_convergence_separate(results, save_plots)

        # Create additional detailed plots
        self._plot_detailed_convergence(results, save_plots)

    def _plot_population_convergence_separate(self, results: Dict, save_plots: bool = True):
        """Create separate population convergence plot."""
        print("Creating population convergence plot...")

        # Create figure with subplots (2x2 layout for population)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Population Convergence Analysis\n'
                     f'{results["num_simulations"]:,} simulations, '
                     f'{results["steps_per_simulation"]:,} steps each, '
                     f'Fission rate: {self.fission_rate}',
                     fontsize=14, fontweight='bold')

        time_points = results['time_points']

        # Plot 1: Population Mean Convergence
        ax1 = axes[0, 0]
        ax1.plot(time_points, results['empirical_pop_mean'], 'b-',
                 label='Empirical Mean', linewidth=2)
        ax1.plot(time_points, results['theoretical_pop_mean'], 'r--',
                 label='Theoretical Mean (S/α)', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Population')
        ax1.set_title('Mean Population Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add text box with final values
        textstr = (f'Final Empirical: {results["final_pop_empirical_mean"]:.2f}\n'
                   f'Final Theoretical: {
                       results["final_pop_theoretical_mean"]:.2f}\n'
                   f'Error: {results["final_pop_mean_error"]:.4f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # Plot 2: Population Variance Convergence
        ax2 = axes[0, 1]
        ax2.plot(time_points, results['empirical_pop_variance'], 'b-',
                 label='Empirical Variance', linewidth=2)
        ax2.plot(time_points, results['theoretical_pop_variance'], 'r--',
                 label='Theoretical Variance', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Variance')
        ax2.set_title('Population Variance Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add text box with final values
        textstr = (f'Final Empirical: {results["final_pop_empirical_var"]:.2f}\n'
                   f'Final Theoretical: {
                       results["final_pop_theoretical_var"]:.2f}\n'
                   f'Error: {results["final_pop_variance_error"]:.4f}')
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # Plot 3: Population Error Evolution
        ax3 = axes[1, 0]
        ax3.semilogy(time_points, results['pop_mean_error'], 'b-',
                     label='Mean Error', linewidth=2)
        ax3.semilogy(time_points, results['pop_variance_error'], 'r-',
                     label='Variance Error', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Absolute Error (log scale)')
        ax3.set_title('Population Error Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Population Standard Deviation Comparison
        ax4 = axes[1, 1]
        ax4.plot(time_points, results['empirical_pop_std'], 'b-',
                 label='Empirical Std Dev', linewidth=2)
        ax4.plot(time_points, results['theoretical_pop_std'], 'r--',
                 label='Theoretical Std Dev', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Population Standard Deviation Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            filename = f'population_convergence_analysis_{
                self.fission_rate}_{self.num_simulations}sims.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Population plots saved as: {filename}")

        plt.show()

    def _plot_detection_convergence_separate(
            self,
            results: Dict,
            save_plots: bool = True):
        """Create separate detection convergence plot."""
        print("Creating detection convergence plot...")

        # Create figure with subplots (2x2 layout for detection)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Detection Count Convergence Analysis\n'
                     f'{results["num_simulations"]:,} simulations, '
                     f'{results["steps_per_simulation"]:,} steps each, '
                     f'Fission rate: {self.fission_rate}',
                     fontsize=14, fontweight='bold')

        time_points = results['time_points']

        # Plot 1: Detection Mean Convergence
        ax1 = axes[0, 0]
        ax1.plot(time_points, results['empirical_det_mean'], 'g-',
                 label='Empirical Mean', linewidth=2)
        ax1.plot(time_points, results['theoretical_det_mean'], 'm--',
                 label='Theoretical Mean (λ_d(S/α)t)', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Detection Count')
        ax1.set_title('Mean Detection Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add text box with final values
        textstr = (f'Final Empirical: {results["final_det_empirical_mean"]:.2f}\n'
                   f'Final Theoretical: {
                       results["final_det_theoretical_mean"]:.2f}\n'
                   f'Error: {results["final_det_mean_error"]:.4f}')
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # Plot 2: Detection Variance Convergence
        ax2 = axes[0, 1]
        ax2.plot(time_points, results['empirical_det_variance'], 'g-',
                 label='Empirical Variance', linewidth=2)
        ax2.plot(time_points, results['theoretical_det_variance'], 'm--',
                 label='Theoretical Variance', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Variance')
        ax2.set_title('Detection Variance Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add text box with final values
        textstr = (f'Final Empirical: {results["final_det_empirical_var"]:.2f}\n'
                   f'Final Theoretical: {
                       results["final_det_theoretical_var"]:.2f}\n'
                   f'Error: {results["final_det_variance_error"]:.4f}')
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # Plot 3: Detection Error Evolution
        ax3 = axes[1, 0]
        ax3.semilogy(time_points, results['det_mean_error'], 'g-',
                     label='Mean Error', linewidth=2)
        ax3.semilogy(time_points, results['det_variance_error'], 'm-',
                     label='Variance Error', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Absolute Error (log scale)')
        ax3.set_title('Detection Error Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Detection Standard Deviation Comparison
        ax4 = axes[1, 1]
        ax4.plot(time_points, results['empirical_det_std'], 'g-',
                 label='Empirical Std Dev', linewidth=2)
        ax4.plot(time_points, results['theoretical_det_std'], 'm--',
                 label='Theoretical Std Dev', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Detection Standard Deviation Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            filename = f'detection_convergence_analysis_{
                self.fission_rate}_{self.num_simulations}sims.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Detection plots saved as: {filename}")

        plt.show()

    def _plot_detailed_convergence(self, results: Dict, save_plots: bool = True):
        """
        Create separate detailed convergence plots for population and detection.
        """
        print("Creating detailed convergence plots...")

        # Sample a few trajectories for visualization
        num_sample_trajectories = min(50, self.num_simulations)
        sample_indices = np.random.choice(self.num_simulations,
                                          num_sample_trajectories,
                                          replace=False)

        time_points = results['time_points']
        det_matrices = results['detection_matrices']

        # Create separate detailed plots
        self._plot_population_detailed(
            time_points, results, sample_indices, save_plots)
        self._plot_detection_detailed(
            time_points, results, det_matrices, sample_indices, save_plots)

    def _plot_population_detailed(self, time_points: np.ndarray, results: Dict,
                                  sample_indices: np.ndarray, save_plots: bool = True):
        """Create detailed population convergence plot."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Sample Population Trajectories
        ax1.set_title(f'Sample Population Trajectories ({
                      len(sample_indices)} of {self.num_simulations})')
        for i, idx in enumerate(sample_indices):
            alpha_val = 0.1 if i < 10 else 0.05
            ax1.plot(time_points, self.population_matrices[idx],
                     'b-', alpha=alpha_val, linewidth=0.5)

        ax1.plot(time_points, results['empirical_pop_mean'], 'r-',
                 label='Empirical Mean', linewidth=3)
        ax1.plot(time_points, results['theoretical_pop_mean'], 'k--',
                 label='Theoretical Mean (S/α)', linewidth=3)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Population')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Population Confidence Intervals
        ax2.set_title('Population Confidence Intervals')

        # Population confidence intervals (±2σ)
        pop_empirical_upper = results['empirical_pop_mean'] + \
            2 * results['empirical_pop_std']
        pop_empirical_lower = results['empirical_pop_mean'] - \
            2 * results['empirical_pop_std']
        pop_theoretical_upper = results['theoretical_pop_mean'] + \
            2 * results['theoretical_pop_std']
        pop_theoretical_lower = results['theoretical_pop_mean'] - \
            2 * results['theoretical_pop_std']

        ax2.fill_between(time_points, pop_empirical_lower, pop_empirical_upper,
                         alpha=0.3, color='blue', label='Empirical ±2σ')
        ax2.fill_between(time_points, pop_theoretical_lower, pop_theoretical_upper,
                         alpha=0.3, color='red', label='Theoretical ±2σ')

        ax2.plot(time_points, results['empirical_pop_mean'], 'b-',
                 label='Empirical Mean', linewidth=2)
        ax2.plot(time_points, results['theoretical_pop_mean'], 'r--',
                 label='Theoretical Mean', linewidth=2)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Population')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            filename = f'detailed_population_convergence_{
                self.fission_rate}_{self.num_simulations}sims.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Detailed population plots saved as: {filename}")

        plt.show()

    def _plot_detection_detailed(self, time_points: np.ndarray, results: Dict,
                                 det_matrices: np.ndarray, sample_indices: np.ndarray,
                                 save_plots: bool = True):
        """Create detailed detection convergence plot."""

        print("\nDEBUGGING PLOT CONFIDENCE INTERVALS:")
        print(f"Final empirical mean: {results['empirical_det_mean'][-1]:.2f}")
        print(f"Final empirical std: {results['empirical_det_std'][-1]:.2f}")

        # Calculate confidence intervals
        # In your plotting function, modify the confidence interval calculation:
        det_empirical_upper = results['empirical_det_mean'] + \
            2 * results['empirical_det_std']
        det_empirical_lower = results['empirical_det_mean'] - \
            2 * results['empirical_det_std']

        # Clip negative values to 0
        det_empirical_lower = np.maximum(det_empirical_lower, 0)

        print(f"Final empirical upper bound: {det_empirical_upper[-1]:.2f}")
        print(f"Final empirical lower bound: {det_empirical_lower[-1]:.2f}")
        print(f"Final empirical range width: {
              det_empirical_upper[-1] - det_empirical_lower[-1]:.2f}")

        # Check if there are any extreme values
        print(f"Max empirical upper: {np.max(det_empirical_upper):.2f}")
        print(f"Min empirical lower: {np.min(det_empirical_lower):.2f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Sample Detection Trajectories
        ax1.set_title(f'Sample Detection Trajectories ({
                      len(sample_indices)} of {self.num_simulations})')
        for i, idx in enumerate(sample_indices):
            alpha_val = 0.1 if i < 10 else 0.05
            ax1.plot(time_points, det_matrices[idx],
                     'g-', alpha=alpha_val, linewidth=0.5)

        ax1.plot(time_points, results['empirical_det_mean'], 'r-',
                 label='Empirical Mean', linewidth=3)
        ax1.plot(time_points, results['theoretical_det_mean'], 'k--',
                 label='Theoretical Mean (λ_d(S/α)t)', linewidth=3)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Detection Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Detection Confidence Intervals
        ax2.set_title('Detection Confidence Intervals')

        # Detection confidence intervals (±2σ)
        det_empirical_upper = results['empirical_det_mean'] + \
            2 * results['empirical_det_std']
        det_empirical_lower = results['empirical_det_mean'] - \
            2 * results['empirical_det_std']
        det_theoretical_upper = results['theoretical_det_mean'] + \
            2 * results['theoretical_det_std']
        det_theoretical_lower = results['theoretical_det_mean'] - \
            2 * results['theoretical_det_std']

        ax2.fill_between(time_points, det_empirical_lower, det_empirical_upper,
                         alpha=0.3, color='green', label='Empirical ±2σ')
        ax2.fill_between(time_points, det_theoretical_lower, det_theoretical_upper,
                         alpha=0.3, color='purple', label='Theoretical ±2σ')

        # Add explicit y-axis limits to focus on the relevant range:
        # 10% above final mean
        ax2.set_ylim([0, results['empirical_det_mean'][-1] * 1.1])

        ax2.plot(time_points, results['empirical_det_mean'], 'g-',
                 label='Empirical Mean', linewidth=2)
        ax2.plot(time_points, results['theoretical_det_mean'], 'm--',
                 label='Theoretical Mean', linewidth=2)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Detection Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            filename = f'detailed_detection_convergence_{
                self.fission_rate}_{self.num_simulations}sims.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Detailed detection plots saved as: {filename}")

        plt.show()

    def print_summary(self, results: Dict):
        """
        Print comprehensive summary of convergence analysis for both population and detection.

        Parameters
        ----------
        results : Dict
            Analysis results from _analyze_results()
        """
        print("\n" + "="*80)
        print("ENHANCED CONVERGENCE ANALYSIS SUMMARY")
        print("="*80)

        params = results['theoretical_params']

        print("Simulation Parameters:")
        print(f"  Number of simulations: {results['num_simulations']:,}")
        print(f"  Steps per simulation: {results['steps_per_simulation']:,}")
        print(f"  Fission rate: {self.fission_rate}")
        print(f"  Simulation duration: {
              results['time_points'][-1]:.3f} seconds")

        print("\nPhysical Parameters:")
        print(f"  Source rate (S): {self.physical_params.source}")
        print(f"  Absorption rate: {self.physical_params.absorb}")
        print(f"  Detection rate (λ_d): {self.physical_params.detect}")
        print(f"  Fission rate (λ_f): {self.physical_params.fission}")
        print(f"  Expected neutron yield (v̄): {params['vbar']:.3f}")
        print(f"  Rossi-alpha (α): {params['alpha']:.6f}")

        print("\n" + "="*50)
        print("POPULATION CONVERGENCE ANALYSIS")
        print("="*50)

        print("Theoretical Values:")
        print(f"  Equilibrium population (S/α): {params['equilibrium']:.2f}")
        print(f"  σ₁² (population noise): {params['sig_1_squared']:.2f}")
        print(f"  σ₂² (detection noise): {params['sig_2_squared']:.2f}")
        print(f"  Final theoretical variance: {
              results['final_pop_theoretical_var']:.2f}")

        print("\nEmpirical Results:")
        print(f"  Final empirical mean: {
              results['final_pop_empirical_mean']:.2f}")
        print(f"  Final empirical variance: {
              results['final_pop_empirical_var']:.2f}")

        print("\nConvergence Metrics:")
        print(f"  Mean error: {results['final_pop_mean_error']:.4f}")
        print(f"  Variance error: {results['final_pop_variance_error']:.4f}")
        print(f"  Relative mean error: {
              results['final_pop_mean_error']/params['equilibrium']*100:.2f}%")
        print(f"  Relative variance error: {
              results['final_pop_variance_error']/results['final_pop_theoretical_var']*100:.2f}%")

        print("\nConvergence Assessment:")
        pop_mean_converged = results['final_pop_mean_error'] < 0.01 * \
            params['equilibrium']
        pop_var_converged = results['final_pop_variance_error'] < 0.01 * \
            results['final_pop_theoretical_var']

        print(f"  Population mean convergence: {
              '✓ PASSED' if pop_mean_converged else '✗ FAILED'}")
        print(f"  Population variance convergence: {
              '✓ PASSED' if pop_var_converged else '✗ FAILED'}")

        print("\n" + "="*50)
        print("DETECTION CONVERGENCE ANALYSIS")
        print("="*50)

        print("Theoretical Values:")
        print(f"  Final theoretical mean: {
              results['final_det_theoretical_mean']:.2f}")
        print(f"  Final theoretical variance: {
              results['final_det_theoretical_var']:.2f}")

        print("\nEmpirical Results:")
        print(f"  Final empirical mean: {
              results['final_det_empirical_mean']:.2f}")
        print(f"  Final empirical variance: {
              results['final_det_empirical_var']:.2f}")

        print("\nConvergence Metrics:")
        print(f"  Mean error: {results['final_det_mean_error']:.4f}")
        print(f"  Variance error: {results['final_det_variance_error']:.4f}")
        print(f"  Relative mean error: {
              results['final_det_mean_error']/results['final_det_theoretical_mean']*100:.2f}%")
        print(f"  Relative variance error: {
              results['final_det_variance_error']/results['final_det_theoretical_var']*100:.2f}%")

        print("\nConvergence Assessment:")
        det_mean_converged = results['final_det_mean_error'] < 0.01 * \
            results['final_det_theoretical_mean']
        det_var_converged = results['final_det_variance_error'] < 0.01 * \
            results['final_det_theoretical_var']

        print(f"  Detection mean convergence: {
              '✓ PASSED' if det_mean_converged else '✗ FAILED'}")
        print(f"  Detection variance convergence: {
              '✓ PASSED' if det_var_converged else '✗ FAILED'}")

        print("="*80)


def main():
    """
    Main function to run the enhanced convergence analysis.
    """
    print("Enhanced Stochastic Neutron Population and Detection Convergence Analysis")
    print("=" * 80)

    # Create analyzer
    analyzer = EnhancedConvergenceAnalyzer(
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

    print("\nEnhanced analysis complete!")


if __name__ == "__main__":
    main()
