"""Written by Tomer279 with the assistance of Cursor.ai.

Weak Convergence Analysis for Numerical SDE Methods.

This module provides comprehensive weak convergence analysis for comparing
the efficiency and accuracy of different numerical methods (Euler-Maruyama,
Taylor 2.0, Runge-Kutta 3.0) used to solve the detection SDE.

The analysis computes weak errors (errors in statistical moments) as a
function of step size, estimates convergence orders, and compares
computational efficiency.

Classes:
    PlottingConfig:
        Configuration container for plotting method labels and colors
    ErrorPlotParams:
        Parameter container for error convergence plots
    ComparisonPlotParams:
        Parameter container for method comparison plots
    PlotCreationParams:
        Parameter container for creating all plots
    ConvergenceAnalysisConfig:
        Configuration parameters for weak convergence analysis
    WeakConvergenceAnalyzer:
        Main analyzer class for weak convergence of numerical SDE methods

Functions:
    main():
        Main function to run weak convergence analysis example

Key Features:
    - Step size refinement studies
    - Weak error computation (mean, variance)
    - Convergence order estimation via linear regression
    - Computational efficiency comparison
    - Publication-ready visualizations

Examples:
    >>> from src.models.core_parameters import (
    ...     PhysicalParameters, DeadTimeParameters, TimeParameters,
    ...     RateConstants, FissionDistribution
    ... )
    >>> physical_params = PhysicalParameters(...)
    >>> dead_time_params = DeadTimeParameters(mean_tau=1e-6)
    >>> time_params = TimeParameters(t_end=0.1)
    >>> config = ConvergenceAnalysisConfig(num_paths=1000)
    >>> analyzer = WeakConvergenceAnalyzer(
    ...     physical_params, dead_time_params, time_params, config
    ... )
    >>> results = analyzer.run_convergence_analysis([1e-4, 1e-5, 1e-6])
    >>> analyzer.plot_convergence_results(results)
"""
# Path setup for imports when running as script
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from typing import Optional
from dataclasses import dataclass
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.models.core_parameters import (
    PhysicalParameters,
    TimeParameters,
    DeadTimeParameters
)
from src.detection.detection_sde_model import DetectionSDEModel
from src.core.euler_maruyama_methods import EulerMaruyamaDetectionSDE
from src.core.taylor_methods import TaylorDetectionSDE
from src.core.runge_kutta_methods import RungeKuttaDetectionSDE
from src.utils.utils import calculate_system_parameters
from src.models.core_parameters import (
    RateConstants, FissionDistribution
)


@dataclass
class PlottingConfig:
    """Configuration for plotting methods."""
    method_labels: dict[str, str]
    method_colors: dict[str, str]


@dataclass
class ErrorPlotParams:
    """Parameters for error convergence plots."""
    step_sizes: np.ndarray
    methods: list[str]
    results: dict
    error_type: str
    title: str
    ylabel: str


@dataclass
class ComparisonPlotParams:
    """Parameters for method comparison plot."""
    step_sizes: np.ndarray
    methods: list[str]
    results: dict


@dataclass
class PlotCreationParams:
    """Parameters for creating all plots."""
    step_sizes: np.ndarray
    methods: list[str]
    results: dict


@dataclass
class ConvergenceAnalysisConfig:
    """
    Configuration parameters for weak convergence analysis.

    This class groups all analysis-specific parameters to reduce
    the number of arguments in WeakConvergenceAnalyzer initialization.

    Attributes
    ----------
    num_paths : int
        Number of Monte Carlo paths per step size
    reference_grid_points : int
        Grid points for reference solution (very fine step size)
    n_0 : Optional[float]
        Initial population. If None, uses equilibrium value.

    Examples
    --------
    >>> config = ConvergenceAnalysisConfig(
    ...     num_paths=1000,
    ...     reference_grid_points=10_000_000
    ... )
    """
    num_paths: int = 1000
    reference_grid_points: int = 10_000_000
    n_0: Optional[float] = None


class WeakConvergenceAnalyzer:
    """
    Analyzer for weak convergence of numerical SDE methods.

    This class performs comprehensive weak convergence analysis by:
    1. Running each numerical method with varying step sizes
    2. Computing weak errors (errors in statistical moments)
    3. Estimating convergence orders
    4. Comparing computational efficiency

    Attributes
    ----------
    physical_params : PhysicalParameters
        Physical parameters for the simulation
    dead_time_params : DeadTimeParameters
        Dead time distribution parameters
    fission : float
        Fission rate constant
    t_end : float
        End time for simulations
    num_paths : int
        Number of Monte Carlo paths per step size
    reference_grid_points : int
        Grid points for reference solution (very fine step size)

    Public Methods
    --------------
    run_convergence_analysis(step_sizes, methods)
        Run complete convergence analysis
    plot_convergence_results(results)
        Create publication-ready convergence plots
    estimate_convergence_orders(results)
        Estimate convergence orders from error data
    compare_efficiency(results)
        Compare computational efficiency of methods

    Examples
    --------
    >>> analyzer = WeakConvergenceAnalyzer(
    ...     fission=33.95,
    ...     num_paths=1000,
    ...     t_end=0.1
    ... )
    >>> step_sizes = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    >>> results = analyzer.run_convergence_analysis(step_sizes)
    >>> analyzer.plot_convergence_results(results)
    """

    def __init__(
            self,
            physical_params: PhysicalParameters,
            dead_time_params: DeadTimeParameters,
            time_params: TimeParameters,
            config: ConvergenceAnalysisConfig):
        """
        Initialize weak convergence analyzer.

        Parameters
        ----------
        physical_params : PhysicalParameters
            Physical parameters for the simulation
        dead_time_params : DeadTimeParameters
            Dead time distribution parameters
        time_params : TimeParameters
            Time parameters (t_end used for simulations)
        config : ConvergenceAnalysisConfig
            Analysis configuration parameters
        """
        self.physical_params = physical_params
        self.dead_time_params = dead_time_params
        self.time_params = time_params
        self.config = config
        self.sde_model = DetectionSDEModel()

        # Compute initial condition if needed
        if config.n_0 is None:
            self._compute_initial_condition()

    def _compute_initial_condition(self) -> None:
        """
        Compute initial population from equilibrium.

        Sets config.n_0 to equilibrium value if not provided.
        """
        theoretical_params = calculate_system_parameters(
            self.physical_params.p_v,
            self.physical_params.fission,
            self.physical_params.absorb,
            self.physical_params.source,
            self.physical_params.detect
        )
        self.config.n_0 = theoretical_params['equilibrium']

    def run_convergence_analysis(
            self,
            step_sizes: list[float],
            methods: Optional[list[str]] = None) -> dict:
        """
        Run complete weak convergence analysis.

        Parameters
        ----------
        step_sizes : list[float]
            list of step sizes (dt values) to test
        methods : Optional[list[str]]
            Methods to test. If None, tests all methods.
            Options: ['euler_maruyama', 'taylor', 'runge_kutta']

        Returns
        -------
        dict
            dictionary containing:
            - 'step_sizes': list of step sizes tested
            - 'methods': dictionary of method results
            - 'reference': Reference solution statistics
            - 'errors': Weak errors for each method and step size
            - 'computation_times': Computation times for each method
        """
        if methods is None:
            methods = ['euler_maruyama', 'taylor', 'runge_kutta']

        self._print_analysis_header(step_sizes, methods)

        # Compute reference solution
        reference_stats = self._compute_reference_solution()

        # Initialize results storage
        results = {
            'step_sizes': step_sizes,
            'methods': {},
            'reference': reference_stats,
            'errors': {},
            'computation_times': {}
        }

        # Test each method
        for method_name in methods:
            method_results = self._test_method(method_name, step_sizes)
            results['methods'][method_name] = method_results

            # Compute weak errors
            mean_errors = np.abs(
                np.array(method_results['means']) - reference_stats['mean']
            )
            variance_errors = np.abs(
                np.array(method_results['variances']) -
                reference_stats['variance']
            )

            results['errors'][method_name] = {
                'mean_error': mean_errors,
                'variance_error': variance_errors
            }
            results['computation_times'][method_name] = (
                method_results['computation_times']
            )

        return results

    def _print_analysis_header(
            self,
            step_sizes: list[float],
            methods: list[str]) -> None:
        """Print analysis header information."""
        print("=" * 80)
        print("WEAK CONVERGENCE ANALYSIS")
        print("=" * 80)
        print(f"Fission rate: {self.physical_params.fission}")
        print(f"Number of paths per step size: {self.config.num_paths}")
        print(f"End time: {self.time_params.t_end} s")
        print(f"Step sizes to test: {step_sizes}")
        print(f"Methods: {methods}")
        print("-" * 80)

    def _test_method(
            self,
            method_name: str,
            step_sizes: list[float]) -> dict:
        """
        Test a single method with multiple step sizes.

        Parameters
        ----------
        method_name : str
            Name of the method to test
        step_sizes : list[float]
            list of step sizes to test

        Returns
        -------
        dict
            dictionary with 'means', 'variances', 'computation_times'
        """
        print(f"\n{'=' * 80}")
        print(f"Testing {method_name.upper()} method")
        print(f"{'=' * 80}")

        method_results = {
            'means': [],
            'variances': [],
            'computation_times': []
        }

        for dt in step_sizes:
            grid_points = self._dt_to_grid_points(dt)
            print(f"\nStep size: {dt:.2e}")
            print(f"Grid points: {grid_points:,}")

            # Run Monte Carlo simulation
            start_time = time.time()
            paths = self._run_method_paths(method_name, dt)
            elapsed_time = time.time() - start_time

            # Compute statistics
            final_values = paths[:, -1]
            mean_val = np.mean(final_values)
            var_val = np.var(final_values)

            method_results['means'].append(mean_val)
            method_results['variances'].append(var_val)
            method_results['computation_times'].append(elapsed_time)

            print(f"  Mean: {mean_val:.6f}")
            print(f"  Variance: {var_val:.6f}")
            print(f"  Computation time: {elapsed_time:.2f} s")

        return method_results

    def _compute_reference_solution(self) -> dict[str, float]:
        """
        Compute reference solution using very fine step size.
        """
        print("\nComputing reference solution...")
        dt_ref = self._grid_points_to_dt(self.config.reference_grid_points)
        print(f"Reference step size: {dt_ref:.2e}")
        print(f"Reference grid points: {self.config.reference_grid_points:,}")
        
        # Use MORE paths for reference to reduce statistical error
        original_num_paths = self.config.num_paths
        reference_num_paths = max(5000, original_num_paths * 3)
        print(f"Using {reference_num_paths} paths for reference solution")
        
        # Temporarily increase num_paths for reference
        temp_config_num_paths = self.config.num_paths
        self.config.num_paths = reference_num_paths
        
        # Use Taylor method for reference (highest order)
        paths = self._run_method_paths('taylor', dt_ref)
        
        # Restore original num_paths
        self.config.num_paths = temp_config_num_paths
        
        final_values = paths[:, -1]
        return {
            'mean': np.mean(final_values),
            'variance': np.var(final_values, ddof=0)  # Population variance
        }

    def _run_method_paths(
            self,
            method_name: str,
            dt: float) -> np.ndarray:
        """
        Run multiple paths for a given method and step size.

        Parameters
        ----------
        method_name : str
            Name of the method ('euler_maruyama', 'taylor', 'runge_kutta')
        dt : float
            Step size

        Returns
        -------
        np.ndarray
            Array of shape (num_paths, grid_points+1) containing all paths
        """
        grid_points = self._dt_to_grid_points(dt)
        time_params = TimeParameters(
            t_0=self.time_params.t_0,
            t_end=self.time_params.t_end,
            grid_points=grid_points
        )

        solver = self._create_solver(method_name)

        # Run multiple paths
        paths = []
        for i in range(self.config.num_paths):
            if (i + 1) % max(1, self.config.num_paths // 10) == 0:
                print(f"  Path {i + 1}/{self.config.num_paths}", end='\r')
            _, _, detect = solver.solve_detection_sde(
                time_params, self.dead_time_params,
                self.physical_params.fission, self.config.n_0
            )
            paths.append(detect)

        print()  # New line after progress
        return np.array(paths)

    def _create_solver(self, method_name: str):
        """
        Create solver instance for given method.

        Parameters
        ----------
        method_name : str
            Name of the method

        Returns
        -------
        BaseDetectionSolver
            Solver instance
        """
        if method_name == 'euler_maruyama':
            return EulerMaruyamaDetectionSDE(
                self.physical_params, self.sde_model, auto_save=False
            )
        if method_name == 'taylor':
            return TaylorDetectionSDE(
                self.physical_params, self.sde_model, auto_save=False
            )
        if method_name == 'runge_kutta':
            return RungeKuttaDetectionSDE(
                self.physical_params, self.sde_model, auto_save=False
            )
        raise ValueError(f"Unknown method: {method_name}")

    def _dt_to_grid_points(self, dt: float) -> int:
        """
        Convert step size to grid points.

        Parameters
        ----------
        dt : float
            Step size

        Returns
        -------
        int
            Number of grid points
        """
        return int(np.ceil(self.time_params.t_end / dt)) + 1

    def _grid_points_to_dt(self, grid_points: int) -> float:
        """
        Convert grid points to step size.

        Parameters
        ----------
        grid_points : int
            Number of grid points

        Returns
        -------
        float
            Step size
        """
        return self.time_params.t_end / (grid_points - 1)

    def estimate_convergence_orders(self, results: dict) -> dict:
        """
        Estimate convergence orders from error data.

        Fits log(error) = log(C) + p * log(dt) to estimate order p.

        Parameters
        ----------
        results : dict
            Results from run_convergence_analysis

        Returns
        -------
        dict
            dictionary with estimated orders for each method
        """
        step_sizes = np.array(results['step_sizes'])
        log_dt = np.log(step_sizes)

        orders = {}

        for method_name in results['errors'].keys():
            mean_errors = results['errors'][method_name]['mean_error']
            var_errors = results['errors'][method_name]['variance_error']

            # Fit linear regression: log(error) = log(C) + p * log(dt)
            mean_order = self._fit_convergence_order(
                log_dt, mean_errors
            )
            var_order = self._fit_convergence_order(
                log_dt, var_errors
            )

            orders[method_name] = {
                'mean_order': mean_order,
                'variance_order': var_order
            }

        return orders

    def _fit_convergence_order(
            self,
            log_dt: np.ndarray,
            errors: np.ndarray) -> Optional[float]:
        """
        Fit convergence order from error data.

        Parameters
        ----------
        log_dt : np.ndarray
            Logarithm of step sizes
        errors : np.ndarray
            Error values

        Returns
        -------
        Optional[float]
            Estimated convergence order, or None if insufficient data
        """
        mask = errors > 0
        if np.sum(mask) < 2:
            return None

        slope, _, _, _, _ = stats.linregress(
            log_dt[mask], np.log(errors[mask])
        )
        return slope

    def plot_convergence_results(
            self,
            results: dict,
            save_plots: bool = True) -> None:
        """Create comprehensive convergence plots."""
        step_sizes = np.array(results['step_sizes'])
        methods = list(results['errors'].keys())
        plotting_config = self._create_plotting_config()

        fig, gs = self._create_figure_with_subplots()
        plot_params = PlotCreationParams(
            step_sizes=step_sizes,
            methods=methods,
            results=results
        )
        self._create_all_plots(fig, gs, plot_params, plotting_config)
        self._finalize_plot(fig, save_plots)

    def _create_plotting_config(self) -> PlottingConfig:
        """Create plotting configuration."""
        return PlottingConfig(
            method_labels={
                'euler_maruyama': 'EM',
                'taylor': 'Taylor 2.0',
                'runge_kutta': 'RK 3.0'
            },
            method_colors={
                'euler_maruyama': 'blue',
                'taylor': 'red',
                'runge_kutta': 'green'
            }
        )

    def _create_figure_with_subplots(self):
        """Create figure with subplot grid."""
        fig = plt.figure(figsize=(16, 12))
        return fig, fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    def _create_all_plots(
            self,
            fig,
            gs,
            params: PlotCreationParams,
            config: PlottingConfig) -> None:
        """Create all subplots."""
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_error_convergence(
            ax1, ErrorPlotParams(
                step_sizes=params.step_sizes,
                methods=params.methods,
                results=params.results,
                error_type='mean_error',
                title='Mean Convergence',
                ylabel='Weak Error in Mean $|E[X_T] - E[X_{ref}]|$'
            ), config
        )

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_error_convergence(
            ax2, ErrorPlotParams(
                step_sizes=params.step_sizes,
                methods=params.methods,
                results=params.results,
                error_type='variance_error',
                title='Variance Convergence',
                ylabel='Weak Error in Variance $|\\text{Var}(X_T) - \\text{Var}(X_{ref})|$'
            ), config
        )

        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_efficiency(ax3, params.methods, params.results, config)

        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_convergence_orders(
            ax4, params.methods, params.results, config)

        ax5 = fig.add_subplot(gs[2, :])
        self._plot_method_comparison(
            ax5,
            ComparisonPlotParams(
                step_sizes=params.step_sizes,
                methods=params.methods,
                results=params.results
            ),
            config
        )

    def _finalize_plot(self, fig, save_plots: bool) -> None:
        """Finalize plot with title and save if needed."""
        fig.suptitle(  # Use fig.suptitle instead of plt.suptitle
            f'Weak Convergence Analysis\n'
            f'Fission Rate: {self.physical_params.fission}, '
            f'Paths per Step Size: {self.config.num_paths}',
            fontsize=16, fontweight='bold', y=0.995
        )
        plt.tight_layout()

        if save_plots:
            filename = (
                f'weak_convergence_analysis_f{self.physical_params.fission}_'
                f'{self.config.num_paths}paths.png'
            )
            # Use fig.savefig
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nPlots saved as: {filename}")

        plt.show()

    def _plot_error_convergence(
            self,
            ax,
            params: ErrorPlotParams,
            config: PlottingConfig) -> None:
        """Plot error convergence for mean or variance."""
        for method_name in params.methods:
            errors = params.results['errors'][method_name][params.error_type]
            label = config.method_labels.get(method_name, method_name)
            color = config.method_colors.get(method_name, 'black')
            marker = 'o' if params.error_type == 'mean_error' else 's'
            ax.loglog(params.step_sizes, errors, marker + '-', label=label,
                      color=color, linewidth=2, markersize=8)

        ax.set_xlabel('Step Size $\\Delta t$', fontsize=12)
        ax.set_ylabel(params.ylabel, fontsize=12)
        ax.set_title(params.title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        self._add_reference_order_lines(ax, params)

    def _add_reference_order_lines(
            self,
            ax,
            params: ErrorPlotParams) -> None:
        """Add reference lines for expected convergence orders."""
        if len(params.step_sizes) > 0:
            dt_ref = params.step_sizes[0]
            error_ref = params.results['errors'][params.methods[0]
                                                 ][params.error_type][0]
            ax.loglog(params.step_sizes, error_ref * (params.step_sizes / dt_ref),
                      'k--', alpha=0.3, label='Order 1.0', linewidth=1)
            ax.loglog(params.step_sizes, error_ref * (params.step_sizes / dt_ref) ** 2,
                      'k:', alpha=0.3, label='Order 2.0', linewidth=1)
            ax.loglog(params.step_sizes, error_ref * (params.step_sizes / dt_ref) ** 3,
                      'k-.', alpha=0.3, label='Order 3.0', linewidth=1)

    def _plot_efficiency(
            self,
            ax,
            methods: list[str],
            results: dict,
            config: PlottingConfig) -> None:
        """Plot efficiency comparison (time vs accuracy)."""
        for method_name in methods:
            times = results['computation_times'][method_name]
            mean_errors = results['errors'][method_name]['mean_error']
            label = config.method_labels.get(method_name, method_name)
            color = config.method_colors.get(method_name, 'black')
            ax.loglog(times, mean_errors, 'o-', label=label, color=color,
                      linewidth=2, markersize=8)

        ax.set_xlabel('Computation Time (s)', fontsize=12)
        ax.set_ylabel('Weak Error in Mean', fontsize=12)
        ax.set_title('Efficiency: Time vs Accuracy', fontsize=14,
                     fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

    def _plot_convergence_orders(
            self,
            ax,
            methods: list[str],
            results: dict,
            config: PlottingConfig) -> None:
        """Plot estimated convergence orders."""
        orders = self.estimate_convergence_orders(results)
        plot_data = self._prepare_order_plot_data(methods, orders, config)

        x_pos = np.arange(len(plot_data['method_names']))
        width = 0.35

        ax.bar(x_pos - width/2, plot_data['mean_orders'], width,
               label='Mean', alpha=0.8)
        ax.bar(x_pos + width/2, plot_data['var_orders'], width,
               label='Variance', alpha=0.8)

        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Estimated Convergence Order', fontsize=12)
        ax.set_title('Convergence Orders', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_data['method_names'], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    def _prepare_order_plot_data(
            self,
            methods: list[str],
            orders: dict,
            config: PlottingConfig) -> dict:
        """Prepare data for convergence order plot."""
        method_names = []
        mean_orders = []
        var_orders = []

        for method_name in methods:
            method_names.append(
                config.method_labels.get(method_name, method_name))
            mean_order = orders[method_name]['mean_order']
            var_order = orders[method_name]['variance_order']
            mean_orders.append(mean_order if mean_order is not None else 0)
            var_orders.append(var_order if var_order is not None else 0)

        return {
            'method_names': method_names,
            'mean_orders': mean_orders,
            'var_orders': var_orders
        }

    def _plot_method_comparison(
            self,
            ax,
            params: ComparisonPlotParams,
            config: PlottingConfig) -> None:
        """Plot method comparison across step sizes."""
        x_pos = np.arange(len(params.step_sizes))
        width = 0.25

        for i, method_name in enumerate(params.methods):
            mean_errors = params.results['errors'][method_name]['mean_error']
            label = config.method_labels.get(method_name, method_name)
            color = config.method_colors.get(method_name, 'black')
            offset = (i - len(params.methods)/2 + 0.5) * width
            ax.bar(x_pos + offset, mean_errors, width, label=label,
                   color=color, alpha=0.8)

        ax.set_xlabel('Step Size Index', fontsize=12)
        ax.set_ylabel('Weak Error in Mean', fontsize=12)
        ax.set_title('Error Comparison Across Step Sizes', fontsize=14,
                     fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{dt:.1e}' for dt in params.step_sizes],
                           rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

    def print_summary(self, results: dict) -> None:
        """
        Print comprehensive summary of convergence analysis.

        Parameters
        ----------
        results : Dict
            Results from run_convergence_analysis
        """
        print("\n" + "=" * 80)
        print("WEAK CONVERGENCE ANALYSIS SUMMARY")
        print("=" * 80)

        print("\nSimulation Parameters:")
        print(f"  Fission rate: {self.physical_params.fission}")
        print(f"  End time: {self.time_params.t_end} s")
        print(f"  Number of paths per step size: {self.config.num_paths}")
        print(f"  Dead time type: {self.dead_time_params.tau_distribution}")
        print(f"  Dead time mean: {self.dead_time_params.mean_tau:.2e} s")

        print("\nReference Solution:")
        ref = results['reference']
        print(f"  Mean: {ref['mean']:.6f}")
        print(f"  Variance: {ref['variance']:.6f}")

        print("\nStep Sizes Tested:")
        for i, dt in enumerate(results['step_sizes']):
            print(f"  {i+1}. {dt:.2e}")

        # Estimate orders
        orders = self.estimate_convergence_orders(results)

        print(f"\n{'=' * 80}")
        print("CONVERGENCE RESULTS BY METHOD")
        print("=" * 80)

        method_labels = {
            'euler_maruyama': 'Euler-Maruyama',
            'taylor': 'Taylor 2.0',
            'runge_kutta': 'Runge-Kutta 3.0'
        }

        for method_name in results['errors'].keys():
            print(f"\n{method_labels.get(method_name, method_name)}:")
            print("-" * 60)

            mean_errors = results['errors'][method_name]['mean_error']
            var_errors = results['errors'][method_name]['variance_error']
            times = results['computation_times'][method_name]

            print("  Mean Errors:")
            for i, (dt, error) in enumerate(
                zip(results['step_sizes'], mean_errors)
            ):
                print(f"    dt={dt:.2e}: {error:.6e} (time: {times[i]:.2f}s)")

            print("  Variance Errors:")
            for i, (dt, error) in enumerate(
                zip(results['step_sizes'], var_errors)
            ):
                print(f"    dt={dt:.2e}: {error:.6e}")

            # Print estimated orders
            mean_order = orders[method_name]['mean_order']
            var_order = orders[method_name]['variance_order']

            if mean_order is not None:
                print(f"  Estimated Mean Convergence Order: {mean_order:.2f}")
            if var_order is not None:
                print(f"  Estimated Variance Convergence Order: {
                      var_order:.2f}")

        print("\n" + "=" * 80)


def main():
    """
    Main function to run weak convergence analysis example.

    This function demonstrates how to use the WeakConvergenceAnalyzer
    to compare numerical methods for solving the detection SDE.
    """

    print("Weak Convergence Analysis for Numerical SDE Methods")
    print("=" * 80)

    # Create physical parameters
    rate_constants = RateConstants(
        absorb=7.0, detect=10.0, source=1000.0, fission=33.95
    )
    fission_distribution = FissionDistribution(p_v=[1/6, 1/3, 1/3, 1/6])
    physical_params = PhysicalParameters(
        rate_constants=rate_constants,
        fission_distribution=fission_distribution
    )

    # Create dead time parameters
    dead_time_params = DeadTimeParameters(
        mean_tau=1e-6, tau_distribution='constant'
    )

    # Create time parameters
    time_params = TimeParameters(t_0=0.0, t_end=0.1)

    # Create analysis configuration
    config = ConvergenceAnalysisConfig(
        num_paths=2000,
        reference_grid_points=20_000_000  # Reduced for faster testing
    )

    # Create analyzer
    analyzer = WeakConvergenceAnalyzer(
        physical_params, dead_time_params, time_params, config
    )

    # Define step sizes to test
    step_sizes = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

    # Run analysis
    results = analyzer.run_convergence_analysis(step_sizes)

    # Create plots
    analyzer.plot_convergence_results(results, save_plots=True)

    # Print summary
    analyzer.print_summary(results)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
