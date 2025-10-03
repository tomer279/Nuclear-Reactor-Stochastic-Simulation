""" Written by Tomer279 with the assistance of Cursor.ai.

Nuclear Reactor Stochastic Simulation - Main Execution Script

This is the main entry point for running comprehensive
nuclear reactor simulations using multiple numerical methods:
Stochastic, Euler-Maruyama, Taylor, and Runge-Kutta.

OVERVIEW:
=========
This script orchestrates the complete simulation pipeline including:
- Stochastic simulations for baseline results
- Euler-Maruyama method simulations with constant dead time
- Taylor method simulations with constant dead time
- Runge-Kutta method simulations with constant dead time
- Count rate analysis and comparison
- Plot generation and visualization

SIMULATION METHODS:
==================
1. STOCHASTIC: Direct simulation
2. EULER-MARUYAMA: First-order stochastic integration
3. TAYLOR: Strong Taylor 1.5 method for higher accuracy
4. RUNGE-KUTTA: Fourth-order numerical integration

DEAD TIME MODELS:
=================
- basic: No dead time effects
- constant: Constant dead time (used for EM, Taylor, RK)
- uniform: Uniformly distributed dead time (stochastic only)
- normal: Normally distributed dead time (stochastic only)
- gamma: Gamma distributed dead time (stochastic only)

USAGE INSTRUCTIONS:
==================
1. Configure simulation parameters in config.py
2. Set run flags (RUN_STOCHASTIC, RUN_EULER_MARUYAMA, etc.)
3. Run: python main_new.py
4. Results are saved to ./data/ directory
5. Plots are generated automatically
"""
import traceback
import matplotlib.pyplot as plt
import numpy as np
from core_parameters import (
    PhysicalParameters,
    TimeParameters,
    DeadTimeParameters,
    FissionParameters)
import simulation_runner as sim_run
from data_management import DataManager
import count_rates as cr
import plot_simulations as ps

# =============================================================================
# SIMULATION CONTROL FLAGS - MODIFY THESE TO CONTROL SIMULATIONS
# =============================================================================

RUN_STOCHASTIC = False       # Run direct stochastic simulations
RUN_EULER_MARUYAMA = False   # Run Euler-Maruyama method simulations
RUN_TAYLOR = False           # Run Taylor method simulations
RUN_RUNGE_KUTTA = False      # Run Runge-Kutta method simulations
RUN_CPS_ANALYSIS = True      # Run count rate analysis and plot comparison


# =============================================================================
# SIMULATION PREFIXES - MODIFY THESE FOR DIFFERENT SIMULATION TYPES
# =============================================================================
# Prefixes determine the naming convention
# and organization of simulation results
STOCHASTIC_PREFIX = 'mil_f'       # Million steps
EM_PREFIX = 'mil_f'                 # Long simulation time for Euler-Maruyama
TAYLOR_PREFIX = 'mil_f'             # Long simulation time for Taylor
RK_PREFIX = 'mil_f'                 # long simulation time for Runge-Kutta
# =============================================================================
# DEAD TIME CONFIGURATION - MODIFY THESE TO CONTROL DEAD TIME MODELS
# =============================================================================
# Control which dead time models are simulated for each method

# Euler-Maruyama Dead Time Models
EM_DEAD_TIME_MODELS = {
    'basic': False,      # No dead time effects
    'constant': True,      # Constant dead time
}

# Taylor Method Dead Time Models
TAYLOR_DEAD_TIME_MODELS = {
    'basic': False,      # No dead time effects
    'constant': True,       # Constant dead time
}

RK_DEAD_TIME_MODELS = {
    'basic': False,
    'constant': True
}

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================
# Control which dead time type to analyze
# Options: 'basic', 'const', 'uniform', 'normal', 'gamma'
ANALYZE_DEAD_TIME_TYPE = 'const'


class SimulationManager:
    """
    Manages the complete simulation workflow for nuclear reactor simulations.

    This class encapsualtes all simulation logic and provides a clean
    interfrace for running stochastic, Euler-Maruyama, Taylor, and Runge-Kutta
    simulations along with comprehensive analysis and plotting.
    """

    def __init__(self):

        self.physical_params = PhysicalParameters()

        # Create time parameters with custom values
        self.time_params = TimeParameters(
            t_0=0.0,
            t_end=0.1,
            steps=1_000_000,  # Adjust as needed (default is 100_000_000)
            grid_points=1_000_000  # Adjust as needed (default is 10_000_000)
        )

        # Create dead time parameters
        self.dead_time_params = DeadTimeParameters(
            mean_tau=1e-6,
            tau_distribution='uniform'
        )

        # Create fission parameters
        self.fission_params = FissionParameters(
            physical_params=self.physical_params)

        self.data_manager = DataManager()
        self.simul_time_vec = None

    def run_simulations(self,
                        run_stochastic: bool = False,
                        run_euler_maruyama: bool = False,
                        run_taylor: bool = False,
                        run_runge_kutta: bool = False):
        """
        Run the complete simulation workflow.

        Parameters
        ----------
        run_stochastic : bool, optional
            Whether to run stochastic simulations. The default is False.
        run_euler_maruyama : bool, optional
            Whether to run Euler-Maruyama simulations. The default is False.
        run_taylor : bool, optional
            Whether to run Taylor simulations. The default is False.
        run_runge_kutta : bool, optional
            Whether to run Runge-Kutta simulations. The default is False.

        Returns
        -------
        None.

        """
        print("=" * 80)
        print("NUCLEAR REACTOR STOCHASTIC SIMULATION")
        print("=" * 80)

        fission_vec = self.fission_params.fission_vec
        steps = self.time_params.steps
        grid_points = self.time_params.grid_points

        print("Configuration loaded: "
              f"{len(fission_vec)} fission values")
        print(f"Simulation parameters: steps={steps}, "
              f"grid_points={grid_points}")

        if run_euler_maruyama or run_taylor or run_runge_kutta:
            self._load_stochastic_time_data(fission_vec)
            if self.simul_time_vec is None:
                return

        # Run simulations
        self._run_stochastic_simulations(run_stochastic)
        self._run_euler_maruyama_simulations(run_euler_maruyama)
        self._run_taylor_simulations(run_taylor)
        self._run_runge_kutta_simulations(run_runge_kutta)

        print("\n" + "="*80)
        print("SIMULATION COMPLETE!")
        print("="*80)

    def run_cps_analysis(self):
        """Run comprehensive CPS analysis and comparison."""
        print("\n" + "="*60)
        print("STEP 3: COUNT RATE ANALYSIS AND COMPARISON")
        print("="*60)

        try:
            fission_vec = self.fission_params.fission_vec

            # Load simulation data
            simulation_data = self._load_simulation_data(fission_vec)
            if simulation_data is None:
                return

            # Calculate CPS values
            cps_data = self._calculate_cps_values(simulation_data)

            # Display and analyze results
            self._display_analysis_results(cps_data)

            print(
                f"CPS analysis for {ANALYZE_DEAD_TIME_TYPE} "
                "completed successfully!")

        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"ERROR in CPS analysis: {e}")
            traceback.print_exc()

    def _load_stochastic_time_data(self, fission_vec):
        """Load stochastic time data once for use by numerical methods"""
        print("\n" + "="*60)
        print("LOADING STOCHASTIC TIME DATA")
        print("="*60)
        try:
            _, self.simul_time_vec, _ = (
                self.data_manager.load_stochastic_data(
                    fission_vec=fission_vec, prefix=STOCHASTIC_PREFIX)
            )
            print("Loaded time data for "
                  f"{len(self.simul_time_vec)} fission values")

        except FileNotFoundError:
            print("ERROR: Stochastic time data not found. "
                  "Run stochastic simulations first.")
            self.simul_time_vec = None

    def _run_stochastic_simulations(self,
                                    should_run: bool):
        """Run stochastic simulations if enabled."""
        if should_run:
            print("\n" + "="*60)
            print("STEP 1: RUNNING STOCHASTIC SIMULATIONS")
            print("="*60)

            sim_run.run_stochastic_simulations(
                physical_params=self.physical_params,
                time_params=self.time_params,
                fission_params=self.fission_params,
                output_prefix=STOCHASTIC_PREFIX
            )
        else:
            print("\n" + "="*60)
            print("STEP 1: SKIPPING STOCHASTIC SIMULATIONS")
            print("="*60)
            print("Set RUN_STOCHASTIC = True to run stochastic simulations")

    def _run_euler_maruyama_simulations(self,
                                        should_run: bool):
        """Run Euler-Maruyama simulations if enabled."""
        if should_run:
            if self.simul_time_vec is None:
                print("ERROR: Time data not available "
                      "for Euler-Maruyama simulations")
                return

            print("\n" + "="*60)
            print("STEP 2: RUNNING EULER-MARUYAMA SIMULATIONS")
            print("="*60)

            for dead_time_type, should_run_dt in EM_DEAD_TIME_MODELS.items():
                if should_run_dt:
                    print(
                        f"\nRunning Euler-Maruyama simulations "
                        f"with {dead_time_type} dead time...")

                    # Set the dead time distribution
                    self.dead_time_params.set_distribution(dead_time_type)

                    # Run simulations for this dead time type
                    sim_run.run_euler_maruyama_simulations(
                        physical_params=self.physical_params,
                        time_params=self.time_params,
                        dead_time_params=self.dead_time_params,
                        fission_params=self.fission_params,
                        output_prefix=EM_PREFIX
                    )

            print("Euler-Maruyama simulations completed!")
        else:
            print("\n" + "="*60)
            print("STEP 2: SKIPPING EULER-MARUYAMA SIMULATIONS")
            print("="*60)
            print("Set RUN_EULER_MARUYAMA = True to run EM simulations")

    def _run_taylor_simulations(self,
                                should_run: bool):
        """Run Taylor method simulations if enabled."""
        if should_run:
            if self.simul_time_vec is None:
                print("ERROR: Time data not available for Taylor simulations")
                return

            print("\n" + "="*60)
            print("STEP 2.5: RUNNING TAYLOR METHOD SIMULATIONS")
            print("="*60)

            for dead_time_type, should_run_dt in (
                    TAYLOR_DEAD_TIME_MODELS.items()):
                if should_run_dt:
                    print(f"\nRunning Taylor simulations "
                          f"with {dead_time_type} dead time...")

                    self.dead_time_params.set_distribution(dead_time_type)

                    sim_run.run_taylor_simulations(
                        physical_params=self.physical_params,
                        time_params=self.time_params,
                        dead_time_params=self.dead_time_params,
                        fission_params=self.fission_params,
                        output_prefix=TAYLOR_PREFIX
                    )

            print("Taylor method simulations completed!")
        else:
            print("\n" + "="*60)
            print("STEP 2.5: SKIPPING TAYLOR METHOD SIMULATIONS")
            print("="*60)
            print("Set RUN_TAYLOR = True to run Taylor simulations")

    def _run_runge_kutta_simulations(self,
                                     should_run: bool):
        """Run Runge-Kutta method simulations if enabled."""
        if should_run:
            if self.simul_time_vec is None:
                print("ERROR: Time data not available "
                      "for Runge-Kutta simulations")
                return

            print("\n" + "="*60)
            print("STEP 2.75: RUNNING RUNGE-KUTTA METHOD SIMULATIONS")
            print("="*60)

            for dead_time_type, should_run_dt in RK_DEAD_TIME_MODELS.items():
                if should_run_dt:
                    print(f"\nRunning Runge-Kutta simulations "
                          f"with {dead_time_type} dead time...")

                    self.dead_time_params.set_distribution(dead_time_type)

                    sim_run.run_runge_kutta_simulations(
                        physical_params=self.physical_params,
                        time_params=self.time_params,
                        dead_time_params=self.dead_time_params,
                        fission_params=self.fission_params,
                        output_prefix=RK_PREFIX
                    )

            print("Runge-Kutta method simulations completed!")
        else:
            print("\n" + "="*60)
            print("STEP 2.75: SKIPPING RUNGE-KUTTA METHOD SIMULATIONS")
            print("="*60)
            print("Set RUN_RUNGE_KUTTA = True to run Runge-Kutta simulations")

    def _load_simulation_data(self, fission_vec):
        """
        Load all simulation data for CPS analysis."""

        print(f"Loading simulation data for {ANALYZE_DEAD_TIME_TYPE} "
              "dead time...")

        # Load stochastic data
        print("Loading stochastic data...")
        stochastic_data = self.data_manager.load_stochastic_data(
            fission_vec=fission_vec, prefix=STOCHASTIC_PREFIX)

        if len(stochastic_data[0]) == 0:
            print("ERROR: No stochastic data loaded!")
            return None

        # Load EM data
        print("Loading Euler-Maruyama data...")
        em_data = self.data_manager.load_euler_maruyama_data(
            fission_vec=fission_vec, dead_time_type=ANALYZE_DEAD_TIME_TYPE,
            prefix=EM_PREFIX)

        if len(em_data[0]) == 0:
            print("ERROR: No Euler-Maruyama data loaded!")
            return None

        # Load Taylor data
        print("Loading Taylor method data...")
        taylor_data = self.data_manager.load_taylor_data(
            fission_vec=fission_vec, dead_time_type=ANALYZE_DEAD_TIME_TYPE,
            prefix=TAYLOR_PREFIX)

        if len(taylor_data[0]) == 0:
            print("ERROR: No Taylor method data loaded!")
            return None

        # Load Runge-Kutta data
        print("Loading Runge-Kutta data...")
        rk_data = self.data_manager.load_runge_kutta_data(
            fission_vec=fission_vec, dead_time_type=ANALYZE_DEAD_TIME_TYPE,
            prefix=RK_PREFIX)

        if len(rk_data[0]) == 0:
            print("ERROR: No Runge-Kutta data loaded!")
            return None

        return stochastic_data, em_data, taylor_data, rk_data

    def _calculate_cps_values(self, simulation_data):
        """Calculate CPS values for all methods."""
        stochastic_data, em_data, taylor_data, rk_data = simulation_data

        # Extract data components
        _, time_matrices, detect_matrices = stochastic_data
        _, em_detect_matrices = em_data
        _, taylor_detect_matrices = taylor_data
        _, rk_detect_matrices = rk_data

        # Calculate time duration for numerical methods
        time_duration = self.time_params.t_end - self.time_params.t_0
        print(f"Using numerical method time duration: "
              f"{time_duration:.6f} seconds")

        # Calculate CPS for numerical methods
        numerical_cps = self._calculate_numerical_cps(
            em_detect_matrices, taylor_detect_matrices,
            rk_detect_matrices, time_duration)

        stochastic_cps = self._calculate_stochastic_cps(
            time_matrices, detect_matrices)

        # Calculate theoretical CPS
        theoretical_cps = self._calculate_theoretical_cps()

        return {
            'stochastic_cps': stochastic_cps,
            'em_cps': numerical_cps['em_cps'],
            'taylor_cps': numerical_cps['taylor_cps'],
            'rk_cps': numerical_cps['rk_cps'],
            'theoretical_cps': theoretical_cps,
            'fission_vec': self.fission_params.fission_vec
        }

    def _calculate_theoretical_cps(self):
        """Calculate theoretical CPS using new parameter classes."""
        print("Calculating theoretical CPS...")

        # Map dead time type to the format expected by theoretical calculation
        dead_time_type_mapping = {
            'const': 'constant',
            'constant': 'constant',
            'uniform': 'uniform',
            'normal': 'normal',
            'gamma': 'gamma',
            'basic': 'constant',
            'without': 'constant'
        }

        theoretical_dead_time_type = dead_time_type_mapping.get(
            ANALYZE_DEAD_TIME_TYPE, 'constant'
        )

        theoretical_cps = cr.calculate_theoretical_cps_for_fission_rates(
            fission_rates=self.fission_params.fission_vec,
            dead_time_type=theoretical_dead_time_type,
            mean_tau_s=self.dead_time_params.mean_tau,
            std_tau_s=0.0,
            detect=self.physical_params.detect,
            absorb=self.physical_params.absorb,
            source=self.physical_params.source,
            p_v=self.physical_params.p_v
        )

        return theoretical_cps

    def _calculate_stochastic_cps(self, time_matrices, detect_matrices):
        """Calculate stochastic CPS."""
        all_cps = cr.calculate_all_count_rates(
            simul_time_vec=time_matrices,
            simul_detect_vec=detect_matrices,
            mean_tau=self.dead_time_params.mean_tau)
        return all_cps['stochastic_const_tau']

    def _calculate_numerical_cps(self,
                                 em_detect_matrices,
                                 taylor_detect_matrices,
                                 rk_detect_matrices,
                                 time_duration):
        """Calculate CPS for numerical methods"""
        em_cps = np.array([
            detect_matrix[-1] / time_duration
            for detect_matrix in em_detect_matrices
        ])

        taylor_cps = np.array([
            detect_matrix[-1] / time_duration
            for detect_matrix in taylor_detect_matrices
        ])

        rk_cps = np.array([
            detect_matrix[-1] / time_duration
            for detect_matrix in rk_detect_matrices
        ])

        return {
            'em_cps': em_cps,
            'taylor_cps': taylor_cps,
            'rk_cps': rk_cps
        }

    def _display_analysis_results(self, cps_data):
        """Display and analyze CPS results."""
        # Create display manager and show all results
        display_manager = CPSDisplayManager(cps_data)
        display_manager.display_all_results()

        # Create comparison plots
        self._create_comparison_plots(cps_data)

    def _create_comparison_plots(self, cps_data):
        """Create comparison plots for CPS analysis."""
        # Create fission parameters object for plotting
        fission_params = FissionParameters(
            physical_params=self.physical_params)

        # Create comparison plot
        print(
            f"\nCreating comparison plot for {ANALYZE_DEAD_TIME_TYPE} "
            "dead time...")
        ps.plot_cps_comparison(
            {
                'stochastic': cps_data['stochastic_cps'],
                'methods': {
                    'em': cps_data['em_cps'],
                    'taylor': cps_data['taylor_cps'],
                    'rk': cps_data['rk_cps']
                },
                'fission_params': fission_params
            },
            {
                'dead_time_type': ANALYZE_DEAD_TIME_TYPE,
                'mean_tau': self.dead_time_params.mean_tau,
                't_end': self.time_params.t_end,
                'grid_points': self.time_params.grid_points,
                'save_path': False
            }
        )

        # Create methods vs theoretical plot
        print(f"\nCreating methods vs theoretical comparison plot "
              f"for {ANALYZE_DEAD_TIME_TYPE} dead time...")
        ps.plot_methods_vs_theoretical(
            {
                'theoretical': cps_data['theoretical_cps'],
                'methods': {
                    'em': cps_data['em_cps'],
                    'taylor': cps_data['taylor_cps'],
                    'rk': cps_data['rk_cps']
                },
                'fission_params': fission_params
            },
            {
                'dead_time_type': ANALYZE_DEAD_TIME_TYPE,
                'mean_tau': self.dead_time_params.mean_tau,
                'grid_points': self.time_params.grid_points,
                't_end': self.time_params.t_end,
                'save_path': None
            }
        )
        plt.show()


class CPSDisplayManager:
    """
    Manages the display and analysis of CPS results.

    This class encapsulates all CPS-related display functionality including
    detailed results, statistics, and plotting.
    """

    def __init__(self, cps_data):
        """
        Initialize with CPS data.

        Parameters
        ----------
        cps_data : dict
            Dictionary containing all CPS data including:
            - fission_vec: Array of fission values
            - stochastic_cps: Stochastic CPS results
            - em_cps: Euler-Maruyama CPS results
            - taylor_cps: Taylor CPS results
            - rk_cps: Runge-Kutta CPS results
            - theoretical_cps: Theoretical CPS results
        """
        self.fission_vec = cps_data['fission_vec']
        self.stochastic_cps = cps_data['stochastic_cps']
        self.em_cps = cps_data['em_cps']
        self.taylor_cps = cps_data['taylor_cps']
        self.rk_cps = cps_data['rk_cps']
        self.theoretical_cps = cps_data['theoretical_cps']

    def display_all_results(self):
        """Display comprehensive CPS analysis results."""
        self.display_detailed_results()
        self.display_summary_statistics()

    def display_detailed_results(self):
        """Display detailed CPS results for each fission value."""
        self._display_first_five_results()
        self._display_detailed_table()

    def _display_first_five_results(self):
        """Display first 5 CPS results with detailed breakdown."""
        print(f"\nCPS Results for {ANALYZE_DEAD_TIME_TYPE} dead time "
              "(first 5 values):")

        for i, fission in enumerate(self.fission_vec[:5]):
            print(f"  Fission {fission}:")

            # Handle stochastic CPS
            stochastic_val = self._get_stochastic_value(self.stochastic_cps[i])
            print(f"    Stochastic CPS: {stochastic_val:.2f}")

            print(f"    Euler-Maruyama CPS: {self.em_cps[i]:.2f}")
            print(f"    Taylor CPS: {self.taylor_cps[i]:.2f}")
            print(f"    Runge-Kutta CPS: {self.rk_cps[i]:.2f}")
            print(f"    Theoretical CPS: {self.theoretical_cps[i]:.2f}")

            # Calculate differences
            print(
                "    EM vs Stochastic: "
                f"{abs(stochastic_val - self.em_cps[i]):.2f}")
            print(
                "    Taylor vs Stochastic: "
                f"{abs(stochastic_val - self.taylor_cps[i]):.2f}")
            print(
                "    RK vs Stochastic: "
                f"{abs(stochastic_val - self.rk_cps[i]):.2f}")

    def _display_detailed_table(self):
        """Display detailed CPS results table."""
        print("\n" + "="*80)
        print("DETAILED CPS RESULTS FOR EACH FISSION VALUE")
        print("="*80)
        print(f"{'Fission':<10} {'Stochastic':<12} {'EM':<12} {'Taylor':<12} "
              f"{'Runge-Kutta':<12} {'Theoretical':<12}")
        print("-" * 80)

        for i, fission in enumerate(self.fission_vec):
            stochastic_val = self._get_stochastic_value(self.stochastic_cps[i])
            print(f"{fission:<10.3f} {stochastic_val:<12.2f} "
                  f"{self.em_cps[i]:<12.2f} {self.taylor_cps[i]:<12.2f} "
                  f"{self.rk_cps[i]:<12.2f} {self.theoretical_cps[i]:<12.2f}")

    def display_summary_statistics(self):
        """Display summary statistics comparing all methods."""
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        methods = {
            'Euler-Maruyama': self.em_cps,
            'Taylor': self.taylor_cps,
            'Runge-Kutta': self.rk_cps
        }

        for method_name, cps in methods.items():
            self._display_single_method_statistics(method_name, cps)

        self._display_correlation_analysis()

    def _display_single_method_statistics(self, method_name, cps):
        """Display statistics for a single method."""
        if not np.all(np.isnan(cps)):
            rel_diff = (cps - self.theoretical_cps) / \
                self.theoretical_cps * 100
            abs_error = np.abs(cps - self.theoretical_cps)

            print(f"\n{method_name}:")
            print(f"  Mean CPS: {np.nanmean(cps):.6f}")
            print(f"  Std CPS: {np.nanstd(cps):.6f}")
            print(f"  Mean absolute error: {np.nanmean(abs_error):.6f}")
            print(f"  Max absolute error: {np.nanmax(abs_error):.6f}")
            print(f"  Mean relative difference from theoretical: "
                  f"{np.nanmean(rel_diff):.4f}%")
            print(f"  Std relative difference: {np.nanstd(rel_diff):.4f}%")
            print(f"  Max relative difference: "
                  f"{np.nanmax(np.abs(rel_diff)):.4f}%")
        else:
            print(f"\n{method_name}: All values are NaN")

    def _display_correlation_analysis(self):
        """Display correlation analysis between methods."""
        print("\nCorrelation Analysis:")

        if (not np.all(np.isnan(self.em_cps)) and
                not np.all(np.isnan(self.taylor_cps))):
            em_taylor_corr = np.corrcoef(
                self.em_cps[~np.isnan(self.em_cps)],
                self.taylor_cps[~np.isnan(self.taylor_cps)])[0, 1]
            print(f"  Euler-Maruyama vs Taylor: {em_taylor_corr:.6f}")

        if (not np.all(np.isnan(self.em_cps)) and
                not np.all(np.isnan(self.rk_cps))):
            em_rk_corr = np.corrcoef(
                self.em_cps[~np.isnan(self.em_cps)],
                self.rk_cps[~np.isnan(self.rk_cps)])[0, 1]
            print(f"  Euler-Maruyama vs Runge-Kutta: {em_rk_corr:.6f}")

        if (not np.all(np.isnan(self.taylor_cps)) and
                not np.all(np.isnan(self.rk_cps))):
            taylor_rk_corr = np.corrcoef(
                self.taylor_cps[~np.isnan(self.taylor_cps)],
                self.rk_cps[~np.isnan(self.rk_cps)])[0, 1]
            print(f"  Taylor vs Runge-Kutta: {taylor_rk_corr:.6f}")

    def _get_stochastic_value(self, stochastic_cps_item):
        """Get stochastic CPS value handling arrays or scalars."""
        if (hasattr(stochastic_cps_item, '__len__') and
                len(stochastic_cps_item) > 0):
            return np.mean(stochastic_cps_item)

        return stochastic_cps_item

    def get_cps_data(self):
        """Return the CPS data dictionary for plotting."""
        return {
            'fission_vec': self.fission_vec,
            'stochastic_cps': self.stochastic_cps,
            'em_cps': self.em_cps,
            'taylor_cps': self.taylor_cps,
            'rk_cps': self.rk_cps,
            'theoretical_cps': self.theoretical_cps
        }


def main():
    """
    Main execution function demonstrating the complete simulation workflow.
    """
    # Create simulation manager
    sim_manager = SimulationManager()

    # Run simulations
    sim_manager.run_simulations(
        run_stochastic=RUN_STOCHASTIC,
        run_euler_maruyama=RUN_EULER_MARUYAMA,
        run_taylor=RUN_TAYLOR,
        run_runge_kutta=RUN_RUNGE_KUTTA
    )

    # Run CPS analysis if requested
    if RUN_CPS_ANALYSIS:
        sim_manager.run_cps_analysis()
    else:
        print("\n" + "="*60)
        print("STEP 3: SKIPPING COUNT RATE ANALYSIS")
        print("="*60)
        print("Set RUN_CPS_ANALYSIS = True to run CPS comparison")


if __name__ == "__main__":
    main()
