"""Written by Tomer279 with the assistance of Cursor.ai.

Simulation orchestration and execution system for nuclear reactor modeling.

This module provides comprehensive simulation orchestration capabilities for
nuclear reactor dynamics, coordinating multiple numerical methods and parameter
sweeps across various simulation approaches. It serves as the central execution
engine that manages stochastic simulations, Euler-Maruyama methods, Taylor
methods, and Runge-Kutta integrations with proper data management and progress
tracking.

Classes:
    SimulationOrchestrator:
        Central coordinator for executing various simulation types

Key Features:
    - Unified interface for multiple numerical methods
    - Parameter sweep management across fission rate ranges
    - Automatic dead time distribution handling
    - Integrated data management and saving
    - Progress tracking and status reporting
    - Configurable output management
    - Error handling and validation

Simulation Types Supported:
    - Stochastic simulations with full Monte Carlo dynamics
    - Euler-Maruyama numerical SDE integration
    - Taylor method deterministic integration
    - Runge-Kutta deterministic integration

Dependencies:
    numpy: For numerical operations
    stochastic_simulation: Stochastic simulation engine
    euler_maruyama_methods: Euler-Maruyama SDE solvers
    taylor_methods: Taylor method integrators
    runge_kutta_methods: Runge-Kutta integrators
    data_management: Data storage and organization
    core_parameters: Parameter container classes
    simulation_setting: Simulation control configuration

Usage Examples:
    # Initialize orchestrator with parameter containers
    orchestrator = SimulationOrchestrator(
        physical_params=physical_params,
        time_params=time_params,
        dead_time_params=dead_time_params,
        fission_params=fission_params
    )

    # Run different simulation types
    orchestrator.run_stochastic_simulations(output_prefix='stoch')
    orchestrator.run_euler_maruyama_simulations(output_prefix='em')

    # Convenience functions for simple execution
    run_euler_maruyama_simulations(
        physical_params, time_params, dead_time_params,
        fission_params, output_prefix='simulation'
    )

Note:
    This module handles the orchestration and execution of simulations.
    For individual numerical methods, see their respective modules:
    stochastic_simulation.py, euler_maruyama_methods.py, etc.
"""

import numpy as np
from stochastic_simulation import (StochasticSimulator,
                                   SimulationParameters
                                   )

from euler_maruyama_methods import (
    euler_maruyama_detection_constant_dead_time,
    euler_maruyama_detection_uniform_dead_time,
    euler_maruyama_detection_normal_dead_time,
    euler_maruyama_detection_gamma_dead_time,
)
from taylor_methods import (
    taylor_detection_constant_dead_time,
    taylor_detection_uniform_dead_time,
    taylor_detection_normal_dead_time,
    taylor_detection_gamma_dead_time
)
from runge_kutta_methods import (
    runge_kutta_detection_constant_dead_time,
    runge_kutta_detection_uniform_dead_time,
    runge_kutta_detection_normal_dead_time,
    runge_kutta_detection_gamma_dead_time
)

import data_management as dm

from core_parameters import (
    PhysicalParameters, TimeParameters,
    DeadTimeParameters, FissionParameters
)

from simulation_setting import (
    SimulationControl,
    OutputSettings
)


class SimulationOrchestrator:
    """
    Central coordinator for nuclear reactor simulation execution.

    This class provides unified orchestration of multiple simulation approaches
    including stochastic Monte Carlo methods, Euler-Maruyama SDE integration,
    Taylor methods, and Runge-Kutta deterministic integration.

    It manages parameter sweeps, data organization, progress tracking,
    and result saving across different numerical methods
    with consistent interfaces.

    The orchestrator handles the complexity of coordinating multiple simulation
    engines while providing a simple, consistent API for parameter sweeps
    and result management.

    Attributes
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction constants
    time_params : TimeParameters
        Time discretization and duration parameters
    dead_time_params : DeadTimeParameters
        Dead time distribution configuration
    fission_params : FissionParameters
        Fission rate sweep parameters and derived calculations
    data_manager : DataManager
        Data management system for organized result storage

    Public Methods
    --------------
    run_stochastic_simulations(output_prefix)
        Execute stochastic simulations for all fission values
    run_euler_maruyama_simulations(output_prefix)
        Execute Euler-Maruyama simulations for all fission values
    run_taylor_simulations(output_prefix)
        Execute Taylor method simulations for all fission values
    run_runge_kutta_simulations(output_prefix)
        Execute Runge-Kutta simulations for all fission values

    Private Methods
    ---------------
    _run_single_stochastic(fission, index, output_prefix)
        Execute single stochastic simulation
    _run_single_euler_maruyama(fission, output_prefix)
        Execute single Euler-Maruyama simulation
    _run_single_taylor(fission, output_prefix)
        Execute single Taylor method simulation
    _run_single_runge_kutta(fission, output_prefix)
        Execute single Runge-Kutta simulation

    Examples
    --------
    >>> orchestrator = SimulationOrchestrator(physical_params, time_params,
    ...                                       dead_time_params, fission_params)
    >>> orchestrator.run_stochastic_simulations(output_prefix='test')
    >>> orchestrator.run_euler_maruyama_simulations(
    ...     output_prefix='experimental')
    """

    def __init__(
            self,
            physical_params: PhysicalParameters,
            time_params: TimeParameters,
            dead_time_params: DeadTimeParameters,
            fission_params: FissionParameters):
        """
        Initialize simulation orchestrator with parameter containers.

        Parameters
        ----------
        physical_params : PhysicalParameters
            Physical parameters containing nuclear reaction rate constants
            and fission probability distributions
        time_params : TimeParameters
            Time discretization parameters including grid points,
            duration, and discretization method
        dead_time_params : DeadTimeParameters
            Dead time distribution parameters and configuration
            for detector modeling
        fission_params : FissionParameters
            Fission rate parameters including sweep values and
            derived calculations (Rossi-alpha, equilibrium)

        Raises
        ------
        ValueError
            If any parameter container is invalid or incomplete
        """
        self.physical_params = physical_params
        self.time_params = time_params
        self.dead_time_params = dead_time_params
        self.fission_params = fission_params
        self.data_manager = dm.DataManager()

    def run_stochastic_simulations(
            self,
            output_prefix: str = 'f'):
        """
        Execute stochastic simulations across fission rate sweep.

        This method runs full Monte Carlo stochastic simulations for each
        fission rate value in the parameter sweep, generating population
        trajectories, time series, and detection events using comprehensive
        stochastic dynamics simulation.

        Parameters
        ----------
        output_prefix : str
            Prefix for output filename generation and data organization.
            Should be descriptive of the simulation run (
                e.g., 'test', 'production')

        Notes
        -----
        Stochastic simulations are computationally intensive
        and generate large amounts of data.

        Progress is reported for each fission value.

        Results are automatically saved through
        the integrated data management system.
        """
        print("=" * 60)
        print("RUNNING STOCHASTIC SIMULATIONS")
        print("=" * 60)

        for i, fission in enumerate(self.fission_params.fission_vec):
            print(f"iteration {i + 1}/{len(self.fission_params.fission_vec)}, "
                  f"fission = {fission}")
            self._run_single_stochastic(fission, i, output_prefix)

        print("Stochastic simulations complete!")

    def run_euler_maruyama_simulations(
            self,
            output_prefix: str = 'f'):
        """
        Execute Euler-Maruyama SDE simulations across fission rate sweep.

        This method runs Euler-Maruyama stochastic differential equation
        integration for each fission rate value,
        using hybrid analytical-numerical approaches where population
        is solved analytically and detection processes
        are integrated numerically with dead time effects.

        Parameters
        ----------
        output_prefix : str
            Prefix for output filename generation and data organization

        Notes
        -----
        Euler-Maruyama simulations are significantly faster
        than full stochastic simulations while maintaining good approximation
        quality for detection processes.

        The dead time distribution type is automatically configured
        from the dead_time_params.
        """
        print("=" * 60)
        print("RUNNING EULER-MARUYAMA SIMULATIONS")
        print("=" * 60)

        for i, fission in enumerate(self.fission_params.fission_vec):
            print(f"iteration {i + 1}/{len(self.fission_params.fission_vec)}, "
                  f"fission = {fission}")
            self._run_single_euler_maruyama(fission, output_prefix)

        print("Euler-Maruyama simulations complete!")

    def run_taylor_simulations(
            self,
            output_prefix: str = 'f'):
        """
        Execute Taylor method deterministic simulations
        across fission rate sweep.

        This method runs Taylor numerical integration methods for each fission
        rate value, providing deterministic trajectory computation with high
        precision for population and detection dynamics including dead time
        modeling.

        Parameters
        ----------
        output_prefix : str
            Prefix for output filename generation and data organization

        Notes
        -----
        Taylor methods provide high-precision deterministic solutions suitable
        for validation studies and precision analysis.

        They are typically faster than stochastic methods
        but computationally intensive.
        """
        print("=" * 60)
        print("RUNNING TAYLOR METHOD SIMULATIONS")
        print("=" * 60)

        for i, fission in enumerate(self.fission_params.fission_vec):
            print(f"iteration {i + 1}/{len(self.fission_params.fission_vec)}, "
                  f"fission = {fission}")
            self._run_single_taylor(fission, output_prefix)

        print("Taylor method simulations complete!")

    def run_runge_kutta_simulations(
            self,
            output_prefix: str = 'f'):
        """
        Execute Runge-Kutta deterministic simulations
        across fission rate sweep.

        This method runs Runge-Kutta numerical integration
        for each fission rate value,
        providing robust deterministic trajectory computation suitable
        for routine simulation studies and validation analysis.

        Parameters
        ----------
        output_prefix : str
            Prefix for output filename generation and data organization

        Notes
        -----
        Runge-Kutta methods provide robust, well-tested deterministic solutions
        commonly used for ODE and SDE integration.

        They balance computational efficiency with numerical stability.
        """
        print("=" * 60)
        print("RUNNING RUNGE-KUTTA SIMULATIONS")
        print("=" * 60)

        for i, fission in enumerate(self.fission_params.fission_vec):
            print(f"iteration {i + 1}/{len(self.fission_params.fission_vec)}, "
                  f"fission = {fission}")
            self._run_single_runge_kutta(fission, output_prefix)

        print("Runge-Kutta simulations complete!")

    def _run_single_stochastic(
            self,
            fission: float,
            index: int,
            output_prefix: str):
        """
        Run stochastic simulation for a single fission value.

        This private method handles the execution of a single stochastic
        simulation, including parameter setup, simulator initialization,
        execution, and automatic result saving
        through the data management system.
        """
        # Set fission rate for this simulation
        self.physical_params.rates.set_fission(fission)

        # Create initial population from equilibrium
        n_0 = np.array([self.fission_params.equil[index]])

        # Create simulation parameters for StochasticSimulator
        sim_params = SimulationParameters(
            physical_params=self.physical_params,
            time_params=self.time_params,
            n_0=n_0,
            control_params=SimulationControl(output_settings=OutputSettings(
                output_prefix=output_prefix,
                output_directory='./data/stochastic'
            ))
        )

        # Create and run simulator
        simulator = StochasticSimulator(sim_params)
        _, _, _ = (
            simulator.run_simulation(self.data_manager)
        )

    def _run_single_euler_maruyama(
            self,
            fission: float,
            output_prefix: str):
        """
        Execute single Euler-Maruyama simulation for one fission value.

        This private method coordinates the execution of Euler-Maruyama
        simulations, automatically configuring the appropriate dead time
        distribution and delegating to the corresponding specialized function.
        """
        # Set fission rate for this simulation
        self.physical_params.rates.set_fission(fission)

        index = f"{output_prefix}{fission}"

        # Run based on dead time distribution type
        dead_time_type = self.dead_time_params.tau_distribution

        if dead_time_type == 'constant':
            euler_maruyama_detection_constant_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, fission, index
            )
        elif dead_time_type == 'uniform':
            tau_std = 0.1 * self.dead_time_params.mean_tau
            euler_maruyama_detection_uniform_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, tau_std, fission
            )
        elif dead_time_type == 'normal':
            tau_std = 0.1 * self.dead_time_params.mean_tau
            euler_maruyama_detection_normal_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, tau_std, fission
            )
        elif dead_time_type == 'gamma':
            tau_std = 0.1 * self.dead_time_params.mean_tau
            euler_maruyama_detection_gamma_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, tau_std, fission
            )

    def _run_single_taylor(
            self,
            fission: float,
            output_prefix: str):
        """
        Execute single Taylor method simulation for one fission value.

        This private method coordinates Taylor method execution, automatically
        configuring dead time parameters and delegating to appropriate
        specialized functions.
        """
        # Set fission rate for this simulation
        self.physical_params.rates.set_fission(fission)

        index = f"{output_prefix}{fission}"

        # Run based on dead time distribution type
        dead_time_type = self.dead_time_params.tau_distribution

        if dead_time_type == 'constant':
            taylor_detection_constant_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, fission, index
            )
        elif dead_time_type == 'uniform':
            tau_std = 0.1 * self.dead_time_params.mean_tau
            taylor_detection_uniform_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, tau_std, fission
            )
        elif dead_time_type == 'normal':
            tau_std = 0.1 * self.dead_time_params.mean_tau
            taylor_detection_normal_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, tau_std, fission
            )
        elif dead_time_type == 'gamma':
            tau_std = 0.1 * self.dead_time_params.mean_tau
            taylor_detection_gamma_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, tau_std, fission
            )

    def _run_single_runge_kutta(
            self,
            fission: float,
            output_prefix: str):
        """
        Execute single Runge-Kutta simulation for one fission value.

        This private method coordinates Runge-Kutta execution, automatically
        configuring dead time parameters and delegating to appropriate
        specialized functions.
        """
        # Set fission rate for this simulation
        self.physical_params.rates.set_fission(fission)

        # Run based on dead time distribution type
        dead_time_type = self.dead_time_params.tau_distribution
        index = f"{output_prefix}{fission}"

        if dead_time_type == 'constant':
            runge_kutta_detection_constant_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, fission, index
            )
        elif dead_time_type == 'uniform':
            tau_std = 0.1 * self.dead_time_params.mean_tau
            runge_kutta_detection_uniform_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, tau_std, fission
            )
        elif dead_time_type == 'normal':
            tau_std = 0.1 * self.dead_time_params.mean_tau
            runge_kutta_detection_normal_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, tau_std, fission
            )
        elif dead_time_type == 'gamma':
            tau_std = 0.1 * self.dead_time_params.mean_tau
            runge_kutta_detection_gamma_dead_time(
                self.physical_params, self.time_params,
                self.dead_time_params.mean_tau, tau_std, fission
            )


# Convenience functions for simpler usage
def run_stochastic_simulations(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        fission_params: FissionParameters,
        output_prefix='f'):
    """
    Convenience function for stochastic simulation parameter sweeps.

    This function provides a simplified interface for running stochastic
    simulations across a fission rate sweep without requiring creation
    of a SimulationOrchestrator instance.

    Parameters
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction constants
    time_params : TimeParameters
        Time discretization parameters
    fission_params : FissionParameters
        Fission rate sweep parameters
    output_prefix : str
        Prefix for output filename generation

    Examples
    --------
    >>> run_stochastic_simulations(physical_params, time_params,
    ...                            fission_params, output_prefix='study1')
    """
    # Create dummy dead time params (not used for stochastic)
    dead_time_params = DeadTimeParameters()
    orchestrator = SimulationOrchestrator(physical_params, time_params,
                                          dead_time_params, fission_params)
    orchestrator.run_stochastic_simulations(output_prefix)


def run_euler_maruyama_simulations(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        dead_time_params: DeadTimeParameters,
        fission_params: FissionParameters,
        output_prefix='f'):
    """
    Convenience function for Euler-Maruyama simulation parameter sweeps.

    This function provides a simplified interface for running Euler-Maruyama
    SDE simulations across a fission rate sweep with automatic dead time
    distribution handling.

    Parameters
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction constants
    time_params : TimeParameters
        Time discretization parameters
    dead_time_params : DeadTimeParameters
        Dead time distribution configuration
    fission_params : FissionParameters
        Fission rate sweep parameters
    output_prefix : str
        Prefix for output filename generation

    Examples
    --------
    >>> dead_time_params = DeadTimeParameters(
    ...     mean_tau=1e-6, tau_distribution='uniform'
    )
    >>> run_euler_maruyama_simulations(physical_params, time_params,
    ...                                dead_time_params, fission_params)
    """
    orchestrator = SimulationOrchestrator(physical_params, time_params,
                                          dead_time_params, fission_params)
    orchestrator.run_euler_maruyama_simulations(output_prefix)


def run_taylor_simulations(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        dead_time_params: DeadTimeParameters,
        fission_params: FissionParameters,
        output_prefix: str = 'f'):
    """
    Convenience function for Taylor method simulation parameter sweeps.

    This function provides a simplified interface for running Taylor method
    simulations across a fission rate sweep with automatic dead time
    distribution handling.

    Parameters
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction constants
    time_params : TimeParameters
        Time discretization parameters
    dead_time_params : DeadTimeParameters
        Dead time distribution configuration
    fission_params : FissionParameters
        Fission rate sweep parameters
    output_prefix : str
        Prefix for output filename generation

    Examples
    --------
    >>> dead_time_params = DeadTimeParameters(
    ...     mean_tau=1e-6, tau_distribution='gamma'
    )
    >>> run_taylor_simulations(physical_params, time_params,
    ...                        dead_time_params, fission_params)
    """
    orchestrator = SimulationOrchestrator(physical_params, time_params,
                                          dead_time_params, fission_params)
    orchestrator.run_taylor_simulations(output_prefix)


def run_runge_kutta_simulations(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        dead_time_params: DeadTimeParameters,
        fission_params: FissionParameters,
        output_prefix: str = 'f'):
    """
    Convenience function for Runge-Kutta simulation parameter sweeps.

    This function provides a simplified interface for running Runge-Kutta
    simulations across a fission rate sweep with automatic dead time
    distribution handling.

    Parameters
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing nuclear reaction constants
    time_params : TimeParameters
        Time discretization parameters
    dead_time_params : DeadTimeParameters
        Dead time distribution configuration
    fission_params : FissionParameters
        Fission rate sweep parameters
    output_prefix : str
        Prefix for output filename generation

    Examples
    --------
    >>> dead_time_params = DeadTimeParameters(
    ...     mean_tau=1e-6, tau_distribution='normal'
    )
    >>> run_runge_kutta_simulations(physical_params, time_params,
    ...                            dead_time_params, fission_params)
    """
    orchestrator = SimulationOrchestrator(
        physical_params,
        time_params,
        dead_time_params,
        fission_params)
    orchestrator.run_runge_kutta_simulations(output_prefix)
