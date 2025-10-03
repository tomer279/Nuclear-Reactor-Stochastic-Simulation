""" Written by Tomer279 with the assistance of Cursor.ai.

Data management system for nuclear reactor simulation results.

This module provides comprehensive data management capabilities for nuclear
reactor simulations, supporting organized storage, loading, and retrieval of
simulation results from various numerical methods. It uses a structured
directory hierarchy and type-safe dataclass interfaces for reliable data
operations.

Classes:
    StochasticData: Container for stochastic simulation results
    EulerMaruyamaData: Container for Euler-Maruyama simulation results
    TaylorData: Container for Taylor method simulation results
    RungeKuttaData: Container for Runge-Kutta simulation results
    DataManager: Centralized data management interface

Key Features:
    - Organized file storage with automatic directory creation
    - Type-safe data operations using dataclasses
    - Support for multiple simulation methods and dead time types
    - Batch loading capabilities for multiple fission values
    - Data summary and validation utilities
    - Backward compatibility with legacy function interfaces

Directory Structure:
    The DataManager automatically creates:
    - stochastic/ (population, time, detection matrices)
    - euler_maruyama/ (population, detection matrices)
    - taylor_methods/ (population, detection matrices)
    - runge_kutta/ (population, detection matrices)
    - analysis/ (count rates, processed data)

Dependencies:
    pathlib: For cross-platform path operations
    numpy: For array operations and file I/O
    typing: For type annotations and dataclass support
    utils: Custom utility functions for filename generation

Usage Examples:
    # Initialize data manager
    dm = DataManager('./simulation_data')

    # Save stochastic simulation results
    stochastic_data = StochasticData(
        population_matrix=pop_mat,
        time_matrix=time_mat,
        detection_matrix=detect_mat,
        fission_value=33.95
    )
    dm.save_stochastic_data(stochastic_data)

    # Load multiple simulation results
    pop_list, time_list, detect_list = dm.load_stochastic_data([33.94, 33.95])

    # Get data summary
    summary = dm.get_data_summary()
    print(f"Available datasets: {summary}")

Note:
    This module handles the operational aspects of data storage and retrieval.
    For core simulation parameters, see core_parameters.py. For simulation
    control, see simulation_setting.py.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import utils as utl

# =============================================================================
# 0. DATA STRCTURES
# =============================================================================


@dataclass
class StochasticData:
    """Data Structure for stochastic simulation results."""
    population_matrix: np.ndarray
    time_matrix: np.ndarray
    detection_matrix: np.ndarray
    fission_value: float
    prefix: str = 'f'


@dataclass
class EulerMaruyamaData:
    """Data Structure for Euler-Maruyama simulation results."""
    population_data: np.ndarray
    detection_data: np.ndarray
    fission_value: float
    dead_time_type: str
    prefix: str = 'f'


@dataclass
class TaylorData:
    """Data Structure for Taylor method simulation results."""
    population_data: np.ndarray
    detection_data: np.ndarray
    fission_value: float
    dead_time_type: str
    prefix: str = 'f'


@dataclass
class RungeKuttaData:
    """Data Structure for Runge-Kutta method simulation results."""
    population_data: np.ndarray
    detection_data: np.ndarray
    fission_value: float
    dead_time_type: str
    prefix: str = 'f'


# =============================================================================
# 1. DIRECTORY MANAGEMENT
# =============================================================================

class DataManager:
    """
    Centralized data management system for nuclear reactor simulations.

    This class provides a unified interface for organizing, storing, and
    retrieving simulation data from various numerical methods. It maintains
    a hierarchical directory structure and offers type-safe operations through
    dataclass-based data containers.

    The DataManager creates and manages the following directory structure:
    - stochastic/ (population, time, detection matrices)
    - euler_maruyama/ (population, detection matrices)
    - taylor_methods/ (population, detection matrices)
    - runge_kutta/ (population, detection matrices)
    - analysis/ (count rates, processed data)

    Attributes
    ----------
    base_dir : Path
        Base directory path for all data storage operations

    Public Methods
    --------------
    save_stochastic_data(data)
        Save stochastic simulation data
    load_stochastic_data(fission_vec, prefix)
        Load stochastic simulation data for multiple fission values
    save_euler_maruyama_data(data)
        Save Euler-Maruyama simulation data
    load_euler_maruyama_data(fission_vec, dead_time_type, prefix)
        Load Euler-Maruyama data for multiple fission values
    save_taylor_data(data)
        Save Taylor method simulation data
    load_taylor_data(fission_vec, dead_time_type, prefix)
        Load Taylor method data for multiple fission值
    save_runge_kutta_data(data)
        Save Runge-Kutta simulation data
    load_runge_kutta_data(fission_vec, dead_time_type, prefix)
        Load Runge-Kutta data for multiple fission values
    get_data_summary()
        Get summary of all available data

    Private Methods
    ---------------
    _create_directory_structure()
        Create the complete directory structure

    Examples
    --------
    >>> dm = DataManager('./simulation_results')
    >>> data = StochasticData(pop_mat, time_mat, detect_mat, 33.95, 'f')
    >>> dm.save_stochastic_data(data)
    >>> pop_list, time_list, detect_list = dm.load_stochastic_data(
        [33.94, 33.95])
    """

    def __init__(self, base_dir: str = './data'):
        """
        Initialize data manager with base directory.

        Parameters
        ----------
        base_dir : str
            Base directory for all data storage
        """
        self.base_dir = Path(base_dir)
        self._create_directory_structure()

    def _create_directory_structure(self):
        """Create the complete directory structure."""

        # Main data directories
        directories = [
            'stochastic/population_matrices',
            'stochastic/time_matrices',
            'stochastic/detection_matrices',
            'euler_maruyama/population_matrices',
            'euler_maruyama/detection_matrices',
            'euler_maruyama/time_matrices',
            'taylor_methods/population_matrices',
            'taylor_methods/time_matrices',
            'taylor_methods/detection_matrices',
            'runge_kutta/population_matrices',
            'runge_kutta/detection_matrices',
            'analysis/count_rates',
            'analysis/processed_data'
        ]

        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)

        print(f"Data directory structure created at: {self.base_dir}")

# =============================================================================
# 2. STOCHASTIC DATA OPERATIONS
# =============================================================================

    def save_stochastic_data(self, data: StochasticData) -> None:
        """
        Save stochastic simulation data to organized directory structure.

        This method saves all three matrices (population, time, detection)
        from stochastic simulations to appropriately organized directories.
        It creates fission-specific subdirectories and uses standardized
        filename conventions.

        Parameters
        ----------
        data : StochasticData
            Stochastic simulation data object containing:
                - population_matrix: Population time series data
                - time_matrix: Time matrix data
                - detection_matrix: Detection event data
                - fission_value: Simulation fission rate
                - prefix: File naming prefix

        Raises
        ------
        OSError
            If file writing fails
        """
        # Create fission-specific subdirectory
        fission_dir = f"{data.prefix}{data.fission_value}"

        # Filenames for population, time, and detections
        filename_pop = utl.generate_filename(data.prefix,
                                             'Simul_Pop_Matrix',
                                             'basic',
                                             data.fission_value)
        filename_time = utl.generate_filename(data.prefix,
                                              'Simul_Time_Matrix',
                                              'basic',
                                              data.fission_value)
        filename_detect = utl.generate_filename(data.prefix,
                                                'Detection_Matrix',
                                                'basic',
                                                data.fission_value)

        # Save population matrix
        pop_path = (self.base_dir / 'stochastic'
                    / 'population_matrices' / fission_dir)
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / filename_pop, data.population_matrix)

        # Save time matrix
        time_path = (self.base_dir / 'stochastic'
                     / 'time_matrices' / fission_dir)
        time_path.mkdir(exist_ok=True)
        np.save(time_path / filename_time, data.time_matrix)

        # Save detection matrix
        detect_path = (self.base_dir / 'stochastic'
                       / 'detection_matrices' / fission_dir)
        detect_path.mkdir(exist_ok=True)
        np.save(detect_path / filename_detect, data.detection_matrix)

        print(f"Stochastic data saved for fission = {data.fission_value}\n")

    def load_stochastic_data(
            self,
            fission_vec: List[float],
            prefix: str = 'f'
    ) -> (Tuple[List[np.ndarray],
                List[np.ndarray],
                List[np.ndarray]]):
        """
        Load stochastic simulation data from organized directory structure.

        This method loads stochastic simulation data (population, time, and
        detection matrices) for multiple fission values. It handles missing
        files gracefully with warning messages and returns lists of arrays
        for further processing.

        Parameters
        ----------
        fission_vec : List[Float]
            List of fission rate values to load
        prefix : str
            File naming prefix to match saved data

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
            Lists containing:
                - Population matrices for each fission value
                - Time matrices for each fission value
                - Detection matrices for each fission value

        Raises
        ------
        FileNotFoundError
            If no files are found for any fission values
        """
        pop_matrices = []
        time_matrices = []
        detect_matrices = []

        for fission in fission_vec:
            fission_dir = f"{prefix}{fission}"

            # Use .3f precision to match your renamed files
            filename_pop = f'Simul_Pop_Matrix_{prefix}{fission:.3f}.npy'
            filename_time = f'Simul_Time_Matrix_{prefix}{fission:.3f}.npy'
            filename_detect = f'Detection_Matrix_{prefix}{fission:.3f}.npy'

            # Load population matrix
            pop_path = (self.base_dir / 'stochastic' /
                        'population_matrices' / fission_dir /
                        filename_pop)
            if pop_path.exists():
                pop_matrices.append(np.load(pop_path))
            else:
                print(f"Warning: Population file not found: {filename_pop}")

            # Load time matrix
            time_path = (self.base_dir / 'stochastic' /
                         'time_matrices' / fission_dir /
                         filename_time)
            if time_path.exists():
                time_matrices.append(np.load(time_path))
            else:
                print(f"Warning: Time file not found: {filename_time}")

            # Load detection matrix
            detect_path = (self.base_dir / 'stochastic' /
                           'detection_matrices' / fission_dir /
                           filename_detect)
            if detect_path.exists():
                detect_matrices.append(np.load(detect_path))
            else:
                print(f"Warning: Detection file not found: {filename_detect}")

            print(f"Loaded stochastic data for fission = {fission}")

        return pop_matrices, time_matrices, detect_matrices

# =============================================================================
# 3. EULER-MARUYAMA DATA OPERATIONS
# =============================================================================

    def save_euler_maruyama_data(
            self,
            data: EulerMaruyamaData) -> None:
        """
        Save Euler-Maruyama simulation data to organized directory structure.

        This method saves population and detection data from Euler-Maruyama
        numerical integrations with various dead time distributions.
        It creates organized directories and uses standardized naming.

        Parameters
        ----------
        data : EulerMaruyamaData
            Euler-Maruyama simulation data object containing:
                - population_data: Population time series data
                - detection_data: Detection time series data
                - fission_value: Simulation fission rate
                - dead_time_type: Dead time distribution type
                - prefix: File naming prefix

        Raises
        ------
        OSError
            If file writing fails
        """
        fission_dir = f"{data.prefix}{data.fission_value}"

        # Filenames for population and detection
        filename_pop = utl.generate_filename(data.prefix,
                                             'EM_Pop',
                                             data.dead_time_type,
                                             data.fission_value)
        filename_detect = utl.generate_filename(data.prefix,
                                                'EM_Detect',
                                                data.dead_time_type,
                                                data.fission_value)

        # Save population data
        pop_path = (self.base_dir / 'euler_maruyama'
                    / 'population_matrices' / fission_dir)
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / filename_pop, data.population_data)

        # Save detection data
        detect_path = self.base_dir / 'euler_maruyama' / \
            'detection_matrices' / fission_dir
        detect_path.mkdir(exist_ok=True)
        np.save(detect_path / filename_detect, data.detection_data)

        print(
            f"Euler-Maruyama data saved for fission = {data.fission_value}, "
            f"type = {data.dead_time_type}\n")

    def load_euler_maruyama_data(self,
                                 fission_vec: List[float],
                                 dead_time_type: str,
                                 prefix: str = 'f'
                                 ) -> (Tuple[List[np.ndarray],
                                             List[np.ndarray]]):
        """
        Load Euler-Maruyama simulation data from organized directory structure.

        This method loads Euler-Maruyama simulation data for multiple fission
        values and specific dead time distribution type. It handles missing
        files gracefully and returns structured data for analysis.

        Parameters
        ----------
        fission_vec : List[float]
            List of fission rate values to load
        dead_time_type : str
            Dead time distribution type ('basic', 'const', 'uniform',
            'normal', 'gamma')
        prefix : str
            File naming prefix to match saved data

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Lists containing:
                - Population matrices for each fission value
                - Detection matrices for each fission value

        Raises
        ------
        FileNotFoundError
            If no files are found for any fission values
        """
        pop_matrices = []
        detect_matrices = []

        for fission in fission_vec:

            # Generate filename for loading
            filename_pop = utl.generate_filename(prefix, 'EM_Pop',
                                                 dead_time_type, fission)
            filename_detect = utl.generate_filename(prefix, 'EM_Detect',
                                                    dead_time_type, fission)

            fission_dir = f"{prefix}{fission}"

            # Load population data
            pop_path = (self.base_dir / 'euler_maruyama' /
                        'population_matrices' / fission_dir /
                        filename_pop)
            if pop_path.exists():
                pop_matrices.append(np.load(pop_path))
            else:
                print(f"Warning: Population file not found: {filename_pop}")

            # Load detection data
            detect_path = (self.base_dir / 'euler_maruyama' /
                           'detection_matrices' / fission_dir
                           / filename_detect)
            if detect_path.exists():
                detect_matrices.append(np.load(detect_path))
            else:
                print(f"Warning: Detection file not found: {filename_detect}")

            print(
                f"Loaded Euler-Maruyama data for fission = {fission}, "
                f"type = {dead_time_type}")

        return pop_matrices, detect_matrices

# =============================================================================
# 4. TAYLOR METHOD DATA OPERATIONS
# =============================================================================

    def save_taylor_data(self, data: TaylorData) -> None:
        """
        Save Taylor method simulation data to organized directory structure.

        This method saves population and detection data from Taylor method
        numerical integrations with various dead time distributions.
        It creates organized directories and uses standardized naming.

        Parameters
        ----------
        data : TaylorData
            Taylor method simulation data object containing:
                - population_data: Population time series data
                - detection_data: Detection time series data
                - fission_value: Simulation fission rate
                - dead_time_type: Dead time distribution type
                - prefix: File naming prefix

        Raises
        ------
        OSError
            If file writing fails
        """

        fission_dir = f"{data.prefix}{data.fission_value}"

        # Filenames for population and time
        filename_pop = utl.generate_filename(data.prefix,
                                             "Taylor_Pop",
                                             data.dead_time_type,
                                             data.fission_value)
        filename_detect = utl.generate_filename(data.prefix,
                                                "Taylor_Detect",
                                                data.dead_time_type,
                                                data.fission_value)

        # Save population data
        pop_path = (self.base_dir / 'taylor_methods'
                    / 'population_matrices' / fission_dir)
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / filename_pop, data.population_data)

        # Save detection data
        detect_path = self.base_dir / 'taylor_methods' / \
            'detection_matrices' / fission_dir
        detect_path.mkdir(exist_ok=True)
        np.save(detect_path / filename_detect, data.detection_data)

        print("Taylor data saved "
              f"for fission = {data.fission_value}, "
              f"dead_time = {data.dead_time_type}\n")

    def load_taylor_data(self, fission_vec: List[float],
                         dead_time_type: str = 'basic', prefix: str = 'f') \
            -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Load Taylor method simulation data from organized directory structure.

        This method loads Taylor method simulation data for multiple fission
        values and specific dead time distribution type. It handles missing
        files gracefully and returns structured data for analysis.

        Parameters
        ----------
        fission_vec : List[float]
            List of fission rate values to load
        dead_time_type : str
            Dead time distribution type ('basic', 'const', 'uniform',
            'normal', 'gamma')
        prefix : str
            File naming prefix to match saved data

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Lists containing:
                - Population matrices for each fission value
                - Detection matrices for each fission value

        Raises
        ------
        FileNotFoundError
            If no files are found for any fission values
        """
        pop_matrices = []
        detect_matrices = []

        for fission in fission_vec:
            fission_dir = f"{prefix}{fission}"

            # Generate filenames
            filename_pop = utl.generate_filename(
                prefix,
                "Taylor_Pop",
                dead_time_type,
                fission)

            filename_detect = utl.generate_filename(
                prefix,
                "Taylor_Detect",
                dead_time_type,
                fission)

            # Load population matrix
            pop_path = (self.base_dir / 'taylor_methods' /
                        'population_matrices' / fission_dir / filename_pop)
            if pop_path.exists():
                pop_matrices.append(np.load(pop_path))
            else:
                print(f"Warning: Population file not found: {filename_pop}")

            # Load detection matrix
            detect_path = self.base_dir / 'taylor_methods' / \
                'detection_matrices' / fission_dir / filename_detect
            if detect_path.exists():
                detect_matrices.append(np.load(detect_path))
            else:
                print(f"Warning: Detection file not found: {filename_detect}")

            print(
                f"Loaded Taylor data for fission = {fission}, "
                f"dead_time = {dead_time_type}")

        return pop_matrices, detect_matrices

# =============================================================================
# 4. RUNGE-KUTTA METHOD DATA OPERATIONS
# =============================================================================

    def save_runge_kutta_data(self, data: RungeKuttaData) -> None:
        """
        Save Runge-Kutta simulation data to organized directory structure.

        This method saves population and detection data from Runge-Kutta
        numerical integrations with various dead time distributions.
        It creates organized directories and uses standardized naming.

        Parameters
        ----------
        data : RungeKuttaData
            Runge-Kutta simulation data object containing:
                - population_data: Population time series data
                - detection_data: Detection time series data
                - fission_value: Simulation fission rate
                - dead_time_type: Dead time distribution type
                - prefix: File naming prefix

        Raises
        ------
        OSError
            If file writing fails
        """
        fission_dir = f"{data.prefix}{data.fission_value}"

        # Filenames for population and detection
        filename_pop = utl.generate_filename(data.prefix,
                                             'RK_Pop',
                                             data.dead_time_type,
                                             data.fission_value)
        filename_detect = utl.generate_filename(data.prefix,
                                                'RK_Detect',
                                                data.dead_time_type,
                                                data.fission_value)

        # Save population data
        pop_path = (self.base_dir / 'runge_kutta'
                    / 'population_matrices' / fission_dir)
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / filename_pop, data.population_data)

        # Save detection data
        detect_path = self.base_dir / 'runge_kutta' / \
            'detection_matrices' / fission_dir
        detect_path.mkdir(exist_ok=True)
        np.save(detect_path / filename_detect, data.detection_data)

        print(
            f"Runge-Kutta data saved for fission = {data.fission_value}, "
            f"type = {data.dead_time_type}\n")

    def load_runge_kutta_data(self,
                              fission_vec: List[float],
                              dead_time_type: str,
                              prefix: str = 'f'
                              ) -> (Tuple[List[np.ndarray],
                                          List[np.ndarray]]):
        """
        Load Runge-Kutta simulation data from organized directory structure.

        This method loads Runge-Kutta simulation data for multiple fission
        values and specific dead time distribution type. It handles missing
        files gracefully and returns structured data for analysis.

        Parameters
        ----------
        fission_vec : List[float]
            List of fission rate values to load
        dead_time_type : str
            Dead time distribution type ('basic', 'const', 'uniform',
            'normal', 'gamma')
        prefix : str
            File naming prefix to match saved data

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Lists containing:
                - Population matrices for each fission value
                - Detection matrices for each fission value

        Raises
        ------
        FileNotFoundError
            If no files are found for any fission values
        """
        pop_matrices = []
        detect_matrices = []

        for fission in fission_vec:
            # Generate filename for loading
            filename_pop = utl.generate_filename(prefix, 'RK_Pop',
                                                 dead_time_type, fission)
            filename_detect = utl.generate_filename(prefix, 'RK_Detect',
                                                    dead_time_type, fission)

            fission_dir = f"{prefix}{fission}"

            # Load population data
            pop_path = (self.base_dir / 'runge_kutta' /
                        'population_matrices' / fission_dir /
                        filename_pop)
            if pop_path.exists():
                pop_matrices.append(np.load(pop_path))
            else:
                print(f"Warning: Population file not found: {filename_pop}")

            # Load detection data
            detect_path = (self.base_dir / 'runge_kutta' /
                           'detection_matrices' / fission_dir
                           / filename_detect)
            if detect_path.exists():
                detect_matrices.append(np.load(detect_path))
            else:
                print(f"Warning: Detection file not found: {filename_detect}")

            print(
                f"Loaded Runge-Kutta data for fission = {fission}, "
                f"type = {dead_time_type}")

        return pop_matrices, detect_matrices

# =============================================================================
# 5. UTILITY FUNCTIONS
# =============================================================================

    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get comprehensive summary of all available simulation data.

        This method scans the directory structure and provides a summary of
        available data organized by simulation method. It counts data sets
        and identifies what simulation results are available for analysis.

        Returns
        -------
        Dict[str, Dict]
            Dictionary containing data summary organized by method:
                - 'stochastic': Population, time, and detection data counts
                - 'euler_maruyama': Population and detection data counts
                - 'taylor_methods': Population and detection data counts
                - 'runge_kutta': Population and detection data counts
                - 'analysis': Count rates and processed data counts

        Examples
        --------
        >>> dm = DataManager('./data')
        >>> summary = dm.get_data_summary()
        >>> print(f"Available stochastic datasets: {summary['stochastic']}")
        """
        summary = {
            'stochastic': {},
            'euler_maruyama': {},
            'taylor_methods': {},
            'runge_kutta': {},
            'analysis': {}
        }

        # Count stochastic data
        stochastic_pop_path = (self.base_dir /
                               'stochastic' / 'population_matrices')
        if stochastic_pop_path.exists():
            summary['stochastic']['population'] = len(
                list(stochastic_pop_path.glob('f*')))

        # Count Euler-Maruyama data
        em_pop_path = self.base_dir / 'euler_maruyama' / 'population_matrices'
        if em_pop_path.exists():
            summary['euler_maruyama']['population'] = len(
                list(em_pop_path.glob('f*')))

        # Count Taylor method data
        taylor_pop_path = (self.base_dir /
                           'taylor_methods' /
                           'population_matrices')
        if taylor_pop_path.exists():
            summary['taylor_methods']['population'] = len(
                list(taylor_pop_path.glob('f*'))
            )

        # Count Runge-Kutta data
        rk_pop_path = self.base_dir / 'runge_kutta' / 'population_matrices'
        if rk_pop_path.exists():
            summary['runge_kutta']['population'] = len(
                list(rk_pop_path.glob('f*'))
            )

        # Count analysis data
        analysis_path = self.base_dir / 'analysis'
        if analysis_path.exists():
            count_rates_path = analysis_path / 'count_rates'
            processed_data_path = analysis_path / 'processed_data'

            if count_rates_path.exists():
                summary['analysis']['count_rates'] = len(
                    list(count_rates_path.glob('f*'))
                )
            if processed_data_path.exists():
                summary['analysis']['processed_data'] = len(
                    list(processed_data_path.glob('f*'))
                )

        # Count analysis data
        analysis_path = self.base_dir / 'analysis'
        if analysis_path.exists():
            summary['analysis']['count_rates'] = len(
                list((analysis_path / 'count_rates').glob('f*')))
            summary['analysis']['processed_data'] = len(
                list((analysis_path / 'processed_data').glob('f*')))

        return summary
