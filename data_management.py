""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Data management system for nuclear reactor simulation results.

This module handles saving, loading, and organizing simulation data
from stochastic, Euler-Maruyama, and Taylor method simulations.

ORGANIZATION:
============
1. DIRECTORY MANAGEMENT - Creating and managing folder structures
2. STOCHASTIC DATA OPERATIONS - Save/load stochastic simulation data
3. EULER-MARUYAMA DATA OPERATIONS - Save/load EM simulation data
4. TAYLOR METHOD DATA OPERATIONS - Save/load Taylor simulation data
5. UTILITY FUNCTIONS - Helper functions for data operations
6. COMPATIBILITY FUNCTIONS - Legacy support functions
"""

import numpy as np
import utils as utl
from pathlib import Path
from typing import Dict, List, Tuple


# =============================================================================
# 1. DIRECTORY MANAGEMENT
# ============================================================================

class DataManager:
    
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
            'analysis/count_rates',
            'analysis/processed_data'
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
        
        print(f"Data directory structure created at: {self.base_dir}")
    
# =============================================================================
# 2. STOCHASTIC DATA OPERATIONS
# =============================================================================
    
    def save_stochastic_data(self, pop_mat: np.ndarray, time_mat: np.ndarray, 
                           detect_mat: np.ndarray, fission_value: float, 
                           prefix: str = 'f') -> None:
        """
        Save stochastic simulation data to organized folders.
        
        Parameters
        ----------
        pop_mat : np.ndarray
            Population matrix
        time_mat : np.ndarray
            Time matrix
        detect_mat : np.ndarray
            Detection matrix
        fission_value : float
            Fission value for file naming
        prefix : str
            Prefix for file names
        """
        # Create fission-specific subdirectory
        fission_dir = f"{prefix}{fission_value}"
        
        # Filenames for population, time, and detections
        filename_pop = utl.generate_filename(prefix, 'Simul_Pop_Matrix',
                                             'basic', fission_value)
        filename_time = utl.generate_filename(prefix, 'Simul_Time_Matrix',
                                              'basic', fission_value)
        filename_detect = utl.generate_filename(prefix, 'Detection_Matrix',
                                                'basic' , fission_value)
        
        # Save population matrix
        pop_path = self.base_dir / 'stochastic' / 'population_matrices' / fission_dir
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / filename_pop, pop_mat)
        
        # Save time matrix
        time_path = self.base_dir / 'stochastic' / 'time_matrices' / fission_dir
        time_path.mkdir(exist_ok=True)
        np.save(time_path / filename_time, time_mat)
        
        # Save detection matrix
        detect_path = self.base_dir / 'stochastic' / 'detection_matrices' / fission_dir
        detect_path.mkdir(exist_ok=True)
        np.save(detect_path / filename_detect, detect_mat)
        
        print(f"Stochastic data saved for fission = {fission_value}")
    
    
    def load_stochastic_data(self, fission_vec: List[float], 
                           prefix: str = 'f') -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Load stochastic simulation data from organized folders.
        
        Parameters
        ----------
        fission_vec : List[float]
            List of fission values to load
        prefix : str
            Prefix for file names
            
        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
            Lists of population, time, and detection matrices
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
    
    def save_euler_maruyama_data(self, pop_data: np.ndarray, detect_data: np.ndarray,
                                fission_value: float, dead_time_type: str,
                                prefix: str = 'f') -> None:
        """
        Save Euler-Maruyama simulation data to organized folders.
        
        Parameters
        ----------
        pop_data : np.ndarray
            Population data
        detect_data : np.ndarray
            Detection data
        fission_value : float
            Fission value for file naming
        dead_time_type : str
            Type of dead time ('basic', 'const', 'exp')
        prefix : str
            Prefix for file names
        """
        fission_dir = f"{prefix}{fission_value}"
        
        # Filenames for population and detection
        filename_pop = utl.generate_filename(prefix, 'EM_Pop',
                                             dead_time_type, fission_value)
        filename_detect = utl.generate_filename(prefix, 'EM_Detect',
                                             dead_time_type, fission_value)
        
        # Save population data
        pop_path = self.base_dir / 'euler_maruyama' / 'population_matrices' / fission_dir
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / filename_pop, pop_data)
        
        # Save detection data
        detect_path = self.base_dir / 'euler_maruyama' / 'detection_matrices' / fission_dir
        detect_path.mkdir(exist_ok=True)
        np.save(detect_path / filename_detect, detect_data)
        
        print(f"Euler-Maruyama data saved for fission = {fission_value}, type = {dead_time_type}")
    
    def load_euler_maruyama_data(self, fission_vec: List[float], 
                                dead_time_type: str, prefix: str = 'f') -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load Euler-Maruyama simulation data from organized folders.
        
        Parameters
        ----------
        fission_vec : List[float]
            List of fission values to load
        dead_time_type : str
            Type of dead time ('basic', 'const', 'exp')
        prefix : str
            Prefix for file names
            
        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Lists of population and detection matrices
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
                        filename_pop )
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
            
            print(f"Loaded Euler-Maruyama data for fission = {fission}, type = {dead_time_type}")
        
        return pop_matrices, detect_matrices
    
# =============================================================================
# 4. TAYLOR METHOD DATA OPERATIONS
# =============================================================================
    
    def save_taylor_data(self, pop_data: np.ndarray, detect_data: np.ndarray,
                        fission_value: float, method: str, dead_time_type : str,
                        prefix: str = 'f') -> None:
        """
        Save Taylor method simulation data to organized folders.
        
        Parameters
        ----------
        pop_data : np.ndarray
            Population data
        time_data : np.ndarray
            Time data
        fission_value : float
            Fission value for file naming
        method : str
            Taylor method type ('strong', 'weak')
        dead_time_type : str
            Dead time type ('basic', 'const', 'uniform', 'exp')
        prefix : str
            Prefix for file names
        """
        
        fission_dir = f"{prefix}{fission_value}"
        
        # Filenames for population and time
        filename_pop = utl.generate_filename(prefix, f"{method}_Taylor_Pop",
                                             dead_time_type, fission_value)
        filename_detect = utl.generate_filename(prefix, f"{method}_Taylor_Detect",
                                                dead_time_type, fission_value)
        
        # Save population data
        pop_path = self.base_dir / 'taylor_methods' / 'population_matrices' / fission_dir
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / filename_pop, pop_data)
        
        # Save detection data
        detect_path = self.base_dir / 'taylor_methods' / 'detection_matrices' / fission_dir
        detect_path.mkdir(exist_ok=True)
        np.save(detect_path / filename_detect, detect_data)
        
        
        print(f"Taylor {method} data saved for fission = {fission_value}, \
              dead_time = {dead_time_type}")

    
    
    def load_taylor_data(self, fission_vec: List[float], method: str,
                        dead_time_type: str = 'basic', prefix: str = 'f') \
                -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Load Taylor method simulation data.
        
        Parameters
        ----------
        fission_vec : List[float]
            List of fission values to load
        method : str
            Taylor method type ('strong', 'weak', 'strong_system')
        dead_time_type : str
            Dead time type ('basic', 'const', 'uniform', 'exp')
        prefix : str
            Prefix for file names
            
        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
            Lists of population, time, and detection matrices
        """
        pop_matrices = []
        detect_matrices = []
        
        for fission in fission_vec:
            fission_dir = f"{prefix}{fission}"
            
            if method == 'strong':
                method_name = 'strong_system'
            else:
                method_name = method
                
            # Generate filenames
            filename_pop = utl.generate_filename(prefix, f"{method_name}_Taylor_Pop", 
                                                 dead_time_type, fission)

            filename_detect = utl.generate_filename(prefix, f"{method_name}_Taylor_Detect",
                                                    dead_time_type, fission)
            
            # Load population matrix
            pop_path = self.base_dir / 'taylor_methods' / 'population_matrices' / fission_dir / filename_pop
            if pop_path.exists():
                pop_matrices.append(np.load(pop_path))
            else:
                print(f"Warning: Population file not found: {filename_pop}")
            
            # Load detection matrix
            detect_path = self.base_dir / 'taylor_methods' / 'detection_matrices' / fission_dir / filename_detect
            if detect_path.exists():
                detect_matrices.append(np.load(detect_path))
            else:
                print(f"Warning: Detection file not found: {filename_detect}")
                
            print(f"Loaded Taylor {method} data for fission = {fission}, dead_time = {dead_time_type}")
        
        return pop_matrices, detect_matrices
    
# =============================================================================
# 5. UTILITY FUNCTIONS
# =============================================================================

    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of all available data.
        
        Returns
        -------
        Dict[str, Dict]
            Summary of available data by type
        """
        summary = {
            'stochastic': {},
            'euler_maruyama': {},
            'taylor_methods': {},
            'analysis': {}
        }
        
        # Count stochastic data
        stochastic_pop_path = self.base_dir / 'stochastic' / 'population_matrices'
        if stochastic_pop_path.exists():
            summary['stochastic']['population'] = len(list(stochastic_pop_path.glob('f*')))
        
        # Count Euler-Maruyama data
        em_pop_path = self.base_dir / 'euler_maruyama' / 'population_matrices'
        if em_pop_path.exists():
            summary['euler_maruyama']['population'] = len(list(em_pop_path.glob('f*')))
        
        # Count analysis data
        analysis_path = self.base_dir / 'analysis'
        if analysis_path.exists():
            summary['analysis']['count_rates'] = len(list((analysis_path / 'count_rates').glob('f*')))
            summary['analysis']['processed_data'] = len(list((analysis_path / 'processed_data').glob('f*')))
            
        return summary

# =============================================================================
# 6. COMPATIBILITY FUNCTIONS
# =============================================================================
    
    def load_euler_maruyama_basic(self, fission_vec, prefix='f'):
        """Compatibility function for loading Euler-Maruyama basic data."""
        return self.load_euler_maruyama_data(fission_vec, 'basic', prefix)

    def load_euler_maruyama_with_const_dead_time_data(self, fission_vec, prefix='f'):
        """Compatibility function for loading Euler-Maruyama const dead time data."""
        return self.load_euler_maruyama_data(fission_vec, 'const', prefix)

    def load_euler_maruyama_with_uniform_dead_time_data(self, fission_vec, prefix = 'f'):
        return self.load_euler_maruyama_data(fission_vec, 'uniform', prefix)
    
    def load_euler_maruyama_with_exp_dead_time_data(self, fission_vec, prefix='f'):
        """Compatibility function for loading Euler-Maruyama exp dead time data."""
        return self.load_euler_maruyama_data(fission_vec, 'exp', prefix)
    
    def load_taylor_with_const_dead_time_data(self, fission_vec, prefix = 'f'):
        """Compatibility function for loading Taylor const dead time data."""
        return self.load_taylor_data(fission_vec, 'strong', 'const', prefix)
    




