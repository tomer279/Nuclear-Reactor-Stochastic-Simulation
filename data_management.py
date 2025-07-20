""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Data management functions for loading and saving simulation results.

This module provides functions to load simulation data from files,
manage data structures, and handle data validation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


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
            'analysis/count_rates',
            'analysis/processed_data'
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
        
        print(f"Data directory structure created at: {self.base_dir}")
    
    
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
        
        # Save population matrix
        pop_path = self.base_dir / 'stochastic' / 'population_matrices' / fission_dir
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / f'Simul_Pop_Matrix_{prefix}{fission_value}.npy', pop_mat)
        
        # Save time matrix
        time_path = self.base_dir / 'stochastic' / 'time_matrices' / fission_dir
        time_path.mkdir(exist_ok=True)
        np.save(time_path / f'Simul_Time_Matrix_{prefix}{fission_value}.npy', time_mat)
        
        # Save detection matrix
        detect_path = self.base_dir / 'stochastic' / 'detection_matrices' / fission_dir
        detect_path.mkdir(exist_ok=True)
        np.save(detect_path / f'Detection_Matrix_{prefix}{fission_value}.npy', detect_mat)
        
        print(f"Stochastic data saved for fission = {fission_value}")
    
    
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
        
        # Save population data
        pop_path = self.base_dir / 'euler_maruyama' / 'population_matrices' / fission_dir
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / f'EM_Pop_{dead_time_type}_Dead_Time_{prefix}{fission_value}.npy', pop_data)
        
        # Save detection data
        detect_path = self.base_dir / 'euler_maruyama' / 'detection_matrices' / fission_dir
        detect_path.mkdir(exist_ok=True)
        np.save(detect_path / f'EM_Detect_{dead_time_type}_Dead_Time_{prefix}{fission_value}.npy', detect_data)
        
        print(f"Euler-Maruyama data saved for fission = {fission_value}, type = {dead_time_type}")
    
    
    def save_taylor_data(self, pop_data: np.ndarray, time_data: np.ndarray,
                        fission_value: float, method: str,
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
        prefix : str
            Prefix for file names
        """
        fission_dir = f"{prefix}{fission_value}"
        
        # Save population data
        pop_path = self.base_dir / 'taylor_methods' / 'population_matrices' / fission_dir
        pop_path.mkdir(exist_ok=True)
        np.save(pop_path / f'{method}_Taylor_Pop_f{fission_value}.npy', pop_data)
        
        # Save time data
        time_path = self.base_dir / 'taylor_methods' / 'time_matrices' / fission_dir
        time_path.mkdir(exist_ok=True)
        np.save(time_path / f'{method}_Taylor_Time_f{fission_value}.npy', time_data)
        
        print(f"Taylor data saved for fission = {fission_value}, method = {method}")
    
    
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
            
            # Load population matrix
            pop_path = (self.base_dir / 'stochastic' /
                        'population_matrices' / fission_dir / 
                        f'Simul_Pop_Matrix_{prefix}{fission}.npy')
            if pop_path.exists():
                pop_matrices.append(np.load(pop_path))
            
            # Load time matrix
            time_path = (self.base_dir / 'stochastic' /
                         'time_matrices' / fission_dir /
                         f'Simul_Time_Matrix_{prefix}{fission}.npy')
            if time_path.exists():
                time_matrices.append(np.load(time_path))
            
            # Load detection matrix
            detect_path = (self.base_dir / 'stochastic' / 
                           'detection_matrices' / fission_dir / 
                           f'Detection_Matrix_{prefix}{fission}.npy')
            if detect_path.exists():
                detect_matrices.append(np.load(detect_path))
            
            print(f"Loaded stochastic data for fission = {fission}")
        
        return pop_matrices, time_matrices, detect_matrices
    

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
            fission_dir = f"{prefix}{fission}"
            
            # Load population data
            pop_path = (self.base_dir / 'euler_maruyama' /
                        'population_matrices' / fission_dir /
                        f'EM_Pop_{dead_time_type}_Dead_Time_{prefix}{fission}.npy')
            if pop_path.exists():
                pop_matrices.append(np.load(pop_path))
            
            # Load detection data
            detect_path = (self.base_dir / 'euler_maruyama' / 
                           'detection_matrices' / fission_dir 
                           / f'EM_Detect_{dead_time_type}_Dead_Time_{prefix}{fission}.npy')
            if detect_path.exists():
                detect_matrices.append(np.load(detect_path))
            
            print(f"Loaded Euler-Maruyama data for fission = {fission}, type = {dead_time_type}")
        
        return pop_matrices, detect_matrices
    
    
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
    
    def load_euler_maruyama_basic(self, fission_vec, prefix='f'):
        """Compatibility function for loading Euler-Maruyama basic data."""
        return self.load_euler_maruyama_data(fission_vec, 'basic', prefix)

    def load_euler_maruyama_with_const_dead_time_data(self, fission_vec, prefix='f'):
        """Compatibility function for loading Euler-Maruyama const dead time data."""
        return self.load_euler_maruyama_data(fission_vec, 'const', prefix)

    def load_euler_maruyama_with_exp_dead_time_data(self, fission_vec, prefix='f'):
        """Compatibility function for loading Euler-Maruyama exp dead time data."""
        return self.load_euler_maruyama_data(fission_vec, 'exp', prefix)




