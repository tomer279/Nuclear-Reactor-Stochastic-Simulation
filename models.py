"""
Data Models for Nuclear Reactor Stochastic Simulation Dashboard

This module provides structured data classes that solve common problems in the 
Streamlit application by replacing complex parameter passing, confusing session 
state management, and unclear return values with clear, self-documenting objects.

PROBLEMS SOLVED:
================

Problem 1: Complex Parameter Passing
------------------------------------
Before: Functions require many individual parameters, making them hard to use and error-prone.
After: Functions accept a single, well-structured parameter object.

Example:
    # Before: Hard to remember parameter order, easy to mix up
    _run_simulations_for_selected_rates(
        selected_fission_rates, detection_rate, absorption_rate, 
        source_rate, simulation_steps, initial_population
    )
    
    # After: Clear, self-documenting parameter object
    def run_simulations(params: SimulationParameters) -> Dict[float, SimulationResults]:
        ...

Problem 2: Confusing Session State Management
---------------------------------------------
Before: Complex nested dictionaries scattered across session state.
After: Single source of truth with clear data access patterns.

Example:
    # Before: Complex nested access patterns
    st.session_state.dead_time_results[analysis_name]['theoretical_cps'][fission_rate]
    st.session_state.simulation_params['detection_rate']
    st.session_state.dead_time_params[analysis_name]['mean_dead_time']
    
    # Complex initialization required:
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'simulation_params' not in st.session_state:
        st.session_state.simulation_params = None
    # ... more initialization code
    
    # After: Simple, clear access patterns
    app_state.simulation_params.detection_rate
    app_state.get_analysis_by_name("Normal τ=1.0μs").get_cps_for_fission_rate(33.94)
    
    # Simple initialization:
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()

Problem 3: Unclear Return Values
--------------------------------
Before: Functions return multiple separate arrays with unclear structure.
After: Functions return structured objects with clear methods.

Example:
    # Before: Unclear what the returned values represent
    time_matrix, population_matrix, detection_matrix = run_stochastic_simulation(...)
    
    # After: Clear, structured return value
    def run_simulation(params: SimulationParameters) -> SimulationResults:
        return SimulationResults(
            time_matrix=time_matrix,
            population_matrix=population_matrix,
            detection_matrix=detection_matrix,
            fission_rate=fission_rate
        )
    
    # Clear usage:
    result = run_simulation(params)
    final_pop = result.get_final_population()
    max_pop = result.get_max_population()

Problem 4: Complex Dead Time Configuration
------------------------------------------
Before: Dead time parameters scattered across multiple places and stored separately.
After: All parameters bundled together with helpful utility methods.

Example:
    # Before: Parameters scattered across UI and stored separately
    mean_dead_time = st.number_input("Mean Dead Time (μs)", 0.1, 10.0, 1.0)
    dead_time_std_percent = st.number_input("Dead Time Std Dev (%)", 1.0, 100.0, 10.0)
    dead_time_type = st.selectbox("Dead Time Type", ["Constant", "Uniform", "Normal", "Gamma"])
    
    # Stored separately:
    st.session_state.dead_time_params[analysis_name] = {
        'mean_dead_time': mean_dead_time,
        'dead_time_std_percent': dead_time_std_percent,
        'dead_time_type': dead_time_type
    }
    
    # After: Everything bundled together with utility methods
    config = DeadTimeConfig(
        mean_dead_time=1.0,
        std_percent=10.0,
        distribution_type="normal",
        analysis_name="Normal τ=1.0μs σ=10%"
    )
    
    # Helpful methods available:
    display_name = config.get_display_name()  # "Normal τ=1.0μs σ=10%"
    tau_seconds = config.get_tau_seconds()     # 0.000001
    uniform_bounds = config.get_uniform_bounds()  # (low, high) for uniform distribution

Problem 5: Scattered Dead Time Results
--------------------------------------
Before: Results stored in nested dictionaries with complex access patterns.
After: Results stored in structured objects with clear access methods.

Example:
    # Before: Complex nested dictionary structure
    st.session_state.dead_time_results[analysis_name] = {
        fission_rate: {
            'cps': cps_value,
            'alpha_inv': alpha_inv_value
        }
    }
    
    # Theoretical CPS added separately:
    st.session_state.dead_time_results[analysis_name]['theoretical_cps'] = {
        fission_rate: theoretical_cps_value
    }
    
    # Complex access patterns:
    cps = st.session_state.dead_time_results[analysis_name][fission_rate]['cps']
    theoretical = st.session_state.dead_time_results[analysis_name]['theoretical_cps'][fission_rate]
    
    # After: Clear, structured access
    analysis = app_state.get_analysis_by_name("Normal τ=1.0μs")
    cps = analysis.get_cps_for_fission_rate(33.94)
    theoretical = analysis.get_theoretical_cps_for_fission_rate(33.94)
    error_percent = analysis.get_error_percentage(33.94)

DATA CLASSES:
=============

SimulationParameters: Bundles all simulation input parameters
DeadTimeConfig: Configuration for dead time analysis with utility methods
SimulationResults: Results from a single simulation with statistics methods
DeadTimeAnalysis: Results from dead time analysis with comparison methods
AppState: Main application state replacing complex session state management

HELPER FUNCTIONS:
=================

create_default_simulation_parameters(): Creates default simulation parameters
create_dead_time_config(): Creates dead time configuration with automatic naming
"""


from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class SimulationParameters:
    """
    Parameters for running stochastic simulations.
    
    This class bundles all the parameters needed to run simulations,
    replacing the need to pass 6+ individual parameters to functions.
    """
    
    fission_rates: List[float]
    detection_rate : float
    absorption_rate: float
    source_rate : float 
    initial_population : Optional[int]
    simulation_steps : int
    use_equilibrium : bool
    
    def __post_init__(self):
        """Automatically handle equilibrium logic"""
        if self.use_equilibrium:
            self.initial_population = None
        
    def get_initial_population_for_fission_rate(self, fission_rate : float,
                                                equilibrium_value : float) -> int:
        """Get initial population for a specific fission rate"""
        if self.use_equilibrium:
            return int(equilibrium_value)
        return self.initial_population or 50_000
    
@dataclass
class DeadTimeConfig:
    """
    Configuration for dead time analysis.
    
    This class bundles dead time parameters and provides helpful methods
    for unit conversions and display formatting.
    """
    mean_dead_time: float  # in microseconds
    std_percent: float    # standard deviation as percentage of mean
    distribution_type: str  # 'constant', 'normal', 'uniform', 'gamma'
    analysis_name: str
    
    def get_display_name(self) -> str:
        """Generate display name for the analysis"""
        if self.distribution_type.lower() == 'constant':
            return f"{self.distribution_type} τ={self.mean_dead_time}μs"
        else:
            return f"{self.distribution_type} τ={self.mean_dead_time}μs σ={self.std_percent}%"
    
    def get_tau_seconds(self) -> float:
        """Convert microseconds to seconds"""
        return self.mean_dead_time * 1e-6
    
    def get_std_seconds(self) -> float:
        """Convert percentage to standard deviation in seconds"""
        return (self.std_percent / 100.0) * self.get_tau_seconds()
    
    def get_uniform_bounds(self) -> tuple[float, float]:
        """Get uniform distribution bounds (low, high)"""
        mean_s = self.get_tau_seconds()
        std_s = self.get_std_seconds()
        low = mean_s - np.sqrt(3) * std_s
        high = mean_s + np.sqrt(3) * std_s
        return low, high
    
    def get_gamma_parameters(self) -> tuple[float, float]:
        """Get gamma distribution parameters (shape, scale)"""
        mean_s = self.get_tau_seconds()
        std_s = self.get_std_seconds()
        shape = (mean_s / std_s) ** 2
        scale = (std_s ** 2) / mean_s
        return shape, scale
    
@dataclass
class SimulationResults:
    """
    Results from a single simulation run.
    
    This class bundles the three matrices returned by simulations
    and provides convenient methods for accessing statistics.
    """
    time_matrix: np.ndarray
    population_matrix: np.ndarray
    detection_matrix: np.ndarray
    fission_rate: float
    
    def get_final_population(self) -> float:
        """Get final population value"""
        return self.population_matrix[0][-1]
    
    def get_max_population(self) -> float:
        """Get maximum population value"""
        return np.max(self.population_matrix[0])
    
    def get_min_population(self) -> float:
        """Get minimum population value"""
        return np.min(self.population_matrix[0])
    
    def get_mean_population(self) -> float:
        """Get mean population value"""
        return np.mean(self.population_matrix[0])
    
    def get_population_std(self) -> float:
        """Get population standard deviation"""
        return np.std(self.population_matrix[0])
    
    def get_simulation_duration(self) -> float:
        """Get total simulation time in seconds"""
        return self.time_matrix[0][-1]
    
    def get_detection_count(self) -> int:
        """Get total number of detections"""
        valid_detections = self.detection_matrix[0]
        valid_detections = valid_detections[~np.isnan(valid_detections)]
        return len(valid_detections)
    
    def get_cps(self) -> float:
        """Get counts per second"""
        detection_count = self.get_detection_count()
        duration = self.get_simulation_duration()
        return detection_count / duration if duration > 0 else 0.0
    
@dataclass
class DeadTimeAnalysis:
    """
    Results from dead time analysis.
    
    This class bundles CPS results for multiple fission rates
    and optionally includes theoretical CPS calculations.
    """
    config: DeadTimeConfig
    cps_results: Dict[float, Dict[str, Any]]  # fission_rate -> {cps, alpha_inv}
    theoretical_cps: Optional[Dict[float, float]] = None  # fission_rate -> theoretical_cps
    
    def has_theoretical_cps(self) -> bool:
        """Check if theoretical CPS has been calculated"""
        return self.theoretical_cps is not None
    
    def get_cps_for_fission_rate(self, fission_rate: float) -> float:
        """Get simulated CPS for a specific fission rate"""
        return self.cps_results[fission_rate]['cps']
    
    def get_alpha_inv_for_fission_rate(self, fission_rate: float) -> float:
        """Get alpha inverse for a specific fission rate"""
        return self.cps_results[fission_rate]['alpha_inv']
    
    def get_theoretical_cps_for_fission_rate(self, fission_rate: float) -> Optional[float]:
        """Get theoretical CPS for a specific fission rate"""
        if self.theoretical_cps:
            return self.theoretical_cps[fission_rate]
        return None
    
    def get_error_percentage(self, fission_rate: float) -> Optional[float]:
        """Get percentage error between theoretical and simulated CPS"""
        theoretical = self.get_theoretical_cps_for_fission_rate(fission_rate)
        if theoretical is None:
            return None
        
        simulated = self.get_cps_for_fission_rate(fission_rate)
        return ((theoretical - simulated) / simulated * 100) if simulated > 0 else None
    
    def get_sorted_fission_rates(self) -> List[float]:
        """Get fission rates sorted numerically"""
        return sorted(self.cps_results.keys())
    
    def get_summary_data(self) -> List[Dict[str, Any]]:
        """Get summary data for table display"""
        summary_data = []
        
        for fission_rate in self.get_sorted_fission_rates():
            row_data = {
                'Fission Rate': fission_rate,
                'Alpha Inverse': f"{self.get_alpha_inv_for_fission_rate(fission_rate):.3f}",
                'CPS (Simulated)': f"{self.get_cps_for_fission_rate(fission_rate):.2f}"
            }
            
            theoretical = self.get_theoretical_cps_for_fission_rate(fission_rate)
            if theoretical is not None:
                error = self.get_error_percentage(fission_rate)
                row_data['CPS (Theoretical)'] = f"{theoretical:.2f}"
                row_data['Error (%)'] = f"{error:.1f}" if error is not None else "N/A"
            
            summary_data.append(row_data)
        
        return summary_data
    
@dataclass
class AppState:
    """
    Main application state for Streamlit app.
    
    This class replaces the complex session state management
    with a single, clear data structure.
    """
    simulation_params: Optional[SimulationParameters] = None
    simulation_results: Optional[Dict[float, SimulationResults]] = None
    dead_time_analyses: List[DeadTimeAnalysis] = field(default_factory=list)
    
    def has_simulation_results(self) -> bool:
        """Check if simulation results are available"""
        return self.simulation_results is not None
    
    def has_dead_time_analyses(self) -> bool:
        """Check if any dead time analyses have been run"""
        return len(self.dead_time_analyses) > 0
    
    def get_analysis_by_name(self, name: str) -> Optional[DeadTimeAnalysis]:
        """Get dead time analysis by name"""
        for analysis in self.dead_time_analyses:
            if analysis.config.analysis_name == name:
                return analysis
        return None
    
    def add_dead_time_analysis(self, analysis: DeadTimeAnalysis):
        """Add a new dead time analysis"""
        self.dead_time_analyses.append(analysis)
    
    def remove_dead_time_analysis(self, name: str) -> bool:
        """Remove dead time analysis by name"""
        for i, analysis in enumerate(self.dead_time_analyses):
            if analysis.config.analysis_name == name:
                del self.dead_time_analyses[i]
                return True
        return False
    
    def clear_dead_time_analyses(self):
        """Clear all dead time analyses"""
        self.dead_time_analyses.clear()
    
    def clear_all(self):
        """Clear all data"""
        self.simulation_params = None
        self.simulation_results = None
        self.dead_time_analyses.clear()
    
    def get_analysis_names(self) -> List[str]:
        """Get list of all analysis names"""
        return [analysis.config.analysis_name for analysis in self.dead_time_analyses]
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall statistics across all simulations"""
        if not self.has_simulation_results():
            return {}
        
        all_final_pops = [result.get_final_population() for result in self.simulation_results.values()]
        all_max_pops = [result.get_max_population() for result in self.simulation_results.values()]
        all_min_pops = [result.get_min_population() for result in self.simulation_results.values()]
        all_cps = [result.get_cps() for result in self.simulation_results.values()]
        
        return {
            'average_final_population': np.mean(all_final_pops),
            'average_max_population': np.mean(all_max_pops),
            'average_min_population': np.mean(all_min_pops),
            'average_cps': np.mean(all_cps),
            'total_simulations': len(self.simulation_results)
        }
    
    def get_dead_time_results_dict(self) -> Dict[str, Dict[float, Dict[str, Any]]]:
        """Convert dead time analyses to dictionary format for plotting"""
        results_dict = {}
        for analysis in self.dead_time_analyses:
            analysis_name = analysis.config.analysis_name
            results_dict[analysis_name] = {}
            
            for fission_rate in analysis.get_sorted_fission_rates():
                results_dict[analysis_name][fission_rate] = {
                    'cps' : analysis.get_cps_for_fission_rate(fission_rate),
                    'alpha_inv' : analysis.get_alpha_inv_for_fission_rate(fission_rate)
                    }
            if analysis.has_theoretical_cps():
                results_dict[analysis_name]['theoretical_cps'] = analysis.theoretical_cps
                
        return results_dict
    
    def update_analysis(self, updated_analysis: DeadTimeAnalysis) -> bool:
        """Update an existing dead time analysis"""
        for i, analysis in enumerate(self.dead_time_analyses):
            if analysis.config.analysis_name == updated_analysis.config.analysis_name:
                self.dead_time_analyses[i] = updated_analysis
                return True
        return False

# Helper functions for creating common configurations
def create_default_simulation_parameters() -> SimulationParameters:
    """Create default simulation parameters"""
    return SimulationParameters(
        fission_rates=[33.94, 33.95, 33.96],
        detection_rate=10.0,
        absorption_rate=7.0,
        source_rate=1000.0,
        initial_population=None,
        simulation_steps=100000,
        use_equilibrium=True
    )

def create_dead_time_config(mean_dead_time: float, std_percent: float, 
                           distribution_type: str, analysis_name: str = None) -> DeadTimeConfig:
    """Create dead time configuration with automatic naming"""
    if analysis_name is None:
        if distribution_type.lower() == 'constant':
            analysis_name = f"{distribution_type} τ={mean_dead_time}μs"
        else:
            analysis_name = f"{distribution_type} τ={mean_dead_time}μs σ={std_percent}%"
    
    return DeadTimeConfig(
        mean_dead_time=mean_dead_time,
        std_percent=std_percent,
        distribution_type=distribution_type,
        analysis_name=analysis_name
    )
