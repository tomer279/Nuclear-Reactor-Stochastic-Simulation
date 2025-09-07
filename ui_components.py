"""
UI Components for Nuclear Reactor Stochastic Simulation Dashboard

This file contains focused UI components that consolidate complex UI logic,
making the code more maintainable and organized.

PROBLEMS ADDRESSED:
==================
1. Complex Parameter Collection: Scattered parameter input logic throughout streamlit_app.py
2. Complex Results Display: Multiple functions handling results display with repetitive patterns
3. Complex Dead Time Analysis UI: Scattered dead time analysis UI logic

COMPONENT CLASSES:
==================
- ParameterInput: Consolidates parameter collection logic
- ResultsDisplay: Consolidates results display logic  
- DeadTimeAnalysisUI: Consolidates dead time analysis UI logic
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from models import SimulationParameters, SimulationResults, DeadTimeConfig, DeadTimeAnalysis
from services import DeadTimeAnalysisService

class ParameterInput:
    """Handles parameter input in sidebar"""
    
    def get_simulation_parameters(self) -> SimulationParameters:
        """Get simulation parameters from user input"""
        st.header("Simulation Parameters")
        
        # Fission rate selection
        available_fission_rates = [33.94, 33.95, 33.96, 33.97, 33.98, 
                                  33.982, 33.984, 33.986, 33.988, 33.99, 33.992]
        
        selected_fission_rates = st.multiselect(
            "Select Fission Rates (s⁻¹)",
            available_fission_rates,
            default=[33.94, 33.95, 33.96],
            help="Select multiple fission rates to compare"
        )
        
        if st.checkbox("All fission rates"):
            selected_fission_rates = available_fission_rates
        
        # Other parameters
        detection_rate = st.number_input("Detection Rate (s⁻¹)", 1.0, 50.0, 10.0)
        absorption_rate = st.number_input("Absorption Rate (s⁻¹)", 1.0, 20.0, 7.0)
        source_rate = st.number_input("Source Rate (s⁻¹)", 100.0, 5000.0, 1000.0)
        
        # Initial population
        st.subheader("Initial Population Settings")
        use_equilibrium = st.checkbox(
            "Use equilibrium values (recommended)",
            help="Set initial population to equilibrium for each fission rate. Required for theoretical CPS calculations."
        )
        
        initial_population = None
        if not use_equilibrium:
            initial_population = st.number_input("Initial Population", 10_000, 200_000, 50_000)
        
        simulation_steps = st.number_input("Simulation Steps", 10000, 1_000_000, 100_000)
        
        return SimulationParameters(
            fission_rates=selected_fission_rates,
            detection_rate=detection_rate,
            absorption_rate=absorption_rate,
            source_rate=source_rate,
            initial_population=initial_population,
            simulation_steps=simulation_steps,
            use_equilibrium=use_equilibrium
        )

class ResultsDisplay:
    """Handles display of simulation results"""
    
    def display_simulation_summary(self, params: SimulationParameters):
        """Display simulation parameters summary"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Number of Fission Rates", len(params.fission_rates))
        with col2:
            st.metric("Detection Rate", f"{params.detection_rate} s⁻¹")
        with col3:
            st.metric("Absorption Rate", f"{params.absorption_rate} s⁻¹")
        with col4:
            st.metric("Source Rate", f"{params.source_rate} s⁻¹")
    
    def display_comparative_plot(self, results: Dict[float, SimulationResults]):
        """Display comparative population evolution plot"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sorted_fission_rates = sorted(results.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        
        for i, fission_rate in enumerate(sorted_fission_rates):
            result = results[fission_rate]
            ax.plot(result.time_matrix[0], result.population_matrix[0], 
                   color=colors[i], linewidth=2, 
                   label=f'Fission Rate: {fission_rate}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Population')
        ax.set_title('Comparative Population Evolution: Multiple Fission Rates')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    def display_summary_table(self, results: Dict[float, SimulationResults]):
        """Display summary table of results"""
        summary_data = []
        
        for fission_rate in sorted(results.keys()):
            result = results[fission_rate]
            summary_data.append({
                'Fission Rate': fission_rate,
                'Final Population': f"{result.get_final_population():.0f}",
                'Mean Population': f"{result.get_mean_population():.1f}",
                'Max Population': f"{result.get_max_population():.0f}",
                'Min Population': f"{result.get_min_population():.0f}"
            })
        
        st.table(summary_data)

class DeadTimeAnalysisUI:
    """Handles dead time analysis UI"""
    
    def __init__(self):
        self.service = DeadTimeAnalysisService()
    
    def display_analysis_form(self, is_first_analysis: bool = True) -> DeadTimeConfig:
        """Display form for dead time analysis configuration"""
        if is_first_analysis:
            st.write("**Configure Dead Time Analysis:**")
        else:
            st.subheader("Add New Dead Time Analysis")
        
        distribution_type = st.selectbox("Dead Time Type", 
                                       ["Constant", "Uniform", "Normal", "Gamma"],
                                       help = "Constant: Fixed dead time value. Random: Variable dead time with specified distribution.")
        
        if distribution_type == 'Constant':
            # For constant dead time, only show mean dead time
            col1, col2 = st.columns(2)
            with col1:
                mean_dead_time = st.number_input("Dead Time (μs)", 0.1, 10.0, 1.0,
                                                 help = "Fixed dead time value for all detections")
            with col2:
                st.info("**Constant Dead Time**: All detections have the same dead time value. No standard deviation needed.")
            
            # Set std_percent to 0 for constant dead time (not used in calculations)
            std_percent = 0.0
        
        else:
            col1, col2 = st.columns(2)
            with col1:
                mean_dead_time = st.number_input("Mean Dead Time (μs)", 0.1, 10.0, 1.0,
                                                 help=f"Average dead time for {distribution_type.lower()} distribution")
            with col2:
                std_percent = st.number_input("Dead Time Std Dev (%)", 1.0, 100.0, 10.0,
                                              help=f"Standard deviation as percentage of mean for {distribution_type.lower()} distribution")
            # Show distribution-specific information
            if distribution_type == "Normal":
                st.info("**Normal Distribution**: Dead time follows Gaussian distribution with specified mean and standard deviation.")
            elif distribution_type == "Uniform":
                st.info("**Uniform Distribution**: Dead time is uniformly distributed between calculated bounds.")
            elif distribution_type == "Gamma":
                st.info("**Gamma Distribution**: Dead time follows Gamma distribution with specified mean and standard deviation.")
        
        config = DeadTimeConfig(
            mean_dead_time=mean_dead_time,
            std_percent=std_percent,
            distribution_type=distribution_type,
            analysis_name=""  # Will be set below
        )
        
        analysis_name = st.text_input("Analysis Name", value=config.get_display_name())
        config.analysis_name = analysis_name
        
        return config
    
    def display_cps_comparison_plot(self, analyses: List[DeadTimeAnalysis]):
        """Display CPS vs alpha inverse comparison plot matching test file style"""
        # Setup the plotting style 
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (14, 9),
            'font.size': 12,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 10,
            'figure.dpi': 100
            })
    
        # Create the plot 
        fig, ax = plt.subplots(figsize=(14, 9))
    
        # Professional color and marker schemes
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        markers = ['o-', 's-', '^-', 'd-', 'v-', 'p-']
    
        for i, analysis in enumerate(analyses):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
        
            # Get fission rates and sort them
            fission_rates = sorted(analysis.cps_results.keys())
            alpha_inv_values = [analysis.cps_results[f]['alpha_inv'] for f in fission_rates]
            cps_values = [analysis.cps_results[f]['cps'] for f in fission_rates]
        
            # Plot simulated CPS with professional styling
            ax.plot(alpha_inv_values, cps_values, marker, 
                    label=f"{analysis.config.analysis_name} (Simulated)", 
                    color=color, markersize=8, linewidth=2)
        
            # Add theoretical CPS if available
            if analysis.has_theoretical_cps():
                theoretical_cps_values = [analysis.theoretical_cps[f] for f in fission_rates]
                ax.plot(alpha_inv_values, theoretical_cps_values, '--', 
                        label=f"{analysis.config.analysis_name} (Theoretical)", 
                        color=color, linewidth=2, alpha=0.7)
    
        # Professional styling
        ax.set_xticks(alpha_inv_values)
        ax.set_xticklabels([f'{val:.1f}' for val in alpha_inv_values], rotation=45, fontsize=10)
        ax.set_title("Count Rate Comparison: Simulation vs Theory (Multiple Dead Time Configurations)", 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel("1/α (inverse Rossi-alpha)", fontsize=14)
        ax.set_ylabel("Counts per Second (CPS)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best', bbox_to_anchor=(1.05, 1))
    
        # Add parameter information box
        app_state = st.session_state.app_state
        params = app_state.simulation_params
        
        info_text = (f"Detection rate: {params.detection_rate} s⁻¹\n"
                     f"Simulation steps: {params.simulation_steps}\n"
                     f"Fission rates: {len(params.fission_rates)} selected\n"
                     f"Dead time analyses: {len(analyses)} configurations")
    
        ax.text(0.02, 0.98, info_text, 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
        plt.tight_layout()
        st.pyplot(fig)