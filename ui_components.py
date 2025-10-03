"""Written by Tomer279 with the assistance of Cursor.ai.

User interface components for nuclear reactor simulation dashboard.

This module provides comprehensive UI components for the nuclear reactor
stochastic simulation dashboard, consolidating complex user interface logic
into focused, maintainable classes. It handles parameter input collection,
results visualization, and dead time analysis configuration through
streamlined Streamlit-based interfaces.

Classes:
    ParameterInput:
        Handles simulation parameter collection and validation
    ResultsDisplay:
        Manages simulation results visualization and summary display
    DeadTimeAnalysisUI:
        Provides dead time analysis configuration and comparison interfaces

Key Features:
    - Streamlined parameter input with validation and help text
    - Interactive simulation configuration with real-time feedback
    - Comprehensive results visualization with comparative plots
    - Professional dead time analysis configuration interface
    - Multi-parameter comparison and statistical analysis display
    - Responsive layout with organized information presentation

UI Components:
    - Parameter input forms with validation and help systems
    - Results visualization with comparative plots and summary tables
    - Dead time analysis configuration with distribution selection
    - Professional plotting with consistent styling and legends
    - Interactive controls for simulation parameter adjustment

Dependencies:
    streamlit: For web-based user interface components
    matplotlib: For professional plotting and visualization
    numpy: For numerical operations and data processing
    models: Data model classes for simulation parameters and results
    services: Business logic services for analysis operations

Usage Examples:
    # Initialize UI components
    param_input = ParameterInput()
    results_display = ResultsDisplay()
    dead_time_ui = DeadTimeAnalysisUI()

    # Collect simulation parameters
    params = param_input.get_simulation_parameters()
    if param_input.validate_parameters(params):
        # Run simulation and display results
        results_display.display_simulation_summary(params)
        results_display.display_comparative_plot(results)

    # Configure dead time analysis
    dead_time_config = dead_time_ui.display_analysis_form()
    dead_time_ui.display_cps_comparison_plot(analyses)

Note:
    This module handles the user interface layer of the simulation dashboard.
    For core simulation logic, see the respective simulation modules.
    For data models, see models.py. For business logic, see services.py.
"""
from typing import Dict, List
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from models import (SimulationParameters,
                    SimulationResults,
                    DeadTimeConfig,
                    DeadTimeAnalysis)
from services import DeadTimeAnalysisService


class ParameterInput:
    """
    User interface component for simulation parameter
    collection and validation.

    This class provides a comprehensive interface for collecting simulation
    parameters from users through Streamlit widgets, including fission rate
    selection, physical constants, initial conditions, and simulation settings.

    It includes built-in validation and helpful guidance
    for parameter selection.

    Public Methods
    --------------
    get_simulation_parameters()
        Collect all simulation parameters from user interface
    validate_parameters(params)
        Validate collected parameters for consistency and correctness

    Examples
    --------
    >>> param_input = ParameterInput()
    >>> params = param_input.get_simulation_parameters()
    >>> if param_input.validate_parameters(params):
    ...     # Parameters are valid, proceed with simulation
    ...     pass
    """

    def get_simulation_parameters(self) -> SimulationParameters:
        """
        Collect comprehensive simulation parameters from user interface.

        This method creates an interactive form for users to configure
        all aspects of nuclear reactor simulation including fission rates,
        physical constants, initial conditions, and simulation settings.
        The interface provides helpful guidance and reasonable defaults.

        Returns
        -------
        SimulationParameters
            Complete simulation configuration object containing:
              - Selected fission rates for parameter sweep
              - Physical constants (detection, absorption, source rates)
              - Initial population settings
              - Simulation duration and step configuration

        Notes
        -----
        The interface includes:
          - Multi-select fission rate picker with common reactor values
          - Number inputs for physical constants with validation ranges
          - Checkbox for equilibrium initial conditions
          - Simulation step configuration with performance guidance
        """
        st.header("Simulation Parameters")

        # Fission rate selection
        available_fission_rates = [33.94, 33.95, 33.96, 33.97, 33.98,
                                   33.982, 33.984, 33.986, 33.988,
                                   33.99, 33.992]

        selected_fission_rates = st.multiselect(
            "Select Fission Rates (s⁻¹)",
            available_fission_rates,
            default=[33.94, 33.95, 33.96],
            help="Select multiple fission rates to compare"
        )

        if st.checkbox("All fission rates"):
            selected_fission_rates = available_fission_rates

        # Other parameters
        detection_rate = st.number_input(
            "Detection Rate (s⁻¹)", 1.0, 50.0, 10.0)
        absorption_rate = st.number_input(
            "Absorption Rate (s⁻¹)", 1.0, 20.0, 7.0)
        source_rate = st.number_input(
            "Source Rate (s⁻¹)", 100.0, 5000.0, 1000.0)

        # Initial population
        st.subheader("Initial Population Settings")
        use_equilibrium = st.checkbox(
            "Use equilibrium values (recommended)",
            help=(
                "Set initial population to equilibrium for each fission rate."
                " Required for theoretical CPS calculations.")
        )

        initial_population = None
        if not use_equilibrium:
            initial_population = st.number_input(
                "Initial Population", 10_000, 200_000, 50_000)

        simulation_steps = st.number_input(
            "Simulation Steps", 10000, 1_000_000, 100_000)

        return SimulationParameters(
            fission_rates=selected_fission_rates,
            detection_rate=detection_rate,
            absorption_rate=absorption_rate,
            source_rate=source_rate,
            initial_population=initial_population,
            simulation_steps=simulation_steps,
            use_equilibrium=use_equilibrium
        )

    def validate_parameters(
            self,
            params: SimulationParameters) -> bool:
        """
        Validate simulation parameters for consistency and correctness.

        This method performs comprehensive validation of user-provided
        parameters to ensure they are physically reasonable and will
        produce meaningful simulation results.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters object to validate

        Returns
        -------
        bool
            True if all parameters are valid, False otherwise.
            Error messages are displayed to the user for invalid parameters.

        Notes
        -----
        Validation checks include:
          - At least one fission rate selected
          - All rate constants are positive
          - Simulation steps meet minimum requirements
          - Parameter combinations are physically reasonable
        """
        if not params.fission_rates:
            st.error("Please select at least one fission rate.")
            return False

        if params.detection_rate <= 0:
            st.error("Detection rate must be positive.")
            return False

        if params.absorption_rate <= 0:
            st.error("Absorption rate must be positive.")
            return False

        if params.source_rate <= 0:
            st.error("Source rate must be positive.")
            return False

        if params.simulation_steps < 1000:
            st.error("Simulation steps should be at least 1000.")
            return False

        return True


class ResultsDisplay:
    """
    User interface component for simulation results visualization and display.

    This class provides comprehensive visualization capabilities for simulation
    results including parameter summaries, comparative plots, and statistical
    analysis tables. It creates professional, publication-ready visualizations
    with consistent styling and clear information presentation.

    Public Methods
    --------------
    display_simulation_summary(params)
        Display simulation parameter summary with key metrics
    display_comparative_plot(results)
        Create comparative population evolution plots
    display_summary_table(results)
        Generate summary statistics table

    Examples
    --------
    >>> results_display = ResultsDisplay()
    >>> results_display.display_simulation_summary(params)
    >>> results_display.display_comparative_plot(results)
    >>> results_display.display_summary_table(results)
    """

    def display_simulation_summary(
            self,
            params: SimulationParameters) -> None:
        """
        Display comprehensive simulation parameter summary.

        This method creates a clean, organized display of key simulation
        parameters using Streamlit metrics for easy comparison and
        verification of simulation configuration.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters to display

        Notes
        -----
        The summary includes:
          - Number of fission rates selected
          - Key physical constants with units
          - Simulation configuration details
          - Parameter validation status
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Number of Fission Rates", len(params.fission_rates))
        with col2:
            st.metric("Detection Rate", f"{params.detection_rate} s⁻¹")
        with col3:
            st.metric("Absorption Rate", f"{params.absorption_rate} s⁻¹")
        with col4:
            st.metric("Source Rate", f"{params.source_rate} s⁻¹")

    def display_comparative_plot(
            self,
            results: Dict[float, SimulationResults]) -> None:
        """
        Create professional comparative population evolution plot.

        This method generates a high-quality comparative plot showing
        population evolution across different fission rates, with
        professional styling, clear legends, and publication-ready formatting.

        Parameters
        ----------
        results : Dict[float, SimulationResults]
            Dictionary mapping fission rates to simulation results

        Notes
        -----
        The plot includes:
          - Population evolution curves for each fission rate
          - Professional color scheme with clear differentiation
          - Comprehensive legend with fission rate values
          - Grid and axis labels for easy interpretation
          - Responsive layout optimized for dashboard display
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        sorted_fission_rates = sorted(results.keys())
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(results)))

        for i, fission_rate in enumerate(sorted_fission_rates):
            result = results[fission_rate]
            ax.plot(result.time_matrix[0], result.population_matrix[0],
                    color=colors[i], linewidth=2,
                    label=f'Fission Rate: {fission_rate}')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Population')
        ax.set_title(
            'Comparative Population Evolution: Multiple Fission Rates')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        st.pyplot(fig)

    def display_summary_table(
            self,
            results: Dict[float, SimulationResults]) -> None:
        """
        Generate comprehensive summary statistics table.

        This method creates a detailed summary table showing key statistics
        for each fission rate simulation, including population statistics
        and performance metrics.

        Parameters
        ----------
        results : Dict[float, SimulationResults]
            Dictionary mapping fission rates to simulation results

        Notes
        -----
        The table includes:
          - Fission rate values
          - Final, mean, maximum, and minimum population statistics
          - Formatted numerical values for easy reading
          - Sorted by fission rate for consistent presentation
        """
        if not results:
            st.warning("No results available for summary table.")
            return

        st.subheader("Simulation Results Summary")

        # Prepare summary data
        summary_data = []
        for fission_rate in sorted(results.keys()):
            result = results[fission_rate]
            pop_range = (result.get_max_population()
                         - result.get_min_population())
            summary_data.append({
                'Fission Rate (s⁻¹)': fission_rate,
                'Final Population': f"{result.get_final_population():.0f}",
                'Mean Population': f"{result.get_mean_population():.1f}",
                'Max Population': f"{result.get_max_population():.0f}",
                'Min Population': f"{result.get_min_population():.0f}",
                'Population Range': f"{pop_range:.0f}"
            })

        # Display formatted table
        st.table(summary_data)


class DeadTimeAnalysisUI:
    """
    User interface component for dead time analysis
    configuration and visualization.

    This class provides comprehensive interfaces for configuring dead time
    analysis parameters, including distribution selection, parameter input,
    and comparative visualization of count rate analysis results. It supports
    multiple dead time distributions and creates professional comparison plots.

    Attributes
    ----------
    service : DeadTimeAnalysisService
        Service layer for dead time analysis operations

    Public Methods
    --------------
    display_analysis_form(is_first_analysis)
        Display configuration form for dead time analysis
    display_cps_comparison_plot(analyses)
        Create professional CPS comparison plots

    Private Methods
    ---------------
    _display_form_header(is_first_analysis)
        Display appropriate form header
    _get_distribution_type_selection()
        Get dead time distribution type from user
    _handle_constant_dead_time_input()
        Handle constant dead time parameter input
    _handle_random_dead_time_input(distribution_type)
        Handle random dead time parameter input
    _display_distribution_info(distribution_type)
        Display distribution-specific information
    _create_and_configure_dead_time_config(mean_dead_time, std_percent, distribution_type)
        Create and configure DeadTimeConfig object
    _add_parameter_info_box(ax, analyses)
        Add parameter information box to plots

    Examples
    --------
    >>> dead_time_ui = DeadTimeAnalysisUI()
    >>> config = dead_time_ui.display_analysis_form()
    >>> dead_time_ui.display_cps_comparison_plot(analyses)
    """

    def __init__(self):
        """Initialize dead time analysis UI with service layer."""
        self.service = DeadTimeAnalysisService()

    def display_analysis_form(
            self,
            is_first_analysis: bool = True) -> DeadTimeConfig:
        """
        Display comprehensive dead time analysis configuration form.

        This method creates an interactive form for configuring dead time
        analysis parameters, including distribution type selection,
        parameter input, and analysis naming. It provides helpful guidance
        and validation for different dead time distribution types.

        Parameters
        ----------
        is_first_analysis : bool, default True
            Whether this is the first analysis being configured

        Returns
        -------
        DeadTimeConfig
            Configured dead time analysis parameters

        Notes
        -----
        The form includes:
          - Distribution type selection (Constant, Uniform, Normal, Gamma)
          - Parameter input with appropriate validation
          - Distribution-specific help and information
          - Analysis naming for identification
        """
        self._display_form_header(is_first_analysis)

        distribution_type = self._get_distribution_type_selection()

        if distribution_type == 'Constant':
            mean_dead_time, std_percent = (
                self._handle_constant_dead_time_input())
        else:
            mean_dead_time, std_percent = (
                self._handle_random_dead_time_input(
                    distribution_type)
            )

        return self._create_and_configure_dead_time_config(
            mean_dead_time, std_percent, distribution_type)

    def display_cps_comparison_plot(
            self,
            analyses: List[DeadTimeAnalysis]) -> None:
        """
        Create professional CPS vs alpha inverse comparison plot.

        This method generates a high-quality comparison plot showing
        count rates per second (CPS) versus inverse Rossi-alpha values
        for multiple dead time configurations, with both simulated and
        theoretical results where available.

        Parameters
        ----------
        analyses : List[DeadTimeAnalysis]
            List of dead time analysis results to compare

        Notes
        -----
        The plot includes:
          - Professional styling with consistent colors and markers
          - Both simulated and theoretical CPS curves
          - Clear legends and axis labels
          - Parameter information box
          - Publication-ready formatting
        """
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
        markers = ['o', 's', '^', 'd', 'v', 'p']

        for i, analysis in enumerate(analyses):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            # Get fission rates and sort them
            fission_rates = sorted(analysis.cps_results.keys())
            alpha_inv_values = [analysis.cps_results[f]
                                ['alpha_inv'] for f in fission_rates]
            cps_values = [analysis.cps_results[f]['cps']
                          for f in fission_rates]

            # Plot simulated CPS with professional styling
            ax.plot(alpha_inv_values, cps_values, marker,
                    label=f"{analysis.config.analysis_name} (Simulated)",
                    color=color, markersize=8, linestyle='None')

            # Add theoretical CPS if available
            if analysis.has_theoretical_cps():
                theoretical_cps_values = [
                    analysis.theoretical_cps[f] for f in fission_rates]
                ax.plot(alpha_inv_values, theoretical_cps_values, '--',
                        label=f"{analysis.config.analysis_name} (Theoretical)",
                        color=color, linewidth=2, alpha=0.7)

        # Professional styling
        ax.set_xticks(alpha_inv_values)
        ax.set_xticklabels(
            [f'{val:.1f}' for val in alpha_inv_values],
            rotation=45, fontsize=10)
        ax.set_title("Count Rate Comparison: "
                     "Simulation vs Theory "
                     "Multiple Dead Time Configurations)",
                     fontsize=16, fontweight='bold')
        ax.set_xlabel("1/α (inverse Rossi-alpha)", fontsize=14)
        ax.set_ylabel("Counts per Second (CPS)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best', bbox_to_anchor=(1.05, 1))

        # Add parameter information box
        self._add_parameter_info_box(ax, analyses)

        plt.tight_layout()
        st.pyplot(fig)

    def _add_parameter_info_box(
            self,
            ax,
            analyses: list[DeadTimeAnalysis]) -> None:
        """Add parameter information box to the plot."""
        app_state = st.session_state.app_state
        params = app_state.simulation_params

        info_text = (
            f"Detection rate: {params.detection_rate} s⁻¹\n"
            f"Simulation steps: {params.simulation_steps}\n"
            f"Fission rates: {len(params.fission_rates)} selected\n"
            f"Dead time analyses: {len(analyses)} configurations")

        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox={"boxstyle": 'round',
                      "facecolor": 'wheat',
                      "alpha": 0.8})

    def _display_form_header(
            self,
            is_first_analysis: bool):
        """Display the header for the analysis form."""
        if is_first_analysis:
            st.write("**Configure Dead Time Analysis:**")
        else:
            st.subheader("Add New Dead Time Analysis")

    def _get_distribution_type_selection(self) -> str:
        """Get the distribution type selection from user."""
        return st.selectbox(
            "Dead Time Type",
            ["Constant", "Uniform", "Normal", "Gamma"],
            help=("Constant: Fixed dead time value. "
                  "Random: Variable dead time with specified distribution.")
        )

    def _handle_constant_dead_time_input(self) -> tuple[float, float]:
        """Handle input for constant dead time configuration."""
        col1, col2 = st.columns(2)
        with col1:
            mean_dead_time = st.number_input(
                "Dead Time (μs)", 0.1, 10.0, 1.0,
                help="Fixed dead time value for all detections")
        with col2:
            st.info(
                "**Constant Dead Time**: "
                "All detections have the same dead time value. "
                "No standard deviation needed.")

        return mean_dead_time, 0.0

    def _handle_random_dead_time_input(
            self,
            distribution_type: str) -> tuple[float, float]:
        """Handle input for random dead time configuration."""
        col1, col2 = st.columns(2)
        with col1:
            mean_dead_time = st.number_input(
                "Mean Dead Time (μs)", 0.1, 10.0, 1.0,
                help=f"Average dead time for {distribution_type.lower()} "
                "distribution")
        with col2:
            std_percent = st.number_input(
                "Dead Time Std Dev (%)", 1.0, 100.0, 10.0,
                help="Standard deviation as percentage of mean for "
                     f"{distribution_type.lower()} distribution")

        self._display_distribution_info(distribution_type)
        return mean_dead_time, std_percent

    def _display_distribution_info(
            self,
            distribution_type: str) -> None:
        """Display distribution-specific information."""
        if distribution_type == "Normal":
            st.info(
                "**Normal Distribution**: Dead time follows "
                "Gaussian distribution with specified mean "
                "and standard deviation.")
        elif distribution_type == "Uniform":
            st.info(
                "**Uniform Distribution**: Dead time is uniformly "
                "distributed between calculated bounds.")
        elif distribution_type == "Gamma":
            st.info(
                "**Gamma Distribution**: Dead time follows Gamma "
                "distribution with specified mean "
                "and standard deviation.")

    def _create_and_configure_dead_time_config(
            self,
            mean_dead_time: float,
            std_percent: float,
            distribution_type: str) -> DeadTimeConfig:
        """Create and configure the DeadTimeConfig object."""
        config = DeadTimeConfig(
            mean_dead_time=mean_dead_time,
            std_percent=std_percent,
            distribution_type=distribution_type,
            analysis_name=""  # Will be set below
        )

        analysis_name = st.text_input("Analysis Name",
                                      value=config.get_display_name())

        config.analysis_name = analysis_name
        return config
