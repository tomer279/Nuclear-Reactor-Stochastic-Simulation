
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from models import SimulationParameters, DeadTimeAnalysis, AppState
from services import SimulationService, DeadTimeAnalysisService
from ui_components import ParameterInput, ResultsDisplay, DeadTimeAnalysisUI

# =============================================================================
# SERVICE INSTANCES
# =============================================================================

# Create service instances once
simulation_service = SimulationService()
dead_time_service = DeadTimeAnalysisService()

st.set_page_config(
    page_title = "Nuclear Reactor Stochastic Simulation Dashboard",
    layout = "wide"
    )

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'app_state' not in st.session_state:
    st.session_state.app_state = AppState()

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_population_plot(time_matrix, population_matrix, fission_rate = None):
    """Create population evolution plot following project plotting standards"""
    # Setup professional plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'figure.dpi': 100
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot population over time
    ax.plot(time_matrix[0], population_matrix[0], 'b-', linewidth=2, 
            label = (f'Neutron Population (Fission : {fission_rate})' if fission_rate
                     else 'Neutron Population'))
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Population')
    title = (f'Population Evolution (Fission Rate: {fission_rate})' if fission_rate 
             else 'Stochastic Simulation: Population Evolution')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Professional grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc = 'best')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)
    
    plt.tight_layout()
    return fig

# =============================================================================
# STATISTICS AND DATA PROCESSING FUNCTIONS
# =============================================================================

def calculate_detection_stats(detection_matrix):
    """Calculate detection statistics"""
    # Clean detection matrix (remove NaN values)
    valid_detections = detection_matrix[0]
    valid_detections = valid_detections[~np.isnan(valid_detections)]
    
    if len(valid_detections) == 0:
        return {
            'total_detections': 0,
            'cps': 0,
            'detection_times': []
        }
    
    # Calculate statistics
    total_detections = len(valid_detections)
    simulation_time = detection_matrix[0][-1] if not np.isnan(detection_matrix[0][-1]) else 0
    cps = total_detections / simulation_time if simulation_time > 0 else 0
    
    return {
        'total_detections': total_detections,
        'cps': cps,
        'detection_times': valid_detections.tolist()
    }

# =============================================================================
# DEAD TIME ANALYSIS FUNCTIONS
# =============================================================================

def _run_dead_time_analysis(results_dict, params, mean_dead_time,
                            dead_time_std_percent, dead_time_type, analysis_name):
    """Run dead time analysis on simulation results"""
    try:
        
        # Run analysis using service
        analysis = dead_time_service.run_analysis(
            results_dict, params, mean_dead_time,
            dead_time_std_percent, dead_time_type, analysis_name
            )
        
        # Store in AppState
        app_state = st.session_state.app_state
        app_state.add_dead_time_analysis(analysis)
        
        st.success(f"Dead time analysis {analysis_name} completed!")
        
    except Exception as e:
        st.error(f"Dead time analysis failed: {str(e)}")
        st.exception(e)
    
def _add_theoretical_cps_to_analysis(analysis_name, params):
    """Add theoretical CPS calculation to existing analysis"""
    try:
        app_state = st.session_state.app_state
        
        # Find the analysis by name
        analysis = app_state.get_analysis_by_name(analysis_name)
        if not analysis:
            st.error(f"Analysis {analysis_name} not found")
            return
        
        updated_analysis = dead_time_service.add_theoretical_cps(analysis, params)
        
        app_state.update_analysis(updated_analysis)
        
        st.success(f"Theoretical CPS added to {analysis_name}!")
            
    except Exception as e:
        st.error(f"Failed to add theoretical CPS: {str(e)}")
        st.exception(e)
        
# =============================================================================
# SIMULATION RUNNER FUNCTIONS
# =============================================================================

def _handle_simulation_run(run_sim, sim_params):
    """Handle simulation run logic"""
    if not run_sim:
        return
    
    if not sim_params.fission_rates:
        st.error("Please select at least one fission rate")
        return
    
    app_state = st.session_state.app_state
    app_state.clear_dead_time_analyses()
    
    _run_simulations_for_selected_rates(sim_params)
    
def _run_simulations_for_selected_rates(params: SimulationParameters):
    """ Run simulations for all selected fission rates using services"""
    try:
       
        results_dict = {}
        total_rates = len(params.fission_rates)
       
        # Create a container for progress updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, fission_rate in enumerate(params.fission_rates):
            # Update progress display
            progress = (i+1) / total_rates
            progress_bar.progress(progress)
            status_text.text(f"Simulating fission rate {fission_rate} ({i+1}/{total_rates})...")
        
            # Create single fission rate parameters
            single_rate_params = SimulationParameters(
                fission_rates = [fission_rate],
                detection_rate = params.detection_rate,
                absorption_rate = params.absorption_rate,
                source_rate = params.source_rate,
                initial_population = params.initial_population,
                simulation_steps = params.simulation_steps,
                use_equilibrium = params.use_equilibrium
            )
            
            single_result = simulation_service.run_multiple_simulations(single_rate_params)
            results_dict.update(single_result)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Store results in AppState
        app_state = st.session_state.app_state
        app_state.simulation_params = params
        app_state.simulation_results = results_dict
        st.success(f"Stochastic simulations completed for {len(params.fission_rates)} fission rates!")
    
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        st.exception(e)
        
    
# =============================================================================
# UI DISPLAY FUNCTIONS
# =============================================================================
    
def _display_simulation_tabs(results, results_display):
    """Display main simulation results tabs"""
    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["Comparative Analysis", "Individual Plots", "Summary Table"])
    
    with tab1:
        results_display.display_comparative_plot(results)
    with tab2:
        _display_individual_plots(results)
    with tab3:
        results_display.display_summary_table(results)
 
def _display_individual_plots(results):
    """Display individual plots tab"""
    st.subheader("Individual Population Plots")
    
    # Create individual plots for each fission rate
    sorted_fission_rates = sorted(results.keys())
    for fission_rate in sorted_fission_rates:
        result = results[fission_rate]
        _display_single_fission_plot(result, fission_rate)
        
def _display_single_fission_plot(result, fission_rate):
    """Display plot and statistics for a single fission rate"""
    st.write(f"**Fission Rate: {fission_rate}**")
    
    # Create and display plot
    fig = create_population_plot(result.time_matrix, result.population_matrix, fission_rate)
    st.pyplot(fig)
    
    # Display statistics
    _display_fission_rate_statistics(result)

def _display_fission_rate_statistics(result):
    """Display statistics for a single fission rate"""
    population_data = result.population_matrix[0]
    detection_stats = calculate_detection_stats(result.detection_matrix)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Population", f"{population_data[-1]:.0f}")
    with col2:
        st.metric("Mean Population", f"{np.mean(population_data):.1f}")
    with col3:
        st.metric("Total Detections", detection_stats['total_detections'])
    with col4:
        st.metric("CPS", f"{detection_stats['cps']:.2f}")
    
    st.markdown("---")
        
def _display_clear_buttons():
    """Display clear results buttons in sidebar"""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear All Results"):
           app_state = st.session_state.app_state
           app_state.clear_all()
           st.rerun() 
    
    with col2:
        if st.button("Clear CPS Analysis only"):
            app_state = st.session_state.app_state
            app_state.clear_dead_time_analyses()
            st.rerun()
    
 
# =============================================================================
# DEAD TIME ANALYSIS UI FUNCTIONS
# =============================================================================

def _handle_dead_time_analysis(results, params):
    """Handle dead time analysis section"""
    # Dead time analysis section
    st.markdown('---')
    st.subheader("Dead Time Analysis")
    
    app_state = st.session_state.app_state
    has_dead_time_results = app_state.has_dead_time_analyses()
    
    if not has_dead_time_results:
        _display_first_dead_time_analysis(results, params)
    else:
        _display_existing_dead_time_analyses(results, params)
    
def _display_first_dead_time_analysis(results, params):
    """ Display first dead time analysis configuration"""
    # Create DeadTimeAnalysisUI instance
    dead_time_ui = DeadTimeAnalysisUI()
    
    config = dead_time_ui.display_analysis_form(is_first_analysis = True)
    
    continue_analysis = st.button("Continue with Dead Time Analysis", type = "secondary")
    
    if continue_analysis:
        _run_dead_time_analysis(results, params, config.mean_dead_time,
                                config.std_percent, config.distribution_type,
                                config.analysis_name)
        st.rerun()
        
def _display_existing_dead_time_analyses(results, params):
    """Display existing dead time analyses with tabs"""
    app_state = st.session_state.app_state
    st.success(f"Dead time analysis completed! ({len(app_state.dead_time_analyses)} analyses)" )
    
    dead_time_ui = DeadTimeAnalysisUI()
    
    # Display all analyses
    tab_dt1, tab_dt2, tab_dt3 = st.tabs(["CPS Comparison", "Individual Analyses", "Add New Analysis"])

    with tab_dt1:
        dead_time_ui.display_cps_comparison_plot(app_state.dead_time_analyses)
    with tab_dt2:
        _display_individual_dead_time_analyses()
    with tab_dt3:
        _display_add_new_dead_time_analysis(results, params)
                
                
def _display_individual_dead_time_analyses():
    """Display individual dead time analysis statistics"""
    st.subheader("Individual Analysis Statistics")

    app_state = st.session_state.app_state
    for analysis in app_state.dead_time_analyses:
        # Display metrics
        _display_single_analysis_metrics(analysis)
        
        # Display summary table
        dead_time_summary = analysis.get_summary_data()
        st.table(dead_time_summary)
        st.markdown("---")

def _display_single_analysis_metrics(analysis: DeadTimeAnalysis):
    """Display metrics for a single dead time analysis"""
    st.write(f"**{analysis.config.analysis_name}**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Dead Time", f"{analysis.config.mean_dead_time} μs")
    with col2:
        st.metric("Std Dev", f"{analysis.config.std_percent}%")
    with col3:
        st.metric("Type", analysis.config.distribution_type)
    with col4:
        _display_theoretical_cps_button(analysis)

def _display_theoretical_cps_button(analysis : DeadTimeAnalysis):
    """Display theoretical CPS button with proper conditions"""
    app_state = st.session_state.app_state
    
    # Early return if not using equilibrium values
    if app_state.simulation_params.initial_population is not None:
        st.info("Theoretical CPS only available with equilibrium values")
        return
    
    # Early return if theoretical CPS already added
    if analysis.has_theoretical_cps():
        st.success("Theoretical CPS Added")
        return
    
    # Show button to add theoretical CPS
    if st.button("Add Theoretical CPS",
            key = f"theo_{analysis.config.analysis_name}"):
        _add_theoretical_cps_to_analysis(analysis.config.analysis_name,
                                         app_state.simulation_params)
        st.rerun()
   
def _display_add_new_dead_time_analysis(results, params):
    """Display form to add new dead time analysis"""
    
    dead_time_ui = DeadTimeAnalysisUI()
    config = dead_time_ui.display_analysis_form(is_first_analysis = False)
    
    add_analysis = st.button("Add Dead Time Analysis", type="secondary")
    if add_analysis:
        _handle_new_analysis_creation(config, results, params)

def _handle_new_analysis_creation(config, results, params):
    """Handle creation of new dead time analysis"""
    app_state = st.session_state.app_state
    
    if app_state.get_analysis_by_name(config.analysis_name):
        st.error("Analysis name already exists! Please choose a different name.")
        return
    
    _run_dead_time_analysis(results, params, config.mean_dead_time,
                           config.std_percent, config.distribution_type,
                           config.analysis_name)
    st.rerun()
            
def _display_main_content(sim_params):
    """Display main content based on simulation results"""
    app_state = st.session_state.app_state
    has_results = app_state.has_simulation_results()
               
    if not has_results:
        _display_welcome_screen(sim_params)
        return
    
    # Display results
    results = app_state.simulation_results
    params = app_state.simulation_params
    
    results_display = ResultsDisplay()
    results_display.display_simulation_summary(params)
    
    _handle_dead_time_analysis(results, params)
    _display_simulation_tabs(results, results_display)
        

# =============================================================================
# WELCOME SCREEN FUNCTION
# =============================================================================
        
def _display_welcome_screen(params: SimulationParameters):
    """Display welcome screen with parameter preview"""
    
    _display_welcome_header()
    _display_parameter_preview(params)
    _display_dead_time_preview()
    _display_theoretical_cps_info(params)
 

def _display_welcome_header():
    """Display welcome header and overview"""
    st.markdown("""
    ## Welcome to the Nuclear Reactor Stochastic Simulation Dashboard!
    
    This dashboard demonstrates **stochastic simulation** of neutron 
    population dynamics in nuclear reactors with comprehensive **dead time analysis**.
    
    ### What You'll See:
    - **Comparative Analysis**: Population evolution for multiple fission rates
    - **Individual Plots**: Detailed plots for each fission rate
    - **Summary Table**: Statistics comparison across all simulations
    - **Dead Time Analysis**: Count rate analysis with multiple dead time distributions
    - **Theoretical CPS**: Analytical count rate calculations (equilibrium only)
    - **Statistical Analysis**: Population and detection statistics
    
    ### Getting Started:
    1. Select multiple fission rates in the sidebar
    2. Adjust simulation parameters (detection, absorption, source rates)
    3. Choose initial population settings (fixed value or equilibrium)
    4. Click "Run Stochastic Simulation"
    5. Explore results in the tabs below
    6. Configure dead time analysis with different distributions
    7. Compare multiple dead time configurations
    8. Add theoretical CPS calculations (equilibrium simulations only)
    
    ### Dead Time Analysis Features:
    - **Constant Dead Time**: Fixed dead time value
    - **Random Distributions**: Normal, Uniform, and Gamma distributions
    - **Multiple Analyses**: Compare different dead time configurations
    - **CPS vs Alpha Inverse**: Visualize count rate relationships
    - **Theoretical Comparison**: Compare simulated vs theoretical CPS
    - **Error Analysis**: Percentage error between simulation and theory
    
    ### Theoretical CPS Calculations:
    - **Available only with equilibrium values**: Theoretical formulas assume steady-state conditions
    - **Multiple distributions**: Constant, Normal, Uniform, and Gamma dead time models
    - **Analytical formulas**: Based on stochastic differential equations and dead time statistics
    - **Comparative analysis**: Side-by-side comparison of simulated and theoretical results
    """)
    
def _display_parameter_preview(params : SimulationParameters):
    """Display current parameter preview"""
    st.subheader("Current Parameter Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Selected Fission Rates:** {params.fission_rates}")
        st.write(f"**Detection Rate:** {params.detection_rate} s⁻¹")
        st.write(f"**Absorption Rate:** {params.absorption_rate} s⁻¹")
        
    with col2:
        st.write(f"**Source Rate:** {params.source_rate} s⁻¹")
        if params.use_equilibrium:
            st.write("**Initial Population:** Equilibrium values (enables theoretical CPS)")
        else:
            st.write(f"**Initial Population:** {params.initial_population} (theoretical CPS not available)")
        st.write(f"**Simulation Steps:** {params.simulation_steps}")

def _display_dead_time_preview():
    """Display dead time analysis preview"""
    st.subheader("Dead Time Analysis Preview")
    st.info("""
    After running simulations, you'll be able to:
    - Configure dead time parameters (mean, standard deviation, distribution type)
    - Run multiple dead time analyses for comparison
    - Visualize CPS vs Alpha Inverse relationships
    - Compare different dead time distributions (Constant, Normal, Uniform, Gamma)
    - Add theoretical CPS calculations for equilibrium simulations
    - Analyze percentage errors between simulation and theory
    """)

def _display_theoretical_cps_info(params: SimulationParameters):
    """Display theoretical CPS availability information"""
    st.subheader("Theoretical CPS Analysis")
    if params.use_equilibrium:
        st.success("""
        ✅ **Theoretical CPS Available**: You're using equilibrium values, so theoretical CPS calculations will be available after running dead time analysis.
        
        **What you can do:**
        - Compare simulated count rates with analytical predictions
        - Analyze the accuracy of your stochastic simulation
        - Study the effect of different dead time distributions on count rates
        - Calculate percentage errors between simulation and theory
        """)
    else:
        st.warning("""
        ⚠️ **Theoretical CPS Not Available**: You're using a fixed initial population, so theoretical CPS calculations won't be available.
        
        **To enable theoretical CPS:**
        - Check "Use equilibrium values" in the sidebar
        - Theoretical formulas require steady-state (equilibrium) conditions
        - This ensures meaningful comparison between simulation and theory
        """)
        
# =============================================================================
# MAIN APPLICATION FUNCTION
# =============================================================================

def main():
    st.title("Nuclear Reactor Stochastic Simulation Dashboard")
    st.markdown("---")
    
    # Sidebar for parameters
    with st.sidebar:
        param_input = ParameterInput()
        sim_params = param_input.get_simulation_parameters() 
        
        st.markdown("---")

        # Run simulation button
        run_sim = st.button("Run Stochastic Simulation", type="primary")
        
        # Clear results button
        _display_clear_buttons()
       
    # Handle simulation run
    _handle_simulation_run(run_sim, sim_params)
            
    _display_main_content(sim_params)
    
if __name__ == "__main__":
    main()