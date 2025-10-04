# Nuclear Reactor Stochastic Simulation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

A comprehensive, professional-grade framework for stochastic simulation of nuclear reactor neutron population dynamics. This project has evolved from a research prototype into a **modern, modular scientific computing platform** featuring multiple numerical methods, advanced dead time analysis, and an intuitive web-based dashboard.

### 🎯 Key Capabilities

- **Multiple Numerical Methods**: Stochastic, Euler-Maruyama, Taylor series, and Runge-Kutta simulations
- **Advanced Dead Time Analysis**: Support for Constant, Normal, Uniform, and Gamma distributions
- **Interactive Web Dashboard**: Professional Streamlit-based interface for real-time analysis
- **Theoretical Validation**: Analytical CPS calculations with simulation comparison
- **Publication-Ready Outputs**: Professional visualizations and statistical analysis

## 🏗️ Architecture

The project has undergone a **complete architectural transformation** from a monolithic structure to a modern, modular framework:

### 📁 Project Structure

```
├── 📊 streamlit_app.py          # Main interactive dashboard
├── 🎯 main.py                   # Command-line simulation orchestrator
├── 🧩 models.py                 # Data models with type safety
├── ⚙️ services.py               # Business logic layer
├── 🎨 ui_components.py          # Reusable UI components
├── 🔬 stochastic_simulation.py  # Core stochastic simulation engine
├── 📈 euler_maruyama_methods.py # Euler-Maruyama numerical methods
├── 📐 taylor_methods.py         # Taylor series expansion methods
├── 🎲 runge_kutta_methods.py    # Runge-Kutta numerical integration
├── 📊 plot_simulations.py       # Professional visualization tools
├── 💾 data_management.py        # Data persistence and organization
├── 🧮 count_rates.py           # Count rate analysis and dead time effects
├── ⚡ core_parameters.py       # Physical parameter management
├── 📁 data/                    # Simulation results storage
└── 📚 Legacy Files/            # Deprecated legacy code
```

### 🎨 Modern Design Patterns

- **Service Layer Architecture**: Clean separation of business logic from UI
- **Data Models**: Type-safe, validated data structures with comprehensive methods
- **Component-Based UI**: Reusable Streamlit components for consistent interface
- **Dependency Injection**: Modular, testable architecture
- **Professional Error Handling**: Comprehensive exception management

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install streamlit numpy matplotlib scipy
```

### Option 1: Interactive Dashboard (Recommended)

Launch the professional web-based dashboard in https://nuclear-reactor-stochastic-simulation-dashboard.streamlit.app/

**Dashboard Features:**
- 🎛️ **Interactive Parameter Configuration** with validation and help systems
- 📊 **Real-time Simulation Execution** with progress tracking
- 📈 **Professional Visualization** with comparative plots and summary tables
- 🔬 **Advanced Dead Time Analysis** supporting multiple distributions
- 📐 **Theoretical vs Simulated Comparison** for equilibrium conditions
- 📋 **Multi-Analysis Comparison** with publication-ready outputs

### Option 2: Command-Line Interface

Run comprehensive simulations from command line:

```bash
python main.py
```

Configure simulation parameters by editing the control flags in `main.py`:

```python
# Simulation Control Flags
RUN_STOCHASTIC = True       # Direct stochastic simulations
RUN_EULER_MARUYAMA = True   # Euler-Maruyama method
RUN_TAYLOR = True           # Taylor series methods
RUN_RUNGE_KUTTA = True      # Runge-Kutta integration
RUN_CPS_ANALYSIS = True     # Count rate analysis
```

## 🔬 Simulation Methods

### 1. Stochastic Simulations
- **Direct Monte Carlo** approach for baseline results
- **Branching process** modeling of neutron populations
- **Multiple dead time distributions** (Constant, Normal, Uniform, Gamma)

### 2. Euler-Maruyama Methods
- **First-order stochastic integration** for SDEs

### 3. Taylor Series Methods
- **Strong Taylor 1.5 / Weak Taylor 2.0** scheme for higher accuracy

### 4. Runge-Kutta Integration
- **Third-order** numerical integration
- **Adaptive step size** control

## 📊 Dead Time Analysis

### Supported Distributions

| Distribution | Description | Use Case |
|-------------|-------------|----------|
| **Constant** | Fixed dead time value |
| **Normal** | Gaussian distribution |
| **Uniform** | Uniform distribution |
| **Gamma** | Gamma distribution |

### Analysis Features

- **CPS vs Alpha Inverse** relationships
- **Theoretical vs Simulated** comparison
- **Statistical error analysis** with percentage differences
- **Multi-configuration** comparison plots

## 🌐 Live Dashboard

Access the deployed dashboard:
**[🔗 Nuclear Reactor Simulation Dashboard](https://nuclear-reactor-stochastic-simulation-dashboard.streamlit.app/)**

### Dashboard Capabilities

1. **Parameter Configuration**
   - Multi-select fission rate picker
   - Physical constants with validation ranges
   - Equilibrium vs fixed initial population
   - Simulation duration and step configuration

2. **Simulation Execution**
   - Real-time progress tracking
   - Multiple fission rate analysis
   - Automatic result storage

3. **Results Visualization**
   - Comparative population evolution plots
   - Individual fission rate analysis
   - Summary statistics tables
   - Professional formatting

4. **Dead Time Analysis**
   - Interactive configuration forms
   - Multiple analysis comparison
   - Theoretical CPS calculations
   - Statistical analysis tools

## 📈 Key Features

### 🎯 Professional Data Models

```python
# Type-safe simulation parameters
params = SimulationParameters(
    fission_rates=[33.94, 33.95, 33.96],
    detection_rate=10.0,
    absorption_rate=7.0,
    source_rate=1000.0,
    simulation_steps=100000,
    use_equilibrium=True
)

# Dead time configuration with utilities
dead_time_config = DeadTimeConfig(
    mean_dead_time=1.0,
    std_percent=10.0,
    distribution_type='normal',
    analysis_name='Normal Dead Time Analysis'
)
```

### 🔧 Service Layer Architecture

```python
# Simulation orchestration
simulation_service = SimulationService()
results = simulation_service.run_multiple_simulations(params)

# Dead time analysis
dead_time_service = DeadTimeAnalysisService()
analysis = dead_time_service.run_analysis(results, params, dead_time_config)
```

### 📊 Advanced Visualization

- **Professional plotting** with consistent styling
- **Publication-ready** figures with proper legends and formatting
- **Interactive plots** in the Streamlit dashboard
- **Comparative analysis** across multiple configurations

## 🧪 Research Applications

### Nuclear Reactor Analysis
- **Neutron population dynamics** simulation
- **Dead time effects** on count rates
- **Reactor kinetics** analysis
- **Statistical validation** of theoretical models

### Educational Use
- **Interactive learning** through web dashboard
- **Multiple numerical methods** comparison
- **Theoretical vs simulated** validation
- **Professional visualization** for presentations

### Industry Applications
- **Detector characterization** with dead time analysis
- **Reactor monitoring** system development
- **Statistical analysis** tools for nuclear engineering

## 🛠️ Development

### Code Quality Standards

- **PEP 257** docstrings for all classes and methods
- **Type hints** throughout the codebase
- **Comprehensive error handling** with specific exceptions
- **Modular design** following single responsibility principle
- **Professional documentation** with usage examples

### Extensibility

The modular architecture makes it easy to:
- Add new numerical methods
- Implement additional dead time distributions
- Create custom visualization components
- Extend the service layer for new analysis types

## 📚 Documentation

- **Comprehensive docstrings** in all modules
- **Usage examples** throughout the codebase
- **Architectural decisions** documented in code comments
- **API documentation** for all public methods

## 🤝 Contributing

We welcome contributions! The modular architecture makes it easy to:
- Add new simulation methods
- Implement additional dead time distributions
- Create custom visualization components
- Extend analysis capabilities

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Cursor.ai** for development assistance
- **Streamlit** for the excellent web framework

---


