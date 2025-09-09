# Fission Chain Stochastic Simulation

## Overview
This project implements stochastic and Euler-Maruyama simulations for fission chain dynamics, including dead time effects and count rate calculations.

## Features

- **Stochastic Simulations**: Branching process simulations of neutron populations
- **Euler-Maruyama Methods**: Numerical solutions to SDEs with dead time effects
- **Multiple Dead Time Models**: Constant, exponential, and random dead time
- **Taylor Methods**: Strong and weak Taylor schemes for improved accuracy
- **Visualization tools**: Plotting population dynamics and CPS comparisons

## Quick Start
``` python
python main.py
```

## Configuration
Edit `config.py` to modify simulation parameters.

## Output
Results are saved in `data/` directory.

## Streamlit Dashboard
A web-based dashboard is available for interactive simulation and analysis:
- Run stochastic simulations with custom parameters
- Perform dead time analysis with multiple distributions
- Compare theoretical vs simulated results
- Interactive visualization of population dynamics

Deploy the dashboard using Streamlit Community Cloud on:
[https://nuclear-reactor-stochastic-simulation-dashboard.streamlit.app/](https://nuclear-reactor-stochastic-simulation-dashboard.streamlit.app/)
or run locally with:
```bash
streamlit run streamlit_app.py
```

## Future Development

This project is actively being developed. See [ROADMAP.md](ROADMAP.md) for planned improvements.

## License
MIT License - see [LICENSE](LICENSE) for details.



