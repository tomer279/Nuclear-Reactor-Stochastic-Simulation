# Nuclear Reactor Stochastic Simulation

## Overview
This project implements stochastic and Euler-Maruyama simulations for nuclear reactor dynamics, including dead time effects and count rate calculations.

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

## Future Development

This project is actively being developed. See [ROADMAP.md](ROADMAP.md) for planned improvements.

## License
MIT License - see [LICENSE](LICENSE) for details.

