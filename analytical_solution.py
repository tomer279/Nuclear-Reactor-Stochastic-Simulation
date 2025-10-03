"""
Written by Tomer279 with the assistance of Cursor.ai.

Analytical solutions for opulation dynamics.

This module provides an analytical solution  for the stochastic differential
equation describing neutron population dynamics.
The main function implements the exact analytical solution
for the population SDE, avoiding numerical integration methods.

The analytical solution is based on the SDE:
    dN_t = (-α * N_t + S)dt + σ₁ * dW₁ - σ₂ * dW₂

where:
    - N_t is the neutron population at time t
    - α is the Rossi-alpha coefficient
    - S is the source term
    - σ₁, σ₂ are noise amplitudes
    - W₁, W₂ are independent Wiener processes

The exact solution uses:
    N_(i+1) = μ(dt) + σ(dt) * Z

where:
    μ(dt) = e^(-α*dt)(N_i - S/α) + S/α
    σ²(dt) = (σ₁² + σ₂²)/(2α) * (1 - e^(-2α*dt))
    Z ~ N(0,1)

"""


import numpy as np
import utils as utl
from core_parameters import PhysicalParameters, TimeParameters

rng = np.random.default_rng()


def analytical_population_solution(
        physical_params: PhysicalParameters,
        time_params: TimeParameters,
        fission: float,
        n_0: float = None) -> np.ndarray:
    """
    Simulate population dynamics using the analytical solution.

    This function uses the exact analytical solution for the SDE:
    dN_t = (-α * N_t + S)dt + σ₁ * dW₁ - σ₂ * dW₂

    Instead of numerical integration, it uses the exact solution:
    N_(i+1) = μ(dt) + σ(dt) * Z

    where:
    μ(dt) = e^(-α*dt)(N_i - S/α) + S/α
    σ²(dt) = (σ₁² + σ₂²)/(2α) * (1 - e^(-2α*dt))
    Z ~ N(0,1)

    Parameters
    ----------
    physical_params : PhysicalParameters
        Physical parameters containing system constants
    time_params : TimeParameters
        Time parameters containing time configuration
    fission: float
        Fission rate constant for this simulation
    n_0: float, optional
        Initial population value.
        If None, defaults to equilibrium value S/α

    Returns
    -------
    np.ndarray
        Array of population values at each time point
    """

    # Set fission rate for time simulation
    physical_params.set_fission(fission)

    # Calculate system parameters
    params = utl.calculate_system_parameters(
        physical_params.p_v,
        physical_params.fission,
        physical_params.absorb,
        physical_params.source,
        physical_params.detect)

    equil = params['equilibrium']

    if n_0 is None:
        n_0 = equil

    # Initialize population array
    pop = np.zeros(time_params.grid_points + 1)
    pop[0] = n_0

    # Calculate time step
    dt = time_params.get_grid_spacing()

    # Pre-calculate terms for efficiency
    exp_term, var_term = _calculate_terms(params, dt)

    # Generate all random numbers at once
    gaussian = rng.normal(size=time_params.grid_points)

    # Analytical solution step
    print("Calculating analytical solution")
    for i in utl.progress_tracker(time_params.grid_points):
        current_pop = pop[i]
        mean_term = exp_term * (current_pop - equil) + equil
        pop[i + 1] = mean_term + np.sqrt(var_term) * gaussian[i]

    print("\n")
    return pop


def _calculate_terms(params: dict, dt: float):
    """Pre-calculate terms for efficiency"""
    alpha = params['alpha']
    sig_1_squared = params['sig_1_squared']
    sig_2_squared = params['sig_2_squared']
    exp_term = np.exp(- alpha * dt)
    var_term = (
        ((sig_1_squared + sig_2_squared) / (2 * alpha))
        * (1 - np.exp(- 2 * alpha * dt))
    )
    return exp_term, var_term
