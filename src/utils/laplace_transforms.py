"""Written by Tomer279 with the assistance of Cursor.ai.

Laplace Transform Calculations for Dead Time Distributions.

This module provides Laplace transforms and their first and second derivatives
for various dead time distributions used in nuclear reactor detection systems.
The Laplace transforms are essential for calculating noise amplitudes and
drift derivatives in the detection SDE.

Mathematical Background
-----------------------
The Psi function is related to the Laplace transform by:
    Ψ(x) = 1 - L{f_τ}(x)
    
where L{f_τ}(x) = ∫₀^∞ e^(-x*s) * f_τ(s) ds

Therefore:
    - L{f_τ}(x) = 1 - Ψ(x)
    - L'{f_τ}(x) = -Ψ'(x)
    - L''{f_τ}(x) = -Ψ''(x)

Classes
-------
LaplaceTransform
    Container for Laplace transform value and its first and second derivatives
DeadTimeLaplaceCalculator
    Calculator for Laplace transforms of different dead time distributions

Notes
-----
Second derivatives are required for Taylor 2.0 methods which use higher-order
terms in the integration formula. All four distribution types (constant,
uniform, normal, gamma) provide second derivatives.

Examples
--------
>>> calculator = DeadTimeLaplaceCalculator()
>>> result = calculator.calculate_laplace_transform(
...     x=10.0, tau=1e-6, distribution='constant'
... )
>>> print(f"L{{f_τ}}(x) = {result.value}")
>>> print(f"L'{{f_τ}}(x) = {result.derivative}")
>>> print(f"L''{{f_τ}}(x) = {result.second_derivative}")
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class LaplaceTransform:
    """
    Container for Laplace transform value and its derivatives.

    Attributes
    ----------
    value: float
        The laplace transform of tau
    derivative: float
        The derivative of the Laplace transform
    second_derivative: float
        The second derivative of the Laplace transform
    """
    value: float
    derivative: float
    second_derivative: Optional[float] = None


class DeadTimeLaplaceCalculator:
    """
    Calculator for Laplace transforms of dead time distributions

    This class provides calculations for the Laplace transform of a specific
    distribution for the dead time, and its derivative. \n
    The class is used to evaluate the diffusion :math:`\\sigma_3`
    associated with the stochastic differential equation 
    for the number of detections, using the formula:

    .. math::
        \\sigma^2 = \\frac{1 + 2 a L'(a)}{a L(a)}
    Where :math:`a = N_t \\lambda_d` is the detection rate.

    Public Methods
    --------------
    calculate_laplace_transform(x, tau, distribution, std)
        Calculate Laplace transform and derivative for given distribution
    calculate_constant_laplace(x, tau)
        Laplace transform for constant dead time
    calculate_uniform_laplace(x, tau, std)
        Laplace transform for uniform dead time
    calculate_normal_laplace(x, tau, std)
        Laplace transform for normal dead time
    calculate_gamma_laplace(x, tau, std)
        Laplace transform for gamma dead time
    Examples
    --------
    >>> calculator = DeadTimeLaplaceCalculator()
    >>> result = calculator.calculate_laplace_transform(
    ...     x=10.0, tau=1e-6, distribution='constant'
    ... )
    >>> print(f"Value: {result.value}, Derivative: {result.derivative}")
    """

    def calculate_laplace_transform(
            self,
            x: float,
            tau: float,
            distribution: str,
            std: Optional[float] = None) -> LaplaceTransform:
        """
        Calculate Laplace transform and derivative for given distribution.

        This method routes to the appropriate distribution-specific 
        calculation based on the distribution type.

        Parameters
        ----------
        x : float
            Detection rate parameter (typically d * N)
        tau : float
            Mean dead time value (seconds)
        distribution : str
            Type of dead time distribution:
            'constant', 'uniform', 'normal', 'gamma'
        std : Optional[float]
            Standard deviation of dead time (required for non-constant)

        Returns
        -------
        LaplaceTransform
            Container with value and derivative

        Raises
        ------
        ValueError
            If distribution type is unknown or std is missing when required
        """
        if distribution == 'constant':
            return self.calculate_constant_laplace(x, tau)
        if std is None:
            raise ValueError(
                f"std required for {distribution} distribution")
        if distribution == 'uniform':
            return self.calculate_uniform_laplace(x, tau, std)
        if distribution == 'normal':
            return self.calculate_normal_laplace(x, tau, std)
        if distribution == 'gamma':
            return self.calculate_gamma_laplace(x, tau, std)
        raise ValueError(f"Unknown distribution type: {distribution}")

    def calculate_constant_laplace(
            self,
            x: float,
            tau: float) -> LaplaceTransform:
        """
        Calculate the Laplace transform and derivative 
        of constant dead time distribution.

        For constant dead time τ:

        .. math::
            f_\\tau(s)  & = \\delta(s - \\tau) \\\\
            L({f_\\tau}(x)) & = e^{- x \\tau} \\\\
            L'({f_\\tau}(x)) & = - \\tau e^{ - x \\tau} \\\\

        Parameters
        ----------
        x : float
            Detection rate parameter
        tau : float
            Constant dead time value

        Returns
        -------
        LaplaceTransform
            Laplace transform value and its derivative
        """
        exp_term = np.exp(-x * tau)

        value = exp_term
        derivative = -tau * exp_term
        second_derivative = (tau ** 2) * exp_term

        return LaplaceTransform(
            value=value,
            derivative=derivative,
            second_derivative=second_derivative)

    def calculate_uniform_laplace(
            self,
            x: float,
            tau: float,
            std: float) -> LaplaceTransform:
        """
        Calculate the Laplace transform and derivative 
        of uniform dead time distribution.

        For uniform dead time :math:`\\tau`, with mean :math:`\\mu` 
        and standard deviation :math:`\\sigma`:

        .. math::
            a & = \\mu - \\sqrt{3} \\sigma \\\\
            b & = \\mu + \\sqrt{3} \\sigma \\\\
            f_\\tau(s) & = \\frac{1}{b-a}, \\quad x \\in [a,b] \\\\
            L({f_\\tau}(x)) & = \\frac{e^{-bx} - e^{-ax}}{x(b-a)} \\\\
            L'({f_\\tau}(x)) & =  
            \\frac{(bx+1)e^{-bx} - (ax + 1)e^{-ax}}
            {(b-a) x^{2}} \\\\
        Parameters
        ----------
        x : float
            Detection rate parameter
        tau : float
            Mean dead time value
        std : float
            Standard deviation of dead time

        Returns
        -------
        LaplaceTransform
            Laplace transform value and its derivative
        """
        low = tau - np.sqrt(3) * std
        high = tau + np.sqrt(3) * std

        exp_low = np.exp(-x * low)

        exp_high = np.exp(-x * high)

        value = (exp_high - exp_low) / (-x * (high - low))

        nominator = (
            (high * x + 1) * exp_high
            - (low * x + 1) * exp_low
        )

        denominator = (high - low) * (x ** 2)

        derivative = nominator / denominator

        second_nominator = ((high * x * (high * x + 2) + 2) * exp_high
                            - (low * x * (low * x + 2) + 2) * exp_low)

        second_denominator = (high - low) * (x ** 3)

        second_derivative = -second_nominator / second_denominator

        return LaplaceTransform(
            value=value,
            derivative=derivative,
            second_derivative=second_derivative)

    def calculate_normal_laplace(
            self,
            x: float,
            tau: float,
            std: float) -> LaplaceTransform:
        """
        Calculate the Laplace transform and derivative 
        of normal dead time distribution.

        For normal dead time :math:`\\tau`, with mean :math:`\\mu` 
        and standard deviation :math:`\\sigma`:

        .. math::
            f_\\tau(s) & = \\frac{1}{\\sqrt{2 \\pi \\sigma ^2}} 
            \\exp \\left( \\frac{-(x - \\mu)^2}{2 \\sigma^2} \\right) \\\\
            L({f_\\tau}(x)) & =
            \\exp \\left(\\frac{(\\sigma x)^2}{2} -x \\mu  \\right) \\\\
            L'({f_\\tau}(x))  & = 
            (\\sigma^2 x - \\mu) 
            \\exp \\left(\\frac{(\\sigma x)^2}{2} -x \\mu  \\right) \\\\

        Parameters
        ----------
        x : float
            Detection rate parameter
        tau : float
            Mean dead time value
        std : float
            Standard deviation of dead time

        Returns
        -------
        LaplaceTransform
            Laplace transform value and its derivative
        """
        exp_term = np.exp(- x * tau + ((std * x) ** 2)/2)

        value = exp_term

        derivative = ((std ** 2) * x - tau) * exp_term

        second_derivative = (
            ((std ** 2) * x - tau) ** 2 + std ** 2
        ) * exp_term

        return LaplaceTransform(
            value=value,
            derivative=derivative,
            second_derivative=second_derivative
        )

    def calculate_gamma_laplace(
            self,
            x: float,
            tau: float,
            std: float) -> LaplaceTransform:
        """
        Calculate the Laplace transform and derivative 
        of gamma dead time distribution.

        For gamma dead time :math:`\\tau`, with mean :math:`\\mu` 
        and standard deviation :math:`\\sigma`:

        .. math::
            \\alpha & = \\left( \\frac{\\mu}{\\sigma} \\right)^2 \\\\
            \\theta & = \\frac{\\sigma^2}{\\mu} \\\\
            f_\\tau(s) & = \\frac{x^{\\alpha - 1} e^{-x/\\theta}}
            {\\Gamma(\\alpha) \\theta^{\\alpha}} \\\\
            L({f_\\tau}(x)) & = \\frac{1}{(1 + \\theta x)^{\\alpha}} \\\\
            L'({f_\\tau}(x)) & = \\frac{- \\alpha \\theta}
            {(1 + \\theta x)^{\\alpha + 1}} \\\\

        Parameters
        ----------
        x : float
            Detection rate parameter
        tau : float
            Mean dead time value
        std : float
            Standard deviation of dead time

        Returns
        -------
        LaplaceTransform
            Laplace transform value and its derivative
        """
        shape = (tau / std) ** 2
        scale = (std ** 2) / tau

        base = scale * x + 1

        value = 1 / (base ** shape)

        derivative = (-shape * scale) / (base ** (shape + 1))

        second_derivative = (
            shape * (shape + 1) * (scale ** 2) / (base ** (shape + 2))
        )

        return LaplaceTransform(
            value=value,
            derivative=derivative,
            second_derivative=second_derivative)


def calculate_noise_amplitude_from_laplace(
        laplace_result: LaplaceTransform,
        detection_rate: float) -> float:
    """
    Calculate noise amplitude σ₃ from Laplace transform values.

    The noise amplitude for the detection SDE can be computed from:

    .. math::
        \\sigma_3^2 = \\frac{1 + 2 a L'(a)}{a L(a)}

    Where :math:`a = N_t \\lambda_d`.

    Parameters
    ----------
    laplace_result : LaplaceTransform
        Laplace transform value and derivative at detection rate
    detection_rate : float
        Detection rate (:math:`N_t \\lambda_d`)

    Returns
    -------
    float
        Noise amplitude :math:`\\sigma_3`

    Examples
    --------
    >>> calculator = DeadTimeLaplaceCalculator()
    >>> laplace = calculator.calculate_constant_laplace(x=10.0, tau=1e-6)
    >>> sigma_3 = calculate_noise_amplitude_from_laplace(laplace, 10.0)
    """
    sig_squared = detection_rate * laplace_result.value * (
        1 + 2 * detection_rate * laplace_result.derivative)

    if sig_squared < 0:
        raise ValueError(
            f"noise amplitude is negative: sig^2 = {sig_squared}")

    return np.sqrt(sig_squared)


def calculate_laplace_and_noise(
    x: float,
    tau: float,
    distribution: str,
    std: Optional[float] = None
) -> tuple[LaplaceTransform, float]:
    """
    Calculate both Laplace transform and noise amplitude in one call.

    Parameters
    ----------
    x : float
        Detection rate parameter (typically :math:`N_t \\lambda_d`)
    tau : float
        Mean dead time value (seconds)
    distribution : str
        Type of dead time distribution
    std : Optional[float]
        Standard deviation of dead time (required for non-constant)

    Returns
    -------
    Tuple[LaplaceTransform, float]
        Laplace transform result and noise amplitude σ₃

    Examples
    --------
    >>> laplace, sigma_3 = calculate_laplace_and_noise(
    ...     x=10.0, tau=1e-6, distribution='constant'
    ... )
    >>> print(f"Noise amplitude: {sigma_3:.6e}")
    """
    calculator = DeadTimeLaplaceCalculator()
    laplace_result = calculator.calculate_laplace_transform(
        x, tau, distribution, std)

    noise_amplitude = calculate_noise_amplitude_from_laplace(
        laplace_result, x)

    return laplace_result, noise_amplitude
