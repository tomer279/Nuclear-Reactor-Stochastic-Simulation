""" Written by Tomer279 with the assistance of Cursor.ai """

"""
Utility functions for nuclear reactor physics calculations.

This module contains mathematical utility functions for calculating
statistical moments, variance, and Diven factors used in neutron
count distribution analysis.
"""


import numpy as np
import math

def mean(p: np.array) -> float:
    """
    "Calculating the mean (or expected value) of a given probability distribution 'p'
    The mean is defined as:
        E[p] = sum_{k=0}^{len(p)} k * p_(k)
    Parameters
    ----------
    p: np.array
        probability distrubtion array
    
    Returns
    -------
    float
        The mean of p
    """
    indices = np.arange(len(p))
    return np.dot(indices,p)

def mean_square(p: np.array) -> float:
    """
    "Calculating the second moment (or mean of square) of a given probability distribution 'p'
    The second moment is defined as:
        E[p^2] = sum_{k=0}^{len(p)} k^2 * p_(k)
    Parameters
    ----------
    p: np.array
        probability distrubtion array
    
    Returns
    -------
    float
        The second moment of p
    """
    indices = np.arange(len(p))
    return np.dot(indices ** 2 , p)


def variance(p: np.array) -> float:
    """
    "Calculating the variance of a given probability distribution 'p'
    The variance is defined as:
        Var(p) = mean_square(p) - mean(p)^2
    Parameters
    ----------
    p: np.array
        probability distrubtion array
    
    Returns
    -------
    float
        The variance of p
        
    Raises
    ------
    """
        
    return mean_square(p) - mean(p) ** 2


def diven_factor(p: np.array, n: int) -> float:
    """
    Calculating the Diven Factor of order 'n' for a given probability distribution 'p.'
    The Diven Factor is defined as:
        D_n = (1/n!) * sum_{k=n}^{len(p)} k!/(n-k)! * p_(k-1)
        
    TODO: this function may be used in future analysis for higher-order moments
    and variance calculations in neutron count distributions.
    
    The Diven Factor is particularly important in nuclear reactor physics for:
    - Calculating variance in neutron count distributions
    - Characterizing the stochastic nature of fission events
    - Analyzing reactor noise and fluctuations
    
    Parameters
    ----------
    p : np.array
        probability distrubtion array.
    n : int
        order of the Diven Factor

    Returns
    -------
    float
        The Diven Factor of order 'n.'
    Notes
    -----
    
    For n=2, this gives the second factorial moment which is related to the variance
    of the particle production distribution.
    """
    if n == 0:
        return 1.0 # D_0 = 1 by definition
    
    if n == 1:
        # D_1 is the mean of the distribution
        return mean(p)
    
    # Calculate Diven factor using vectorized operations
    k_values = np.arange(n, len(p))
    
    # Calculate k! / (k-n)! = k * (k-1) * ... (k - n + 1) efficiently
    # This is the falling factorial (k)_n
    falling_factorial = np.ones_like(k_values, dtype = float)
    for i in range(n):
        falling_factorial *= (k_values - i)
    
    # Calculate the sum
    result = np.sum(falling_factorial * p[k_values]) / math.factorial(n)
    return result
