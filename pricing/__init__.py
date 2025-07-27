# pricing/__init__.py
from .monte_carlo import monte_carlo_simulation, monte_carlo_pricing, simple_monte_carlo_pricing
from .black_scholes import black_scholes_price

__all__ = [
    "monte_carlo_simulation",
    "monte_carlo_pricing", 
    "simple_monte_carlo_pricing",
    "black_scholes_price"
]




"""
Pricing Lib - Une librairie de pricing pour les dérivés financiers.
"""

__version__ = "1.0.0"
__author__ = "Simonin"
