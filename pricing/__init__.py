# __init__.py dans pricing_lib/pricing/
from .black_scholes import black_scholes_price
from .binomial_tree import binomial_tree_price
from .monte_carlo import monte_carlo_pricing, monte_carlo_simulation

__all__ = ["black_scholes_price", "binomial_tree_price", "monte_carlo_pricing", "monte_carlo_simulation"]




"""
Pricing Lib - Une librairie de pricing pour les dérivés financiers.
"""

__version__ = "1.0.0"
__author__ = "Simonin"
