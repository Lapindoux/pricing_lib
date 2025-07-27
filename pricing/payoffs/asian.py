import numpy as np
import numpy.typing as npt
from typing import Literal


def asian_payoff(ST: npt.NDArray[np.float64], K: float, option_type: str = "call") -> npt.NDArray[np.float64]:
    """
    Payoff d'une option asiatique (basée sur la moyenne du sous-jacent).
    
    Args:
        ST: Trajectoires du sous-jacent (num_simulations, num_steps)
        K: Prix d'exercice
        option_type: "call" ou "put"
        
    Returns:
        Array des payoffs pour chaque simulation
        
    Raises:
        ValueError: Si option_type n'est pas "call" ou "put"
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type doit être 'call' ou 'put'")
        
    avg_price = np.mean(ST, axis=1)  # Calcul de la moyenne par trajectoire
    
    if option_type == "call":
        return np.maximum(avg_price - K, 0)
    else:  # put
        return np.maximum(K - avg_price, 0)


def asian_geometric_payoff(ST: npt.NDArray[np.float64], K: float, option_type: str = "call") -> npt.NDArray[np.float64]:
    """
    Payoff d'une option asiatique géométrique (basée sur la moyenne géométrique).
    
    Args:
        ST: Trajectoires du sous-jacent (num_simulations, num_steps)
        K: Prix d'exercice
        option_type: "call" ou "put"
        
    Returns:
        Array des payoffs pour chaque simulation
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    # Moyenne géométrique = exp(moyenne des log)
    # Éviter log(0) en ajoutant une petite valeur
    ST_safe = np.maximum(ST, 1e-10)
    geometric_avg = np.exp(np.mean(np.log(ST_safe), axis=1))
    
    if option_type == "call":
        return np.maximum(geometric_avg - K, 0)
    else:  # put
        return np.maximum(K - geometric_avg, 0)


def asian_strike_payoff(
    ST: npt.NDArray[np.float64], 
    option_type: str = "call",
    fixed_strike: bool = False,
    K: float = 0.0
) -> npt.NDArray[np.float64]:
    """
    Payoff d'une option asiatique à strike flottant.
    
    Args:
        ST: Trajectoires du sous-jacent (num_simulations, num_steps)
        option_type: "call" ou "put"
        fixed_strike: Si True, utilise K comme strike fixe
        K: Strike fixe (utilisé seulement si fixed_strike=True)
        
    Returns:
        Array des payoffs pour chaque simulation
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    final_price = ST[:, -1]
    
    if fixed_strike:
        strike = K
    else:
        # Strike flottant = moyenne arithmétique
        strike = np.mean(ST, axis=1)
    
    if option_type == "call":
        return np.maximum(final_price - strike, 0)
    else:  # put
        return np.maximum(strike - final_price, 0)
