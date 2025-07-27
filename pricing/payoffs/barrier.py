import numpy as np
import numpy.typing as npt
from typing import Callable, Union


def barrier_knock_out(
    ST: npt.NDArray[np.float64], 
    K: float, 
    barrier: float, 
    payoff_sousjacent: Callable,
    barrier_type: str = "up"
) -> npt.NDArray[np.float64]:
    """
    Payoff d'une option knock-out (annulée si barrière atteinte).
    
    Args:
        ST: Trajectoires du sous-jacent (num_simulations, num_steps)
        K: Prix d'exercice
        barrier: Niveau de barrière
        payoff_sousjacent: Fonction de payoff sous-jacent
        barrier_type: "up" pour up-and-out, "down" pour down-and-out
        
    Returns:
        Array des payoffs (0 si barrière atteinte)
        
    Raises:
        ValueError: Si barrier_type n'est pas "up" ou "down"
    """
    if barrier_type not in ["up", "down"]:
        raise ValueError("barrier_type doit être 'up' ou 'down'")
    
    # Calcul vectorisé pour déterminer les trajectoires valides
    if barrier_type == "up":
        valid_paths = np.all(ST < barrier, axis=1)  # Up-and-out (strict inequality)
    else:  # down
        valid_paths = np.all(ST > barrier, axis=1)  # Down-and-out (strict inequality)
    
    # Calcul du payoff standard sur la valeur finale
    final_prices = ST[:, -1]
    payoffs = payoff_sousjacent(final_prices, K)
    
    # Application de la condition de barrière
    return payoffs * valid_paths


def barrier_knock_in(
    ST: npt.NDArray[np.float64], 
    K: float, 
    barrier: float, 
    payoff_sousjacent: Callable,
    barrier_type: str = "up"
) -> npt.NDArray[np.float64]:
    """
    Payoff d'une option knock-in (activée si barrière atteinte).
    
    Args:
        ST: Trajectoires du sous-jacent (num_simulations, num_steps)
        K: Prix d'exercice
        barrier: Niveau de barrière
        payoff_sousjacent: Fonction de payoff sous-jacent
        barrier_type: "up" pour up-and-in, "down" pour down-and-in
        
    Returns:
        Array des payoffs (0 si barrière non atteinte)
        
    Raises:
        ValueError: Si barrier_type n'est pas "up" ou "down"
    """
    if barrier_type not in ["up", "down"]:
        raise ValueError("barrier_type doit être 'up' ou 'down'")
    
    # Calcul vectorisé pour déterminer les trajectoires activées
    if barrier_type == "up":
        activated_paths = np.any(ST >= barrier, axis=1)  # Up-and-in (>= barrier)
    else:  # down
        activated_paths = np.any(ST <= barrier, axis=1)  # Down-and-in (<= barrier)
    
    # Calcul du payoff standard sur la valeur finale
    final_prices = ST[:, -1]
    payoffs = payoff_sousjacent(final_prices, K)
    
    # Application de la condition de barrière
    return payoffs * activated_paths


def double_barrier_knock_out(
    ST: npt.NDArray[np.float64], 
    K: float, 
    lower_barrier: float,
    upper_barrier: float, 
    payoff_sousjacent: Callable
) -> npt.NDArray[np.float64]:
    """
    Payoff d'une option double knock-out (annulée si une des barrières est atteinte).
    
    Args:
        ST: Trajectoires du sous-jacent (num_simulations, num_steps)
        K: Prix d'exercice
        lower_barrier: Barrière inférieure
        upper_barrier: Barrière supérieure
        payoff_sousjacent: Fonction de payoff sous-jacent
        
    Returns:
        Array des payoffs (0 si une barrière est atteinte)
    """
    if lower_barrier >= upper_barrier:
        raise ValueError("lower_barrier doit être < upper_barrier")
    
    # Les trajectoires sont valides si elles restent dans le corridor
    valid_paths = np.all((ST >= lower_barrier) & (ST <= upper_barrier), axis=1)
    
    # Calcul du payoff standard sur la valeur finale
    final_prices = ST[:, -1]
    payoffs = payoff_sousjacent(final_prices, K)
    
    return payoffs * valid_paths
