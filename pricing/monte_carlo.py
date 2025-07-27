import numpy as np
from typing import Optional, Callable
import numpy.typing as npt


def monte_carlo_simulation(
    S: float, 
    T: float, 
    r: float, 
    sigma: float, 
    num_simulations: int,
    num_steps: int,
    seed: Optional[int] = None
) -> npt.NDArray[np.float64]:
    """
    Génère les trajectoires du sous-jacent avec un processus de Brownien géométrique.

    Args:
        S: Prix initial du sous-jacent (doit être > 0)
        T: Durée jusqu'à échéance en années (doit être > 0)
        r: Taux sans risque
        sigma: Volatilité du sous-jacent (doit être > 0)
        num_simulations: Nombre de simulations Monte Carlo (doit être > 0)
        num_steps: Nombre de pas de temps (doit être > 0)
        seed: Graine pour la reproductibilité (optionnel)

    Returns:
        Matrice des trajectoires (num_simulations, num_steps)
        
    Raises:
        ValueError: Si les paramètres sont invalides
    """
    # Validation des paramètres
    if S <= 0:
        raise ValueError("Le prix initial S doit être positif")
    if T <= 0:
        raise ValueError("La durée T doit être positive")
    if sigma <= 0:
        raise ValueError("La volatilité sigma doit être positive")
    if num_simulations <= 0:
        raise ValueError("Le nombre de simulations doit être positif")
    if num_steps <= 0:
        raise ValueError("Le nombre de pas doit être positif")
    
    # Configuration du générateur aléatoire
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / num_steps
    Z = np.random.standard_normal((num_simulations, num_steps))
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    ST = S * np.exp(np.cumsum(increments, axis=1))

    return ST


def monte_carlo_pricing(
    ST: npt.NDArray[np.float64], 
    K: float, 
    r: float, 
    T: float, 
    payoff_function: Optional[Callable] = None, 
    payoff_sousjacent: Optional[Callable] = None, 
    barrier: Optional[float] = None
) -> float:
    """
    Calcule le prix d'une option par Monte Carlo.
    
    Args:
        ST: Trajectoires du sous-jacent
        K: Prix d'exercice
        r: Taux sans risque
        T: Durée jusqu'à échéance
        payoff_function: Fonction de payoff avec barrière (optionnel)
        payoff_sousjacent: Fonction de payoff sous-jacent
        barrier: Niveau de barrière (optionnel)
        
    Returns:
        Prix actualisé de l'option
        
    Raises:
        ValueError: Si les paramètres sont incompatibles
    """
    # Validation des paramètres
    if T <= 0:
        raise ValueError("La durée T doit être positive")
    if K <= 0:
        raise ValueError("Le prix d'exercice K doit être positif")
    
    # 🔍 Cas sans barrière → Payoff sous-jacent seul
    if payoff_sousjacent is not None and payoff_function is None:
        payoff = payoff_sousjacent(ST[:, -1], K)

    # 🚀 Cas avec barrière → Applique la fonction de barrière
    elif payoff_function is not None and payoff_sousjacent is not None:
        if barrier is None:
            raise ValueError("Une barrière doit être spécifiée pour les options à barrière")
        payoff = payoff_function(ST, K, barrier, payoff_sousjacent)

    # 🔍 Cas option asiatique → Payoff sur toute la trajectoire
    elif payoff_sousjacent is not None and payoff_function is None:
        payoff = payoff_sousjacent(ST, K)

    else:
        raise ValueError("Il faut fournir au moins un payoff sous-jacent")

    # Validation du payoff
    if not isinstance(payoff, np.ndarray):
        raise ValueError("Le payoff doit être un array numpy")
    
    price = np.exp(-r * T) * np.mean(payoff)
    return price


def simple_monte_carlo_pricing(
    S: float,
    K: float, 
    T: float,
    r: float,
    sigma: float,
    payoff_function: Callable,
    num_simulations: int = 10000,
    num_steps: int = 252,
    seed: Optional[int] = None
) -> float:
    """
    Fonction simplifiée de pricing Monte Carlo qui génère les trajectoires en interne.
    
    Args:
        S: Prix initial du sous-jacent
        K: Prix d'exercice
        T: Durée jusqu'à échéance
        r: Taux sans risque
        sigma: Volatilité
        payoff_function: Fonction de payoff à appliquer
        num_simulations: Nombre de simulations
        num_steps: Nombre de pas de temps
        seed: Graine aléatoire (optionnel)
        
    Returns:
        Prix de l'option
    """
    # Génération des trajectoires
    ST = monte_carlo_simulation(S, T, r, sigma, num_simulations, num_steps, seed)
    
    # Calcul du payoff
    if hasattr(payoff_function, '__code__') and payoff_function.__code__.co_argcount == 2:
        # Fonction payoff simple (vanilla)
        payoffs = payoff_function(ST[:, -1], K)
    else:
        # Fonction payoff complexe (asian, etc.)
        payoffs = payoff_function(ST, K)
    
    # Prix actualisé
    return np.exp(-r * T) * np.mean(payoffs)

