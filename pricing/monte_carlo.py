import numpy as np


def monte_carlo_simulation(S: float, T: float, r: float, sigma: float, num_simulations: int,
                           num_steps: int) -> np.ndarray:
    """
    Génère les trajectoires du sous-jacent avec un processus de Brownien géométrique.

    Paramètres :
    - S : Prix initial du sous-jacent
    - T : Durée jusqu'à échéance (en années)
    - r : Taux sans risque
    - sigma : Volatilité du sous-jacent
    - num_simulations : Nombre de simulations Monte Carlo
    - num_steps : Nombre de pas de temps

    Retourne :
    - Matrice des trajectoires (num_simulations, num_steps)
    """
    np.random.seed(42)
    dt = T / num_steps
    Z = np.random.standard_normal((num_simulations, num_steps))
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    ST = S * np.exp(np.cumsum(increments, axis=1))

    return ST

import numpy as np

def monte_carlo_pricing(ST: np.ndarray, K: float, r: float, T: float, payoff_function, barrier=None) -> float:
    """
    Applique un payoff aux trajectoires générées et calcule le prix de l'option.

    Paramètres :
    - ST : Matrice des trajectoires (num_simulations, num_steps)
    - K : Prix d'exercice
    - r : Taux sans risque
    - T : Durée jusqu'à échéance
    - payoff_function : Fonction qui applique un payoff
    - barrier : Niveau de barrière (si applicable)

    Retourne :
    - Prix estimé de l'option
    """
    payoff = payoff_function(ST, K, barrier) if barrier is not None else payoff_function(ST, K)
    price = np.exp(-r * T) * np.mean(payoff)

    return price
