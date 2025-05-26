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

def monte_carlo_pricing(ST: np.ndarray, K: float, r: float, T: float, payoff_function=None, payoff_sousjacent=None, barrier=None) -> float:
    """Calcule le prix en évitant la double application du payoff sous-jacent."""

    # 🔍 Cas sans barrière → Payoff sous-jacent seul
    if payoff_sousjacent is not None and payoff_function is None:
        payoff = payoff_sousjacent(ST[:, -1], K)

    # 🚀 Cas avec barrière → Applique seulement `payoff_function()`
    elif payoff_function is not None and payoff_sousjacent is not None:
        payoff = payoff_function(ST, K, barrier, payoff_sousjacent)  # ✅ Supprime la double multiplication

    else:
        raise ValueError("Il faut fournir un payoff sous-jacent, avec ou sans barrière.")

    price = np.exp(-r * T) * np.mean(payoff)
    return price

