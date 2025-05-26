import numpy as np

def asian_payoff(ST: np.ndarray, K: float) -> np.ndarray:
    """Payoff d'une option asiatique (basée sur la moyenne du sous-jacent)."""
    avg_price = np.mean(ST)  # Calcul de la moyenne sur toute la trajectoire
    return np.maximum(avg_price - K, 0)
