import numpy as np

def barrier_knock_out(ST: np.ndarray, K: float, barrier: float) -> np.ndarray:
    """Payoff d'une option knock-out (annulée si barrière atteinte)."""
    valid_paths = np.all(ST < barrier, axis=1)
    #payoff = payoff_sousjacent(ST[:, -1], K) * valid_paths
    payoff = np.maximum(ST[:, -1] - K, 0) * valid_paths
    return payoff

def barrier_knock_in(ST: np.ndarray, K: float, barrier: float) -> np.ndarray:
    """Payoff d'une option knock-in, activée si la barrière est atteinte."""
    valid_paths = np.any(ST >= barrier, axis=1)  # Vérifie si la barrière est atteinte

    # 🔥 Multiplication du payoff du sous-jacent par `valid_paths`
    #payoff = payoff_sousjacent(ST[:, -1], K) * valid_paths
    payoff = np.maximum(ST[:, -1] - K, 0) * valid_paths
    return payoff

