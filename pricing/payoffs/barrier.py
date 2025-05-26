import numpy as np

def barrier_knock_out(ST: np.ndarray, K: float, barrier: float, payoff_sousjacent) -> np.ndarray:
    """Payoff d'une option knock-out (annulée si barrière atteinte)."""
    valid_paths = np.all(ST < barrier, axis=1)  # Knock-out si la barrière est atteinte
    payoff = payoff_sousjacent(ST[:, -1], K) * valid_paths  # 🔥 Applique le payoff du sous-jacent
    return payoff

def barrier_knock_in(ST: np.ndarray, K: float, barrier: float, payoff_sousjacent) -> np.ndarray:
    """Payoff d'une option knock-in (activée si barrière est atteinte)."""
    valid_paths = np.any(ST >= barrier, axis=1)  # Knock-in activé si la barrière est atteinte
    payoff = payoff_sousjacent(ST[:, -1], K) * valid_paths  # 🔥 Applique le payoff sous-jacent uniquement sur les trajectoires validées
    return payoff
