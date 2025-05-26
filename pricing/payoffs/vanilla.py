import numpy as np

def vanilla_call(ST: np.ndarray, K: float) -> np.ndarray:
    """Payoff d'une option CALL européenne."""
    return np.maximum(ST[:, -1] - K, 0)

def vanilla_put(ST: np.ndarray, K: float) -> np.ndarray:
    """Payoff d'une option PUT européenne."""
    return np.maximum(K - ST[:, -1], 0)
