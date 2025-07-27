import numpy as np
import numpy.typing as npt


def vanilla_call(ST: npt.NDArray[np.float64], K: float) -> npt.NDArray[np.float64]:
    """
    Payoff d'une option CALL européenne.
    
    Args:
        ST: Prix du sous-jacent à l'échéance
        K: Prix d'exercice
        
    Returns:
        Payoff max(ST - K, 0)
    """
    return np.maximum(ST - K, 0)


def vanilla_put(ST: npt.NDArray[np.float64], K: float) -> npt.NDArray[np.float64]:
    """
    Payoff d'une option PUT européenne.
    
    Args:
        ST: Prix du sous-jacent à l'échéance
        K: Prix d'exercice
        
    Returns:
        Payoff max(K - ST, 0)
    """
    return np.maximum(K - ST, 0)
