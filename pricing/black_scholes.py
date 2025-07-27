# black_scholes.py - Implémentation du modèle Black-Scholes pour le pricing des options

import math
import scipy.stats as si


def black_scholes_price(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    option_type: str = "call"
) -> float:
    """
    Calcule le prix d'une option avec le modèle Black-Scholes.

    Args:
        S: Prix du sous-jacent (doit être > 0)
        K: Prix d'exercice (doit être > 0)
        T: Durée jusqu'à l'échéance en années (doit être > 0)
        r: Taux sans risque
        sigma: Volatilité du sous-jacent (doit être > 0)
        option_type: "call" pour option d'achat, "put" pour option de vente

    Returns:
        Prix de l'option (float)
        
    Raises:
        ValueError: Si les paramètres sont invalides
    """
    # Validation des paramètres
    if S <= 0:
        raise ValueError("Le prix du sous-jacent S doit être positif")
    if K <= 0:
        raise ValueError("Le prix d'exercice K doit être positif")
    if T <= 0:
        raise ValueError("La durée T doit être positive")
    if sigma <= 0:
        raise ValueError("La volatilité sigma doit être positive")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * math.exp(-r * T) * si.norm.cdf(d2)
    else:  # put
        price = K * math.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

    return price
