# black_scholes.py - Implémentation du modèle Black-Scholes pour le pricing des options

import math
import scipy.stats as si


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Calcule le prix d'une option avec le modèle Black-Scholes.

    Arguments :
    - S : Prix du sous-jacent
    - K : Prix d'exercice
    - T : Durée jusqu'à l'échéance (en années)
    - r : Taux sans risque
    - sigma : Volatilité du sous-jacent
    - option_type : "call" pour option d'achat, "put" pour option de vente

    Retourne :
    - Prix de l'option (float)
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * math.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    return price
