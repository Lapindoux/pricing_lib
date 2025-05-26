import math

def binomial_tree_price(S: float, K: float, T: float, r: float, sigma: float, N: int, option_type: str = "call") -> float:
    """
    Calcule le prix d'une option avec le modèle binomial.

    Paramètres :
    - S : Prix du sous-jacent
    - K : Prix d'exercice
    - T : Durée jusqu'à échéance (en années)
    - r : Taux sans risque
    - sigma : Volatilité du sous-jacent
    - N : Nombre d'étapes de l'arbre binomial
    - option_type : "call" pour option d'achat, "put" pour option de vente

    Retourne :
    - Prix de l'option (float)
    """
    dt = T / N  # Durée d'une étape
    u = math.exp(sigma * math.sqrt(dt))  # Facteur de hausse
    d = 1 / u  # Facteur de baisse
    p = (math.exp(r * dt) - d) / (u - d)  # Probabilité de hausse

    # Construction de l'arbre des prix
    stock_prices = [[0] * (i + 1) for i in range(N + 1)]
    stock_prices[0][0] = S

    for i in range(1, N + 1):
        for j in range(i + 1):
            stock_prices[i][j] = S * (u ** j) * (d ** (i - j))

    # Calcul des valeurs finales des options
    option_values = [max(0, (stock_prices[N][j] - K) if option_type == "call" else (K - stock_prices[N][j])) for j in range(N + 1)]

    # Remontée dans l'arbre pour obtenir le prix initial
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = math.exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])

    return option_values[0]
