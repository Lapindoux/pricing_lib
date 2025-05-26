import unittest
import numpy as np
from pricing import monte_carlo_simulation, monte_carlo_pricing
from pricing.payoffs import vanilla_call

class TestMonteCarloConvergence(unittest.TestCase):
    """Test de convergence du modèle Monte Carlo."""

    @classmethod
    def setUpClass(cls):
        """Initialisation des paramètres."""
        cls.S, cls.K, cls.T, cls.r, cls.sigma = 100, 100, 1, 0.05, 0.2
        cls.num_steps = 252  # Nombre de jours simulés

    def test_convergence_monte_carlo(self):
        """Vérifie que le prix se stabilise quand `num_simulations` augmente."""
        num_simulations_values = [10000, 50000]
        prices = []

        for num_simulations in num_simulations_values:
            ST = monte_carlo_simulation(self.S, self.T, self.r, self.sigma, num_simulations, self.num_steps)
            price = monte_carlo_pricing(ST, self.K, self.r, self.T, vanilla_call)
            prices.append(price)

        # 📈 Vérification de la convergence : les prix doivent être proches du dernier prix
        for i in range(len(prices) - 1):
            self.assertAlmostEqual(prices[i], prices[-1], places=0, msg=f"Convergence faible entre {num_simulations_values[i]} et {num_simulations_values[-1]} simulations")

if __name__ == "__main__":
    unittest.main()
