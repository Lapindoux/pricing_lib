import unittest
import numpy as np

from pricing import monte_carlo_simulation, monte_carlo_pricing
from pricing.payoffs import vanilla_call, vanilla_put, barrier_knock_in, barrier_knock_out, asian_payoff


class TestPayoffs(unittest.TestCase):
    """Tests unitaires pour les fonctions de payoff."""

    def setUp(self):
        """Définition des paramètres de test."""
        self.K = 100
        self.ST_standard = np.array([[90, 100, 110], [95, 105, 120]])  # Trajectoires classiques
        self.ST_barrier = np.array([[90, 100, 110], [130, 120, 115]])  # Une trajectoire atteint la barrière
        self.barrier = 120

    def test_vanilla_call(self):
        """Test du payoff d'un CALL européen."""
        expected_payoff = np.array([10, 20])  # Payoff attendu
        np.testing.assert_array_equal(vanilla_call(self.ST_standard[:, -1], self.K), expected_payoff)

    def test_vanilla_put(self):
        """Test du payoff d'un PUT européen."""
        expected_payoff = np.array([0, 0])  # Payoff attendu
        np.testing.assert_array_equal(vanilla_put(self.ST_standard[:, -1], self.K), expected_payoff)

    def test_barrier_knock_out(self):
        """Test du payoff knock-out (annulé si barrière atteinte)."""
        expected_payoff = np.array([10, 0])  # La deuxième trajectoire est invalidée
        np.testing.assert_array_equal(barrier_knock_out(self.ST_barrier, self.K, self.barrier), expected_payoff)

    def test_barrier_knock_in(self):
        """Test du payoff knock-in (activé si barrière atteinte)."""
        expected_payoff = np.array([0, 15])  # Seulement la deuxième trajectoire est activée
        np.testing.assert_array_equal(barrier_knock_in(self.ST_barrier, self.K, self.barrier), expected_payoff)

    def test_asian_payoff(self):
        """Test du payoff asiatique basé sur la moyenne des trajectoires."""
        expected_payoff = np.array([0, 10 / 3 * 2])  # Payoff basé sur la moyenne
        np.testing.assert_array_almost_equal_nulp(asian_payoff(self.ST_standard, 100), expected_payoff, nulp=5)

    def test_non_regression_knock_in_out(self):
        """Test de non-régression : Knock-In + Knock-Out ne dépasse pas le prix du Call classique."""
        call_price = vanilla_call(self.ST_standard[:, -1], self.K)
        knock_in_price = barrier_knock_in(self.ST_standard, self.K, self.barrier)
        knock_out_price = barrier_knock_out(self.ST_standard, self.K, self.barrier)

        total_knock = np.sum(knock_in_price) + np.sum(knock_out_price)
        total_call = np.sum(call_price)
        self.assertLessEqual(total_knock, total_call, "Knock-In + Knock-Out dépasse le Call classique !")


if __name__ == "__main__":
    unittest.main()
