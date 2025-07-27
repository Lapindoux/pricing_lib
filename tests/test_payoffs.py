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
        result = barrier_knock_out(self.ST_barrier, self.K, self.barrier, vanilla_call)
        np.testing.assert_array_equal(result, expected_payoff)

    def test_barrier_knock_in(self):
        """Test du payoff knock-in (activé si barrière atteinte)."""
        expected_payoff = np.array([0, 15])  # Seulement la deuxième trajectoire est activée
        result = barrier_knock_in(self.ST_barrier, self.K, self.barrier, vanilla_call)
        np.testing.assert_array_equal(result, expected_payoff)

    def test_asian_payoff(self):
        """Test du payoff asiatique basé sur la moyenne des trajectoires."""
        # Avec la correction: moyenne par trajectoire
        # Trajectoire 1: [90, 100, 110] -> moyenne = 100 -> payoff = max(100-100, 0) = 0
        # Trajectoire 2: [95, 105, 120] -> moyenne = 106.67 -> payoff = max(106.67-100, 0) = 6.67
        expected_payoff = np.array([0, 6.666666666666667])
        result = asian_payoff(self.ST_standard, self.K)
        np.testing.assert_array_almost_equal(result, expected_payoff)

    def test_non_regression_knock_in_out(self):
        """Test de non-régression : Knock-In + Knock-Out égale le prix du Call classique."""
        call_price = vanilla_call(self.ST_standard[:, -1], self.K)
        knock_in_price = barrier_knock_in(self.ST_standard, self.K, self.barrier, vanilla_call)
        knock_out_price = barrier_knock_out(self.ST_standard, self.K, self.barrier, vanilla_call)

        # Pour une barrière up, KI + KO devrait égaler le call vanilla
        total_knock = knock_in_price + knock_out_price
        np.testing.assert_array_almost_equal(total_knock, call_price, 
                                           err_msg="Knock-In + Knock-Out ne donne pas le Call classique")

    def test_barrier_types(self):
        """Test des différents types de barrières."""
        # Test down-and-out
        ST_down = np.array([[110, 90, 95], [105, 100, 110]])  # Une trajectoire touche la barrière down
        barrier_down = 95
        
        # Down-and-out: invalidé si on touche la barrière par le bas
        result_down_out = barrier_knock_out(ST_down, self.K, barrier_down, vanilla_call, "down")
        expected_down_out = np.array([0, 10])  # Première trajectoire invalidée
        np.testing.assert_array_equal(result_down_out, expected_down_out)
        
        # Down-and-in: activé si on touche la barrière par le bas
        result_down_in = barrier_knock_in(ST_down, self.K, barrier_down, vanilla_call, "down")
        expected_down_in = np.array([0, 0])  # Première trajectoire activée mais OTM
        np.testing.assert_array_equal(result_down_in, expected_down_in)


if __name__ == "__main__":
    unittest.main()
