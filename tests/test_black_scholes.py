import unittest
from pricing.black_scholes import black_scholes_price

class TestBlackScholes(unittest.TestCase):
    """Tests unitaires pour la fonction Black-Scholes."""

    def test_call_option(self):
        """Test du prix d'une option CALL."""
        prix = black_scholes_price(100, 100, 1, 0.05, 0.2, "call")
        self.assertAlmostEqual(prix, 10.45, places=2)  # Prix attendu (approximatif)

    def test_put_option(self):
        """Test du prix d'une option PUT."""
        prix = black_scholes_price(100, 100, 1, 0.05, 0.2, "put")
        self.assertAlmostEqual(prix, 5.57, places=2)  # Prix attendu (approximatif)

    def test_invalid_option_type(self):
        """Test d'une entrée invalide."""
        with self.assertRaises(ValueError):
            black_scholes_price(100, 100, 1, 0.05, 0.2, "invalid")

if __name__ == "__main__":
    unittest.main()
