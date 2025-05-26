import unittest
from pricing.binomial_tree import binomial_tree_price

class TestBinomialTree(unittest.TestCase):
    """Tests unitaires pour le modèle binomial."""

    def test_call_option(self):
        """Test du prix d'une option CALL avec 100 étapes."""
        prix = binomial_tree_price(100, 100, 1, 0.05, 0.2, 100, "call")
        self.assertAlmostEqual(prix, 10.43, places=2)  # Prix attendu approximatif

    def test_put_option(self):
        """Test du prix d'une option PUT avec 100 étapes."""
        prix = binomial_tree_price(100, 100, 1, 0.05, 0.2, 100, "put")
        self.assertAlmostEqual(prix, 5.55, places=2)  # Prix attendu approximatif

    def test_invalid_option_type(self):
        """Test d'une entrée invalide."""
        with self.assertRaises(ValueError):
            binomial_tree_price(100, 100, 1, 0.05, 0.2, 100, "invalid")

    def test_few_steps(self):
        """Test du modèle avec peu d'étapes (10)."""
        prix_call = binomial_tree_price(100, 100, 1, 0.05, 0.2, 10, "call")
        prix_put = binomial_tree_price(100, 100, 1, 0.05, 0.2, 10, "put")
        self.assertTrue(prix_call > 0)
        self.assertTrue(prix_put > 0)

if __name__ == "__main__":
    unittest.main()
