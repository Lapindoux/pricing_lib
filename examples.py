"""
Exemples complets d'utilisation de la bibliothèque de pricing d'options.

Ce fichier démontre les différentes fonctionnalités disponibles et les améliorations apportées.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from pricing import monte_carlo_simulation, monte_carlo_pricing, simple_monte_carlo_pricing
from pricing.payoffs import (
    vanilla_call, vanilla_put, 
    asian_payoff, asian_geometric_payoff, asian_strike_payoff,
    barrier_knock_out, barrier_knock_in, double_barrier_knock_out
)
from pricing.black_scholes import black_scholes_price
from config import PricingConfig


def example_1_vanilla_options():
    """Exemple 1: Options vanilla avec comparaison Black-Scholes vs Monte Carlo."""
    print("=" * 60)
    print("EXEMPLE 1: Options Vanilla - Black-Scholes vs Monte Carlo")
    print("=" * 60)
    
    # Paramètres du marché
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    num_sims = 100000
    num_steps = 252
    
    print(f"Paramètres: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    
    # Prix Black-Scholes
    bs_call = black_scholes_price(S, K, T, r, sigma, "call")
    bs_put = black_scholes_price(S, K, T, r, sigma, "put")
    
    # Prix Monte Carlo
    mc_call = simple_monte_carlo_pricing(S, K, T, r, sigma, vanilla_call, num_sims, num_steps)
    mc_put = simple_monte_carlo_pricing(S, K, T, r, sigma, vanilla_put, num_sims, num_steps)
    
    print("\nRésultats:")
    print(f"CALL - Black-Scholes: {bs_call:.4f}")
    print(f"CALL - Monte Carlo:   {mc_call:.4f} (erreur: {abs(mc_call-bs_call)/bs_call*100:.2f}%)")
    print(f"PUT  - Black-Scholes: {bs_put:.4f}")
    print(f"PUT  - Monte Carlo:   {mc_put:.4f} (erreur: {abs(mc_put-bs_put)/bs_put*100:.2f}%)")
    
    # Vérification de la parité put-call
    parity_bs = bs_call - bs_put - (S - K * np.exp(-r * T))
    parity_mc = mc_call - mc_put - (S - K * np.exp(-r * T))
    print(f"\nParité put-call:")
    print(f"Black-Scholes: {parity_bs:.6f}")
    print(f"Monte Carlo:   {parity_mc:.6f}")


def example_2_asian_options():
    """Exemple 2: Options asiatiques avec différents types de moyennes."""
    print("\n" + "=" * 60)
    print("EXEMPLE 2: Options Asiatiques")
    print("=" * 60)
    
    # Paramètres du marché
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.3
    num_sims = 50000
    num_steps = 252
    
    print(f"Paramètres: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    
    # Génération des trajectoires
    ST = monte_carlo_simulation(S, T, r, sigma, num_sims, num_steps)
    
    # Différents types d'options asiatiques
    asian_arith_call = np.mean(asian_payoff(ST, K, "call"))
    asian_arith_put = np.mean(asian_payoff(ST, K, "put"))
    asian_geom_call = np.mean(asian_geometric_payoff(ST, K, "call"))
    asian_geom_put = np.mean(asian_geometric_payoff(ST, K, "put"))
    asian_float_call = np.mean(asian_strike_payoff(ST, "call", fixed_strike=False))
    asian_float_put = np.mean(asian_strike_payoff(ST, "put", fixed_strike=False))
    
    print("\nPrix des options asiatiques:")
    print(f"Call arithmétique:     {asian_arith_call:.4f}")
    print(f"Put arithmétique:      {asian_arith_put:.4f}")
    print(f"Call géométrique:      {asian_geom_call:.4f}")
    print(f"Put géométrique:       {asian_geom_put:.4f}")
    print(f"Call strike flottant:  {asian_float_call:.4f}")
    print(f"Put strike flottant:   {asian_float_put:.4f}")


def example_3_barrier_options():
    """Exemple 3: Options à barrière avec différents types."""
    print("\n" + "=" * 60)
    print("EXEMPLE 3: Options à Barrière")
    print("=" * 60)
    
    # Paramètres du marché
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.3
    barrier_up = 120
    barrier_down = 80
    num_sims = 50000
    num_steps = 252
    
    print(f"Paramètres: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Barrière haute: {barrier_up}, Barrière basse: {barrier_down}")
    
    # Génération des trajectoires
    ST = monte_carlo_simulation(S, T, r, sigma, num_sims, num_steps)
    
    # Options à barrière
    ko_up_call = np.mean(barrier_knock_out(ST, K, barrier_up, vanilla_call, "up"))
    ko_down_call = np.mean(barrier_knock_out(ST, K, barrier_down, vanilla_call, "down"))
    ki_up_call = np.mean(barrier_knock_in(ST, K, barrier_up, vanilla_call, "up"))
    ki_down_call = np.mean(barrier_knock_in(ST, K, barrier_down, vanilla_call, "down"))
    double_ko_call = np.mean(double_barrier_knock_out(ST, K, barrier_down, barrier_up, vanilla_call))
    
    # Option vanilla pour comparaison
    vanilla_call_price = np.mean(vanilla_call(ST[:, -1], K))
    
    print("\nPrix des options à barrière (CALL):")
    print(f"Vanilla (référence):        {vanilla_call_price:.4f}")
    print(f"Up-and-out (120):          {ko_up_call:.4f}")
    print(f"Down-and-out (80):         {ko_down_call:.4f}")
    print(f"Up-and-in (120):           {ki_up_call:.4f}")
    print(f"Down-and-in (80):          {ki_down_call:.4f}")
    print(f"Double knock-out (80-120): {double_ko_call:.4f}")
    
    # Vérification: KO + KI = Vanilla
    print(f"\nVérifications:")
    print(f"Up KO + Up KI = {ko_up_call + ki_up_call:.4f} vs Vanilla {vanilla_call_price:.4f}")
    print(f"Down KO + Down KI = {ko_down_call + ki_down_call:.4f} vs Vanilla {vanilla_call_price:.4f}")


def example_4_sensitivity_analysis():
    """Exemple 4: Analyse de sensibilité (Greeks approximatifs)."""
    print("\n" + "=" * 60)
    print("EXEMPLE 4: Analyse de Sensibilité")
    print("=" * 60)
    
    # Paramètres de base
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    num_sims = 100000
    num_steps = 252
    
    # Calculs des Greeks approximatifs
    h = 0.01  # Petite variation
    
    # Delta (sensibilité au prix du sous-jacent)
    price_up = simple_monte_carlo_pricing(S + h, K, T, r, sigma, vanilla_call, num_sims, num_steps)
    price_down = simple_monte_carlo_pricing(S - h, K, T, r, sigma, vanilla_call, num_sims, num_steps)
    delta_mc = (price_up - price_down) / (2 * h)
    
    # Gamma (convexité)
    price_base = simple_monte_carlo_pricing(S, K, T, r, sigma, vanilla_call, num_sims, num_steps)
    gamma_mc = (price_up - 2 * price_base + price_down) / (h ** 2)
    
    # Vega (sensibilité à la volatilité)
    price_vol_up = simple_monte_carlo_pricing(S, K, T, r, sigma + h, vanilla_call, num_sims, num_steps)
    price_vol_down = simple_monte_carlo_pricing(S, K, T, r, sigma - h, vanilla_call, num_sims, num_steps)
    vega_mc = (price_vol_up - price_vol_down) / (2 * h)
    
    # Comparaison avec Black-Scholes
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta_bs = norm.cdf(d1)
    gamma_bs = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega_bs = S * norm.pdf(d1) * np.sqrt(T)
    
    print(f"Prix de base: {price_base:.4f}")
    print("\nGreeks (Monte Carlo vs Black-Scholes):")
    print(f"Delta: {delta_mc:.4f} vs {delta_bs:.4f} (erreur: {abs(delta_mc-delta_bs)/delta_bs*100:.1f}%)")
    print(f"Gamma: {gamma_mc:.4f} vs {gamma_bs:.4f} (erreur: {abs(gamma_mc-gamma_bs)/gamma_bs*100:.1f}%)")
    print(f"Vega:  {vega_mc:.4f} vs {vega_bs:.4f} (erreur: {abs(vega_mc-vega_bs)/vega_bs*100:.1f}%)")


def example_5_configuration_management():
    """Exemple 5: Gestion de configuration avancée."""
    print("\n" + "=" * 60)
    print("EXEMPLE 5: Gestion de Configuration")
    print("=" * 60)
    
    # Création d'une configuration personnalisée
    config = PricingConfig()
    
    # Modification des paramètres
    config.set_market_param('S', 120)
    config.set_market_param('sigma', 0.25)
    config.set_simulation_param('num_simulations', 200000)
    
    print("Configuration personnalisée:")
    print(f"Prix initial: {config.get_market_param('S')}")
    print(f"Volatilité: {config.get_market_param('sigma')}")
    print(f"Simulations: {config.get_simulation_param('num_simulations')}")
    
    # Utilisation de la configuration
    market_params = config.get_market_params()
    sim_params = config.get_simulation_params()
    
    price = simple_monte_carlo_pricing(
        market_params['S'], 
        market_params['K'], 
        market_params['T'], 
        market_params['r'], 
        market_params['sigma'],
        vanilla_call,
        sim_params['num_simulations'],
        sim_params['num_steps']
    )
    
    print(f"\nPrix calculé avec la configuration: {price:.4f}")


def example_6_convergence_analysis():
    """Exemple 6: Analyse de convergence Monte Carlo."""
    print("\n" + "=" * 60)
    print("EXEMPLE 6: Analyse de Convergence")
    print("=" * 60)
    
    # Paramètres
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    num_steps = 252
    
    # Prix théorique Black-Scholes
    bs_price = black_scholes_price(S, K, T, r, sigma, "call")
    
    # Test de convergence
    simulation_sizes = [1000, 5000, 10000, 25000, 50000, 100000, 200000]
    mc_prices = []
    errors = []
    
    print(f"Prix théorique Black-Scholes: {bs_price:.4f}")
    print("\nConvergence Monte Carlo:")
    print("Simulations | Prix MC | Erreur | Erreur %")
    print("-" * 45)
    
    for num_sims in simulation_sizes:
        mc_price = simple_monte_carlo_pricing(S, K, T, r, sigma, vanilla_call, num_sims, num_steps, seed=42)
        error = abs(mc_price - bs_price)
        error_pct = error / bs_price * 100
        
        mc_prices.append(mc_price)
        errors.append(error)
        
        print(f"{num_sims:>10,} | {mc_price:>7.4f} | {error:>6.4f} | {error_pct:>6.2f}%")
    
    # Tracé optionnel (si matplotlib disponible)
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.semilogx(simulation_sizes, mc_prices, 'b-o', label='Monte Carlo')
        plt.axhline(y=bs_price, color='r', linestyle='--', label='Black-Scholes')
        plt.xlabel('Nombre de simulations')
        plt.ylabel('Prix de l\'option')
        plt.title('Convergence du prix')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.loglog(simulation_sizes, errors, 'g-o')
        # Ligne théorique 1/sqrt(N)
        theoretical = errors[0] * np.sqrt(simulation_sizes[0] / np.array(simulation_sizes))
        plt.loglog(simulation_sizes, theoretical, 'r--', label='1/√N théorique')
        plt.xlabel('Nombre de simulations')
        plt.ylabel('Erreur absolue')
        plt.title('Vitesse de convergence')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
        print("\nGraphique sauvegardé: convergence_analysis.png")
        
    except ImportError:
        print("\nMatplotlib non disponible pour les graphiques")


def main():
    """Fonction principale exécutant tous les exemples."""
    print("🚀 EXEMPLES DE LA BIBLIOTHÈQUE DE PRICING D'OPTIONS")
    print("📊 Démonstration des fonctionnalités améliorées")
    
    try:
        example_1_vanilla_options()
        example_2_asian_options()
        example_3_barrier_options()
        example_4_sensitivity_analysis()
        example_5_configuration_management()
        example_6_convergence_analysis()
        
        print("\n" + "=" * 60)
        print("✅ TOUS LES EXEMPLES EXÉCUTÉS AVEC SUCCÈS!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()