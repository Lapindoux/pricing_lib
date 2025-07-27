import time
import numpy as np
from typing import Dict, Callable
import sys
import os

# Add pricing module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing import monte_carlo_simulation, monte_carlo_pricing, simple_monte_carlo_pricing
from pricing.payoffs import vanilla_call, vanilla_put, asian_payoff, barrier_knock_out
from pricing.black_scholes import black_scholes_price


def benchmark_function(func: Callable, *args, num_runs: int = 5, **kwargs) -> Dict[str, float]:
    """
    Benchmark une fonction et retourne les statistiques de performance.
    
    Args:
        func: Fonction à benchmarker
        *args: Arguments positionnels
        num_runs: Nombre d'exécutions pour le benchmark
        **kwargs: Arguments nommés
        
    Returns:
        Dictionnaire avec les statistiques de temps
    """
    times = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times)
    }


def benchmark_monte_carlo_scaling():
    """Teste la performance en fonction du nombre de simulations."""
    print("=== Benchmark Monte Carlo Scaling ===")
    
    # Paramètres de base
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    num_steps = 252
    
    simulation_sizes = [1000, 5000, 10000, 50000, 100000]
    
    for num_sims in simulation_sizes:
        print(f"\nNombre de simulations: {num_sims:,}")
        
        # Test simulation des trajectoires
        sim_stats = benchmark_function(
            monte_carlo_simulation,
            S, T, r, sigma, num_sims, num_steps
        )
        print(f"  Simulation trajectoires: {sim_stats['mean_time']:.4f}s ± {sim_stats['std_time']:.4f}s")
        
        # Test pricing option vanilla
        pricing_stats = benchmark_function(
            simple_monte_carlo_pricing,
            S, K, T, r, sigma, vanilla_call, num_sims, num_steps
        )
        print(f"  Pricing option vanilla: {pricing_stats['mean_time']:.4f}s ± {pricing_stats['std_time']:.4f}s")


def benchmark_payoff_functions():
    """Compare la performance des différentes fonctions de payoff."""
    print("\n=== Benchmark Payoff Functions ===")
    
    # Génération de trajectoires de test
    num_sims, num_steps = 10000, 252
    ST = np.random.lognormal(mean=0, sigma=0.2, size=(num_sims, num_steps)) * 100
    K = 100
    
    payoff_functions = {
        'Vanilla Call': (vanilla_call, (ST[:, -1], K)),
        'Vanilla Put': (vanilla_put, (ST[:, -1], K)),
        'Asian Call': (asian_payoff, (ST, K, "call")),
        'Asian Put': (asian_payoff, (ST, K, "put")),
        'Barrier Knock-Out': (barrier_knock_out, (ST, K, 120, vanilla_call, "up"))
    }
    
    for name, (func, args) in payoff_functions.items():
        stats = benchmark_function(func, *args)
        print(f"{name:<20}: {stats['mean_time']:.6f}s ± {stats['std_time']:.6f}s")


def benchmark_vs_black_scholes():
    """Compare Monte Carlo vs Black-Scholes pour les options vanilla."""
    print("\n=== Benchmark Monte Carlo vs Black-Scholes ===")
    
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    num_sims = 100000
    num_steps = 252
    
    # Black-Scholes analytique
    bs_stats = benchmark_function(
        black_scholes_price,
        S, K, T, r, sigma, "call"
    )
    bs_price = black_scholes_price(S, K, T, r, sigma, "call")
    
    # Monte Carlo
    mc_stats = benchmark_function(
        simple_monte_carlo_pricing,
        S, K, T, r, sigma, vanilla_call, num_sims, num_steps
    )
    mc_price = simple_monte_carlo_pricing(S, K, T, r, sigma, vanilla_call, num_sims, num_steps)
    
    print(f"Black-Scholes:")
    print(f"  Prix: {bs_price:.4f}")
    print(f"  Temps: {bs_stats['mean_time']:.6f}s")
    
    print(f"Monte Carlo ({num_sims:,} simulations):")
    print(f"  Prix: {mc_price:.4f}")
    print(f"  Temps: {mc_stats['mean_time']:.4f}s")
    print(f"  Erreur relative: {abs(mc_price - bs_price) / bs_price * 100:.2f}%")
    print(f"  Ratio de temps: {mc_stats['mean_time'] / bs_stats['mean_time']:.0f}x plus lent")


def memory_usage_test():
    """Teste l'utilisation mémoire pour différentes tailles de simulation."""
    print("\n=== Test d'utilisation mémoire ===")
    
    try:
        import psutil
        process = psutil.Process()
        
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
        num_steps = 252
        
        for num_sims in [10000, 50000, 100000]:
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulation
            ST = monte_carlo_simulation(S, T, r, sigma, num_sims, num_steps)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            print(f"{num_sims:,} simulations: {mem_used:.1f} MB utilisés")
            
            # Libération mémoire
            del ST
            
    except ImportError:
        print("psutil non disponible pour le test mémoire")


if __name__ == "__main__":
    print("🚀 Démarrage des benchmarks de performance")
    print("=" * 50)
    
    benchmark_monte_carlo_scaling()
    benchmark_payoff_functions()
    benchmark_vs_black_scholes()
    memory_usage_test()
    
    print("\n✅ Benchmarks terminés!")