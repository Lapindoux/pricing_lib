import numpy as np
from pricing import monte_carlo_simulation, monte_carlo_pricing
from pricing.payoffs import vanilla_call, vanilla_put, barrier_knock_out, barrier_knock_in, asian_payoff
import matplotlib.pyplot as plt
from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """Configuration par défaut pour les paramètres de pricing."""
    return {
        "market_params": {
            "S": 100,      # Prix initial
            "K": 100,      # Prix d'exercice
            "T": 0.25,     # Durée (3 mois)
            "r": 0.05,     # Taux sans risque (5%)
            "sigma": 0.2   # Volatilité (20%)
        },
        "simulation_params": {
            "num_simulations": 10000,
            "num_steps": 252,
            "seed": 42  # Pour la reproductibilité
        },
        "barrier_params": {
            "barrier": 120,
            "barrier_type": "up"
        }
    }


def plot_trajectories(ST: np.ndarray, barrier: float, num_paths_to_plot: int = 100) -> None:
    """
    Visualise les trajectoires Monte Carlo.
    
    Args:
        ST: Trajectoires du sous-jacent
        barrier: Niveau de barrière pour affichage
        num_paths_to_plot: Nombre de trajectoires à afficher
    """
    plt.figure(figsize=(12, 6))
    
    # Limiter le nombre de trajectoires pour la lisibilité
    paths_to_plot = min(num_paths_to_plot, ST.shape[0])
    
    # Tracé des trajectoires individuelles
    for i in range(paths_to_plot):
        plt.plot(ST[i], alpha=0.3, color='lightblue', linewidth=0.5)

    # Trajectoire moyenne
    mean_path = np.mean(ST, axis=0)
    plt.plot(mean_path, color="darkblue", linewidth=2, label="Trajectoire moyenne")

    # Barrière
    plt.axhline(y=barrier, color='red', linewidth=2, linestyle='--', 
                label=f'Barrière ({barrier})')

    # Formatage
    plt.xlabel("Temps (jours)")
    plt.ylabel("Prix du sous-jacent")
    plt.title(f"Simulation Monte Carlo - {paths_to_plot} trajectoires sur {ST.shape[0]}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_pricing(config: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Exécution des différents scénarios de pricing.
    
    Args:
        config: Configuration des paramètres (optionnel)
        
    Returns:
        Dictionnaire des prix calculés
    """
    if config is None:
        config = get_default_config()

    # Extraction des paramètres
    market = config["market_params"]
    simulation = config["simulation_params"]
    barrier_config = config["barrier_params"]

    print("=== Configuration de Pricing ===")
    print(f"Sous-jacent: S={market['S']}, K={market['K']}")
    print(f"Marché: T={market['T']}, r={market['r']:.1%}, σ={market['sigma']:.1%}")
    print(f"Simulation: {simulation['num_simulations']:,} paths, {simulation['num_steps']} steps")
    print(f"Barrière: {barrier_config['barrier']} ({barrier_config['barrier_type']})")
    print("=" * 40)

    try:
        # 🎲 Génération des trajectoires
        ST = monte_carlo_simulation(
            S=market['S'], 
            T=market['T'], 
            r=market['r'], 
            sigma=market['sigma'],
            num_simulations=simulation['num_simulations'], 
            num_steps=simulation['num_steps'],
            seed=simulation['seed']
        )

        # 📊 Visualisation des trajectoires
        plot_trajectories(ST, barrier_config['barrier'])

        # 💰 Pricing des différentes options
        results = {}
        
        # Options vanilles
        results["CALL européen"] = monte_carlo_pricing(
            ST, market['K'], market['r'], market['T'], 
            payoff_sousjacent=vanilla_call
        )
        
        results["PUT européen"] = monte_carlo_pricing(
            ST, market['K'], market['r'], market['T'], 
            payoff_sousjacent=vanilla_put
        )
        
        # Options à barrière
        results["CALL Knock-Out"] = monte_carlo_pricing(
            ST, market['K'], market['r'], market['T'], 
            payoff_function=barrier_knock_out,
            payoff_sousjacent=vanilla_call,
            barrier=barrier_config['barrier']
        )
        
        results["CALL Knock-In"] = monte_carlo_pricing(
            ST, market['K'], market['r'], market['T'], 
            payoff_function=barrier_knock_in,
            payoff_sousjacent=vanilla_call,
            barrier=barrier_config['barrier']
        )
        
        results["PUT Knock-Out"] = monte_carlo_pricing(
            ST, market['K'], market['r'], market['T'], 
            payoff_function=barrier_knock_out,
            payoff_sousjacent=vanilla_put,
            barrier=barrier_config['barrier']
        )
        
        results["PUT Knock-In"] = monte_carlo_pricing(
            ST, market['K'], market['r'], market['T'], 
            payoff_function=barrier_knock_in,
            payoff_sousjacent=vanilla_put,
            barrier=barrier_config['barrier']
        )
        
        # Option asiatique
        results["Option asiatique"] = monte_carlo_pricing(
            ST, market['K'], market['r'], market['T'], 
            payoff_sousjacent=asian_payoff
        )

        # 📊 Affichage des résultats
        print("\n=== Résultats du Pricing Monte Carlo ===")
        for option, price in results.items():
            print(f"{option:20s}: {price:8.2f} €")

        # Vérification de cohérence
        print("\n=== Vérifications de Cohérence ===")
        call_vanilla = results["CALL européen"]
        call_ko = results["CALL Knock-Out"]
        call_ki = results["CALL Knock-In"]
        
        print(f"Call Vanilla:     {call_vanilla:.2f} €")
        print(f"Call KO + KI:     {call_ko + call_ki:.2f} €")
        print(f"Différence:       {abs(call_vanilla - (call_ko + call_ki)):.2f} €")
        
        if abs(call_vanilla - (call_ko + call_ki)) < 0.5:
            print("✅ Cohérence KO + KI ≈ Vanilla respectée")
        else:
            print("⚠️  Incohérence détectée dans les prix barrière")

        return results

    except Exception as e:
        print(f"❌ Erreur lors du pricing: {e}")
        raise


if __name__ == "__main__":
    run_pricing()
