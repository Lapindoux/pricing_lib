import numpy as np
from pricing import monte_carlo_simulation, monte_carlo_pricing
from pricing.payoffs import vanilla_call, vanilla_put, barrier_knock_out, barrier_knock_in, asian_payoff
import matplotlib.pyplot as plt

def run_pricing():
    """Exécution des différents scénarios de pricing."""

    # 📌 Définition des paramètres du marché
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    num_simulations, num_steps = 4, 1
    barrier = 120  # Seuil pour options barrières

    # 🎲 Génération des trajectoires
    ST = monte_carlo_simulation(S, T, r, sigma, num_simulations, num_steps)
    # 📊 Tracé des trajectoires
    plt.figure(figsize=(10, 5))
    for i in range(num_simulations):
        plt.plot(ST[i], alpha=0.6)

    # 🔥 Ajout de la trajectoire moyenne
    plt.plot(np.mean(ST, axis=0), color="black", linewidth=2, label="Moyenne des trajectoires")

    # 🔧 Formatage du graphique
    plt.xlabel("Temps (jours)")
    plt.ylabel("Prix du sous-jacent")
    plt.title("Simulation Monte Carlo du sous-jacent")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=120, color='red')
    plt.show()



    # 💰 Pricing des différentes options
    results = {
        "CALL européen": monte_carlo_pricing(ST, K, r, T, vanilla_call),
        "PUT européen": monte_carlo_pricing(ST, K, r, T, vanilla_put),
        "CALL Knock-Out": monte_carlo_pricing(ST, K, r, T, barrier_knock_out, barrier),
        "CALL Knock-In": monte_carlo_pricing(ST, K, r, T, barrier_knock_in, barrier),
        "Option asiatique": monte_carlo_pricing(ST, K, r, T, asian_payoff),
    }

    # 📊 Affichage des résultats
    print("\n=== Résultats du Pricing Monte Carlo ===")
    for option, price in results.items():
        print(f"{option}: {price:.2f} €")

if __name__ == "__main__":
    run_pricing()
