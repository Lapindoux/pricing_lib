"""
Configuration management for pricing library.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class PricingConfig:
    """Gestionnaire de configuration pour les paramètres de pricing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise la configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration (optionnel)
        """
        self.config_path = config_path
        self._config = self._load_default_config()
        
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut."""
        return {
            "market_params": {
                "S": 100.0,      # Prix initial du sous-jacent
                "K": 100.0,      # Prix d'exercice
                "T": 0.25,       # Durée jusqu'à échéance (années)
                "r": 0.05,       # Taux sans risque
                "sigma": 0.20    # Volatilité
            },
            "simulation_params": {
                "num_simulations": 10000,  # Nombre de simulations
                "num_steps": 252,          # Nombre de pas de temps
                "seed": None,              # Graine aléatoire (None = pas de seed fixe)
                "antithetic": False        # Utiliser les variates antithétiques
            },
            "barrier_params": {
                "barrier": 120.0,          # Niveau de barrière
                "barrier_type": "up"       # Type: "up" ou "down"
            },
            "visualization": {
                "show_plots": True,        # Afficher les graphiques
                "num_paths_plot": 100,     # Nombre de trajectoires à afficher
                "figure_size": [12, 6]     # Taille des figures
            },
            "validation": {
                "tolerance": 0.5,          # Tolérance pour les tests de cohérence
                "run_checks": True         # Exécuter les vérifications
            }
        }
    
    def load_from_file(self, config_path: str) -> None:
        """
        Charge la configuration depuis un fichier JSON.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # Mise à jour récursive de la configuration
            self._update_config(self._config, file_config)
            
        except FileNotFoundError:
            print(f"⚠️  Fichier de configuration introuvable: {config_path}")
        except json.JSONDecodeError as e:
            print(f"❌ Erreur de parsing JSON: {e}")
    
    def save_to_file(self, config_path: str) -> None:
        """
        Sauvegarde la configuration dans un fichier JSON.
        
        Args:
            config_path: Chemin de destination
        """
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            print(f"✅ Configuration sauvegardée: {config_path}")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
    
    def _update_config(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Mise à jour récursive de la configuration."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """
        Récupère une valeur de configuration.
        
        Args:
            section: Section de configuration
            key: Clé spécifique (optionnel)
            
        Returns:
            Valeur de configuration
        """
        if section not in self._config:
            raise KeyError(f"Section '{section}' non trouvée")
        
        if key is None:
            return self._config[section]
        
        if key not in self._config[section]:
            raise KeyError(f"Clé '{key}' non trouvée dans '{section}'")
        
        return self._config[section][key]
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Définit une valeur de configuration.
        
        Args:
            section: Section de configuration
            key: Clé à modifier
            value: Nouvelle valeur
        """
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section][key] = value
    
    def get_market_param(self, key: str) -> Any:
        """
        Récupère un paramètre de marché spécifique.
        
        Args:
            key: Nom du paramètre (S, K, T, r, sigma)
            
        Returns:
            Valeur du paramètre
        """
        return self.get("market_params", key)
    
    def set_market_param(self, key: str, value: Any) -> None:
        """
        Définit un paramètre de marché.
        
        Args:
            key: Nom du paramètre (S, K, T, r, sigma)
            value: Nouvelle valeur
        """
        self.set("market_params", key, value)
    
    def get_market_params(self) -> Dict[str, Any]:
        """Récupère tous les paramètres de marché."""
        return self.get("market_params")
    
    def get_simulation_param(self, key: str) -> Any:
        """
        Récupère un paramètre de simulation spécifique.
        
        Args:
            key: Nom du paramètre (num_simulations, num_steps, seed, etc.)
            
        Returns:
            Valeur du paramètre
        """
        return self.get("simulation_params", key)
    
    def set_simulation_param(self, key: str, value: Any) -> None:
        """
        Définit un paramètre de simulation.
        
        Args:
            key: Nom du paramètre (num_simulations, num_steps, seed, etc.)
            value: Nouvelle valeur
        """
        self.set("simulation_params", key, value)
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """Récupère tous les paramètres de simulation."""
        return self.get("simulation_params")

    def validate(self) -> bool:
        """
        Valide la configuration actuelle.
        
        Returns:
            True si la configuration est valide
        """
        try:
            market = self._config["market_params"]
            simulation = self._config["simulation_params"]
            
            # Validation des paramètres du marché
            assert market["S"] > 0, "Prix initial doit être positif"
            assert market["K"] > 0, "Prix d'exercice doit être positif"
            assert market["T"] > 0, "Durée doit être positive"
            assert market["sigma"] > 0, "Volatilité doit être positive"
            
            # Validation des paramètres de simulation
            assert simulation["num_simulations"] > 0, "Nombre de simulations doit être positif"
            assert simulation["num_steps"] > 0, "Nombre de pas doit être positif"
            
            return True
            
        except (KeyError, AssertionError) as e:
            print(f"❌ Configuration invalide: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Retourne la configuration sous forme de dictionnaire."""
        return self._config.copy()
    
    def __str__(self) -> str:
        """Représentation string de la configuration."""
        return json.dumps(self._config, indent=2, ensure_ascii=False)