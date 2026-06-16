"""Client de test pour l'API FastAPI du modèle.

Lancement (depuis la racine du projet) :
    PYTHONPATH=. python scripts/predict_client.py
"""
from __future__ import annotations

import argparse
import json
import logging

import httpx

# Imports adaptés à votre structure src/
from src.config import TARGET
from src.data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Nombre de clients de test envoyés à l'API.
N_SAMPLES = 3

def build_payloads(n: int = N_SAMPLES) -> list[dict]:
    """Construire n payloads de test à partir du jeu de données (sans valeurs manquantes)."""
    df = load_data()
    
    # On s'assure de ne pas avoir la target dans les features
    if TARGET in df.columns:
        features = df.drop(columns=[TARGET])
    else:
        features = df
        
    # Nettoyage : on supprime les lignes avec des valeurs manquantes (NaN)
    # pour garantir que le payload envoyé à l'API est valide.
    features_clean = features.dropna()
    
    # On échantillonne uniquement parmi les lignes propres
    if len(features_clean) < n:
        logger.warning("Pas assez de lignes propres, réduction du nombre d'échantillons.")
        n = len(features_clean)
        
    sample = features_clean.sample(n=n)
    return [json.loads(row.to_json()) for _, row in sample.iterrows()]

def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000", help="URL de base de l'API"
    )
    args = parser.parse_args()

    payloads = build_payloads()

    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        # S15-1 : Appel GET /health
        try:
            health = client.get("/health")
            logger.info("GET /health -> %s %s", health.status_code, health.json())
        except httpx.ConnectError:
            logger.error("Impossible de se connecter à l'API. Est-elle lancée ?")
            return

        # S15-2 : Appels POST /predict et GET /model-info
        for i, payload in enumerate(payloads):
            response = client.post("/predict", json=payload)
            logger.info("POST /predict (#%d) -> %s %s", i, response.status_code, response.json())
        
        info = client.get("/model-info")
        logger.info("GET /model-info -> %s %s", info.status_code, info.json())

if __name__ == "__main__":
    main()