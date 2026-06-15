"""Configuration centrale du projet de classification."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# CORRECTION : On remonte d'un seul cran pour atteindre la racine (TP-MLOPS/)
# Avant, avec config.py dans todo/mlproject/, il fallait 2 parents. 
# Maintenant, dans src/, il n'en faut qu'un seul.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# TODO (S0-1) : chemin vers votre fichier de donnees (CSV)
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_DIR = ROOT / "models"

# TODO (S0-2) : nom de la colonne cible binaire
TARGET = "Potability"

# TODO (S0-3) : colonnes numeriques
NUMERIC_FEATURES: list[str] = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

# TODO (S0-4) : colonnes categorielles
CATEGORICAL_FEATURES: list[str] = []

RANDOM_STATE = 42

# Surcouche via variables d'environnement
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "classification-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "classifier")