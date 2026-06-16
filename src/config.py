"""Configuration centrale du projet de classification."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# On remonte d'un seul cran pour atteindre la racine depuis src/
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Chemins
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_DIR = ROOT / "models"

# Cible et features
TARGET = "Potability"
NUMERIC_FEATURES: list[str] = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]
CATEGORICAL_FEATURES: list[str] = []

RANDOM_STATE = 42

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "classification-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "classifier")

# Paramètres requis pour le tracking centralisé
MLFLOW_EXPERIMENT_DESCRIPTION = "Projet de prédiction de la potabilité de l'eau via capteurs chimiques."
MLFLOW_EXPERIMENT_TAGS = {
    "project_name": "mlops-water",
    "team": "data-science",
    "version": "1.0"
}

# Seuils pour la porte qualité (S11)
EVAL_F1_MIN = 0.40      # Ajuste selon tes performances réelles
EVAL_ROC_AUC_MIN = 0.55 # Ajuste selon tes performances réelles