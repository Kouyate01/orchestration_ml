"""Chargement, preparation et decoupage des donnees."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, RANDOM_STATE, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Charge le dataset."""
    return pd.read_csv(path)


def prepare_data(path=DATA_PATH) -> None:
    """Charge et nettoie les donnees (imputation des valeurs manquantes)."""
    df = load_data(path)
    # Remplissage des valeurs manquantes par la mediane (standard pour ce dataset)
    df = df.fillna(df.median())
    # Sauvegarde eventuelle du fichier nettoye si besoin, ou simplement confirmation
    print(f"Donnees preparees : {df.shape[0]} lignes.")
    # On sauvegarde une version propre pour l'entrainement si nécessaire
    df.to_csv(path, index=False)


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Effectue le split train/test."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
