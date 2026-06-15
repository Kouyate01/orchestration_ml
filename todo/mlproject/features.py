"""Construction du pre-processing."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    # On crée un mini-pipeline pour les variables numériques : 
    # 1. On remplace les NaN par la médiane
    # 2. On applique le StandardScaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer(
        transformers=[
            # On utilise notre nouveau num_pipeline au lieu du simple StandardScaler
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )