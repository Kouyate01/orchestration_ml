"""API d'inference pour la potabilite de l'eau (FastAPI)."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import depuis votre configuration
from src.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Chargement du modele au demarrage et verification des colonnes attendues."""
    model_path = MODEL_DIR / "model.joblib"
    try:
        model = joblib.load(model_path)
        ml["model"] = model
        # Log des colonnes attendues pour faciliter le debug
        if hasattr(model, "feature_names_in_"):
            logger.info(f"Modele charge. Colonnes attendues : {model.feature_names_in_}")
        logger.info(f"Modele charge depuis {model_path}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modele : {e}")
    yield
    ml.clear()


app = FastAPI(title="Water Potability API", version="0.1.0", lifespan=lifespan)


class Features(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


class PredictionOut(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    # 1. Conversion en DataFrame
    data_dict = features.model_dump()
    row = pd.DataFrame([data_dict])

    # 2. ALIGNEMENT DES COLONNES (Correction cruciale)
    if hasattr(model, "feature_names_in_"):
        try:
            # Réordonne et sélectionne les colonnes selon ce que le modèle attend
            row = row[model.feature_names_in_]
        except KeyError as e:
            logger.error(f"Colonnes manquantes dans la requete : {e}")
            raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {e}")

    # 3. Prediction
    try:
        proba = float(model.predict_proba(row)[0, 1])
        return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))
    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
