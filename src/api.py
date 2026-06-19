"""API d'inference pour la potabilite de l'eau (FastAPI)."""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from src.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"

    app.state.model = None
    app.state.explainer = None

    try:
        if model_path.exists():
            model = joblib.load(model_path)
            app.state.model = model
            logger.info("Modèle chargé avec succès.")

            # --- FIX SHAP : Création d'une base de référence variée ---
            # Au lieu d'une seule ligne, on génère 50 lignes de données synthétiques
            # pour donner au SHAP la matière nécessaire pour calculer l'importance
            n_features = len(model.feature_names_in_)
            background_data = pd.DataFrame(
                np.random.normal(loc=0, scale=1, size=(50, n_features)),
                columns=model.feature_names_in_,
            )

            # Utilisation de la nouvelle API Explainer
            app.state.explainer = shap.Explainer(model.predict_proba, background_data)
            logger.info("Explainer SHAP initialisé avec échantillon de référence.")
        else:
            logger.error(f"Fichier {model_path} introuvable.")
    except Exception as e:
        logger.error(f"Erreur critique lors du chargement : {e}")

    yield
    app.state.model = None
    app.state.explainer = None


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
    shap_values: list[float]


@app.get("/health")
def health(request: Request) -> dict:
    if request.app.state.model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(request: Request, features: Features) -> PredictionOut:
    model = request.app.state.model
    explainer = request.app.state.explainer

    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    data_dict = features.model_dump()
    row = pd.DataFrame([data_dict])

    if hasattr(model, "feature_names_in_"):
        row = row[model.feature_names_in_]

    try:
        proba = float(model.predict_proba(row)[0, 1])

        shap_values = [0.0] * 9
        if explainer is not None:
            try:
                # Calcul SHAP avec le nouvel Explainer
                shap_res = explainer(row)
                # .values[0] contient les impacts pour les 2 classes
                # On prend les impacts pour la classe 1 (Potable)
                shap_values = shap_res.values[0][:, 1]
            except Exception as e:
                logger.warning(f"Calcul SHAP a échoué : {e}")

        return PredictionOut(
            prediction=int(proba >= 0.5),
            probability=round(proba, 4),
            shap_values=[float(x) for x in shap_values],
        )
    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))
