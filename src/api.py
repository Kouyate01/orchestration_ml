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
    """Chargement du modele au demarrage."""
    model_path = MODEL_DIR / "model.joblib"
    try:
        ml["model"] = joblib.load(model_path)
        logger.info(f"Modele charge depuis {model_path}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modele : {e}")
    yield
    ml.clear()

app = FastAPI(title="Water Potability API", version="0.1.0", lifespan=lifespan)

class Features(BaseModel):
    ph: float = Field(..., description="pH de l'eau")
    Hardness: float = Field(..., description="Durete de l'eau")
    Solids: float = Field(..., description="Solides dissous totaux")
    Chloramines: float = Field(..., description="Concentration en chloramines")
    Sulfate: float = Field(..., description="Concentration en sulfates")
    Conductivity: float = Field(..., description="Conductivite")
    Organic_carbon: float = Field(..., description="Carbone organique")
    Trihalomethanes: float = Field(..., description="Trihalomethanes")
    Turbidity: float = Field(..., description="Turbidite")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ph": 7.08, "Hardness": 204.89, "Solids": 20791.3,
                "Chloramines": 7.3, "Sulfate": 368.5, "Conductivity": 564.3,
                "Organic_carbon": 10.38, "Trihalomethanes": 86.99, "Turbidity": 2.96
            }]
        }
    }

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
    
    # Transformation des donnees d'entree en DataFrame
    row = pd.DataFrame([features.model_dump()])
    
    # Prediction
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(
        prediction=int(proba >= 0.5), 
        probability=round(proba, 4)
    )

@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}