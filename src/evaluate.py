"""Evaluation automatisee et validation du modele."""
from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.data
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

# Import depuis tes modules src/
from src.config import (
    DATA_PATH,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    TARGET,
)
from src.data import load_data, split
from src.tracking import setup_experiment # On réutilise notre setup centralisé

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def latest_model_uri() -> str:
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"Aucune version enregistree pour '{MODEL_NAME}'.")
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{MODEL_NAME}/{latest.version}"

def build_thresholds() -> dict[str, MetricThreshold]:
    # S11-1 : Définition des seuils
    return {
        "roc_auc": MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True),
        "f1_score": MetricThreshold(threshold=EVAL_F1_MIN, greater_is_better=True),
    }

def evaluate_model(model_uri: str | None = None, validate: bool = True):
    # Préparation des données
    df = load_data()
    _, x_test, _, y_test = split(df)
    eval_df = x_test.copy()
    eval_df[TARGET] = y_test.values

    # Configuration centralisée
    setup_experiment()
    model_uri = model_uri or latest_model_uri()
    logger.info("Evaluation de %s", model_uri)

    with mlflow.start_run(run_name="evaluate"):
        # S11-2 : Tracabilité et évaluation
        dataset = mlflow.data.from_pandas(eval_df, source=str(DATA_PATH), targets=TARGET, name="eval")
        mlflow.log_input(dataset, context="evaluation")
        
        result = mlflow.models.evaluate(
            model_uri, 
            data=eval_df,
            targets=TARGET, 
            model_type="classifier", 
            evaluators=["default"]
        )
        logger.info("f1_score=%.3f roc_auc=%.3f", result.metrics["f1_score"], result.metrics["roc_auc"])
        
        # S11-3 : Porte qualité
        if validate:
            mlflow.validate_evaluation_results(build_thresholds(), result)
            logger.info("Validation réussie : seuils respectés.")
            
        return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    args = parser.parse_args()

    try:
        evaluate_model(model_uri=args.model_uri, validate=args.validate)
    except MlflowException as exc:
        logger.error("Validation echouee : %s", exc)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()