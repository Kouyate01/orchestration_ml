"""Entrainement du modele de classification (baseline)."""

from __future__ import annotations

import argparse
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

# Imports pour les composants src/
from src.config import MODEL_DIR
from src.data import load_data, split
from src.features import build_preprocessor
from src.tracking import setup_experiment, log_dataset

import mlflow
import mlflow.sklearn


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    # 1. Configuration centralisée
    setup_experiment()

    # SÉCURITÉ : Fermer tout run zombie avant de commencer
    if mlflow.active_run():
        mlflow.end_run()

    # 2. Chargement des données
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    # 3. Entraînement avec suivi MLflow
    # Le run doit être ouvert AVANT d'appeler log_dataset
    with mlflow.start_run():
        log_dataset(df, context="training")  # Maintenant dans le run actif

        model = build_model(c=c, max_iter=max_iter)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

        # Logging MLflow
        mlflow.log_params({"C": c, "max_iter": max_iter})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "logistic_regression_model")

        # Sauvegarde de la matrice de confusion
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, cmap="Blues")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "model.joblib")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
