"""Entrainement du modele de classification (baseline)."""

from __future__ import annotations
import argparse
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from src.config import MODEL_DIR
from src.data import load_data, split
from src.features import build_preprocessor
from src.tracking import setup_experiment, log_dataset
import mlflow
import mlflow.sklearn

PLOTS_DIR = "/app/data/plots"


def build_model(c=1.0, max_iter=1000):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def train(c=1.0, max_iter=1000):
    setup_experiment()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if mlflow.active_run():
        mlflow.end_run()
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    with mlflow.start_run():
        log_dataset(df, context="training")
        model = build_model(c=c, max_iter=max_iter)
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "model.joblib")
        fig1, ax1 = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax1, cmap="Blues")
        fig1.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
        plt.close(fig1)
        fig2, ax2 = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, proba, ax=ax2)
        fig2.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
        plt.close(fig2)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)
