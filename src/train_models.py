"""Entrainement et optimisation de plusieurs modeles de classification (AutoML + SHAP)."""

from __future__ import annotations
import argparse
import logging
import os
import warnings
from dataclasses import dataclass
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODEL_DIR, MODEL_NAME, RANDOM_STATE
from src.data import load_data, split
from src.evaluation import log_shap_summary
from src.features import build_preprocessor
from src.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

PLOTS_DIR = "/app/data/plots"


@dataclass
class ModelSpec:
    name: str
    estimator: ClassifierMixin
    param_grid: dict


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "random_forest",
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [100], "clf__max_depth": [None, 10]},
        ),
        ModelSpec(
            "xgboost",
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
            {"clf__n_estimators": [100], "clf__learning_rate": [0.1]},
        ),
        ModelSpec(
            "lightgbm",
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            {"clf__n_estimators": [100], "clf__learning_rate": [0.1]},
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("clf", estimator)])


@dataclass
class FitResult:
    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


def optimize_model(spec, x_train, y_train, x_test, y_test, cv=5, scoring="roc_auc") -> FitResult:
    search = GridSearchCV(
        build_pipeline(spec.estimator),
        spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return FitResult(
        spec.name,
        best,
        search.best_params_,
        float(search.best_score_),
        float(f1_score(y_test, preds)),
        float(roc_auc_score(y_test, proba)),
        preds,
    )


def describe_registered_version(name, version, result, cv, scoring):
    client = mlflow.MlflowClient()
    client.update_model_version(
        name=name, version=str(version), description=f"Modele {result.name}"
    )


def log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=None, is_best=False) -> None:
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.log_params(result.best_params)
        mlflow.log_metrics(
            {f"cv_{scoring}": result.cv_score, "f1": result.f1, "roc_auc": result.roc_auc}
        )

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
        if is_best:
            os.makedirs(PLOTS_DIR, exist_ok=True)
            fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), bbox_inches="tight")
        mlflow.log_figure(fig, f"cm_{result.name}.png")
        plt.close(fig)

        log_shap_summary(result.best_estimator, x_test, result.name, is_best=is_best)

        try:
            trusted = [
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "numpy.dtype",
                "xgboost.sklearn.XGBClassifier",
                "xgboost.core.Booster",
            ]
            mlflow.sklearn.log_model(
                result.best_estimator,
                "model",
                registered_model_name=register_as,
                skops_trusted_types=trusted,
            )
        except Exception as e:
            logger.warning(
                f"Impossible de sauvegarder le modele dans MLflow a cause d'une restriction de securite : {e}"
            )


def train_all(cv=5, scoring="roc_auc", use_mlflow=True):
    if use_mlflow:
        setup_experiment()
        if mlflow.active_run():
            mlflow.end_run()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv, scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)
    best = results[0]

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models"):
            log_dataset(df, context="training")
            mlflow.set_tag("best_model", best.name)
            for r in results:
                log_run_to_mlflow(
                    r,
                    x_test,
                    y_test,
                    cv,
                    scoring,
                    register_as=MODEL_NAME if r is best else None,
                    is_best=(r is best),
                )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info(
        "Entrainement complet ! Meilleur modele sauvegarde localement et graphiques generes."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    train_all(cv=args.cv)
