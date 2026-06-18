"""Optimisation d'hyperparametres avec Optuna."""
from __future__ import annotations
import argparse
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from optuna import Trial, create_study, samplers
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODEL_DIR, MODEL_NAME, RANDOM_STATE
from src.data import load_data, split
from src.evaluation import log_shap_summary
from src.features import build_preprocessor
from src.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
PLOTS_DIR = "/app/data/plots"

@dataclass
class ModelSpec:
    name: str
    suggest_params: Callable[[Trial], dict]
    build_estimator: Callable[[dict], ClassifierMixin]

def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("xgboost", lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 150), "max_depth": t.suggest_int("max_depth", 3, 7)}, lambda p: XGBClassifier(random_state=RANDOM_STATE, **p)),
    ]

def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("clf", estimator)])

def objective(trial: Trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    pipeline = build_pipeline(spec.build_estimator(spec.suggest_params(trial)))
    return float(cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc").mean())

def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int):
    study = create_study(direction="maximize", sampler=samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, spec, x_train, y_train, cv), n_trials=n_trials)
    return study

@dataclass
class FamilyResult:
    spec: ModelSpec
    study: Any
    best_pipeline: Pipeline
    test_roc_auc: float
    preds: np.ndarray

def optimize_family(spec, x_train, y_train, x_test, y_test, n_trials, cv) -> FamilyResult:
    study = run_study(spec, x_train, y_train, n_trials=n_trials, cv=cv)
    best_pipeline = build_pipeline(spec.build_estimator(study.best_params))
    best_pipeline.fit(x_train, y_train)
    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return FamilyResult(spec, study, best_pipeline, float(roc_auc_score(y_test, proba)), preds)

def log_family_to_mlflow(result, x_test, y_test, n_trials, cv, register_as=None, is_best=False) -> None:
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.log_params({"n_trials": n_trials, "cv": cv, **result.study.best_params})
        mlflow.log_metrics({"cv_roc_auc": result.study.best_value, "test_roc_auc": result.test_roc_auc})

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
        if is_best:
            os.makedirs(PLOTS_DIR, exist_ok=True)
            fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), bbox_inches="tight")
        mlflow.log_figure(fig, f"cm_{result.spec.name}.png")
        plt.close(fig)

        log_shap_summary(result.best_pipeline, x_test, result.spec.name, is_best=is_best)
        mlflow.sklearn.log_model(result.best_pipeline, "model", registered_model_name=register_as)

def optimize(n_trials=10, cv=3, use_mlflow=True):
    if use_mlflow:
        setup_experiment()
        if mlflow.active_run():
            mlflow.end_run()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    results = [optimize_family(spec, x_train, y_train, x_test, y_test, n_trials, cv) for spec in build_model_specs()]
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)
    best = results[0]

    if use_mlflow:
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name="optuna-compare"):
            log_dataset(df, context="training")
            for result in results:
                log_family_to_mlflow(result, x_test, y_test, n_trials, cv, register_as=MODEL_NAME if result is best else None, is_best=(result is best))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, MODEL_DIR / "model.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()
    optimize(args.n_trials, args.cv)
