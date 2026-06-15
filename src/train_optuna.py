"""Optimisation d'hyperparametres avec Optuna."""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from optuna import Trial, create_study, samplers
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
    RANDOM_STATE,
)
from src.data import load_data, split
from src.evaluation import log_shap_summary
from src.features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ModelSpec:
    name: str
    suggest_params: Callable[[Trial], dict]
    build_estimator: Callable[[dict], ClassifierMixin]

def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="random_forest",
            suggest_params=lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            },
            build_estimator=lambda params: RandomForestClassifier(random_state=RANDOM_STATE, **params),
        ),
        ModelSpec(
            name="xgboost",
            suggest_params=lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            },
            build_estimator=lambda params: XGBClassifier(random_state=RANDOM_STATE, **params),
        ),
        ModelSpec(
            name="lightgbm",
            suggest_params=lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
            },
            build_estimator=lambda params: cast(ClassifierMixin, LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **params)),
        ),
    ]

def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("clf", estimator)])

def objective(trial: Trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    params = spec.suggest_params(trial)
    pipeline = build_pipeline(spec.build_estimator(params))
    scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc")
    return float(scores.mean())

def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int):
    study = create_study(
        direction="maximize",
        sampler=samplers.TPESampler(seed=RANDOM_STATE),
    )
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

def describe_registered_version(name: str, version: int, result: FamilyResult, n_trials: int, cv: int) -> None:
    client = mlflow.MlflowClient()
    description = (
        f"Modele {result.spec.name} optimise par Optuna (n_trials={n_trials}, cv={cv}).\n"
        f"ROC AUC: {result.test_roc_auc:.3f}"
    )
    client.update_model_version(name=name, version=str(version), description=description)
    tags = {"model_family": result.spec.name, "cv_roc_auc": f"{result.study.best_value:.4f}", "test_roc_auc": f"{result.test_roc_auc:.4f}"}
    for k, v in tags.items(): client.set_model_version_tag(name=name, version=str(version), key=k, value=v)

def log_family_to_mlflow(result, x_test, y_test, n_trials, cv, register_as=None) -> None:
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.log_params({"n_trials": n_trials, "cv": cv, **result.study.best_params})
        mlflow.log_metrics({"cv_roc_auc": result.study.best_value, "test_roc_auc": result.test_roc_auc})
        
        for trial in result.study.trials:
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric("cv_roc_auc", trial.value)

        log_shap_summary(result.best_pipeline, x_test, result.spec.name)
        model_info = mlflow.sklearn.log_model(result.best_pipeline, "model", registered_model_name=register_as)
        if register_as and model_info.registered_model_version:
            describe_registered_version(register_as, int(model_info.registered_model_version), result, n_trials, cv)

def optimize(n_trials=30, cv=5, use_mlflow=True) -> list[FamilyResult]:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    results = [optimize_family(spec, x_train, y_train, x_test, y_test, n_trials, cv) for spec in build_model_specs()]
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)
    best = results[0]

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            for result in results:
                log_family_to_mlflow(result, x_test, y_test, n_trials, cv, register_as=MODEL_NAME if result is best else None)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, MODEL_DIR / "model.joblib")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    optimize(args.n_trials, args.cv)