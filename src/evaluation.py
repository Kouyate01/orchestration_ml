"""Outils d'evaluation partages : graphiques loggues comme artefacts MLflow."""

from __future__ import annotations
import logging
import os
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
PLOTS_DIR = "/opt/airflow/data/plots"


def log_shap_summary(
    pipeline: Pipeline, x_test, name: str, max_samples: int = 200, is_best: bool = False
) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]

    transformed = preprocessor.transform(x_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    sample = transformed[:max_samples]

    try:
        explainer = shap.LinearExplainer(clf, sample)
        shap_values = explainer.shap_values(sample)
    except Exception:
        try:
            explainer = shap.Explainer(clf, sample)
            shap_values = explainer(sample).values
        except Exception:
            logger.warning("SHAP indisponible pour %s", name)
            return

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Importance des variables (SHAP) : {name}")

    if is_best:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), bbox_inches="tight")

    mlflow.log_figure(fig, f"shap_summary_{name}.png")
    plt.close(fig)
