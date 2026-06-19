import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

from src.data import prepare_data
from src.train import train

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

def task_prepare_data(**context) -> None:
    prepare_data()

def task_train(**context) -> None:
    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])

def task_check_quality(**context) -> None:
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 is None or f1 < QUALITY_THRESHOLD:
        raise ValueError(f"Performance insuffisante : F1={f1}")
    logger.info("Modèle validé avec F1 = %.2f", f1)

with DAG(
    dag_id="model_retraining",
    description="Réentraînement et contrôle qualité",
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    prepare >> train_task >> check
