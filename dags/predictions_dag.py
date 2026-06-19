import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import httpx

from src.config import API_URL, TARGET
from src.data import load_data

logger = logging.getLogger(__name__)

N_PREDICTIONS = 20

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

def task_send_predictions(**context) -> None:
    # On retire la colonne cible
    features = load_data().drop(columns=[TARGET])
    sample = features.sample(n=N_PREDICTIONS)

    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        client.get("/health").raise_for_status()
        for _, row in sample.iterrows():
            payload = row.to_dict()
            response = client.post("/predict", json=payload)
            response.raise_for_status()

    logger.info("%d prévisions envoyées à %s", N_PREDICTIONS, API_URL)

with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 prévisions par jour à l'API",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 10 * * *",
    catchup=False,
    tags=["classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
