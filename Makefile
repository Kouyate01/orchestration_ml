# ==============================================================================
# Projet Qualité de l'Eau - Makefile
# ==============================================================================

SHELL         := /bin/sh
PYTHON        := uv run python
RUN           := uv run
PYTHONPATH    ?= .
export PYTHONPATH

# Variables ajoutées/nécessaires
API_HOST      ?= 127.0.0.1
MLFLOW_PORT   ?= 5000

# Paramètres par défaut
C             ?= 1.0
MAX_ITER      ?= 1000
CV            ?= 5
N_TRIALS      ?= 30
ENTRY         ?= train
PARAMS        ?= 

# Couleurs
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help install train train-models train-optuna evaluate mlflow-ui mlflow-run lint format mlflow-local mlflow-down api predict-client

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(YELLOW)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Installe les dépendances avec uv
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement prêt$(RESET)"

train: ## Entraîne le modèle baseline
	$(PYTHON) -m src.train --c $(C) --max-iter $(MAX_ITER)

train-models: ## Entraîne plusieurs modèles (AutoML)
	$(PYTHON) -m src.train_models --cv $(CV)

train-optuna: ## Optimise avec Optuna
	$(PYTHON) -m src.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

evaluate: ## Évalue et valide la dernière version du modèle
	$(PYTHON) -m src.evaluate

mlflow-ui: ## Démarre le serveur MLflow (via docker-compose)
	docker compose up -d mlflow
	@echo "$(GREEN)MLflow UI : http://localhost:$(MLFLOW_PORT)$(RESET)"

mlflow-local: ## Démarre MLflow en local (SQLite)
	$(PYTHON) -m mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--artifacts-destination ./mlartifacts --serve-artifacts \
		--host $(API_HOST) --port $(MLFLOW_PORT)

mlflow-run: ## Lance un entry point MLproject
	MLFLOW_TRACKING_URI="$$($(PYTHON) -c 'from src.config import MLFLOW_TRACKING_URI; print(MLFLOW_TRACKING_URI)')" \
	$(PYTHON) -m mlflow run . --env-manager local \
		--experiment-name "$$($(PYTHON) -c 'from src.config import MLFLOW_EXPERIMENT; print(MLFLOW_EXPERIMENT)')" \
		-e $(ENTRY) $(PARAMS)

mlflow-down: ## Arrête le serveur MLflow
	docker compose down

api: ## Lance l'API FastAPI (développement)
	$(PYTHON) -m uvicorn src.api:app --reload

predict-client: ## Lance le client de test pour l'API
	PYTHONPATH=. $(PYTHON) scripts/predict_client.py

lint: ## Vérifie le style du code (ruff) sur src et scripts
	$(RUN) ruff check src scripts

format: ## Formate le code (ruff) sur src et scripts
	$(RUN) ruff format src scripts