# ==============================================================================
# Projet Qualité de l'Eau - Makefile
# ==============================================================================

SHELL         := /bin/sh
PYTHON        := uv run python
RUN           := uv run
PYTHONPATH    ?= .
export PYTHONPATH

# Variables
API_HOST      ?= 127.0.0.1
MLFLOW_PORT   ?= 5000

# Paramètres par défaut
C             ?= 1.0
MAX_ITER      ?= 1000
CV            ?= 5
N_TRIALS      ?= 30

# Couleurs
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help install train train-models train-optuna evaluate mlflow-ui mlflow-down api streamlit up down clean lint format

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(YELLOW)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Installe les dépendances avec uv
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement prêt$(RESET)"

# --- ORCHESTRATION ---
up: ## Lance toute la stack (API, Streamlit, MLflow) via Docker
	docker compose up -d --build

down: ## Arrête toute la stack
	docker compose down

# --- DÉVELOPPEMENT & UI ---
streamlit: ## Lance Streamlit en local pour le développement
	$(PYTHON) -m streamlit run frontend/app.py

api: ## Lance l'API FastAPI (développement)
	$(PYTHON) -m uvicorn src.api:app --reload

# --- TRAIN & EVAL ---
train: ## Entraîne le modèle baseline
	$(PYTHON) -m src.train --c $(C) --max-iter $(MAX_ITER)

train-optuna: ## Optimise avec Optuna
	$(PYTHON) -m src.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

evaluate: ## Évalue et valide la dernière version du modèle
	$(PYTHON) -m src.evaluate

# --- MLFLOW ---
mlflow-ui: ## Démarre le serveur MLflow (via docker-compose)
	docker compose up -d mlflow

mlflow-down: ## Arrête le serveur MLflow
	docker compose stop mlflow

# --- UTILS ---
clean: ## Nettoie les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

lint: ## Vérifie le style du code (ruff)
	$(RUN) ruff check src scripts

format: ## Formate le code (ruff)
	$(RUN) ruff format src scripts
