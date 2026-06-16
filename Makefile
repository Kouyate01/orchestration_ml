# ==============================================================================
# Projet Qualité de l'Eau - Makefile
# ==============================================================================

SHELL         := /bin/sh
PYTHON        := uv run python
RUN           := uv run
PYTHONPATH    ?= .
export PYTHONPATH

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

.PHONY: help install train train-models train-optuna evaluate mlflow lint format

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(YELLOW)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Installe les dépendances avec uv
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement prêt$(RESET)"

train: ## Entraîne le modèle baseline (usage: make train C=1.0)
	$(PYTHON) -m src.train --c $(C) --max-iter $(MAX_ITER)

train-models: ## Entraîne plusieurs modèles (AutoML) (usage: make train-models CV=5)
	$(PYTHON) -m src.train_models --cv $(CV)

train-optuna: ## Optimise avec Optuna (usage: make train-optuna N_TRIALS=30 CV=5)
	$(PYTHON) -m src.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

evaluate: ## Évalue le dernier modèle enregistré avec porte qualité
	$(PYTHON) -m src.evaluate

mlflow: ## Lance le serveur MLflow local sur le port 5000
	$(RUN) mlflow server --host 127.0.0.1 --port 5000

lint: ## Vérifie le style du code (ruff)
	$(RUN) ruff check src

format: ## Formate le code (ruff)
	$(RUN) ruff format src