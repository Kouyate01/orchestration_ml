# 🚀 Rapport d'Avancement - Projet MLOps (Qualité de l'Eau)

## 🎯 Résumé du Projet
Adaptation du squelette MLOps pour un cas d'usage environnemental : **La prédiction de la potabilité de l'eau** à partir de relevés de capteurs physico-chimiques (pH, Dureté, Sulfates, etc.). 
* **Cible :** `Potability` (Binaire : 1 = Potable, 0 = Non potable).
* **Type de données :** Exclusivement numériques.

---

## ✅ 1. Configuration Initiale et Environnement (TP S0)
* **Initialisation de l'environnement :** Utilisation de `uv` pour gérer le virtualenv et les dépendances.
* **Refactorisation structurelle :** Migration vers une structure `src/` pour garantir l'importabilité des modules et la séparation des responsabilités.
* **Nettoyage :** Adaptation de `config.py` (cibles, features) et du `Makefile`.

## 🛠️ 2. Ingénierie des Données et Pipeline Robuste
* **`data.py` :** Gestion du chargement CSV et split Train/Test stratifié.
* **`features.py` :** Création d'un `ColumnTransformer` (Imputation par la médiane + Standardisation) pour garantir la robustesse des données en production.

## 📊 3. Tracking des Expériences avec MLflow (TP S5)
* **Instrumentation :** Paramétrage de `classification-baseline` pour tracker les modèles.
* **Logging :** Enregistrement systématique des paramètres, métriques, artefacts (matrice de confusion) et du modèle (`mlflow.sklearn.log_model`).
* **Validation :** Comparaison des runs via l'UI MLflow.

## 📈 4. Optimisation d'Hyperparamètres avec Optuna (TP S6)

* **Passage à l'AutoML bayésien :** Utilisation du `TPESampler` d'Optuna.
* **Architecture multi-familles :** Optimisation simultanée de Random Forest, XGBoost et LightGBM.
* **Traçabilité :** Chaque essai (trial) est un run imbriqué dans MLflow, permettant une analyse fine de la convergence.

## 🛠️ 5. AutoML, GridSearchCV et Explicabilité SHAP (TP S7)

* **Industrialisation :** Pipeline `train_models.py` utilisant `GridSearchCV` pour l'exploration systématique.
* **Explicabilité (SHAP) :** Intégration de `log_shap_summary` pour générer automatiquement l'importance globale des variables (artefact `shap_summary.png`).
* **Model Registry :** Automatisation du versioning (`v1`) avec tags de performance et descriptions métier pour une gouvernance complète.

## 🐙 6. Versioning et Bonnes Pratiques Git
* Création d'un `.gitignore` robuste.
* Workflow complet : `git add` -> `git commit` -> `git push` pour assurer la sécurité et la reproductibilité du code source.

---

## 🚀 État Actuel du Projet
* **Pipeline d'entraînement :** ✅ (AutoML + Optimisation + SHAP)
* **Tracking MLflow :** ✅ (Centralisé et instrumenté)
* **Model Registry :** ✅ (Version `v1` prête pour production)

*Prochaine étape : Industrialisation via une API (FastAPI) pour l'inférence en temps réel.*