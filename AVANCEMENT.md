# 🚀 Rapport d'Avancement - Projet MLOps (Qualité de l'Eau)

## 🎯 Résumé du Projet
Adaptation du squelette MLOps pour un cas d'usage environnemental : **La prédiction de la potabilité de l'eau** à partir de relevés de capteurs physico-chimiques (pH, Dureté, Sulfates, etc.). 
* **Cible :** `Potability` (Binaire : 1 = Potable, 0 = Non potable).
* **Type de données :** Exclusivement numériques (simplification du pipeline pour se concentrer sur l'orchestration).

---

## ✅ 1. Configuration Initiale et Environnement (TP S0)
* **Initialisation de l'environnement :** Utilisation de `uv` pour gérer le virtualenv et les dépendances (`uv sync --extra dev`).
* **Correction des configurations :** * Adaptation de `pyproject.toml` : renommage du projet en `mlops-water` et correction du chemin de build vers le package `mlproject` (au lieu de `src/churn`).
  * Adaptation de `config.py` : Déclaration du chemin de données, de la cible `Potability` et de la liste complète des `NUMERIC_FEATURES`.
  * Nettoyage du `Makefile` pour décommenter les cibles d'exécution.

## 🛠️ 2. Ingénierie des Données et Pipeline Robuste
Pour respecter les bonnes pratiques logicielles (séparation des responsabilités), l'architecture de traitement a été refactorisée :
* **`data.py` (I/O & Splitting) :** Gère uniquement le chargement du fichier CSV et le découpage Train/Test avec stratification (`stratify=y`) pour respecter l'équilibre des classes.
* **`features.py` (Pré-traitement MLOps) :** Création d'un pipeline complet pour les données numériques gérant nativement les valeurs manquantes (`NaN`). Utilisation d'un `SimpleImputer(strategy="median")` couplé à un `StandardScaler` dans le `ColumnTransformer`. Cela garantit que l'API de production (Séance 12) ne plantera pas si un capteur renvoie une valeur vide.

## 📊 3. Tracking des Expériences avec MLflow (TP S5)
Le script d'entraînement de la baseline (`train.py`) a été entièrement instrumenté :
* **Connexion au serveur :** Paramétrage de l'URI et création de l'expérience `classification-baseline`.
* **Logging systématique :**
  * Paramètres du modèle (`C`, `max_iter`).
  * Métriques de performance (`f1_score`, `roc_auc`).
  * Sauvegarde automatique du modèle dans le format standard (`mlflow.sklearn.log_model`).
* **Artefact visuel :** Génération et envoi de la matrice de confusion (`confusion_matrix.png`) directement dans les artefacts du run MLflow.
* **Validation :** Lancement de plusieurs itérations avec des hyperparamètres différents (`C=0.1`, `C=1.0`, `C=10.0`) et comparaison des résultats via le *Parallel Coordinates Plot* sur l'interface UI de MLflow (Port 5000).

## 🐙 4. Versioning et Bonnes Pratiques Git
* Création d'un fichier `.gitignore` robuste pour exclure les environnements (`.venv/`), les données (`data/`, `*.csv`) et les historiques locaux (`mlruns/`, `models/`).
* Dépôt du code source fonctionnel et documenté (`README.md`) sur le repository distant GitHub `Kouyate01/orchestration_ml`.

