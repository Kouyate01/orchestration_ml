# 💧 Projet MLOps : Prédiction de la Potabilité de l'Eau

## 📝 Description du Projet
Ce projet s'inscrit dans le cadre du module d'Orchestration Machine Learning. Il exploite un jeu de données environnemental regroupant diverses mesures physico-chimiques de la qualité de l'eau (pH, dureté, chloramines, sulfates, etc.) provenant de différentes sources.

L'objectif est de concevoir, suivre et déployer un modèle de Machine Learning de bout en bout pour évaluer la sécurité de l'eau.

## 🎯 Définition de la Cible (Target)
Le modèle effectue une classification binaire pour déterminer la viabilité de l'eau :
* **`1`** : L'eau est **potable** (propre à la consommation humaine).
* **`0`** : L'eau **n'est pas potable**.

## 🌍 Cas d'Usage & Utilité Métier
Prédire la potabilité par le Machine Learning permet d'**automatiser la surveillance des réserves d'eau douce** directement à partir des relevés de capteurs chimiques. 
Cette approche garantit la sécurité sanitaire de manière proactive et en temps réel, évitant l'attente et le coût de longs tests en laboratoire traditionnels.

## ⚙️ Architecture & Focus MLOps
La force de ce jeu de données réside dans sa structure : les colonnes étant presque exclusivement numériques, la phase de pré-traitement des données est grandement simplifiée. 

Ce choix stratégique permet de concentrer **100% des efforts sur l'ingénierie MLOps**, notamment :
* **L'environnement & Gestion des dépendances** : Isolation via `uv`.
* **Le Suivi d'Expérimentations** : Tracking des paramètres, métriques et modèles avec **MLflow**.
* **L'Optimisation** : Recherche des hyperparamètres via **Optuna**.
* **Le Déploiement** : Mise à disposition du modèle via une API **FastAPI**.
* **L'Interface Utilisateur** : Consommation de l'API via une application **Streamlit**.
* Lien du dataset : https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability
