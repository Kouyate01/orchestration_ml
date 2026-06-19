import streamlit as st
import httpx
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(page_title="Projet MLOps - KOUYATE MOHAMED ALI", layout="wide")

# --- STYLE CSS ---
st.markdown("""
    <style>
    .name-box { background-color: #0e1117; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #1E88E5; margin-bottom: 20px; }
    .author-name { font-size: 28px; color: #1E88E5; font-weight: bold; margin: 0; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="name-box"><p class="author-name">KOUYATE MOHAMED ALI</p></div>', unsafe_allow_html=True)
st.title("💧 Projet MLOps : Prédiction de la Potabilité de l'eau")

# --- README INTÉGRAL ---
with st.expander("📝 Description du Projet", expanded=True):
    st.markdown("""
### 📝 Description du Projet
Ce projet s'inscrit dans le cadre du module d'Orchestration Machine Learning. Il exploite un jeu de données environnemental regroupant diverses mesures physico-chimiques de la qualité de l'eau (pH, dureté, chloramines, sulfates, etc.) provenant de différentes sources. 
L'objectif est de concevoir, suivre et déployer un modèle de Machine Learning de bout en bout pour évaluer la sécurité de l'eau.

### 🎯 Définition de la Cible (Target)
Le modèle effectue une classification binaire pour déterminer la viabilité de l'eau :
- **1** : L'eau est potable (propre à la consommation humaine).
- **0** : L'eau n'est pas potable.

### 🌍 Cas d'Usage & Utilité Métier
Prédire la potabilité par le Machine Learning permet d'automatiser la surveillance des réserves d'eau douce directement à partir des relevés de capteurs chimiques. Cette approche garantit la sécurité sanitaire de manière proactive et en temps réel, évitant l'attente et le coût de longs tests en laboratoire traditionnels.

### ⚙️ Architecture & Focus MLOps
La force de ce jeu de données réside dans sa structure : les colonnes étant presque exclusivement numériques, la phase de pré-traitement des données est grandement simplifiée.
Ce choix stratégique permet de concentrer 100% des efforts sur l'ingénierie MLOps, notamment :
- **L'environnement & Gestion des dépendances** : Isolation via uv.
- **Le Suivi d'Expérimentations** : Tracking des paramètres, métriques et modèles avec MLflow.
- **L'Optimisation** : Recherche des hyperparamètres via Optuna.
- **Le Déploiement** : Mise à disposition du modèle via une API FastAPI.
- **L'Interface Utilisateur** : Consommation de l'API via une application Streamlit.

🔗 **Lien du dataset** : [Water Quality and Potability](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability)
    """)

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Prédiction", "📈 Évaluation", "📊 Monitoring", "🔗 Liens"])

# --- TAB 1 : PRÉDICTION ---
with tab1:
    st.header("Analyse de la qualité de l'eau")
    api_url = os.environ.get("API_URL", "http://api:8000")
    
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            ph = st.number_input("pH", 0.0, 14.0, 7.0)
            hardness = st.number_input("Hardness", 0.0, 500.0, 200.0)
            solids = st.number_input("Solids", 0.0, 50000.0, 20000.0)
        with c2:
            chloramines = st.number_input("Chloramines", 0.0, 20.0, 7.0)
            sulfate = st.number_input("Sulfate", 0.0, 500.0, 300.0)
            conductivity = st.number_input("Conductivity", 0.0, 1000.0, 400.0)
        with c3:
            organic_carbon = st.number_input("Organic Carbon", 0.0, 50.0, 10.0)
            trihalomethanes = st.number_input("Trihalomethanes", 0.0, 200.0, 80.0)
            turbidity = st.number_input("Turbidity", 0.0, 10.0, 3.0)
        
        submitted = st.form_submit_button("Lancer l'analyse", use_container_width=True)

    if submitted:
        try:
            payload = {"ph": ph, "Hardness": hardness, "Solids": solids, "Chloramines": chloramines, 
                       "Sulfate": sulfate, "Conductivity": conductivity, "Organic_carbon": organic_carbon, 
                       "Trihalomethanes": trihalomethanes, "Turbidity": turbidity}
            
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            
            if response.status_code == 200:
                res = response.json()
                st.success(f"Statut : {'Potable' if res['prediction'] == 1 else 'Non Potable'}")
                st.metric("Confiance", f"{res['probability']:.1%}")
                st.progress(res['probability'])
                
                # --- Visualisation SHAP ---
                st.subheader("🔍 Analyse d'impact (SHAP)")
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#28a745' if x > 0 else '#dc3545' for x in res['shap_values']]
                ax.barh(list(payload.keys()), res['shap_values'], color=colors)
                st.pyplot(fig)
                plt.close()
            else:
                st.error(f"Erreur API ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Erreur : {e}")

# --- TAB 2 : ÉVALUATION ---
with tab2:
    st.header("Évaluation du Modèle")
    plots_dir = "/app/data/plots"
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    roc_path = os.path.join(plots_dir, "roc_curve.png")
    shap_path = os.path.join(plots_dir, "shap_summary.png")
    
    c1, c2 = st.columns(2)
    if os.path.exists(cm_path): c1.image(cm_path, caption="Matrice de Confusion", use_container_width=True)
    if os.path.exists(roc_path): c2.image(roc_path, caption="Courbe ROC", use_container_width=True)
    
    if os.path.exists(shap_path):
        st.subheader("🔍 Analyse globale des variables (SHAP)")
        st.image(shap_path, caption="Impact global des caractéristiques", use_container_width=True)

# --- TAB 3 & 4 ---
with tab3: st.components.v1.iframe("http://88.96.55.189:5001", height=600, scrolling=True)
with tab4:
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.subheader("🔗 Liens Utiles")
        st.link_button("🧪 MLflow", "http://88.96.55.189:5001", use_container_width=True)
        st.link_button("✈️ Airflow", "http://88.96.55.189:8080", use_container_width=True)
        st.link_button("🐙 GitHub", "https://github.com/Kouyate01/orchestration_ml.git", use_container_width=True)
        st.link_button("⚡ Documentation API", "http://88.96.55.189:8000/docs", use_container_width=True)
