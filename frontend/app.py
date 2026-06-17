"""Frontend Streamlit pour tester l'API de classification (Water Potability)."""

from __future__ import annotations
import os
import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="Potabilité de l'eau", layout="wide")
st.title("Prédiction de la potabilité de l'eau")

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, history_tab = st.tabs(["Prediction", "Historique"])

with predict_tab:
    st.subheader("Entrez les paramètres de l'eau")

    with st.form("predict_form"):
        # Champs basés sur votre dataset water_potability
        col1, col2 = st.columns(2)
        with col1:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
            hardness = st.number_input("Hardness", min_value=0.0, value=200.0)
            solids = st.number_input("Solids", min_value=0.0, value=20000.0)
            chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
            sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
        with col2:
            conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
            organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
            trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=80.0)
            turbidity = st.number_input("Turbidity", min_value=0.0, value=3.0)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        payload = {
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity,
        }
        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel à l'API impossible : {exc}")
        else:
            # Affichage des résultats
            pred = "Potable" if result["prediction"] == 1 else "Non Potable"
            st.success(f"Résultat : **{pred}**")
            st.metric(label="Probabilité de potabilité", value=f"{result['probability']:.2%}")

with history_tab:
    st.subheader("Historique")
    st.info("Aucun journal de prévisions disponible.")
