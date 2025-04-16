import streamlit as st
import pandas as pd
import requests
import json
import os

# Chemin vers ton fichier CSV

csv_path = os.path.join(os.path.dirname(__file__), "recommendation_function", "user_article_interaction_scaled.csv")


# Charger les user_id uniques
@st.cache_data
def load_user_ids(path):
    df = pd.read_csv(path)
    return sorted(df['user_id'].unique())

# Récupérer les recommandations via l'Azure Function
def get_recommendations(user_id):
    url = f"https://reco-function-ndv.azurewebsites.net/api/recommend?user_id={user_id}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()  # retourne le dict entier
        else:
            return f"Erreur {response.status_code} : {response.text}"
    except Exception as e:
        return f"Erreur de requête : {str(e)}"


# UI Streamlit
st.set_page_config(page_title="Recommandations", layout="centered")

st.title("🎯 Recommandations utilisateur (Azure Function)")
user_ids = load_user_ids(csv_path)

with st.form("reco_form"):
    selected_user = st.selectbox("Sélectionnez un utilisateur :", user_ids)
    submit = st.form_submit_button("Obtenir des recommandations")

if submit:
    with st.spinner("🔄 Récupération des recommandations..."):
        result = get_recommendations(selected_user)

    if isinstance(result, str):  # une erreur sous forme de string
        st.error(result)
    else:
        recos = result.get("recommendations", [])
        st.subheader("📦 Articles recommandés :")
        if recos:
            for reco in recos:
                st.markdown(
                    f"📰 Article ID : **{reco['article_id']}**"
                )
        else:
            st.info("Aucune recommandation trouvée.")
