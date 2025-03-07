import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Fonction pour charger la liste des équipes depuis le CSV
@st.cache_data
def load_teams():
    try:
        return pd.read_csv("data/teams_competitions.csv")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier teams_competitions.csv: {e}")
        return pd.DataFrame()

# Fonction pour charger le modèle et le scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open("models/logistic_regression.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle ou du scaler : {e}")
        return None, None

# Fonction de prédiction basée sur les entrées utilisateur
def predict_result(inputs, model, scaler):
    # inputs doit être de forme (1, 44)
    X_scaled = scaler.transform(inputs)
    prediction = model.predict(X_scaled)
    return prediction[0]

st.title("Prédiction de Match")

st.markdown("### Sélectionnez les équipes")
teams_df = load_teams()
if teams_df.empty:
    st.error("Aucune donnée d'équipes disponible.")
    st.stop()

# Extraction des noms d'équipes uniques
team_names = sorted(teams_df["team_name"].unique())

# Sélection des équipes dans deux colonnes
col1, col2 = st.columns(2)
with col1:
    equipe1 = st.selectbox("Équipe 1", team_names, key="equipe1")
with col2:
    equipe2 = st.selectbox("Équipe 2", team_names, key="equipe2")

if equipe1 == equipe2:
    st.error("Veuillez sélectionner deux équipes différentes.")
else:
    st.markdown("### Compléter les variables manquantes")
    with st.form(key="form_missing_vars"):
        # Variables pour l'équipe 1
        st.subheader(f"Variables pour {equipe1}")
        classement1 = st.number_input(f"Classement pour {equipe1} :", min_value=0, value=0, step=1)
        points1 = st.number_input(f"Points pour {equipe1} :", min_value=0, value=0, step=1)
        victoires1 = st.number_input(f"Nombre de victoires pour {equipe1} :", min_value=0, value=0, step=1)
        
        # Variables pour l'équipe 2
        st.subheader(f"Variables pour {equipe2}")
        classement2 = st.number_input(f"Classement pour {equipe2} :", min_value=0, value=0, step=1)
        points2 = st.number_input(f"Points pour {equipe2} :", min_value=0, value=0, step=1)
        victoires2 = st.number_input(f"Nombre de victoires pour {equipe2} :", min_value=0, value=0, step=1)
        
        submit_form = st.form_submit_button("Valider et lancer la prédiction")
    
    if submit_form:
        # Afficher un résumé des données saisies
        data_summary = {
            "Équipe": [equipe1, equipe2],
            "Classement": [classement1, classement2],
            "Points": [points1, points2],
            "Victoires": [victoires1, victoires2]
        }
        summary_df = pd.DataFrame(data_summary)
        st.markdown("#### Données saisies")
        st.dataframe(summary_df)
        
        # Charger le modèle et le scaler
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            st.error("Erreur lors du chargement du modèle. La prédiction ne peut pas être effectuée.")
        else:
            # Création d'un vecteur de caractéristiques de taille 44
            feature_vector = np.zeros(44)
            # On place les 3 valeurs saisies pour l'équipe 1 dans les positions 0,1,2
            feature_vector[0] = classement1
            feature_vector[1] = points1
            feature_vector[2] = victoires1
            # On place les 3 valeurs saisies pour l'équipe 2 dans les positions 22,23,24
            feature_vector[22] = classement2
            feature_vector[23] = points2
            feature_vector[24] = victoires2
            
            # Redimensionner pour correspondre à la forme (1, 44)
            feature_vector = feature_vector.reshape(1, -1)
            
            # Effectuer la prédiction
            prediction = predict_result(feature_vector, model, scaler)
            
            # Interpréter la prédiction
            if prediction == 1:
                result_text = f"{equipe1} devrait gagner."
            elif prediction == 0:
                result_text = "Match nul."
            elif prediction == 2:
                result_text = f"{equipe1} devrait perdre."
            else:
                result_text = "Résultat inconnu."
            
            st.success(f"Prédiction : {result_text}")
