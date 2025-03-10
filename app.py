import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Forcer le répertoire de travail vers le dossier du dépôt
st.write("Répertoire de travail actuel :", os.getcwd())

# Charger la liste des équipes depuis le fichier CSV
@st.cache_data
def load_teams():
    try:
        return pd.read_csv(os.path.join("data", "teams_competitions.csv"))
    except Exception as e:
        st.error(f"Erreur lors du chargement de teams_competitions.csv: {e}")
        return pd.DataFrame()

# Charger le modèle et le scaler depuis le dossier "models"
@st.cache_resource
def load_model_and_scaler():
    try:
        model_path = os.path.join("models", "logistic_regression.pkl")
        scaler_path = os.path.join("models", "scaler.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle ou du scaler: {e}")
        return None, None

def predict_result(features, model, scaler):
    X_scaled = scaler.transform(features)
    prediction = model.predict(X_scaled)
    return prediction[0]

st.title("Prédiction de Match")

st.markdown("### Sélectionnez les équipes")
teams_df = load_teams()
if teams_df.empty:
    st.error("Aucune donnée d'équipes disponible.")
    st.stop()

# Extraire et trier les noms d'équipes
team_names = sorted(teams_df["team_name"].unique())
col1, col2 = st.columns(2)
with col1:
    equipe1 = st.selectbox("Équipe 1", team_names, key="equipe1")
with col2:
    equipe2 = st.selectbox("Équipe 2", team_names, key="equipe2")

if equipe1 == equipe2:
    st.error("Veuillez sélectionner deux équipes différentes.")
else:
    st.markdown("### Complétez les variables manquantes (3 entrées par équipe)")
    with st.form(key="form_missing_vars"):
        st.subheader(f"Variables pour {equipe1}")
        classement1 = st.number_input(f"Classement pour {equipe1} :", min_value=0, value=0, step=1)
        points1 = st.number_input(f"Points pour {equipe1} :", min_value=0, value=0, step=1)
        victoires1 = st.number_input(f"Victoires pour {equipe1} :", min_value=0, value=0, step=1)
        
        st.subheader(f"Variables pour {equipe2}")
        classement2 = st.number_input(f"Classement pour {equipe2} :", min_value=0, value=0, step=1)
        points2 = st.number_input(f"Points pour {equipe2} :", min_value=0, value=0, step=1)
        victoires2 = st.number_input(f"Victoires pour {equipe2} :", min_value=0, value=0, step=1)
        
        submit_form = st.form_submit_button("Valider et lancer la prédiction")
    
    if submit_form:
        data_summary = {
            "Équipe": [equipe1, equipe2],
            "Classement": [classement1, classement2],
            "Points": [points1, points2],
            "Victoires": [victoires1, victoires2]
        }
        st.markdown("#### Données saisies")
        st.write(pd.DataFrame(data_summary))
        
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            st.error("Erreur lors du chargement du modèle. La prédiction ne peut pas être effectuée.")
        else:
            # Récupérer le nombre de features attendu par le scaler
            n_features = scaler.mean_.shape[0]
            st.write(f"Le modèle attend {n_features} features.")
            
            # Création d'un vecteur de caractéristiques de taille n_features
            feature_vector = np.zeros(n_features)
            half = n_features // 2
            # Placer les 3 valeurs saisies pour l'équipe 1 dans les 3 premières positions
            feature_vector[0] = classement1
            feature_vector[1] = points1
            feature_vector[2] = victoires1
            # Placer les 3 valeurs saisies pour l'équipe 2 dans les 3 positions à la moitié du vecteur
            feature_vector[half] = classement2
            feature_vector[half+1] = points2
            feature_vector[half+2] = victoires2
            
            # Redimensionner pour obtenir la forme (1, n_features)
            feature_vector = feature_vector.reshape(1, -1)
            
            prediction = predict_result(feature_vector, model, scaler)
            
            if prediction == 1:
                result_text = f"{equipe1} devrait gagner."
            elif prediction == 0:
                result_text = "Match nul."
            elif prediction == 2:
                result_text = f"{equipe1} devrait perdre."
            else:
                result_text = "Résultat inconnu."
            
            st.success(f"Prédiction : {result_text}")
