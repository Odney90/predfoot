import streamlit as st
import pandas as pd

# Charger la liste des équipes depuis un fichier CSV (généré par votre script API)
@st.cache_data
def load_teams():
    # Assurez-vous que 'data/teams_competitions.csv' se trouve dans votre dépôt GitHub
    return pd.read_csv("data/teams_competitions.csv")

st.title("Prédiction de Match")

st.markdown("### Sélectionnez les équipes")
teams_df = load_teams()
team_names = sorted(teams_df["team_name"].unique())

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
        # Exemple de variables à saisir manuellement (vous ajouterez les 52 variables selon vos besoins)
        classement1 = st.number_input(f"Classement pour {equipe1} :", min_value=0, value=0, step=1)
        points1 = st.number_input(f"Points pour {equipe1} :", min_value=0, value=0, step=1)
        victoires1 = st.number_input(f"Nombre de victoires pour {equipe1} :", min_value=0, value=0, step=1)
        
        classement2 = st.number_input(f"Classement pour {equipe2} :", min_value=0, value=0, step=1)
        points2 = st.number_input(f"Points pour {equipe2} :", min_value=0, value=0, step=1)
        victoires2 = st.number_input(f"Nombre de victoires pour {equipe2} :", min_value=0, value=0, step=1)
        
        # Ajoutez d'autres champs pour couvrir l'ensemble des variables nécessaires...
        
        submit_form = st.form_submit_button("Valider et lancer la prédiction")
    
    if submit_form:
        # Création d'un résumé des données saisies
        data_summary = {
            "Équipe": [equipe1, equipe2],
            "Classement": [classement1, classement2],
            "Points": [points1, points2],
            "Victoires": [victoires1, victoires2]
            # Ajoutez d'autres variables au résumé si besoin
        }
        summary_df = pd.DataFrame(data_summary)
        st.markdown("#### Données saisies")
        st.dataframe(summary_df)
        
        # Ici, vous pouvez appeler votre fonction de prédiction en lui passant les données combinées.
        st.success("Prédiction lancée (fonctionnalité à implémenter).")
