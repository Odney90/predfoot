import streamlit as st
import pandas as pd
import subprocess

# ----------------------------
# Bouton de Mise à Jour des Données
# ----------------------------
st.sidebar.header("Mise à jour des données")
if st.sidebar.button("Mettre à jour les données"):
    with st.spinner("Mise à jour en cours, veuillez patienter..."):
        # Exécute le script de mise à jour. Assurez-vous que build_dataset_all_leagues.py se trouve à la racine.
        result = subprocess.run(["python", "build_dataset_all_leagues.py"], capture_output=True, text=True)
        st.sidebar.success("Mise à jour terminée.")
        st.sidebar.text(result.stdout)
        # Recharger l'application pour prendre en compte les nouvelles données
        st.experimental_rerun()

# ----------------------------
# Chargement des données d'équipes
# ----------------------------
@st.cache_data
def load_teams():
    try:
        return pd.read_csv("data/teams_competitions.csv")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier teams_competitions.csv: {e}")
        return pd.DataFrame()

st.title("Prédiction de Match")
st.markdown("### Sélectionnez les équipes")

teams_df = load_teams()

# Vérifier que le DataFrame n'est pas vide
if teams_df.empty:
    st.error("Aucune donnée d'équipes disponible.")
    st.stop()

# Extraire et trier les noms d'équipes uniques
team_names = sorted(teams_df["team_name"].unique())

# ----------------------------
# Sélection des équipes
# ----------------------------
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
        
        # Vous pouvez ajouter d'autres variables si nécessaire
        submit_form = st.form_submit_button("Valider et lancer la prédiction")
    
    if submit_form:
        # Création d'un résumé des données saisies
        data_summary = {
            "Équipe": [equipe1, equipe2],
            "Classement": [classement1, classement2],
            "Points": [points1, points2],
            "Victoires": [victoires1, victoires2]
        }
        summary_df = pd.DataFrame(data_summary)
        st.markdown("#### Données saisies")
        st.dataframe(summary_df)
        
        # Ici, appelez votre fonction de prédiction (à implémenter) en lui passant les données combinées.
        st.success("Prédiction lancée (fonctionnalité à implémenter).")
