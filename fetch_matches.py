import requests
import pandas as pd
import os

# Votre clé API
API_KEY = "79378a342b639a3063867f69ad5e78d5"

# Paramètre pour activer ou non le remplissage interactif
INTERACTIVE_FILL = True

def complete_missing_variables(match_data):
    """
    Pour chaque variable manquante (None) dans match_data,
    si INTERACTIVE_FILL est activé, l'utilisateur est invité à saisir une valeur.
    Laisser vide pour conserver None.
    """
    if INTERACTIVE_FILL:
        for key, value in match_data.items():
            if value is None:
                user_input = input(f"La valeur pour '{key}' est manquante (match {match_data.get('fixture_id', 'N/A')}). Entrez une valeur (laisser vide pour None) : ")
                if user_input.strip() != "":
                    try:
                        # Tente de convertir en float si c'est un nombre
                        match_data[key] = float(user_input) if ('.' in user_input or user_input.isdigit()) else user_input
                    except Exception as e:
                        match_data[key] = user_input
    return match_data

def fetch_match_data(season="2024"):
    """
    Récupère les données de match via l'endpoint fixtures pour la saison indiquée,
    et construit un DataFrame avec 52 variables (26 par équipe).
    Les variables non disponibles sont initialisées à None et peuvent être complétées manuellement.
    """
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_KEY}
    # Exemple de paramètres pour la Premier League pour la saison 2024-2025
    params = {
        "league": 39,
        "season": season
    }
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    matches = []
    
    for fixture in data.get("response", []):
        fixture_info = fixture.get("fixture", {})
        teams_info = fixture.get("teams", {})
        goals_info = fixture.get("goals", {})
        
        # Construction du dictionnaire avec les 52 variables attendues.
        match_data = {
            # Informations de base sur le match
            "fixture_id": fixture_info.get("id"),
            "date": fixture_info.get("date"),
            "home_team": teams_info.get("home", {}).get("name"),
            "away_team": teams_info.get("away", {}).get("name"),
            
            # Variables Générales (Statistiques de Base)
            "Classement_1": None,
            "Classement_2": None,
            "Points_1": None,
            "Points_2": None,
            
            # Historique des Résultats
            "Victoires_1": None,
            "Victoires_2": None,
            "Défaites_1": None,
            "Défaites_2": None,
            "Nuls_1": None,
            "Nuls_2": None,
            
            # Attaque & Défense
            "ButsMarques_1": goals_info.get("home"),
            "ButsMarques_2": goals_info.get("away"),
            "ButsEncaisses_1": None,
            "ButsEncaisses_2": None,
            "DiffButs_1": None,
            "DiffButs_2": None,
            
            # Contexte du match
            "Domicile_1": 1,    # L'équipe à domicile
            "Domicile_2": 0,    # L'équipe visiteuse
            "JoursDepuisDernierMatch_1": None,
            "JoursDepuisDernierMatch_2": None,
            "ImportanceMatch_1": None,
            "ImportanceMatch_2": None,
            "H2H_1": None,
            "H2H_2": None,
            
            # Variables Avancées (Données Offensives et Défensives)
            "ButsDomicile_1": None,
            "ButsDomicile_2": None,
            "ButsExterieur_1": None,
            "ButsExterieur_2": None,
            
            # Performance Tactique et Technique
            "Possession_1": None,
            "Possession_2": None,
            "Tirs_1": None,
            "Tirs_2": None,
            "TirsCadrés_1": None,
            "TirsCadrés_2": None,
            "TirsSubis_1": None,
            "TirsSubis_2": None,
            
            # Expected Goals
            "xG_1": None,
            "xG_2": None,
            "xGA_1": None,
            "xGA_2": None,
            
            # Performance Physique et Défensive
            "DuelsGagnes_1": None,
            "DuelsGagnes_2": None,
            "Interceptions_1": None,
            "Interceptions_2": None,
            
            # Discipline
            "CartonsJaunes_1": None,
            "CartonsJaunes_2": None,
            "Fautes_1": None,
            "Fautes_2": None,
            
            # État Physique et Absences
            "Blessures_1": None,
            "Blessures_2": None,
            
            # Performance des Meilleurs Buteurs
            "MeilleursButeurs_1": None,
            "MeilleursButeurs_2": None,
        }
        
        # Permet à l'utilisateur de compléter les variables manquantes pour ce match
        match_data = complete_missing_variables(match_data)
        matches.append(match_data)
    
    return pd.DataFrame(matches)

def main():
    df = fetch_match_data(season="2024")
    if not os.path.exists("data"):
        os.makedirs("data")
    output_file = "data/matchs_52variables.csv"
    df.to_csv(output_file, index=False)
    print(f"Les données des matchs (52 variables) ont été sauvegardées dans {output_file}")

if __name__ == "__main__":
    main()

