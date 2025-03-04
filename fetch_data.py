import requests
import pandas as pd
import os

# Votre clé API
API_KEY = "79378a342b639a3063867f69ad5e78d5"

def fetch_leagues(season="2024"):
    """
    Récupère les informations sur les compétitions pour la saison donnée.
    Cette fonction utilise l'endpoint /leagues de l'API Football.
    """
    url = "https://v3.football.api-sports.io/leagues"
    headers = {"x-apisports-key": API_KEY}
    params = {
        "season": season
        # Vous pouvez ajouter d'autres filtres si nécessaire (country, etc.)
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    leagues_data = data.get("response", [])
    leagues_list = []
    for item in leagues_data:
        league = item.get("league", {})
        country = item.get("country", {})
        season_info = item.get("season", {})
        leagues_list.append({
            "league_id": league.get("id"),
            "league_name": league.get("name"),
            "country": country.get("name"),
            "logo": league.get("logo"),
            "season_start": season_info.get("start"),
            "season_end": season_info.get("end")
        })
    return pd.DataFrame(leagues_list)

def fetch_matches(season="2024", league_id=39):
    """
    Récupère les données de match pour une ligue donnée (exemple : Premier League avec league_id=39)
    et construit un DataFrame avec 52 variables (26 par équipe).
    
    Les variables disponibles directement de l'API (via /fixtures) sont utilisées,
    les autres sont initialisées à None et pourront être complétées manuellement.
    """
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_KEY}
    params = {
        "league": league_id,
        "season": season
        # Vous pouvez ajouter d'autres filtres (date, status, etc.) si nécessaire.
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    matches = []
    for fixture in data.get("response", []):
        fixture_info = fixture.get("fixture", {})
        teams_info = fixture.get("teams", {})
        goals_info = fixture.get("goals", {})
        
        # Extraction des informations disponibles depuis l'API
        match_data = {
            "fixture_id": fixture_info.get("id"),
            "date": fixture_info.get("date"),
            "home_team": teams_info.get("home", {}).get("name"),
            "away_team": teams_info.get("away", {}).get("name"),
            "ButsMarques_1": goals_info.get("home"),
            "ButsMarques_2": goals_info.get("away"),
        }
        # Les 52 variables attendues (26 par équipe) :
        placeholders = [
            "Classement_1", "Classement_2", "Points_1", "Points_2",
            "Victoires_1", "Victoires_2", "Défaites_1", "Défaites_2", "Nuls_1", "Nuls_2",
            "ButsEncaisses_1", "ButsEncaisses_2", "DiffButs_1", "DiffButs_2",
            "Domicile_1", "Domicile_2", "JoursDepuisDernierMatch_1", "JoursDepuisDernierMatch_2",
            "ImportanceMatch_1", "ImportanceMatch_2", "H2H_1", "H2H_2",
            "ButsDomicile_1", "ButsDomicile_2", "ButsExterieur_1", "ButsExterieur_2",
            "Possession_1", "Possession_2", "Tirs_1", "Tirs_2", "TirsCadrés_1", "TirsCadrés_2",
            "TirsSubis_1", "TirsSubis_2", "xG_1", "xG_2", "xGA_1", "xGA_2",
            "DuelsGagnes_1", "DuelsGagnes_2", "Interceptions_1", "Interceptions_2",
            "CartonsJaunes_1", "CartonsJaunes_2", "Fautes_1", "Fautes_2",
            "Blessures_1", "Blessures_2", "MeilleursButeurs_1", "MeilleursButeurs_2"
        ]
        for var in placeholders:
            if var not in match_data:
                match_data[var] = None
        matches.append(match_data)
    return pd.DataFrame(matches)

def main():
    season = "2024"
    
    # Récupérer et sauvegarder les données des compétitions
    leagues_df = fetch_leagues(season)
    if not os.path.exists("data"):
        os.makedirs("data")
    leagues_csv = "data/leagues_data.csv"
    leagues_df.to_csv(leagues_csv, index=False)
    print(f"Les données des compétitions ont été sauvegardées dans {leagues_csv}")
    
    # Récupérer et sauvegarder les données de match pour une ligue (exemple : Premier League, league_id=39)
    matches_df = fetch_matches(season, league_id=39)
    matches_csv = "data/matches_52variables.csv"
    matches_df.to_csv(matches_csv, index=False)
    print(f"Les données des matchs (52 variables) ont été sauvegardées dans {matches_csv}")

if __name__ == "__main__":
    main()
