import requests
import pandas as pd
import os

# Votre clé API
API_KEY = "79378a342b639a3063867f69ad5e78d5"

# Dictionnaire des compétitions avec leurs IDs dans l’API Football
# Les IDs ci-dessous sont ceux généralement utilisés par l'API Football,
# mais vérifiez la documentation si nécessaire.
competitions = {
    # 7 meilleures ligues et leurs seconds niveaux (League 2)
    "Premier League": 39,
    "Championship": 40,             # 2ème division de la Premier League
    "LaLiga": 140,
    "Segunda Division": 141,         # 2ème division de LaLiga
    "Bundesliga": 78,
    "2. Bundesliga": 79,             # 2ème division de la Bundesliga
    "Serie A": 135,
    "Serie B": 136,                  # 2ème division de la Serie A
    "Ligue 1": 61,
    "Ligue 2": 62,                   # 2ème division de la Ligue 1
    "Primeira Liga": 94,
    "Liga Portugal 2": 95,           # 2ème division de la Primeira Liga
    "Eredivisie": 88,
    "Eerste Divisie": 89,            # 2ème division de l'Eredivisie

    # Autres compétitions
    "Belgian Pro League": 195,
    "Turkish Süper Lig": 53,
    "UEFA Champions League": 2,
    "UEFA Europa League": 3
}

def fetch_teams_for_competition(comp_name, comp_id, season="2024"):
    """
    Récupère les équipes pour une compétition donnée et une saison.
    Le paramètre 'season' est ici fixé à "2024" pour représenter la saison 2024-2025.
    """
    url = "https://v3.football.api-sports.io/teams"
    headers = {
        "x-apisports-key": API_KEY
    }
    params = {
        "league": comp_id,
        "season": season
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        teams = data.get("response", [])
        team_list = []
        for item in teams:
            team = item.get("team", {})
            team_id = team.get("id")
            team_name = team.get("name")
            country = team.get("country")
            founded = team.get("founded")
            logo = team.get("logo")
            
            team_list.append({
                "competition": comp_name,
                "team_id": team_id,
                "team_name": team_name,
                "country": country,
                "founded": founded,
                "logo": logo
            })
        return team_list
    except Exception as e:
        print(f"Erreur lors de la récupération des équipes pour {comp_name} : {e}")
        return []

def main():
    all_teams = []
    for comp_name, comp_id in competitions.items():
        print(f"Récupération des équipes pour {comp_name}...")
        teams = fetch_teams_for_competition(comp_name, comp_id, season="2024")
        all_teams.extend(teams)
    
    if all_teams:
        df = pd.DataFrame(all_teams)
        # Créer le dossier "data" s'il n'existe pas
        if not os.path.exists("data"):
            os.makedirs("data")
        df.to_csv("data/teams_competitions.csv", index=False)
        print("Les données des équipes ont été sauvegardées dans data/teams_competitions.csv")
    else:
        print("Aucune donnée récupérée.")

if __name__ == "__main__":
    main()
