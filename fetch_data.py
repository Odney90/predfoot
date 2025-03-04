import requests
import pandas as pd
import os

# Votre clé API (remplacez-la par votre clé réelle)
API_KEY = "79378a342b639a3063867f69ad5e78d5"

def fetch_data_from_api():
    # URL de l'endpoint pour récupérer les fixtures
    url = "https://v3.football.api-sports.io/fixtures"
    
    # En-têtes nécessaires pour l'API Football V3
    headers = {
        "x-apisports-key": API_KEY
    }
    
    # Paramètres de requête (à adapter selon vos besoins, ici un exemple avec la Premier League et la saison 2022)
    params = {
        "league": "39",   # Exemple: Premier League
        "season": "2022"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Lève une exception en cas de problème HTTP
        data = response.json()
        
        # Récupérer la liste des fixtures
        fixtures = data.get("response", [])
        if not fixtures:
            print("Aucune donnée retournée par l'API.")
            return None
        
        rows = []
        for fixture in fixtures:
            # Extraction de quelques données de base ; vous pouvez ajouter les autres variables selon vos besoins
            fixture_id = fixture.get("fixture", {}).get("id")
            home_team = fixture.get("teams", {}).get("home", {}).get("name")
            away_team = fixture.get("teams", {}).get("away", {}).get("name")
            score_home = fixture.get("goals", {}).get("home")
            score_away = fixture.get("goals", {}).get("away")
            
            # Construire le dictionnaire d'une ligne (complétez avec les autres variables nécessaires)
            row = {
                "fixture_id": fixture_id,
                "home_team": home_team,
                "away_team": away_team,
                "score_home": score_home,
                "score_away": score_away
            }
            rows.append(row)
        
        # Conversion de la liste en DataFrame pandas
        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        print("Erreur lors de la récupération des données via l'API :", e)
        return None

def saisie_manuelle():
    # En cas d'échec de l'API, vous pouvez saisir manuellement quelques données exemples.
    data = {
        "fixture_id": [1],
        "home_team": ["Equipe A"],
        "away_team": ["Equipe B"],
        "score_home": [2],
        "score_away": [1]
    }
    df = pd.DataFrame(data)
    return df

def main():
    # Tente de récupérer les données via l'API
    df = fetch_data_from_api()
    
    # Si la récupération échoue, on bascule sur la saisie manuelle
    if df is None or df.empty:
        print("Utilisation de la saisie manuelle...")
        df = saisie_manuelle()
    
    # Crée le dossier "data" s'il n'existe pas
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Sauvegarde des données dans data/matchs.csv
    df.to_csv("data/matchs.csv", index=False)
    print("Données sauvegardées dans data/matchs.csv")

if __name__ == "__main__":
    main()
