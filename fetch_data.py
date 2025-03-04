import requests
import pandas as pd
import os

# Clé API fournie
API_KEY = "79378a342b639a3063867f69ad5e78d5"

def fetch_data_from_api():
    # URL de l'API Football V3 pour récupérer les fixtures
    url = "https://v3.football.api-sports.io/fixtures"
    
    # Configuration des headers avec la clé API
    headers = {
        "x-apisports-key": API_KEY
    }
    
    # Exemple de paramètres (à adapter en fonction des besoins : saison, ligue, etc.)
    params = {
        "league": "39",    # Par exemple, Premier League (à adapter)
        "season": "2022"     # Exemple d'année de saison
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Lève une exception en cas d'erreur HTTP
        data = response.json()
        
        # Vérification de la présence de données dans la réponse
        fixtures = data.get("response", [])
        if not fixtures:
            print("Aucun fixture trouvé dans la réponse.")
            return None
        
        # Construction d'une DataFrame avec quelques variables d'exemple
        # À compléter pour extraire toutes les 52 variables attendues
        rows = []
        for fixture in fixtures:
            fixture_id = fixture.get("fixture", {}).get("id")
            home_team = fixture.get("teams", {}).get("home", {}).get("name")
            away_team = fixture.get("teams", {}).get("away", {}).get("name")
            score_home = fixture.get("goals", {}).get("home")
            score_away = fixture.get("goals", {}).get("away")
            
            # Exemple de dictionnaire ; vous devez ajouter ici les autres variables
            row = {
                "fixture_id": fixture_id,
                "home_team": home_team,
                "away_team": away_team,
                "score_home": score_home,
                "score_away": score_away,
                # ... ajoutez ici les 52 variables (26 par équipe)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        print("Erreur lors de la récupération des données via l'API :", e)
        return None

def saisie_manuelle():
    # Exemple de saisie manuelle pour quelques variables (à étendre aux 52 variables)
    data = {
        "fixture_id": [1],
        "home_team": ["Equipe A"],
        "away_team": ["Equipe B"],
        "score_home": [2],
        "score_away": [1],
        # ... ajouter les autres variables manuellement
    }
    df = pd.DataFrame(data)
    return df

def main():
    # Tente de récupérer les données via l'API
    df = fetch_data_from_api()
    
    # Si l'API ne renvoie rien ou en cas d'erreur, bascule sur la saisie manuelle
    if df is None or df.empty:
        print("Utilisation de la saisie manuelle...")
        df = saisie_manuelle()
    
    # Création du dossier 'data' s'il n'existe pas
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Sauvegarde des données dans data/matchs.csv
    df.to_csv("data/matchs.csv", index=False)
    print("Données sauvegardées dans data/matchs.csv")

if __name__ == "__main__":
    main()
