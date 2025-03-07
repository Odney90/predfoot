#!/usr/bin/env python
import requests
import pandas as pd
import os
import time

# Votre clé API Football
API_KEY = "79378a342b639a3063867f69ad5e78d5"
headers = {"x-apisports-key": API_KEY}

# Dictionnaire des ligues sélectionnées pour la saison 2024-2025
leagues = {
    "Premier League": 39,
    "Championship": 40,
    "LaLiga": 140,
    "Segunda Division": 141,
    "Bundesliga": 78,
    "2. Bundesliga": 79,
    "Serie A": 135,
    "Serie B": 136,
    "Ligue 1": 61,
    "Ligue 2": 62,
    "Primeira Liga": 94,
    "Liga Portugal 2": 95,
    "Eredivisie": 88,
    "Eerste Divisie": 89,
    "Belgian Pro League": 195,
    "Turkish Süper Lig": 53,
    "UEFA Champions League": 2,
    "UEFA Europa League": 3
}

def fetch_standings(league_id, season):
    url = "https://v3.football.api-sports.io/standings"
    params = {"league": league_id, "season": season}
    print(f"Fetching standings for league_id={league_id}, season={season}...")
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def extract_standings(data):
    standings_dict = {}
    for item in data.get("response", []):
        league_data = item.get("league", {})
        standings_groups = league_data.get("standings", [])
        for group in standings_groups:
            for team in group:
                team_name = team.get("team", {}).get("name")
                standings_dict[team_name] = {
                    "Classement": team.get("rank"),
                    "Points": team.get("points"),
                    "Victoires": team.get("all", {}).get("win"),
                    "Défaites": team.get("all", {}).get("lose"),
                    "Matchs nuls": team.get("all", {}).get("draw"),
                    "Buts marqués": team.get("all", {}).get("goals", {}).get("for"),
                    "Buts encaissés": team.get("all", {}).get("goals", {}).get("against"),
                    "Différence de buts": team.get("all", {}).get("goalDiff")
                }
    return standings_dict

def fetch_fixtures(league_id, season, status="FT"):
    url = "https://v3.football.api-sports.io/fixtures"
    params = {"league": league_id, "season": season, "status": status}
    print(f"Fetching fixtures for league_id={league_id}, season={season}, status={status}...")
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def fetch_match_statistics(fixture_id):
    url = "https://v3.football.api-sports.io/fixtures/statistics"
    params = {"fixture": fixture_id}
    print(f"Fetching match statistics for fixture_id={fixture_id}...")
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def extract_match_stats(data):
    mapping = {
        "Tirs au but": "Shots on Goal",
        # "Tirs non cadrés": "Shots off Goal",  # Exclu
        "Coups de feu à l'intérieur de la boîte": "Shots insidebox",
        "Tirs en dehors de la surface": "Shots outsidebox",
        "Nombre total de coups": "Total Shots",
        "Tirs bloqués": "Blocked Shots",
        "Fautes": "Fouls",
        "Coups de pied de coin": "Corner Kicks",
        "Hors-jeu": "Offsides",
        "Possession du ballon": "Ball Possession",
        "Cartons jaunes": "Yellow Cards",
        "Cartons rouges": "Red Cards",
        "Arrêts du gardien de but": "Goalkeeper Saves",
        "Nombre total de passes": "Total passes",
        "Passes précises": "Pass Accuracy"
    }
    stats = {}
    for team_data in data.get("response", []):
        team_name = team_data.get("team", {}).get("name")
        team_stats = {}
        for stat in team_data.get("statistics", []):
            api_stat = stat.get("type")
            value = stat.get("value")
            for var, api_field in mapping.items():
                if api_stat == api_field:
                    team_stats[var] = value
        stats[team_name] = team_stats
    return stats

def combine_team_data(standings_dict, match_stats, league_name):
    combined = {}
    for team, stats in match_stats.items():
        combined[team] = {"Ligue": league_name}
        if team in standings_dict:
            combined[team].update(standings_dict[team])
        else:
            combined[team].update({
                "Classement": None,
                "Points": None,
                "Victoires": None,
                "Défaites": None,
                "Matchs nuls": None,
                "Buts marqués": None,
                "Buts encaissés": None,
                "Différence de buts": None
            })
        combined[team].update(match_stats.get(team, {}))
    return combined

def build_dataset_for_league(league_name, league_id, season):
    print(f"\n=== Traitement de la ligue : {league_name} ===")
    standings_json = fetch_standings(league_id, season)
    standings_dict = extract_standings(standings_json)
    fixtures_json = fetch_fixtures(league_id, season, status="FT")
    fixtures = fixtures_json.get("response", [])
    print(f"Nombre de matchs récupérés pour {league_name} : {len(fixtures)}")
    data_rows = []
    for fixture in fixtures:
        fixture_id = fixture.get("fixture", {}).get("id")
        match_date = fixture.get("fixture", {}).get("date")
        try:
            stats_json = fetch_match_statistics(fixture_id)
            match_stats = extract_match_stats(stats_json)
        except Exception as e:
            print(f"Erreur pour fixture {fixture_id} : {e}")
            continue
        if not match_stats:
            print(f"Aucune statistique pour fixture {fixture_id}.")
            continue
        combined = combine_team_data(standings_dict, match_stats, league_name)
        for team, data in combined.items():
            row = {
                "fixture_id": fixture_id,
                "date": match_date,
                "Équipe": team,
                "Ligue": league_name
            }
            row.update(data)
            data_rows.append(row)
        time.sleep(1)
    return pd.DataFrame(data_rows)

def build_global_dataset(leagues_dict, season):
    global_rows = []
    for league_name, league_id in leagues_dict.items():
        df_league = build_dataset_for_league(league_name, league_id, season)
        if not df_league.empty:
            global_rows.append(df_league)
        else:
            print(f"Aucune donnée pour la ligue {league_name}.")
    if global_rows:
        return pd.concat(global_rows, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    season = "2024"  # Pour la saison 2024-2025
    print(f"Construction du dataset global pour la saison {season}...")
    dataset = build_global_dataset(leagues, season)
    if dataset.empty:
        print("Aucune donnée récupérée pour aucune des ligues.")
        return
    if not os.path.exists("data"):
        os.makedirs("data")
    output_file = "data/combined_dataset_all_leagues_reduit.csv"
    dataset.to_csv(output_file, index=False)
    print(f"Dataset global sauvegardé dans {output_file}")
    print(dataset.head())

if __name__ == "__main__":
    main()
