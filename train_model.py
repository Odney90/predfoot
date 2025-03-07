#!/usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

def create_resultat_column_from_group(df):
    """
    Pour chaque match (identifié par fixture_id), compare les scores de chaque équipe et crée
    la colonne 'resultat' pour chaque ligne.
    On suppose que chaque match a deux lignes.
      - Pour l'équipe à domicile (assumée avoir score dans 'ButsMarques_1'): 1 = victoire, 0 = nul, 2 = défaite.
      - Pour l'équipe à l'extérieur (score dans 'ButsMarques_2'): 1 = victoire, 0 = nul, 2 = défaite.
    """
    def compute_result(group):
        if len(group) != 2:
            group['resultat'] = np.nan
            return group
        score1 = group.iloc[0]['Buts marqués']
        score2 = group.iloc[1]['Buts marqués']
        if pd.isna(score1) or pd.isna(score2):
            group['resultat'] = np.nan
        elif score1 > score2:
            group.iloc[0, group.columns.get_loc('resultat')] = 1
            group.iloc[1, group.columns.get_loc('resultat')] = 2
        elif score1 < score2:
            group.iloc[0, group.columns.get_loc('resultat')] = 2
            group.iloc[1, group.columns.get_loc('resultat')] = 1
        else:
            group['resultat'] = 0
        return group

    df['resultat'] = np.nan
    df = df.groupby('fixture_id', group_keys=False).apply(compute_result)
    return df

def load_and_preprocess_data(csv_path):
    print("Chargement du dataset depuis :", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier CSV: {e}")
    
    print("Premières lignes du dataset :")
    print(df.head())
    print("Colonnes disponibles :", df.columns.tolist())
    
    if "resultat" not in df.columns or df['resultat'].isna().all():
        print("La colonne 'resultat' est absente ou vide. Création de la colonne à partir des scores...")
        df = create_resultat_column_from_group(df)
        print("Premières lignes après ajout de 'resultat' :")
        print(df[['fixture_id', 'Équipe', 'Buts marqués', 'resultat']].head())
    
    # On retire les colonnes contextuelles : fixture_id, date, Équipe, Ligue, home_team, away_team
    cols_to_drop = ["fixture_id", "date", "Équipe", "Ligue", "home_team", "away_team"]
    X = df.drop(columns=cols_to_drop + ["resultat"], errors='ignore')
    y = df["resultat"]
    
    # Garder uniquement les colonnes numériques
    X = X.select_dtypes(include=[np.number])
    
    print("Dimensions de X après sélection numérique :", X.shape)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_all_models(X, y):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    }
    
    trained_models = {}
    cv_scores = {}
    
    for name, model in models.items():
        print(f"\nEntraînement du modèle {name} avec validation croisée à 5 plis...")
        try:
            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            cv_scores[name] = np.mean(scores)
            print(f"{name} - CV Accuracy = {cv_scores[name]:.4f}")
        except Exception as e:
            print(f"Erreur lors de la validation croisée pour {name} : {e}")
            cv_scores[name] = None
        model.fit(X, y)
        trained_models[name] = model
        
    return trained_models, cv_scores

def save_models(models, scaler, models_dir="models"):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    for name, model in models.items():
        filename = f"{models_dir}/{name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"Modèle '{name}' sauvegardé dans {filename}")
    
    scaler_filename = f"{models_dir}/scaler.pkl"
    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler sauvegardé dans {scaler_filename}")

def main():
    print("=== Début du script d'entraînement ===")
    data_path = "data/combined_dataset_all_leagues_reduit.csv"
    print("Chargement et prétraitement des données...")
    try:
        X, y, scaler = load_and_preprocess_data(data_path)
        print("Données chargées avec succès.")
    except Exception as e:
        print("Erreur lors du chargement/pretraitement :", e)
        return
    
    print("\nDivision des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dimensions de X_train :", X_train.shape)
    print("Dimensions de X_test  :", X_test.shape)
    
    print("\nEntraînement des modèles sur le jeu d'entraînement...")
    trained_models, cv_scores = train_all_models(X_train, y_train)
    
    print("\nÉvaluation sur le jeu de test:")
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            print(f"{name}: Test Accuracy = {test_acc:.4f}")
        except Exception as e:
            print(f"Erreur lors de l'évaluation pour {name} : {e}")
    
    print("\nSauvegarde des modèles et du scaler...")
    save_models(trained_models, scaler)
    
    print("\n=== Entraînement terminé ===")

if __name__ == "__main__":
    main()
