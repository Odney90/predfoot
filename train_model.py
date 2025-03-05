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

def create_resultat_column(df):
    """
    Crée la colonne 'resultat' dans le DataFrame en se basant sur
    les buts marqués par l'équipe à domicile et l'équipe à l'extérieur.
    Pour chaque ligne :
      - Si la colonne 'Équipe' correspond à 'home_team', alors :
           1 si ButsMarques_1 > ButsMarques_2 (victoire),
           0 si égaux (match nul),
           2 si ButsMarques_1 < ButsMarques_2 (défaite).
      - Si 'Équipe' correspond à 'away_team', alors :
           1 si ButsMarques_2 > ButsMarques_1 (victoire),
           0 si égaux,
           2 si ButsMarques_2 < ButsMarques_1 (défaite).
    """
    def determine_result(row):
        if row['Équipe'] == row['home_team']:
            if row['ButsMarques_1'] > row['ButsMarques_2']:
                return 1
            elif row['ButsMarques_1'] == row['ButsMarques_2']:
                return 0
            else:
                return 2
        elif row['Équipe'] == row['away_team']:
            if row['ButsMarques_2'] > row['ButsMarques_1']:
                return 1
            elif row['ButsMarques_2'] == row['ButsMarques_1']:
                return 0
            else:
                return 2
        else:
            return np.nan
    df['resultat'] = df.apply(determine_result, axis=1)
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
    
    # Si la colonne 'resultat' n'existe pas, on la crée.
    if "resultat" not in df.columns:
        print("La colonne 'resultat' n'est pas présente. Création de la colonne à partir des scores...")
        df = create_resultat_column(df)
        print("Premières lignes après ajout de 'resultat' :")
        print(df[['Équipe', 'home_team', 'away_team', 'ButsMarques_1', 'ButsMarques_2', 'resultat']].head())
    
    # On garde uniquement les colonnes numériques pour l'entraînement.
    # Les colonnes contextuelles à exclure sont : "fixture_id", "date", "Équipe", "Ligue", "home_team", "away_team"
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
