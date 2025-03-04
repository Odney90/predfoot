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

# Chemin vers vos données (assurez-vous que le CSV existe et est complet)
DATA_PATH = "data/matchs.csv"

def load_and_preprocess_data():
    # Charger le fichier CSV
    df = pd.read_csv(DATA_PATH)
    
    # Vérifiez que la colonne cible "resultat" existe
    if "resultat" not in df.columns:
        raise ValueError("La colonne 'resultat' n'est pas présente dans vos données.")
    
    # Séparer les features et la cible
    X = df.drop(columns=["resultat"])
    y = df["resultat"]
    
    # Optionnel : remplir ou supprimer les valeurs manquantes si nécessaire
    # Exemple : X.fillna(X.mean(), inplace=True)
    
    # Normalisation des features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_models(X, y):
    models = {}
    
    # Définir les modèles à entraîner
    models["LogisticRegression"] = LogisticRegression(max_iter=1000)
    models["SVM"] = SVC(probability=True)
    models["RandomForest"] = RandomForestClassifier(n_estimators=100, random_state=42)
    models["XGBoost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    
    cv_scores = {}
    # Validation croisée et entraînement sur le jeu d'entraînement
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        cv_scores[name] = np.mean(scores)
        print(f"{name} - CV Accuracy: {cv_scores[name]:.4f}")
        model.fit(X, y)
        models[name] = model  # Entraîné sur l'ensemble complet
    
    return models, cv_scores

def save_models(models, scaler):
    # Créer le dossier "models" s'il n'existe pas
    if not os.path.exists("models"):
        os.makedirs("models")
    
    for name, model in models.items():
        filename = f"models/model_{name.lower()}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"{name} sauvegardé dans {filename}")
    
    # Sauvegarder le scaler utilisé pour normaliser les données
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler sauvegardé dans models/scaler.pkl")

def main():
    print("Chargement et prétraitement des données...")
    X, y, scaler = load_and_preprocess_data()
    
    # Vous pouvez aussi diviser vos données pour une évaluation plus fine si besoin
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entraînement des modèles...")
    models, cv_scores = train_models(X_train, y_train)
    
    # Évaluation sur le jeu de test
    for name, model in models.items():
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"{name} - Test Accuracy: {test_acc:.4f}")
    
    print("Sauvegarde des modèles et du scaler...")
    save_models(models, scaler)
    
    print("Entraînement terminé.")

if __name__ == "__main__":
    main()
