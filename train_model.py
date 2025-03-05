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

def load_and_preprocess_data(csv_path):
    """
    Charge le dataset combiné réduit (contenant 44 variables numériques par match, 
    soit 22 par équipe) et effectue le prétraitement.
    
    On suppose que le dataset contient une colonne 'resultat' pour la cible.
    Les autres colonnes (les 44 variables) sont utilisées comme features.
    """
    df = pd.read_csv(csv_path)
    
    if "resultat" not in df.columns:
        raise ValueError("La colonne 'resultat' n'est pas présente dans le dataset.")
    
    # On suppose que le dataset contient exactement 44 variables pour les features (sans 'resultat')
    X = df.drop(columns=["resultat"], errors='ignore')
    y = df["resultat"]
    
    # Sélectionner uniquement les colonnes numériques
    X = X.select_dtypes(include=[np.number])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_all_models(X, y):
    """
    Entraîne plusieurs modèles de classification sur le jeu de données X et y.
    Utilise une validation croisée à 5 plis pour évaluer les performances.
    Retourne un dictionnaire de modèles entraînés et leurs scores.
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    }
    
    trained_models = {}
    cv_scores = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        cv_scores[name] = np.mean(scores)
        print(f"{name}: CV Accuracy = {cv_scores[name]:.4f}")
        model.fit(X, y)
        trained_models[name] = model
        
    return trained_models, cv_scores

def save_models(models, scaler, models_dir="models"):
    """
    Sauvegarde les modèles entraînés et le scaler dans le dossier spécifié.
    """
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
    data_path = "data/combined_dataset_all_leagues_reduit.csv"  # Chemin vers votre dataset combiné réduit
    print("Chargement et prétraitement des données...")
    X, y, scaler = load_and_preprocess_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entraînement des modèles...")
    trained_models, cv_scores = train_all_models(X_train, y_train)
    
    print("\nÉvaluation sur le jeu de test:")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"{name}: Test Accuracy = {test_acc:.4f}")
    
    print("Sauvegarde des modèles et du scaler...")
    save_models(trained_models, scaler)
    
    print("Entraînement terminé.")

if __name__ == "__main__":
    main()
