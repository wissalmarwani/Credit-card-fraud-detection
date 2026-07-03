import os
import pickle
from config import BASE_DIR
from data_loader import load_data
from data_preprocessing import preprocess_data
from eda import run_eda
from model_evaluation import evaluate_hybrid_model
from model_training import train_hybrid_model

def main():
    # 1. Chargement des données
    df = load_data()
    
    # 2. Analyse exploratoire graphique
    run_eda(df)

    # 3. Prétraitement et split stratifié (avec récupération des scalers)
    X_train, X_test, y_train, y_test, scaler_amount, scaler_time = preprocess_data(df)
    
    # 4. Entraînement du modèle hybride (XGBoost + feature Isolation Forest)
    xgb_model, iso_forest, y_pred, y_proba = train_hybrid_model(X_train, y_train, X_test, y_test)
    
    # 5. Évaluation et tracés des courbes
    evaluate_hybrid_model(y_test, y_pred, y_proba)

    # 6. Sauvegarde des modèles sérialisés pour la production (API/UI)
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("\n>>> Sauvegarde des modèles sérialisés pour la production...")
    with open(os.path.join(models_dir, "scaler_amount.pkl"), "wb") as f:
        pickle.dump(scaler_amount, f)
    with open(os.path.join(models_dir, "scaler_time.pkl"), "wb") as f:
        pickle.dump(scaler_time, f)
    with open(os.path.join(models_dir, "iso_forest.pkl"), "wb") as f:
        pickle.dump(iso_forest, f)
    with open(os.path.join(models_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)

    print(f"Modèles sauvegardés avec succès dans le dossier : {models_dir}")
    print("\nPipeline hybride exécuté avec succès.")

if __name__ == "__main__":
    main()
