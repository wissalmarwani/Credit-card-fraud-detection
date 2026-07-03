import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import BASE_DIR

# Initialisation de l'API FastAPI
app = FastAPI(
    title="API de Détection de Fraude Bancaire (MLOps)",
    description="API temps réel pour évaluer la probabilité de fraude d'une transaction de carte de crédit.",
    version="1.0"
)

# Variables pour stocker les modèles chargés
scaler_amount = None
scaler_time = None
iso_forest = None
xgb_model = None

@app.on_event("startup")
def load_production_models():
    """Charge les modèles sérialisés lors du démarrage du serveur API."""
    global scaler_amount, scaler_time, iso_forest, xgb_model
    models_dir = os.path.join(BASE_DIR, "models")
    
    try:
        print(f"Chargement des modèles depuis : {models_dir}...")
        with open(os.path.join(models_dir, "scaler_amount.pkl"), "rb") as f:
            scaler_amount = pickle.load(f)
        with open(os.path.join(models_dir, "scaler_time.pkl"), "rb") as f:
            scaler_time = pickle.load(f)
        with open(os.path.join(models_dir, "iso_forest.pkl"), "rb") as f:
            iso_forest = pickle.load(f)
        with open(os.path.join(models_dir, "xgb_model.pkl"), "rb") as f:
            xgb_model = pickle.load(f)
        print("Tous les modèles ont été chargés avec succès !")
    except Exception as e:
        print(f"Erreur lors du chargement des modèles : {str(e)}")
        raise RuntimeError(f"Impossible de démarrer l'API car les modèles sont absents ou corrompus. Lancez d'abord scripts/main.py. Détails : {str(e)}")

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.get("/health")
def health_check():
    """Permet de vérifier le bon fonctionnement de l'API et du chargement des modèles."""
    models_ready = all([scaler_amount, scaler_time, iso_forest, xgb_model])
    return {
        "status": "healthy" if models_ready else "unhealthy",
        "models_loaded": models_ready
    }

@app.post("/predict")
def predict_transaction(tx: Transaction):
    """
    Reçoit les données brutes d'une transaction, applique les prétraitements
    et renvoie si la transaction est frauduleuse ainsi que sa probabilité.
    """
    if not all([scaler_amount, scaler_time, iso_forest, xgb_model]):
        raise HTTPException(status_code=503, detail="Modèles non chargés.")

    try:
        # 1. Mise à l'échelle (scaling) de Amount et Time
        amount_scaled = float(scaler_amount.transform([[tx.Amount]])[0][0])
        time_scaled = float(scaler_time.transform([[tx.Time]])[0][0])

        # 2. Création d'un dictionnaire avec les features dans l'ordre attendu pour l'Isolation Forest
        features = {}
        for i in range(1, 29):
            features[f'V{i}'] = getattr(tx, f'V{i}')
        
        # Isolation Forest s'attend à V1..V28, Amount_scaled, Time_scaled
        features['Amount_scaled'] = amount_scaled
        features['Time_scaled'] = time_scaled

        # Conversion en DataFrame pour conserver l'ordre et le nom des colonnes
        df_features = pd.DataFrame([features])

        # 3. Calcul du score d'anomalie de l'Isolation Forest
        # decision_function renvoie le score brut (plus il est bas, plus c'est anormal)
        anomaly_score = float(iso_forest.decision_function(df_features)[0])
        
        # 4. Ajout du score d'anomalie à notre set de features
        features['anomaly_score'] = anomaly_score

        # Conversion finale en DataFrame pour XGBoost
        df_final = pd.DataFrame([features])

        # 5. Prédiction avec XGBoost
        is_fraud = bool(xgb_model.predict(df_final)[0])
        probability = float(xgb_model.predict_proba(df_final)[0][1])

        return {
            "is_fraud": is_fraud,
            "probability": round(probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {str(e)}")
