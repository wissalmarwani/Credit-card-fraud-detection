import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb

def create_mocks():
    """Génère de faux modèles sérialisés pour permettre aux tests CI de s'exécuter sans le dataset de 150 Mo."""
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Fit de faux scalers
    scaler_amount = StandardScaler().fit([[10.0], [500.0]])
    scaler_time = StandardScaler().fit([[0.0], [86400.0]])
    
    # 2. Fit d'un faux modèle Isolation Forest
    # 20 lignes factices, 30 colonnes (V1..V28, Amount_scaled, Time_scaled)
    cols = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
    X_mock = pd.DataFrame(np.random.normal(0, 1, (20, 30)), columns=cols)
    iso_forest = IsolationForest(contamination=0.1, random_state=42).fit(X_mock)
    
    # 3. Fit d'un faux modèle XGBoost
    X_mock_hybrid = X_mock.copy()
    X_mock_hybrid['anomaly_score'] = iso_forest.decision_function(X_mock)
    y_mock = np.array([0]*18 + [1]*2)
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='aucpr').fit(X_mock_hybrid, y_mock)
    
    # 4. Enregistrement des pickles
    with open(os.path.join(models_dir, "scaler_amount.pkl"), "wb") as f:
        pickle.dump(scaler_amount, f)
    with open(os.path.join(models_dir, "scaler_time.pkl"), "wb") as f:
        pickle.dump(scaler_time, f)
    with open(os.path.join(models_dir, "iso_forest.pkl"), "wb") as f:
        pickle.dump(iso_forest, f)
    with open(os.path.join(models_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
        
    print("Modèles factices générés avec succès pour les tests !")

if __name__ == "__main__":
    create_mocks()
