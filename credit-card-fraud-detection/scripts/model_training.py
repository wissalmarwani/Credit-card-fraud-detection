import xgboost as xgb
from sklearn.ensemble import IsolationForest
from config import RANDOM_STATE

def train_hybrid_model(X_train, y_train, X_test, y_test):
    """
    Entraîne le modèle hybride optimal :
    1. Entraîne une Isolation Forest sur les données d'entraînement.
    2. Utilise cette Isolation Forest pour extraire le score d'anomalie
       et l'ajoute comme nouvelle colonne (feature) sur le train et le test set.
    3. Entraîne XGBoost sur ces données enrichies en gérant le déséquilibre de classe.
    """
    # 1. Isolation Forest (Non supervisé) pour extraire le score d'anomalie
    print("\n>>> 1. Entraînement de l'Isolation Forest (Non supervisé) pour calculer le score d'anomalie...")
    iso_forest = IsolationForest(
        contamination=0.0017,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    # 2. Ajout du score d'anomalie comme caractéristique supplémentaire
    X_train_hybrid = X_train.copy()
    X_test_hybrid = X_test.copy()
    X_train_hybrid['anomaly_score'] = iso_forest.decision_function(X_train)
    X_test_hybrid['anomaly_score'] = iso_forest.decision_function(X_test)
    
    # 3. Entraînement de XGBoost (Supervisé)
    print(">>> 2. Entraînement de XGBoost (Supervisé) avec la feature d'anomalie...")
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='aucpr',
        n_jobs=-1
    )
    xgb_model.fit(X_train_hybrid, y_train)
    
    y_pred = xgb_model.predict(X_test_hybrid)
    y_proba = xgb_model.predict_proba(X_test_hybrid)[:, 1]
    
    return xgb_model, iso_forest, y_pred, y_proba