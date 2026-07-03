import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_preprocessing import preprocess_data

def test_preprocess_data():
    """
    Test unitaire pour s'assurer que le prétraitement et le partitionnement
    des données se font correctement avec les deux scalers.
    """
    # 1. Création d'un mini DataFrame fictif (20 transactions)
    np.random.seed(42)
    data = {
        'Time': np.linspace(0, 100, 20),
        'Amount': np.linspace(10, 1000, 20),
        'Class': [0] * 18 + [1] * 2  # 2 fraudes et 18 normales (ratio 10%)
    }
    
    # Ajouter les 28 features V1 à V28 fictives
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, 20)
        
    df = pd.DataFrame(data)
    
    # 2. Exécution du pré-traitement
    X_train, X_test, y_train, y_test, scaler_amount, scaler_time = preprocess_data(df)
    
    # 3. Assertions
    # Vérifier que Time et Amount d'origine ont été droppés et remplacés par leur version normalisée
    assert 'Time' not in X_train.columns
    assert 'Amount' not in X_train.columns
    assert 'Time_scaled' in X_train.columns
    assert 'Amount_scaled' in X_train.columns
    
    # Vérifier les dimensions du split (test_size = 20% par défaut)
    assert X_train.shape[0] == 16
    assert X_test.shape[0] == 4
    assert len(y_train) == 16
    assert len(y_test) == 4
    
    # Vérifier que les deux scalers renvoyés sont bien des instances de StandardScaler
    assert isinstance(scaler_amount, StandardScaler)
    assert isinstance(scaler_time, StandardScaler)
