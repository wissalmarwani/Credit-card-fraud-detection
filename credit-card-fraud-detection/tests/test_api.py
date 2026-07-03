from fastapi.testclient import TestClient
from api import app

def test_health_check():
    """
    Test pour vérifier le bon fonctionnement du point d'accès santé (/health)
    en utilisant le context manager de TestClient pour déclencher le chargement des modèles.
    """
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True

def test_predict_endpoint_success():
    """
    Test pour vérifier que l'API renvoie bien une prédiction au format attendu
    lorsqu'on lui envoie une transaction factice valide.
    """
    with TestClient(app) as client:
        # 1. Préparation d'une transaction fictive (Time=100.0, Amount=50.0, features V à 0.0)
        tx_data = {
            "Time": 100.0,
            "Amount": 50.0
        }
        for i in range(1, 29):
            tx_data[f"V{i}"] = 0.0

        # 2. Appel POST sur la route /predict
        response = client.post("/predict", json=tx_data)
        
        # 3. Assertions
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "probability" in data
        assert isinstance(data["is_fraud"], bool)
        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0
