# 🔍 Détection de Fraude Bancaire — Pipeline MLOps & Modèle Hybride

Ce projet implémente un système complet de détection de fraudes sur les transactions de cartes de crédit. Il repose sur une **approche hybride innovante** combinant l'apprentissage non supervisé (**Isolation Forest**) et supervisé (**XGBoost**), enveloppé dans une architecture **MLOps moderne** (FastAPI, Streamlit, Docker et CI/CD avec GitHub Actions).

---

## 🚀 Liens de Production (Démos en Ligne)

* **💻 Application Web Interactive (Streamlit Cloud)** : [Accéder au Dashboard](https://credit-card-fraud-detection-ibfcktdz6jjmctn7datjpp.streamlit.app)
* **🔌 API de Prédiction en Temps Réel (Render)** : [Consulter l'API Swagger](https://credit-card-fraud-detection-jzms.onrender.com/docs)

---

## 📊 Performances du Modèle Hybride

Le modèle hybride combine un expert en détection d'anomalies (Isolation Forest) qui génère un score d'anomalie, et un classifieur supervisé (XGBoost) qui apprend à partir de ce score et des caractéristiques d'origine.

| Métrique | Score Obtenu | Importance pour le Métier |
| :--- | :---: | :--- |
| **Précision** | **93,10 %** | Limite les fausses alertes (seulement 6.9% de clients légitimes bloqués par erreur). |
| **Rappel (Recall)** | **82,65 %** | Pourcentage de fraudes interceptées avec succès (81 fraudes capturées sur 98). |
| **F1-Score** | **87,57 %** | Synthèse harmonique équilibrée de la performance globale. |
| **PR-AUC (AUPRC)** | **0,8628** | La métrique la plus robuste pour évaluer un modèle sur données fortement déséquilibrées. |

---

## 🏗️ Architecture du Projet

Le code est structuré de manière modulaire selon les bonnes pratiques de génie logiciel :

```text
├── .github/workflows/
│   └── ci.yml                   # Pipeline d'Intégration Continue (GitHub Actions)
├── credit-card-fraud-detection/
│   ├── data/                    # Dataset de transactions (creditcard.csv)
│   ├── models/                  # Modèles sérialisés (Pickles : XGBoost, Isolation Forest, Scalers)
│   ├── plots/                   # Graphiques d'évaluation (courbes ROC et Precision-Recall)
│   ├── scripts/
│   │   ├── api.py               # Serveur d'API de production (FastAPI)
│   │   ├── app_ui.py            # Dashboard utilisateur (Streamlit)
│   │   ├── config.py            # Constantes et configuration globale
│   │   ├── data_loader.py       # Chargement des données
│   │   ├── data_preprocessing.py# Standardisation et split des données (2 scalers séparés)
│   │   ├── eda.py               # Analyse exploratoire graphique
│   │   ├── main.py              # Orchestration et exportation des modèles
│   │   ├── model_evaluation.py  # Évaluation métrique et traçage graphique
│   │   └── model_training.py    # Entraînement du modèle hybride unique
│   ├── tests/
│   │   ├── conftest.py          # Configuration de test pytest
│   │   ├── create_mock_models.py# Générateur de modèles fictifs légers pour la CI
│   │   ├── test_api.py          # Tests d'intégration FastAPI
│   │   └── test_preprocessing.py# Tests unitaires de pré-traitement
│   └── requirements.txt         # Dépendances du projet
└── Dockerfile                   # Recette de conteneurisation de production
```

---

## 🛠️ Concepts Avancés & MLOps Implémentés

* **Correction de Fuite de Données (Data Leakage)** : Utilisation de deux instances séparées de `StandardScaler` pour normaliser indépendamment le Montant et le Temps afin d'éviter toute collision de paramètres.
* **API REST Temps Réel (FastAPI)** : Un point d'accès `/predict` (POST) reçoit les données de transaction au format JSON, exécute les prétraitements à la volée, calcule le score d'anomalie de l'Isolation Forest et l'injecte dans le classifieur XGBoost pour renvoyer le score en temps réel.
* **Dashboard Interactif (Streamlit)** : Interface ergonomique permettant d'ajuster les caractéristiques d'une transaction bancaire pour interroger l'API déployée en ligne.
* **Conteneurisation (Docker)** : Fichier `Dockerfile` configuré pour créer une image autonome de production hébergée sur Render.
* **Mocking (CI/CD Avancée)** : Pour exécuter le pipeline de test sur GitHub Actions sans le jeu de données d'origine (150 Mo), un script de génération de modèles simulés légers (`create_mock_models.py`) permet de valider le fonctionnement de l'API en quelques secondes de compilation.

---

## 💻 Installation et Lancement Local

### 1. Cloner le projet et créer l'environnement virtuel
```bash
git clone https://github.com/wissalmarwani/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection
python -m venv venv
.\venv\Scripts\activate      # Sur Windows
source venv/bin/activate    # Sur Linux/macOS
pip install -r credit-card-fraud-detection/requirements.txt
```

### 2. Lancer la suite de tests avec Pytest
```bash
.\venv\Scripts\pytest.exe credit-card-fraud-detection/
```

### 3. Lancer l'API locale
```bash
.\venv\Scripts\uvicorn.exe credit-card-fraud-detection.scripts.api:app --reload --port 8000
```
L'API sera accessible sur [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Lancer le Dashboard local
```bash
.\venv\Scripts\streamlit.exe run credit-card-fraud-detection/scripts/app_ui.py
```
L'application s'ouvrira sur [http://localhost:8501](http://localhost:8501).
