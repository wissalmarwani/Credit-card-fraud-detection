import streamlit as st
import requests

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fraude Bancaire",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS personnalisé pour un look premium
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 16px;
        color: #555555;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .secure-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #00C851;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
    }
</style>
""", unsafe_allow_html=True)

# Contenu principal
st.markdown("<h1 class='main-title'>🔍 Détecteur de Fraude sur Cartes Bancaires</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Saisissez les informations de la transaction ci-dessous pour évaluer sa légitimité en temps réel grâce au Modèle Hybride (FastAPI + XGBoost + Isolation Forest).</p>", unsafe_allow_html=True)

# URL de l'API FastAPI
API_URL = "http://localhost:8000/predict"

# Layout en deux colonnes : Formulaire de saisie à gauche, résultats à droite
col_input, col_result = st.columns([2, 1.5])

with col_input:
    st.subheader("📋 Détails de la Transaction")
    
    # Paramètres principaux (Time et Amount)
    col_main1, col_main2 = st.columns(2)
    with col_main1:
        amount = st.number_input("Montant de la transaction (€)", min_value=0.0, value=120.0, step=10.0, help="Montant total débité.")
    with col_main2:
        time = st.slider("Temps écoulé (sec)", min_value=0, max_value=172800, value=86400, step=1, help="Temps en secondes écoulé depuis la première transaction de la base.")
    
    st.markdown("---")
    
    # Paramètres PCA V1 à V28 (masqués par défaut dans un expander pour ne pas encombrer)
    with st.expander("⚙️ Caractéristiques PCA Anonymisées (V1 à V28)", expanded=False):
        st.info("Ces caractéristiques sont issues d'une Analyse en Composantes Principales (PCA) pour anonymiser l'identité des clients. La valeur par défaut (0.0) représente le comportement moyen d'une transaction.")
        
        # Organisation en 4 colonnes pour un rendu propre
        pca_cols = st.columns(4)
        v_values = {}
        for i in range(1, 29):
            col_index = (i - 1) % 4
            with pca_cols[col_index]:
                # On met des sliders allant de -5.0 à 5.0 avec une valeur par défaut de 0.0
                v_values[f"V{i}"] = st.slider(f"V{i}", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

with col_result:
    st.subheader("📊 Résultats de l'analyse")
    
    # Création du bouton pour lancer la prédiction
    predict_button = st.button("Analyser la Transaction 🚀", use_container_width=True)
    
    if predict_button:
        # Construction du JSON à envoyer à l'API
        payload = {
            "Time": float(time),
            "Amount": float(amount)
        }
        for i in range(1, 29):
            payload[f"V{i}"] = float(v_values[f"V{i}"])
            
        with st.spinner("Analyse de la transaction en cours..."):
            try:
                # Appel POST à l'API FastAPI
                response = requests.post(API_URL, json=payload, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    is_fraud = data["is_fraud"]
                    probability = data["probability"]
                    
                    # Affichage visuel en fonction du statut de fraude
                    if is_fraud:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3>🚨 ALERTE : Fraude Probable !</h3>
                            <p>Cette transaction présente des schémas d'activité hautement suspects.</p>
                            <h2 style='color:#FF4B4B; margin-top:10px;'>Probabilité de fraude : {probability * 100:.2f} %</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("Action recommandée : Bloquer la transaction et alerter le client.")
                    else:
                        st.markdown(f"""
                        <div class='secure-card'>
                            <h3>✅ Transaction Sécurisée</h3>
                            <p>Aucun comportement suspect détecté. La transaction est conforme.</p>
                            <h2 style='color:#00C851; margin-top:10px;'>Risque de fraude : {probability * 100:.2f} %</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("Action recommandée : Autoriser la transaction.")
                    
                    # Barre de progression du risque de fraude
                    st.markdown("**Niveau de Risque Global :**")
                    st.progress(probability)
                else:
                    st.error(f"Erreur API ({response.status_code}) : {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API FastAPI. Veuillez démarrer l'API avec uvicorn d'abord !")
                st.info("ℹ️ Pour lancer l'API, exécutez dans un terminal : \n`.\\venv\\Scripts\\uvicorn.exe credit-card-fraud-detection.scripts.api:app --port 8000`")
            except Exception as e:
                st.error(f"Une erreur s'est produite : {str(e)}")
    else:
        st.info("Cliquez sur le bouton 'Analyser la Transaction 🚀' pour lancer l'évaluation en direct.")
