# 🔍 Détection de Fraude Bancaire — Approche Hybride Supervisée & Non Supervisée

Pipeline complet de machine learning pour détecter des fraudes sur des transactions bancaires, combinant une approche **non supervisée** (Isolation Forest) et **supervisée** (XGBoost), sur un dataset réel extrêmement déséquilibré.

**Résultats clés :**
- 🎯 ROC-AUC : **0.979** (XGBoost)
- 🎯 F1-score fraude : **0.87** (XGBoost) vs **0.35** (Isolation Forest seul)
- 📊 Dataset : 284 807 transactions, seulement 0.17% de 

## 📌 Contexte

La détection de fraude bancaire est un problème classique en data science, caractérisé par :
- Un **déséquilibre extrême** des classes (les fraudes sont rares)
- La nécessité de détecter des fraudes **jamais vues** (d'où l'intérêt du non supervisé)
- Un enjeu métier fort : un faux négatif (fraude non détectée) coûte cher, un faux positif (fausse alerte) frustre le client

Ce projet explore les deux grandes familles d'approches en machine learning et les compare sur un même dataset.

## 📂 Dataset

- **Source** : [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **284 807 transactions**, dont **492 fraudes** (0.17%)
- Features **V1 à V28** : anonymisées via PCA (pour protéger l'identité des clients)
- Features **Time** et **Amount** : conservées en clair
- **Class** : variable cible (0 = normal, 1 = fraude)
