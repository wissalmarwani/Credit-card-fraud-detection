# Utiliser une image Python officielle légère et stable
FROM python:3.10-slim

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements de notre projet
COPY credit-card-fraud-detection/requirements.txt requirements.txt

# Installer les dépendances Python requises
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'intégralité du projet (scripts et modèles sérialisés)
COPY credit-card-fraud-detection/ /app/credit-card-fraud-detection/

# Ajouter le dossier 'scripts' au PYTHONPATH pour que l'API trouve les modules importés
ENV PYTHONPATH=/app/credit-card-fraud-detection/scripts

# Exposer le port 8000 sur lequel l'API FastAPI va écouter
EXPOSE 8000

# Lancer le serveur d'API uvicorn de production
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
