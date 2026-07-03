import os
import sys

# Ajouter le dossier 'scripts' au sys.path pour que pytest trouve les modules importés.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
