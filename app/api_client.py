"""La communication avec l'API FastAPI."""
import os

import requests

# Adresse de l'API (modifiable avec la variable d'environnement API_URL, utile pour Docker)
API_URL = os.getenv("API_URL", "http://localhost:8000")


class ErreurAPI(Exception):
    """Levée quand l'API refuse le SMS ou ne répond pas."""


def analyser_sms(texte):
    """Envoie le SMS à l'API et renvoie "spam" ou "légitime"."""
    # 1. On envoie le SMS à l'API
    try:
        reponse = requests.post(f"{API_URL}/predict", json={"text": texte}, timeout=5)
    except requests.exceptions.ConnectionError:
        raise ErreurAPI(f"Impossible de joindre l'API sur {API_URL}. Est-elle lancée ?")

    # 2. On vérifie que l'API a accepté le SMS
    if reponse.status_code == 422:
        raise ErreurAPI(f"SMS refusé : {reponse.json()['detail'][0]['msg']}")
    if reponse.status_code != 200:
        raise ErreurAPI(f"Erreur de l'API ({reponse.status_code})")

    # 3. On renvoie la réponse : "spam" ou "légitime"
    return reponse.json()["label"]