"""La communication avec l'API FastAPI."""
import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Mets False quand l'API sera prête : l'interface l'appellera vraiment
UTILISER_FAUSSE_API = True


class ErreurAPI(Exception):
    """Levée quand l'API refuse le SMS ou ne répond pas."""


def analyser_sms(texte):
    """Renvoie "spam" ou "légitime" pour un SMS."""
    if UTILISER_FAUSSE_API:
        return _fausse_api(texte)
    return _vraie_api(texte)


def _fausse_api(texte):
    # FAUSSE API : une règle bidon pour tester l'interface sans l'API
    mots_suspects = ["free", "win", "winner", "prize", "urgent", "claim", "cash"]
    if any(mot in texte.lower() for mot in mots_suspects):
        return "spam"
    return "légitime"


def _vraie_api(texte):
    try:
        reponse = requests.post(f"{API_URL}/predict", json={"text": texte}, timeout=5)
    except requests.exceptions.ConnectionError:
        raise ErreurAPI(f"Impossible de joindre l'API sur {API_URL}. Est-elle lancée ?")

    if reponse.status_code == 422:
        raise ErreurAPI(f"SMS refusé : {reponse.json()['detail'][0]['msg']}")
    if reponse.status_code != 200:
        raise ErreurAPI(f"Erreur de l'API ({reponse.status_code})")
    return reponse.json()["label"]
