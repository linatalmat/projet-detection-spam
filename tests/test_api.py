"""Tests de l'API FastAPI (/health et /predict)."""
import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import api.main as api_main

TEXTES = [
    "Félicitations ! Vous avez gagné un cadeau gratuit. Cliquez ici.",
    "URGENT : votre compte sera bloqué, confirmez vos données maintenant.",
    "Gagnez 500 euros gratuit, cliquez sur le lien.",
    "Bonjour, nous avons rendez-vous demain à 14 heures.",
    "Ton colis arrive demain matin, merci de rester disponible.",
    "Rendez-vous confirmé pour la réunion de demain.",
]
LABELS = ["spam", "spam", "spam", "légitime", "légitime", "légitime"]


@pytest.fixture(scope="module")
def petit_modele():
    """Un mini-modèle entraîné à la volée : les tests ne dépendent pas du vrai fichier .joblib."""
    modele = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB())])
    modele.fit(TEXTES, LABELS)
    return modele


@pytest.fixture
def client(petit_modele, monkeypatch):
    """Client de test avec un modèle déjà chargé."""
    monkeypatch.setattr(api_main, "model", petit_modele)
    return TestClient(api_main.app)


@pytest.fixture
def client_sans_modele(monkeypatch):
    monkeypatch.setattr(api_main, "model", None)
    return TestClient(api_main.app)


# --- /health ---------------------------------------------------------------

def test_health_modele_charge(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json() == {"status": "ok", "model_loaded": True}


def test_health_sans_modele(client_sans_modele):
    reponse = client_sans_modele.get("/health")
    assert reponse.status_code == 200
    assert reponse.json() == {"status": "ok", "model_loaded": False}


# --- /predict : cas valides -------------------------------------------------

def test_predict_spam(client):
    reponse = client.post("/predict", json={"text": "Félicitations ! Cadeau gratuit, cliquez ici."})
    assert reponse.status_code == 200
    assert reponse.json()["label"] == "spam"


def test_predict_legitime(client):
    reponse = client.post("/predict", json={"text": "Bonjour, rendez-vous demain à 14 heures."})
    assert reponse.status_code == 200
    assert reponse.json()["label"] == "légitime"


def test_predict_renvoie_une_probabilite(client):
    donnees = client.post("/predict", json={"text": "Cadeau gratuit, cliquez ici."}).json()
    assert set(donnees) == {"label", "probability"}
    assert 0.5 <= donnees["probability"] <= 1  # c'est la confiance dans le label renvoyé


def test_predict_probabilite_coherente_avec_le_modele(client, petit_modele):
    texte = "Cadeau gratuit, cliquez ici."
    donnees = client.post("/predict", json={"text": texte}).json()
    attendu = petit_modele.predict_proba([texte])[0].max()
    assert donnees["probability"] == pytest.approx(attendu, abs=1e-4)


def test_predict_texte_de_1000_caracteres_accepte(client):
    reponse = client.post("/predict", json={"text": "a" * 1000})
    assert reponse.status_code == 200


def test_predict_traduit_les_labels_anglais(client, monkeypatch):
    """Un ancien modèle qui répond « legitimate » doit être traduit en « légitime »."""
    modele = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB())])
    modele.fit(TEXTES, ["spam"] * 3 + ["legitimate"] * 3)
    monkeypatch.setattr(api_main, "model", modele)
    reponse = client.post("/predict", json={"text": "Bonjour, rendez-vous demain."})
    assert reponse.json()["label"] == "légitime"


# --- /predict : erreurs -----------------------------------------------------

@pytest.mark.parametrize(
    "corps",
    [
        {},                        # champ manquant
        {"text": ""},              # vide
        {"text": "   \n\t "},       # uniquement des espaces
        {"text": "a" * 1001},      # trop long
        {"text": 123},             # pas du texte
        {"text": None},            # null
    ],
)
def test_predict_refuse_les_textes_invalides(client, corps):
    reponse = client.post("/predict", json=corps)
    assert reponse.status_code == 422


def test_predict_sans_modele_renvoie_503(client_sans_modele):
    reponse = client_sans_modele.post("/predict", json={"text": "Bonjour"})
    assert reponse.status_code == 503


def test_predict_refuse_get(client):
    assert client.get("/predict").status_code == 405


# --- Chargement du modèle au démarrage ---------------------------------------

def test_le_modele_est_charge_au_demarrage(petit_modele, tmp_path, monkeypatch):
    chemin = tmp_path / "modele.joblib"
    joblib.dump(petit_modele, chemin)
    monkeypatch.setattr(api_main, "MODEL_PATH", chemin)
    monkeypatch.setattr(api_main, "model", None)

    with TestClient(api_main.app) as c:  # le "with" déclenche le démarrage (lifespan)
        assert c.get("/health").json()["model_loaded"] is True
        assert c.post("/predict", json={"text": "Cadeau gratuit"}).status_code == 200


def test_demarrage_sans_fichier_modele(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "MODEL_PATH", tmp_path / "absent.joblib")
    monkeypatch.setattr(api_main, "model", None)

    with TestClient(api_main.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json={"text": "Bonjour"}).status_code == 503
