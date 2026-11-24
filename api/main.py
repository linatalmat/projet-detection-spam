import os
from contextlib import asynccontextmanager
from pathlib import Path

# joblib : l'outil qui charge le modèle sauvegardé par Data 2
import joblib
from fastapi import FastAPI, HTTPException
# Pydantic : l'outil qui vérifie les données reçues
from pydantic import BaseModel, Field, field_validator


# 1. LE MODÈLE : où il est et comment on le charge
# Chemin du modèle (modifiable avec la variable d'environnement MODEL_PATH, utile pour Docker)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/spam_model.joblib"))

# Le modèle répond en anglais ("legitimate") : on traduit pour l'interface
TRADUCTION = {"legitimate": "légitime", "legitime": "légitime", "ham": "légitime"}

# Le modèle chargé sera rangé ici (None = pas encore chargé)
model = None


# Le code avant "yield" tourne UNE SEULE FOIS, au démarrage de l'API.
# On charge le modèle à ce moment-là, et pas à chaque SMS (sinon l'API serait lente).
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Modèle chargé depuis {MODEL_PATH}")
    else:
        print(f"ATTENTION : aucun modèle trouvé à {MODEL_PATH}")
    yield


# On crée l'application. "app" est le nom que uvicorn cherche au lancement
app = FastAPI(title="Détecteur de spam", lifespan=lifespan)


# 2. LA VALIDATION : ce que l'API accepte
class SMS(BaseModel):
    text: str = Field(max_length=1000, description="Le texte du SMS à analyser")

    @field_validator("text")
    @classmethod
    def pas_vide(cls, valeur: str) -> str:
        if not valeur.strip():
            raise ValueError("Le texte du SMS ne peut pas être vide")
        return valeur


# 3. LA ROUTE /health : vérifier que l'API est en vie (et que le modèle est chargé)
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# 4. LA ROUTE /predict : analyser un SMS avec le modèle
@app.post("/predict")
def predict(sms: SMS):
    # Pas de modèle → erreur 503 ("service indisponible")
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé")

    # Le modèle attend une LISTE de textes : on lui donne [le SMS] et on prend la 1re réponse
    # predict_proba donne une probabilité par classe (dans l'ordre de model.classes_)
    probabilites = model.predict_proba([sms.text])[0]
    indice = int(probabilites.argmax())
    label = str(model.classes_[indice])

    # On traduit si besoin ("legitimate" → "légitime")
    # "probability" = confiance du modèle dans le label renvoyé (entre 0 et 1)
    return {
        "label": TRADUCTION.get(label, label),
        "probability": round(float(probabilites[indice]), 4),
    }