from fastapi import FastAPI
# Pydantic : l'outil qui vérifie les données reçues
from pydantic import BaseModel, Field, field_validator


# On crée l'application. "app" est le nom que uvicorn cherche au lancement
app = FastAPI(title="Détecteur de spam")


# 1. LA VALIDATION : ce que l'API accepte
class SMS(BaseModel):
    text: str = Field(max_length=1000, description="Le texte du SMS à analyser")
    @field_validator("text")
    @classmethod
    def pas_vide(cls, valeur: str) -> str:
        if not valeur.strip():
            raise ValueError("Le texte du SMS ne peut pas être vide")
        return valeur



# 2. LA ROUTE /health : vérifier que l'API est en vie
@app.get("/health")
def health():
    return {"status": "ok"}


# 3. LA ROUTE /predict : analyser un SMS
@app.post("/predict")
def predict(sms: SMS):
    mots_suspects = ["free", "win", "winner", "prize", "urgent", "claim", "cash"]
    if any(mot in sms.text.lower() for mot in mots_suspects):
        return {"label": "spam"}
    return {"label": "légitime"}
