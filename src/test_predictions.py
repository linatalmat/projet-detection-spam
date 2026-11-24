
from pathlib import Path
import joblib

# Chemin du projet
ROOT = Path(__file__).resolve().parent.parent

# Charger le modèle sauvegardé
model_path = ROOT / "models" / "spam_model.joblib"

model = joblib.load(model_path)

# Nouveaux SMS jamais utilisés pendant l'entraînement
messages = [
    "Félicitations ! Vous avez gagné un cadeau gratuit. Cliquez ici.",
    "Bonjour, nous avons rendez-vous demain à 14 heures.",
    "URGENT : Votre compte bancaire sera bloqué. Cliquez sur ce lien.",
    "Coucou, tu es disponible ce soir pour aller au cinéma ?",
]

# Effectuer les prédictions
predictions = model.predict(messages)

# Afficher les résultats
for message, prediction in zip(messages, predictions):

    print("\nSMS :", message)
    print("Prédiction :", prediction)
