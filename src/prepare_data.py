# Préparation des données : SMS en français
# Je lance ce script depuis la racine du projet : python src/prepare_data.py
from pathlib import Path
import pandas as pd

# Chemins : je pars du dossier du projet pour que ça marche d'où qu'on lance le script
racine = Path(__file__).parent.parent
chemin_csv = racine / "data" / "messages.csv"
dossier_sortie = racine / "data" / "processed"

# 1. Import
df = pd.read_csv(chemin_csv)
print("Données brutes :", df.shape)

# 2. Je garde seulement les SMS
sms = df[df["domain"] == "sms"].copy()

# 3. J'harmonise les labels
sms["label"] = sms["label"].replace({"legitimate": "légitime"})

# 4. Je retire les textes vides et les doublons (les accents sont gardés)
sms = sms.dropna(subset=["text"])
sms = sms[sms["text"].str.strip() != ""]
sms = sms.drop_duplicates(subset="text")
print("SMS après nettoyage :", sms.shape)
print(sms["label"].value_counts())

# 5. Je vérifie qu'un même template_group n'est pas dans plusieurs partitions
nb_partitions = sms.groupby("template_group")["split"].nunique()
print("Groupes présents dans plusieurs partitions :", (nb_partitions > 1).sum())

# 6. Export : un fichier complet et un fichier par partition
dossier_sortie.mkdir(parents=True, exist_ok=True)
sms[["text", "label", "split"]].to_csv(dossier_sortie / "sms_clean.csv", index=False)
for nom, partie in sms.groupby("split"):
    partie[["text", "label"]].to_csv(dossier_sortie / f"{nom}.csv", index=False)

print("Fichiers écrits dans", dossier_sortie)
