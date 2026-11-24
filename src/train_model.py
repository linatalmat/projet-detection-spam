#  IMPORTATION DES BIBLIOTHEQUES


from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score
)



#  DEFINITION DES CHEMINS DU PROJET


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


#  CHARGEMENT DES DONNEES


train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print("Données chargées avec succès !")

print("Train :", train_df.shape)
print("Validation :", validation_df.shape)
print("Test :", test_df.shape)



#  SEPARATION DES TEXTES ET DES LABELS


X_train = train_df["text"]
y_train = train_df["label"]

X_validation = validation_df["text"]
y_validation = validation_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]



#  CREATION DE LA BASELINE


baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)


#  CONSTRUCTION DU MODELE


model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", MultinomialNB())
])



#  ENTRAINEMENT DU MODELE


print("\nEntraînement du modèle en cours...")

model.fit(X_train, y_train)

print("Entraînement terminé !")



#  FONCTION D'EVALUATION


def evaluate_model(name, estimator, X, y):

    predictions = estimator.predict(X)

    precision = precision_score(
        y,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    accuracy = accuracy_score(y, predictions)

    matrix = confusion_matrix(
        y,
        predictions,
        labels=["légitime", "spam"]
    )

    faux_positifs = matrix[0, 1]

    print(f"\n========== {name} ==========")

    print(f"Accuracy        : {accuracy:.3f}")
    print(f"Précision spam  : {precision:.3f}")
    print(f"Rappel spam     : {recall:.3f}")
    print(f"Score F1 spam   : {f1:.3f}")

    print(
        "SMS légitimes classés comme spam :",
        faux_positifs
    )

    print("\nMatrice de confusion :")
    print(matrix)

    print("\nInterprétation de la matrice :")
    print("Ligne 1 : SMS réellement légitimes")
    print("Ligne 2 : SMS réellement spam")
    print("Colonne 1 : prédiction légitime")
    print("Colonne 2 : prédiction spam")



#  EVALUATION SUR LES DONNEES DE VALIDATION


print("\n\n====================================")
print("EVALUATION SUR LA VALIDATION")
print("====================================")

evaluate_model(
    "BASELINE",
    baseline,
    X_validation,
    y_validation
)

evaluate_model(
    "TF-IDF + NAIVE BAYES",
    model,
    X_validation,
    y_validation
)



#  EVALUATION FINALE SUR LES DONNEES DE TEST


print("\n\n====================================")
print("EVALUATION FINALE SUR LE TEST")
print("====================================")

evaluate_model(
    "BASELINE",
    baseline,
    X_test,
    y_test
)

evaluate_model(
    "TF-IDF + NAIVE BAYES",
    model,
    X_test,
    y_test
)



#  SAUVEGARDE DU MODELE


MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "spam_model.joblib"

joblib.dump(model, model_path)

print("\n====================================")
print("MODELE SAUVEGARDE AVEC SUCCES !")
print("====================================")

print("Emplacement :", model_path)
