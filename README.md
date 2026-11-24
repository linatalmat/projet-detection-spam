# Projet de détection de spam

![Détection de spam](docs/detection-spam.gif)

Projet du mini-hackathon Data & IA (HETIC, 23 au 27 novembre 2026).
Objectif : classer un SMS en français comme **spam** ou **légitime**.

## Organisation de l'équipe

| Partie | Contenu | Responsable |
|---|---|---|
| Données | Chargement, nettoyage et export du jeu de données | Dylan RAMI |
| Modèle | Baseline et modèle scikit-learn (TF-IDF + MultinomialNB) | À compléter |
| Application | API FastAPI (/health, /predict) et interface Streamlit | À compléter |
| DevOps | Git, pytest, Docker Compose, GitLab CI | À compléter |

## Structure du dépôt

```
projet-detection-spam/
├── data/
│   ├── messages.csv        # jeu de données brut
│   └── processed/          # fichiers propres (train, validation, test)
├── notebooks/
│   └── analyse_fichier.ipynb   # exploration et nettoyage
├── src/
│   └── prepare_data.py     # script de préparation des données
├── docs/
│   └── detection-spam.gif
└── README.md
```

## Installation

Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install pandas
```

Les dépendances complètes (modèle, API, tests) sont à compléter par les autres parties.

## 1. Partie Données : Dylan RAMI

### Source et droits

- Jeu de données : French Spam Text Dataset
- Auteur : Djochrist
- Lien : https://github.com/Djochrist/french-spam-text-dataset-v1.0 (fichier `data/messages.csv`)
- Licence : aucune licence indiquée sur le dépôt
- Nature des données : messages synthétiques (générés), donc pas de vrais SMS de personnes réelles

Le fichier `data/messages.csv` est téléchargé depuis le dépôt de l'auteur. Chaque personne de l'équipe doit en avoir sa propre copie.

### Description des colonnes

| Colonne | Contenu |
|---|---|
| id | Identifiant du message |
| text | Texte du message |
| label | `spam` ou `legitimate` (devient `légitime` après nettoyage) |
| domain | Type de message : `sms` ou `email` |
| spam_category | Catégorie du spam (par exemple phishing) |
| split | Partition fournie : train, validation ou test |
| template_group | Groupe de messages issus du même modèle de texte |

Le fichier brut contient 10 000 lignes et 7 colonnes, sans valeur manquante.

### Ce que fait la préparation

1. Je garde uniquement les SMS (`domain == "sms"`) : 5 000 lignes.
2. Je harmonise les labels en `spam` et `légitime`.
3. Je vérifie les valeurs manquantes, les textes vides et les doublons : il n'y en a aucun.
4. Je garde les accents.
5. Je respecte les partitions fournies : train

Résultat : 2 500 spams et 2 500 SMS légitimes (classes équilibrées).

### Régénérer les fichiers

Depuis la racine du dépôt :

```bash
python src/prepare_data.py
```

### Limites

- Les données sont synthétiques : les scores du modèle risquent d'être plus élevés que sur de vrais SMS.
- Il n'y a que 52 modèles de texte (`template_group`), donc les messages se ressemblent beaucoup.
- La longueur seule sépare déjà bien les classes (spam : environ 171 caractères en moyenne, légitime : environ 116).

## 2. Partie Modèle


**Responsable : Lina TALMAT**

### Objectif

L'objectif de cette partie est de développer un modèle de Machine Learning capable de classifier automatiquement un SMS en deux catégories :

- `spam` : message indésirable ou frauduleux ;
- `légitime` : message normal.

Le modèle analyse le contenu textuel du SMS et prédit sa catégorie.

### Méthodologie

Nous avons utilisé deux approches pour réaliser et évaluer la classification.

**1. Baseline — DummyClassifier**

La baseline est notre modèle de référence. Nous avons utilisé `DummyClassifier(strategy="most_frequent")`.

Cette stratégie prédit systématiquement la catégorie la plus fréquente dans les données d'entraînement.

Comme le jeu d'entraînement contient 1 667 spams et 1 500 messages légitimes, la baseline prédit toujours `spam`.

Elle permet de vérifier si notre modèle principal apporte une réelle amélioration.

**2. Modèle principal — TF-IDF + Multinomial Naive Bayes**

Notre modèle repose sur deux composants :

- **TF-IDF (`TfidfVectorizer`)** : transforme les SMS en représentations numériques exploitables par un algorithme de Machine Learning. Nous utilisons `ngram_range=(1, 2)` pour prendre en compte les mots seuls et les groupes de deux mots.
- **Multinomial Naive Bayes (`MultinomialNB`)** : algorithme de classification probabiliste adapté aux données textuelles. Il apprend à distinguer les caractéristiques des messages spam et légitimes.

Les deux composants sont réunis dans un `Pipeline` Scikit-learn.

### Entraînement et évaluation

Le modèle est entraîné uniquement sur le jeu `train.csv`.

Les jeux `validation.csv` et `test.csv` sont utilisés respectivement pour observer les performances et réaliser l'évaluation finale.

| Partition | Nombre de SMS | Spam | Légitimes |
|---|---:|---:|---:|
| Train | 3 167 | 1 667 | 1 500 |
| Validation | 917 | 417 | 500 |
| Test | 916 | 416 | 500 |
| **Total** | **5 000** | **2 583** | **2 000** |

### Résultats obtenus

Comparaison des performances sur le jeu de test :

| Métrique | Baseline | TF-IDF + Naive Bayes |
|---|---:|---:|
| Accuracy | 45,4 % | 100 % |
| Précision spam | 45,4 % | 100 % |
| Rappel spam | 100 % | 100 % |
| F1-score spam | 62,5 % | 100 % |
| SMS légitimes classés comme spam | 500 | 0 |

**Matrice de confusion du modèle principal :**

| Réalité / Prédiction | Légitime | Spam |
|---|---:|---:|
| Réellement légitime | 500 | 0 |
| Réellement spam | 0 | 416 |

Le modèle a correctement classé les 916 SMS du jeu de test.

### Tests sur de nouveaux messages

Quatre messages supplémentaires ont été utilisés pour vérifier le fonctionnement du modèle :

| Exemple | Prédiction |
|---|---|
| Félicitations ! Vous avez gagné un cadeau gratuit. Cliquez ici. | `spam` |
| Bonjour, nous avons rendez-vous demain à 14 heures. | `légitime` |
| URGENT : Votre compte bancaire sera bloqué. Cliquez sur ce lien. | `spam` |
| Coucou, tu es disponible ce soir pour aller au cinéma ? | `légitime` |

Résultat : 4 prédictions correctes sur 4 exemples testés.

### Sauvegarde du modèle

Le modèle entraîné est sauvegardé avec la bibliothèque `joblib`.

Fichier généré :

`models/spam_model.joblib`

Ce fichier contient le pipeline complet (TF-IDF et Multinomial Naive Bayes). Il pourra être chargé par l'API FastAPI pour effectuer des prédictions sans réentraîner le modèle.

**Scripts développés :**

- `src/train_model.py` : chargement des données, entraînement, comparaison avec la baseline, évaluation et sauvegarde.
- `src/test_predictions.py` : tests de prédiction sur de nouveaux SMS.

### Limites et améliorations possibles

Malgré les excellents résultats obtenus, plusieurs limites doivent être prises en compte :

- **Données synthétiques :** les messages ne représentent pas nécessairement la diversité des SMS réellement reçus.
- **Similarité des templates :** le dataset contient 52 groupes de modèles de texte. Même si les groupes sont séparés entre les partitions, des formulations similaires peuvent faciliter la classification.
- **Généralisation :** les résultats obtenus ne garantissent pas les mêmes performances sur de vrais SMS.

Pistes d'amélioration :
- Tester le modèle sur un jeu de SMS réels et indépendants.
- Comparer Naive Bayes avec la régression logistique et les SVM.
- Tester différentes configurations de TF-IDF.
- Analyser les erreurs sur des messages plus variés.

### Conclusion

Le modèle TF-IDF + Multinomial Naive Bayes obtient un F1-score de 100 % sur le jeu de test, contre 62,5 % pour la baseline.

Ces résultats montrent une excellente performance sur le dataset utilisé. Toutefois, ils doivent être interprétés avec prudence en raison du caractère synthétique des données et du risque de similarité entre les messages.

Le modèle est sauvegardé et prêt à être intégré dans l'application.

## 3. Partie Application

À compléter.

## 4. Partie DevOps

À compléter.