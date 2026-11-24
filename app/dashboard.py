"""Le tableau de bord : statistiques sur les données."""
import re
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from style import ROUGE, VERT

DATA_PATH = Path("data/processed/sms_clean.csv")  # fourni par Data 1

# Les labels peuvent être écrits de plusieurs façons : on les ramène tous à "légitime"
TRADUCTION = {"legitimate": "légitime", "legitime": "légitime", "ham": "légitime"}

# Petits mots sans intérêt, retirés du graphique des mots fréquents
# (les mots de moins de 3 lettres comme "le", "de", "un" sont déjà ignorés)
MOTS_VIDES = {
    "les", "des", "une", "est", "pour", "vous", "votre", "vos", "sur", "par",
    "avec", "dans", "qui", "que", "pas", "aux", "ces", "son", "ses", "nous",
    "elle", "mais", "plus", "tout", "été", "avant", "cette",
}


@st.cache_data
def charger_donnees():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # On traduit les labels pour l'affichage (le fichier de Data 1 n'est pas modifié)
        df["label"] = df["label"].replace(TRADUCTION)
        df["longueur"] = df["text"].str.len()
        mots = Counter()
        for texte in df.loc[df["label"] == "spam", "text"]:
            for mot in re.findall(r"[a-zA-Zàâçéèêëîïôûùüÿ]{3,}", texte.lower()):
                if mot not in MOTS_VIDES:
                    mots[mot] += 1
        top_mots = pd.DataFrame(mots.most_common(10), columns=["mot", "nombre"])
        comptes = df["label"].value_counts()
        longueurs = df[["label", "longueur"]]
        return comptes, top_mots, longueurs, True

    # --- Données d'exemple ---
    comptes = pd.Series({"légitime": 4516, "spam": 653})
    top_mots = pd.DataFrame(
        {
            "mot": ["call", "free", "txt", "mobile", "claim", "text", "stop", "reply", "prize", "win"],
            "nombre": [345, 216, 150, 123, 113, 112, 108, 101, 93, 76],
        }
    )
    hasard = np.random.default_rng(42)
    longueurs = pd.DataFrame(
        {
            "label": ["légitime"] * 600 + ["spam"] * 150,
            "longueur": np.concatenate(
                [hasard.normal(70, 45, 600).clip(5, 400), hasard.normal(140, 25, 150).clip(30, 220)]
            ).astype(int),
        }
    )
    return comptes, top_mots, longueurs, False


def afficher_dashboard():
    comptes, top_mots, longueurs, vraies_donnees = charger_donnees()

    st.header(":material/monitoring: Tableau de bord")
    # Petite ligne grise qui indique d'où viennent les chiffres
    if vraies_donnees:
        st.caption(f"Source : `{DATA_PATH.as_posix()}` · données réelles")
    else:
        st.caption(f"Source : données d'exemple (fichier `{DATA_PATH.as_posix()}` introuvable)")

    # --- Les chiffres clés ---
    total = int(comptes.sum())
    part_spam = comptes.get("spam", 0) / total
    moyennes = longueurs.groupby("label")["longueur"].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(":material/sms: SMS dans les données", f"{total:,}".replace(",", " "))
    k2.metric(":material/block: Part de spam", f"{part_spam:.0%}")
    k3.metric(":material/mail: Long. moy. SMS normal", f"{moyennes.get('légitime', 0):.0f} car.")
    k4.metric(":material/straighten: Long. moy. spam", f"{moyennes.get('spam', 0):.0f} car.")

    couleurs = alt.Scale(domain=["légitime", "spam"], range=[VERT, ROUGE])

    g1, g2 = st.columns(2, gap="large")

    with g1:
        st.markdown("**Répartition des SMS**")
        df_comptes = comptes.rename_axis("label").reset_index(name="nombre")
        donut = (
            alt.Chart(df_comptes)
            .mark_arc(innerRadius=70)
            .encode(theta="nombre:Q", color=alt.Color("label:N", scale=couleurs, title=None),
                    tooltip=["label", "nombre"])
            .properties(height=280)
        )
        st.altair_chart(donut, width="stretch")

    with g2:
        st.markdown("**Les 10 mots les plus fréquents dans les spams**")
        barres = (
            alt.Chart(top_mots)
            .mark_bar(color=ROUGE, cornerRadiusEnd=4)
            .encode(x=alt.X("nombre:Q", title="Nombre d'apparitions"),
                    y=alt.Y("mot:N", sort="-x", title=None), tooltip=["mot", "nombre"])
            .properties(height=280)
        )
        st.altair_chart(barres, width="stretch")

    st.markdown("**Longueur des messages** : les spams sont souvent plus longs")
    histo = (
        alt.Chart(longueurs)
        .mark_bar(opacity=0.6, binSpacing=0)
        .encode(x=alt.X("longueur:Q", bin=alt.Bin(maxbins=40), title="Nombre de caractères"),
                y=alt.Y("count():Q", stack=None, title="Nombre de SMS"),
                color=alt.Color("label:N", scale=couleurs, title=None))
        .properties(height=240)
    )
    st.altair_chart(histo, width="stretch")