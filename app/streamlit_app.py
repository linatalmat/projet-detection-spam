"""Page principale : elle assemble tous les morceaux de l'interface."""
import pandas as pd
import streamlit as st

from api_client import ErreurAPI, analyser_sms
from contenu import CONSEILS_LEGITIME, CONSEILS_SPAM, EXEMPLES, SIGNES_SPAM
from dashboard import afficher_dashboard
from style import appliquer_style, icone

# ---------------------------------------------------------------
# 1. RÉGLAGES DE LA PAGE
# ---------------------------------------------------------------
st.set_page_config(page_title="Détecteur de spam", page_icon=":material/shield:", layout="wide")
appliquer_style()

if "historique" not in st.session_state:
    st.session_state["historique"] = []


# ---------------------------------------------------------------
# 2. BARRE LATÉRALE
# ---------------------------------------------------------------
with st.sidebar:
    st.header(":material/lightbulb: Repérer un spam")
    st.markdown(SIGNES_SPAM)
    st.divider()
    st.caption("Projet Mini-hackathon Data & IA · HETIC 2026")


# ---------------------------------------------------------------
# 3. LE GRAND TITRE
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-surtitre">Détection de spam </div>
        <div class="hero-titre">
            Un SMS suspect ?<br>
            <span class="accent">On te dit s'il faut s'en méfier.</span>
        </div>
        <div class="hero-sous-titre">Colle ton message, l'IA l'analyse et te donne les bons réflexes.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 4. ANALYSER UN SMS (à gauche) + LA SESSION (à droite)
# ---------------------------------------------------------------
def afficher_resultat(resultat):
    if resultat == "spam":
        classe, titre, conseils = "spam", f"{icone('gpp_bad')} Attention, c'est un SPAM", CONSEILS_SPAM
    else:
        classe, titre, conseils = "legitime", f"{icone('verified_user')} SMS légitime", CONSEILS_LEGITIME
    lignes = "".join(f'<p class="conseil">{icone(nom)}<span>{txt}</span></p>' for nom, txt in conseils)
    st.markdown(f'<div class="carte {classe}"><h2>{titre}</h2>{lignes}</div>', unsafe_allow_html=True)


gauche, droite = st.columns([3, 2], gap="large")

with gauche:
    colonnes = st.columns(len(EXEMPLES))
    for colonne, (nom, icone_bouton, texte_exemple) in zip(colonnes, EXEMPLES):
        if colonne.button(nom, icon=icone_bouton, width="stretch"):
            st.session_state["sms"] = texte_exemple

    texte = st.text_area(
        "Ton SMS", key="sms", height=120, max_chars=1000,
        placeholder="Exemple : Félicitations ! Vous avez gagné un bon d'achat de 500 €...",
    )

    if st.button("Analyser le SMS", icon=":material/search:", type="primary", width="stretch"):
        if not texte.strip():
            st.warning("Écris ou colle un SMS avant de lancer l'analyse.", icon=":material/edit:")
        else:
            try:
                resultat = analyser_sms(texte)
            except ErreurAPI as erreur:
                st.error(str(erreur), icon=":material/error:")
            else:
                apercu = texte if len(texte) <= 40 else texte[:40] + "…"
                st.session_state["historique"].insert(0, {"Résultat": resultat.capitalize(), "SMS": apercu})
                afficher_resultat(resultat)

with droite:
    st.subheader(":material/history: Ta session")
    historique = st.session_state["historique"]
    nb_spams = sum(1 for h in historique if h["Résultat"] == "Spam")
    c1, c2 = st.columns(2)
    c1.metric(":material/forum: SMS analysés", len(historique))
    c2.metric(":material/gpp_bad: Spams détectés", nb_spams)
    if historique:
        st.dataframe(pd.DataFrame(historique), hide_index=True, width="stretch", height=180)
    else:
        st.caption("Analyse un SMS pour le voir apparaître ici.")


# ---------------------------------------------------------------
# 5. LE TABLEAU DE BORD
# ---------------------------------------------------------------
st.divider()
afficher_dashboard()
