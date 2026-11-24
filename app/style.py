"""Tout le style (CSS) de l'interface."""
import streamlit as st

ROUGE = "#dc2626"
VERT = "#16a34a"

CSS = """
    <style>
    /* Couleurs semi-transparentes : elles marchent en mode clair ET en mode sombre */
    /* --- Le grand titre --- */
    .hero { text-align: center; padding: 2rem 0 1.5rem 0; }
    .hero-surtitre {
        color: #e5484d; font-size: 0.85rem; font-weight: 600;
        letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 1rem;
    }
    .hero-titre {
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
        font-size: clamp(2.4rem, 5vw, 4.2rem); font-weight: 700;
        line-height: 1.05; letter-spacing: -0.035em;
    }
    .hero-titre .accent { color: #e5484d; }
    .hero-sous-titre { opacity: 0.65; font-size: 1.1rem; margin-top: 1.2rem; }
    .carte {
        border-radius: 16px; padding: 1.4rem 1.6rem; margin-top: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .carte h2 {
        margin: 0 0 0.4rem 0; font-size: 1.6rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .spam { background: rgba(220, 38, 38, 0.12); border-left: 6px solid #dc2626; }
    .spam h2 { color: #ef4444; }
    .legitime { background: rgba(22, 163, 74, 0.12); border-left: 6px solid #16a34a; }
    .legitime h2 { color: #22c55e; }
    .conseil { margin: 0.45rem 0; display: flex; align-items: center; gap: 0.6rem; }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px; padding: 0.8rem 1rem;
    }
    /* Icônes Material (la police est fournie par Streamlit) */
    .mi {
        font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal;
        font-size: 1.25em; line-height: 1; display: inline-block; flex-shrink: 0;
        -webkit-font-feature-settings: "liga"; font-feature-settings: "liga";
    }
    </style>
"""


def appliquer_style():
    """Injecte le CSS dans la page."""
    st.markdown(CSS, unsafe_allow_html=True)


def icone(nom):
    """Renvoie une icône Material pour le HTML (liste : fonts.google.com/icons)."""
    return f'<span class="mi">{nom}</span>'
