"""
Point d'entrée Streamlit — AWB Agent Crédit Conso.

Lancement :
    cd appetence-model
    streamlit run interface/app.py

Prérequis :
    - python main.py train  (pour générer les artefacts ML)
    - ollama serve + ollama pull mistral  (pour la narration IA)
    - pip install streamlit reportlab pyyaml
"""
import sys
import os

# Ajouter la racine du projet au path pour que les imports src/ et agent/ fonctionnent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from agent.config_agent import UI_CONFIG
from interface.components.client_loader import load_clients_file, find_client
from interface.components.scoring_runner import run_pipeline
from interface.components.agent_runner import run_agent
from interface.components.fiche_display import (
    display_header, display_profil, display_risk_analysis,
    display_offers, display_narration, display_shap,
)
from interface.components.pdf_generator import generate_pdf, REPORTLAB_AVAILABLE

# ── Configuration de la page ──
st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout'],
    initial_sidebar_state='expanded',
)

# ── Injection du CSS ──
css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path, 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.image(
        os.path.join(os.path.dirname(__file__), 'assets', 'awb_logo.png')
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'assets', 'awb_logo.png'))
        else "https://placeholder.pics/svg/200x60/C8102E/FFFFFF/AWB",
        use_column_width=True,
    )
    st.markdown("---")
    st.markdown("### Paramètres")

    excel_path = st.text_input(
        "Fichier clients",
        value=UI_CONFIG['default_excel_path'],
        help="Chemin vers le fichier Excel ou CSV contenant les clients",
    )

    models_dir = st.text_input(
        "Dossier modèles",
        value="models",
        help="Dossier contenant lstm_encoder.pt, *.cbm, etc.",
    )

    use_llm = st.checkbox(
        "Activer Mistral (Ollama)",
        value=True,
        help="Si décoché, utilise les templates de fallback (pas de LLM)",
    )

    st.markdown("---")
    st.markdown("**À propos**")
    st.caption(
        "Interface de scoring crédit AWB.\n"
        "Pipeline : LSTM + CatBoost + Agent IA.\n"
        "Usage interne uniquement."
    )

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
st.markdown(
    f"<h1 style='color:#C8102E;margin-bottom:0'>AWB — Agent Crédit Consommation</h1>",
    unsafe_allow_html=True,
)
st.markdown("Saisissez un identifiant client pour générer sa fiche de scoring complète.")
st.markdown("---")

# ── Formulaire de saisie ──
col_input, col_btn = st.columns([3, 1])
with col_input:
    id_input = st.text_input(
        "Identifiant client",
        placeholder="Ex : 123456",
        key="id_client_input",
        label_visibility="collapsed",
    )
with col_btn:
    go = st.button("Analyser", type="primary", use_container_width=True)

# ── Lancement de l'analyse ──
if go and id_input.strip():
    id_client = id_input.strip()

    # Étape 1 : Charger le fichier clients
    with st.spinner("Recherche du client..."):
        try:
            df_clients = load_clients_file(excel_path)
            client_row = find_client(df_clients, id_client)
        except Exception as e:
            st.error(f"Erreur chargement fichier : {e}")
            st.stop()

    if client_row is None:
        st.warning(f"Client `{id_client}` introuvable dans `{excel_path}`.")
        st.stop()

    st.success(f"Client `{id_client}` trouvé.")

    # Étape 2 : Pipeline ML
    with st.spinner("Calcul du score ML (LSTM + CatBoost + SHAP)..."):
        try:
            ml_result = run_pipeline(client_row, models_dir=models_dir)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Erreur pipeline ML : {e}")
            st.exception(e)
            st.stop()

    # Étape 3 : Agent IA
    with st.spinner("Génération de la fiche agent IA..."):
        try:
            # Désactiver LLM si demandé
            if not use_llm:
                _orig = None
                import agent.llm_narrator as _mod
                _orig = _mod.LLMNarrator.retries.__class__
                # Forcer retries=0 → fallback immédiat
                from agent import config_agent as _cfg
                _cfg.LLM_CONFIG = dict(_cfg.LLM_CONFIG, retries=0)

            fiche = run_agent(
                client_row=client_row,
                proba=ml_result['proba'],
                top_5_shap=ml_result['top_5_shap'],
                lstm_shap_aggregated=ml_result['lstm_shap_aggregated'],
            )
        except Exception as e:
            st.error(f"Erreur agent IA : {e}")
            st.exception(e)
            st.stop()

    # ── Affichage ──
    st.markdown("---")
    display_header(fiche)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Profil & Risque", "Offre Commerciale", "Fiche Argumentée", "Facteurs SHAP"]
    )

    with tab1:
        display_profil(fiche)
        display_risk_analysis(fiche)

    with tab2:
        display_offers(fiche)

    with tab3:
        display_narration(fiche)

    with tab4:
        display_shap(fiche)

    # ── Téléchargement PDF ──
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        try:
            pdf_bytes = generate_pdf(fiche)
            ext = 'pdf' if REPORTLAB_AVAILABLE else 'txt'
            st.download_button(
                label=f"Télécharger la fiche ({ext.upper()})",
                data=pdf_bytes,
                file_name=f"fiche_{id_client}.{ext}",
                mime=f"application/{ext}",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur génération PDF : {e}")

    with col_dl2:
        import json
        json_bytes = json.dumps(fiche, ensure_ascii=False, indent=2, default=str).encode('utf-8')
        st.download_button(
            label="Télécharger la fiche (JSON)",
            data=json_bytes,
            file_name=f"fiche_{id_client}.json",
            mime="application/json",
        )

elif go and not id_input.strip():
    st.warning("Veuillez saisir un identifiant client.")
