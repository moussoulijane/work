"""
Application Streamlit — Agent Crédit Conso AWB

Usage :
    streamlit run interface/app.py

Écrans :
1. Recherche client (page d'accueil)
2. Fiche client (résultat de l'analyse)
3. Historique (fiches déjà générées)
"""
import streamlit as st
import sys
from pathlib import Path

# Ajouter le root au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.components.client_loader import (
    load_clients_database, load_enrichment_files, find_client
)
from interface.components.scoring_runner import ScoringRunner
from interface.components.fiche_display import display_fiche
from interface.components.pdf_generator import generate_pdf
from agent.orchestrator import AgentOrchestrator
from agent.config_agent import UI_CONFIG


# ════════════════════════════════════════════════════════════
# Configuration de la page
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout'],
    initial_sidebar_state='collapsed',
)

# CSS custom
st.markdown("""
<style>
    .main-title { color: #C8102E; font-size: 2rem; font-weight: bold; }
    .note-a { background: #28a745; color: white; padding: 8px 16px; border-radius: 8px; font-weight: bold; }
    .note-b { background: #5cb85c; color: white; padding: 8px 16px; border-radius: 8px; font-weight: bold; }
    .note-c { background: #f0ad4e; color: white; padding: 8px 16px; border-radius: 8px; font-weight: bold; }
    .note-d { background: #ff7043; color: white; padding: 8px 16px; border-radius: 8px; font-weight: bold; }
    .note-e { background: #d9534f; color: white; padding: 8px 16px; border-radius: 8px; font-weight: bold; }
    .offer-box { background: #f8f9fa; border-left: 4px solid #C8102E; padding: 20px; border-radius: 8px; }
    .script-box { background: #fff3cd; border: 1px solid #ffc107; padding: 20px; border-radius: 8px; font-style: italic; }
    .point-fort { color: #28a745; }
    .point-attention { color: #f0ad4e; }
    .red-flag { color: #d9534f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Initialisation session
# ════════════════════════════════════════════════════════════
if 'fiche' not in st.session_state:
    st.session_state.fiche = None

if 'scoring_runner' not in st.session_state:
    try:
        st.session_state.scoring_runner = ScoringRunner()
    except FileNotFoundError as e:
        st.error(f"**Erreur d'initialisation** : {e}")
        st.stop()

if 'agent' not in st.session_state:
    st.session_state.agent = AgentOrchestrator()


# ════════════════════════════════════════════════════════════
# En-tête
# ════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.markdown(
        '<div class="main-title">🏦 Attijariwafa Bank — Agent Crédit Conso</div>',
        unsafe_allow_html=True
    )
    st.caption("Analyse automatisée : risque + offre commerciale + argumentation")

st.divider()


# ════════════════════════════════════════════════════════════
# Sidebar — Configuration
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Upload de fichier Excel custom
    uploaded_file = st.file_uploader(
        "Base clients Excel",
        type=['xlsx', 'xls'],
        help="Laisser vide pour utiliser la base par défaut"
    )
    
    excel_path = UI_CONFIG['default_excel_path']
    if uploaded_file is not None:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())
        excel_path = temp_path
        st.success(f"✅ Fichier chargé : {uploaded_file.name}")
    
    # Charger la base
    df_clients = load_clients_database(excel_path)
    if df_clients is None:
        st.error(f"❌ Base introuvable : {excel_path}")
        st.stop()
    
    st.metric("Clients en base", f"{len(df_clients):,}")
    st.caption(f"📁 {Path(excel_path).name}")
    
    # Charger les fichiers d'enrichissement
    dfs_enrichment = load_enrichment_files()
    
    st.divider()
    st.caption("💡 L'analyse combine :")
    st.caption("• LSTM (91 jours de soldes)")
    st.caption("• CatBoost (52 features)")
    st.caption("• Règles risque AWB")
    st.caption("• Grille tarifaire")
    st.caption("• Narration Mistral 7B")


# ════════════════════════════════════════════════════════════
# Écran principal — Recherche / Saisie Manuelle
# ════════════════════════════════════════════════════════════
if st.session_state.fiche is None:
    
    tab_search, tab_manual = st.tabs(["🔍 Recherche (Base Excel)", "✍️ Saisie Manuelle (Simulation)"])
    
    # --- TAB 1: RECHERCHE ---
    with tab_search:
        st.subheader("🔍 Rechercher un client")
        col1, col2 = st.columns([3, 1])
        with col1:
            id_input = st.text_input("Numéro client", placeholder="Ex: 12345", label_visibility="collapsed")
        with col2:
            analyser = st.button("Analyser", type="primary", use_container_width=True)
        
        if analyser and id_input:
            with st.spinner("Recherche en cours..."):
                client = find_client(df_clients, dfs_enrichment, id_input)
            
            if client is None:
                st.error(f"❌ Client {id_input} introuvable dans la base.")
            else:
                progress = st.progress(0, text="Initialisation...")
                try:
                    progress.progress(20, text="⏳ Pipeline ML (LSTM + CatBoost + SHAP)...")
                    scoring_result = st.session_state.scoring_runner.score_client(client)
                    progress.progress(60, text="⏳ Analyse risque...")
                    progress.progress(75, text="⏳ Génération de l'offre commerciale...")
                    progress.progress(85, text="⏳ Rédaction par l'agent IA (Mistral)...")
                    fiche = st.session_state.agent.run(
                        scoring_result['client_row'],
                        scoring_result['proba'],
                        scoring_result['top_5_shap'],
                        scoring_result['lstm_shap_aggregated']
                    )
                    progress.progress(100, text="✅ Fiche générée")
                    st.session_state.fiche = fiche
                    st.rerun()
                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Erreur : {e}")
                    import traceback
                    with st.expander("Détails techniques"): st.code(traceback.format_exc())

    # --- TAB 2: SAISIE MANUELLE ---
    with tab_manual:
        st.subheader("✍️ Saisie manuelle des caractéristiques (Simulation Agent)")
        st.info("Simulez l'entrée des features pour exécuter unitairement l'Agent IA (sans repasser par le pipeline ML).")
        
        with st.form("manual_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.number_input("Âge", min_value=18, max_value=90, value=35)
                type_revenu = st.selectbox("Type revenu", ["SALARIE", "RETRAITE", "PROFESSION_LIBERALE", "COMMERCANT", "AUTRE"])
                segment = st.selectbox("Segment", ["MASS", "PREMIUM", "PRIVE"])
                revenu_principal = st.number_input("Revenu principal (MAD/mois)", min_value=0, value=6000)
                mensualite_immo = st.number_input("Mensualité immo actuelle", min_value=0, value=0)
                total_mensualite_actif = st.number_input("Autres mensualités", min_value=0, value=0)
            
            with col_b:
                solde_moyen = st.number_input("Solde moyen (3 mois)", value=5000)
                solde_std = st.number_input("Volatilité solde (std)", value=1000)
                solde_tendance = st.number_input("Tendance solde (MAD/jour)", value=15)
                solde_nb_negatif = st.number_input("Jours de découvert (3 mois)", min_value=0, value=0)
                solde_min = st.number_input("Solde minimal", value=1000)
                solde_max = st.number_input("Solde maximal", value=8000)
                
            st.divider()
            
            col_c, col_d = st.columns(2)
            with col_c:
                count_simul = st.number_input("Nb simulations (total)", min_value=0, value=3)
                count_simul_mois_n_1 = st.number_input("Nb simulations (dernier mois)", min_value=0, value=1)
            with col_d:
                proba_ml = st.slider("Score Appétence ML simulé (0-100%)", min_value=0, max_value=100, value=75) / 100.0
            
            submit_manual = st.form_submit_button("Lancer l'Agent IA", type="primary", use_container_width=True)
            
        if submit_manual:
            progress = st.progress(0, text="Construction du profil...")
            try:
                # Construire le client_row depuis les inputs du formulaire
                client_row_manual = {
                    'id_client': 'SIMULATION',
                    'age': age,
                    'type_revenu': type_revenu,
                    'segment': segment,
                    'revenu_principal': float(revenu_principal),
                    'mensualite_immo': float(mensualite_immo),
                    'total_mensualite_actif': float(total_mensualite_actif),
                    'total_mensualite_conso_immo': float(mensualite_immo) + float(total_mensualite_actif),
                    'taux_endettement': (float(mensualite_immo) + float(total_mensualite_actif)) / max(float(revenu_principal), 1),
                    'solde_moyen': float(solde_moyen),
                    'solde_std': float(solde_std),
                    'solde_tendance': float(solde_tendance),
                    'solde_nb_negatif': int(solde_nb_negatif),
                    'solde_min': float(solde_min),
                    'solde_max': float(solde_max),
                    'solde_volatilite': float(solde_std) / max(abs(float(solde_moyen)), 1),
                    'solde_dernier_jour': float(solde_moyen),
                    'solde_variation_moy': float(solde_tendance),
                    'count_simul': int(count_simul),
                    'count_simul_mois_n_1': int(count_simul_mois_n_1),
                    'duree_restante_ponderee': 0,
                    # Avancées
                    'marge_mensuelle': float(revenu_principal) - float(mensualite_immo) - float(total_mensualite_actif),
                    'capacite_credit_supp': max(0, float(revenu_principal) * 0.33 - float(mensualite_immo) - float(total_mensualite_actif)),
                    'simul_par_kMAD': int(count_simul) / (float(revenu_principal) / 1000 + 1),
                    'ratio_simul_recents': int(count_simul_mois_n_1) / (int(count_simul) + 1),
                    'score_fragilite': int(solde_nb_negatif) * (float(solde_std) / max(abs(float(solde_moyen)), 1)),
                    'solde_acceleration': 0.0,
                }
                # SHAP simulé : features les plus impactantes basées sur les valeurs saisies
                top_5_simul = [
                    {'rank': 1, 'feature': 'revenu_principal', 'feature_label': 'Revenu principal',
                     'feature_value': float(revenu_principal), 'shap_value': 0.12 if revenu_principal >= 5000 else -0.10,
                     'direction': '+' if revenu_principal >= 5000 else '-', 'contribution_pct': 30.0},
                    {'rank': 2, 'feature': 'taux_endettement', 'feature_label': "Taux d'endettement",
                     'feature_value': client_row_manual['taux_endettement'],
                     'shap_value': -0.08 if client_row_manual['taux_endettement'] > 0.3 else 0.06,
                     'direction': '-' if client_row_manual['taux_endettement'] > 0.3 else '+', 'contribution_pct': 22.0},
                    {'rank': 3, 'feature': 'solde_moyen', 'feature_label': 'Solde moyen 3 mois',
                     'feature_value': float(solde_moyen), 'shap_value': 0.07 if solde_moyen >= 3000 else -0.05,
                     'direction': '+' if solde_moyen >= 3000 else '-', 'contribution_pct': 18.0},
                    {'rank': 4, 'feature': 'count_simul', 'feature_label': 'Nombre de simulations',
                     'feature_value': float(count_simul), 'shap_value': 0.05 * min(count_simul, 5) / 5,
                     'direction': '+', 'contribution_pct': 15.0},
                    {'rank': 5, 'feature': 'solde_nb_negatif', 'feature_label': 'Jours de découvert',
                     'feature_value': float(solde_nb_negatif), 'shap_value': -0.04 * min(solde_nb_negatif, 10) / 10,
                     'direction': '-' if solde_nb_negatif > 0 else '+', 'contribution_pct': 10.0},
                ]

                progress.progress(40, text="⏳ Analyse risque...")
                progress.progress(70, text="⏳ Génération de l'offre commerciale...")
                progress.progress(85, text="⏳ Rédaction par l'agent IA (Mistral)...")

                fiche = st.session_state.agent.run(
                    client_row_manual,
                    proba_ml,
                    top_5_simul,
                    lstm_shap_aggregated=0.0,
                )
                # Injecter le score ML simulé dans la fiche pour l'affichage
                fiche['_simulation'] = True
                fiche['_proba_ml'] = proba_ml

                progress.progress(100, text="✅ Fiche générée")
                st.session_state.fiche = fiche
                st.rerun()

            except Exception as e:
                progress.empty()
                st.error(f"❌ Erreur : {e}")
                import traceback
                with st.expander("Détails techniques"):
                    st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════
# Écran Fiche Client
# ════════════════════════════════════════════════════════════
else:
    fiche = st.session_state.fiche
    
    # Boutons d'action en haut
    col_back, col_pdf, col_json = st.columns([2, 1, 1])
    with col_back:
        if st.button("← Retour à la recherche", use_container_width=True):
            st.session_state.fiche = None
            st.rerun()
    with col_pdf:
        pdf_bytes = generate_pdf(fiche)
        st.download_button(
            "📄 Télécharger PDF",
            data=pdf_bytes,
            file_name=f"fiche_client_{fiche['metadata']['id_client']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with col_json:
        import json
        json_str = json.dumps(fiche, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            "💾 JSON",
            data=json_str,
            file_name=f"fiche_client_{fiche['metadata']['id_client']}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.divider()
    
    # Affichage de la fiche
    display_fiche(fiche)

