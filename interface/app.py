"""
AWB — Agent Crédit Consommation
Interface Streamlit — v2.0

Lancement :
    cd appetence-model && streamlit run interface/app.py
"""
import sys, os, json, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from agent.config_agent import UI_CONFIG
from agent.orchestrator import AgentOrchestrator
from interface.components.client_loader import (
    load_clients_database, load_enrichment_files, find_client
)
from interface.components.scoring_runner import ScoringRunner
from interface.components.fiche_display import display_fiche
from interface.components.pdf_generator import generate_pdf

# ═══════════════════════════════════════════════
# Config page
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="AWB — Agent Crédit Conso",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS
_css_path = Path(__file__).parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════
for key, default in [
    ("fiche", None),
    ("scoring_runner", None),
    ("agent", None),
    ("models_ok", False),
    ("history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Charger les modèles une seule fois
if not st.session_state.models_ok:
    try:
        st.session_state.scoring_runner = ScoringRunner()
        st.session_state.agent = AgentOrchestrator()
        st.session_state.models_ok = True
    except FileNotFoundError as e:
        st.session_state.models_ok = False
        st.session_state._models_error = str(e)

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px'>
        <div style='font-size:2rem;font-weight:900;color:white;letter-spacing:-0.04em'>AWB</div>
        <div style='font-size:0.7rem;color:#94A3B8;letter-spacing:0.15em;text-transform:uppercase'>
            Agent Crédit Conso
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Statut modèles
    if st.session_state.models_ok:
        st.markdown("""
        <div style='background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.4);
                    border-radius:8px;padding:10px 14px;margin-bottom:12px'>
            <div style='font-size:0.75rem;font-weight:700;color:#34D399'>✓ MODÈLES CHARGÉS</div>
            <div style='font-size:0.7rem;color:#94A3B8;margin-top:3px'>CatBoost · LGBM · Agent IA</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.4);
                    border-radius:8px;padding:10px 14px;margin-bottom:12px'>
            <div style='font-size:0.75rem;font-weight:700;color:#FBBF24'>⚠ MODÈLES MANQUANTS</div>
            <div style='font-size:0.7rem;color:#94A3B8;margin-top:3px'>Simulation uniquement</div>
        </div>""", unsafe_allow_html=True)

    # Statut LLM
    try:
        import requests as _r
        _ok = _r.get("http://localhost:11434", timeout=2).status_code == 200
    except Exception:
        _ok = False

    if _ok:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.4);
                    border-radius:8px;padding:10px 14px;margin-bottom:16px'>
            <div style='font-size:0.75rem;font-weight:700;color:#A5B4FC'>✦ MISTRAL 7B EN LIGNE</div>
            <div style='font-size:0.7rem;color:#94A3B8;margin-top:3px'>Narration IA active</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(148,163,184,0.1);border:1px solid rgba(148,163,184,0.3);
                    border-radius:8px;padding:10px 14px;margin-bottom:16px'>
            <div style='font-size:0.75rem;font-weight:700;color:#94A3B8'>○ MISTRAL HORS LIGNE</div>
            <div style='font-size:0.7rem;color:#64748B;margin-top:3px'>Fallback templates</div>
        </div>""", unsafe_allow_html=True)

    # Upload fichier
    st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#94A3B8;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px'>Base de données</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Charger un fichier Excel",
        type=["xlsx", "xls"],
        help="Laissez vide pour utiliser la base par défaut",
        label_visibility="collapsed",
    )

    excel_path = UI_CONFIG["default_excel_path"]
    if uploaded_file:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        excel_path = temp_path

    df_clients = load_clients_database(excel_path)
    dfs_enrichment = load_enrichment_files()

    if df_clients is not None:
        st.metric("Clients", f"{len(df_clients):,}")
    else:
        st.markdown("<div style='color:#F87171;font-size:0.8rem'>Base introuvable</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline description
    st.markdown("""
    <div style='font-size:0.75rem;font-weight:700;color:#94A3B8;letter-spacing:0.08em;
                text-transform:uppercase;margin-bottom:10px'>Pipeline IA</div>
    """, unsafe_allow_html=True)

    pipeline_steps = [
        ("1", "Features Temporelles", "91 jours → 55 signaux comportementaux"),
        ("2", "CatBoost + LGBM", "Ensemble → score appétence calibré"),
        ("3", "SHAP", "Explicabilité des décisions"),
        ("4", "Expert Risque", "Règles AWB + note A→E"),
        ("5", "Expert Commercial", "Grille tarifaire"),
        ("6", "Mistral 7B", "Narration personnalisée"),
    ]
    for num, label, sub in pipeline_steps:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;padding:7px 0;
                    border-bottom:1px solid rgba(255,255,255,0.07)'>
            <div style='width:22px;height:22px;background:rgba(200,16,46,0.8);border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.65rem;font-weight:800;color:white;flex-shrink:0'>{num}</div>
            <div>
                <div style='font-size:0.82rem;font-weight:600;color:#E2E8F0'>{label}</div>
                <div style='font-size:0.7rem;color:#64748B'>{sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.history:
        st.markdown(f"<div style='font-size:0.75rem;color:#64748B'>Fiches générées : {len(st.session_state.history)}</div>", unsafe_allow_html=True)

    if st.session_state.fiche:
        if st.button("← Nouvelle analyse", use_container_width=True):
            st.session_state.fiche = None
            st.rerun()


# ═══════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════
st.markdown("""
<div class="awb-navbar">
    <div>
        <div class="awb-navbar-title">🏦 Attijariwafa Bank</div>
        <div class="awb-navbar-sub">Agent d'Analyse Crédit Consommation — Propulsé par CatBoost · LGBM · Mistral 7B</div>
    </div>
    <div class="awb-badge">USAGE INTERNE</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# VUE PRINCIPALE
# ═══════════════════════════════════════════════
if st.session_state.fiche is None:
    _render_search_screen(df_clients, dfs_enrichment, excel_path)
else:
    _render_fiche_screen(st.session_state.fiche)


# ═══════════════════════════════════════════════
# FONCTIONS D'ÉCRAN
# ═══════════════════════════════════════════════
def _render_search_screen(df_clients, dfs_enrichment, excel_path):
    tab_search, tab_manual = st.tabs(["🔍  Recherche client", "✍️  Saisie manuelle (simulation)"])

    with tab_search:
        _tab_search(df_clients, dfs_enrichment)

    with tab_manual:
        _tab_manual()


def _tab_search(df_clients, dfs_enrichment):
    col_center, _, _ = st.columns([2, 1, 1])
    with col_center:
        st.markdown("""
        <div style='background:white;border-radius:16px;padding:36px 40px;
                    box-shadow:0 4px 24px rgba(0,0,0,0.06);
                    border:1px solid #E2E8F0;margin-top:20px'>
            <div style='font-size:1.1rem;font-weight:700;color:#1C2333;margin-bottom:4px'>
                Recherche par identifiant
            </div>
            <div style='font-size:0.85rem;color:#64748B;margin-bottom:20px'>
                Saisissez le numéro client pour lancer l'analyse complète
            </div>
        """, unsafe_allow_html=True)

        id_input = st.text_input(
            "Numéro client",
            placeholder="Ex : 123456",
            label_visibility="collapsed",
        )
        go = st.button("Analyser ce client", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if go and id_input.strip():
        if df_clients is None:
            st.error("Base clients introuvable. Vérifiez le chemin dans la sidebar.")
            return

        with st.spinner("Recherche du client..."):
            client = find_client(df_clients, dfs_enrichment, id_input.strip())

        if client is None:
            st.error(f"Client **{id_input}** introuvable dans la base.")
            return

        _run_full_pipeline(client)

    elif go:
        st.warning("Veuillez saisir un identifiant client.")


def _tab_manual():
    st.markdown("""
    <div style='background:white;border-radius:12px;padding:24px 28px;
                border:1px solid #E2E8F0;box-shadow:0 2px 12px rgba(0,0,0,0.05);
                margin-bottom:16px'>
        <div style='font-size:1rem;font-weight:700;color:#1C2333;margin-bottom:4px'>
            ✍️ Simulation Agent IA
        </div>
        <div style='font-size:0.85rem;color:#64748B'>
            Renseignez les caractéristiques d'un profil pour simuler l'analyse complète de l'Agent.
            Le score ML est saisi directement — aucun pipeline ML n'est déclenché.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("manual_form", clear_on_submit=False):

        # ── Section 1 : Signalétique ──
        st.markdown("#### 👤 Signalétique")
        c1, c2, c3, c4 = st.columns(4)
        with c1: age = st.number_input("Âge", 18, 90, 35)
        with c2: type_revenu = st.selectbox("Type de revenu", ["SALARIE", "RETRAITE", "PROFESSION_LIBERALE", "COMMERCANT", "AUTRE"])
        with c3: segment = st.selectbox("Segment client", ["MASS", "PREMIUM", "PRIVE"])
        with c4: revenu_principal = st.number_input("Revenu (MAD/mois)", 0, 500000, 6000, step=500)

        # ── Section 2 : Engagements ──
        st.markdown("#### 💳 Engagements actuels")
        c1, c2 = st.columns(2)
        with c1: mensualite_immo = st.number_input("Mensualité crédit immo (MAD)", 0, 50000, 0, step=100)
        with c2: total_mensualite_actif = st.number_input("Autres mensualités actives (MAD)", 0, 50000, 0, step=100)

        taux_calcule = (mensualite_immo + total_mensualite_actif) / max(revenu_principal, 1)
        color_te = "#10B981" if taux_calcule < 0.33 else "#F59E0B" if taux_calcule < 0.40 else "#EF4444"
        st.markdown(
            f"<div style='padding:8px 14px;background:#F8FAFC;border-radius:8px;border:1px solid #E2E8F0;"
            f"font-size:0.85rem;color:#64748B'>Taux d'endettement calculé : "
            f"<b style='color:{color_te}'>{taux_calcule:.1%}</b> "
            f"{'✅ Conforme BAM' if taux_calcule <= 0.40 else '🚫 Dépasse le plafond BAM (40%)'}</div>",
            unsafe_allow_html=True,
        )

        # ── Section 3 : Comportement compte ──
        st.markdown("#### 🏦 Comportement bancaire (3 derniers mois)")
        c1, c2, c3 = st.columns(3)
        with c1:
            solde_moyen = st.number_input("Solde moyen (MAD)", -50000, 500000, 5000, step=500)
            solde_min   = st.number_input("Solde minimal (MAD)", -100000, 500000, 1000, step=500)
        with c2:
            solde_std      = st.number_input("Volatilité solde (std)", 0, 100000, 1000, step=100)
            solde_max      = st.number_input("Solde maximal (MAD)", 0, 1000000, 8000, step=500)
        with c3:
            solde_tendance   = st.number_input("Tendance (MAD/jour)", -500, 500, 15)
            solde_nb_negatif = st.number_input("Jours de découvert", 0, 90, 0)

        # ── Section 4 : Appétence ──
        st.markdown("#### 📊 Signaux d'appétence")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1: count_simul = st.number_input("Simulations crédit (total)", 0, 100, 3)
        with c2: count_simul_mois_n_1 = st.number_input("Simulations (dernier mois)", 0, 50, 1)
        with c3:
            proba_pct = st.slider(
                "Score d'appétence ML simulé",
                0, 100, 72,
                help="En conditions réelles, ce score est calculé par l'ensemble CatBoost + LGBM"
            )
            proba_ml = proba_pct / 100.0

        st.markdown(
            f"<div style='padding:8px 14px;background:#F8FAFC;border-radius:8px;"
            f"border:1px solid #E2E8F0;font-size:0.85rem;color:#64748B;margin-top:4px'>"
            f"Score affiché : <b>{proba_pct}%</b> → "
            f"{'Forte appétence' if proba_ml >= 0.7 else 'Appétence modérée' if proba_ml >= 0.4 else 'Faible appétence'}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        submit = st.form_submit_button("🚀  Lancer l'analyse Agent IA", type="primary", use_container_width=True)

    if submit:
        mensualite_totale = float(mensualite_immo) + float(total_mensualite_actif)
        client_row = {
            "id_client": "SIMULATION",
            "age": int(age),
            "type_revenu": type_revenu,
            "segment": segment,
            "revenu_principal": float(revenu_principal),
            "mensualite_immo": float(mensualite_immo),
            "total_mensualite_actif": float(total_mensualite_actif),
            "total_mensualite_conso_immo": mensualite_totale,
            "taux_endettement": mensualite_totale / max(float(revenu_principal), 1),
            "solde_moyen": float(solde_moyen),
            "solde_std": float(solde_std),
            "solde_tendance": float(solde_tendance),
            "solde_nb_negatif": int(solde_nb_negatif),
            "solde_min": float(solde_min),
            "solde_max": float(solde_max),
            "solde_volatilite": float(solde_std) / max(abs(float(solde_moyen)), 1),
            "solde_dernier_jour": float(solde_moyen),
            "solde_variation_moy": float(solde_tendance),
            "count_simul": int(count_simul),
            "count_simul_mois_n_1": int(count_simul_mois_n_1),
            "duree_restante_ponderee": 0,
            "marge_mensuelle": float(revenu_principal) - mensualite_totale,
            "capacite_credit_supp": max(0, float(revenu_principal) * 0.33 - mensualite_totale),
            "simul_par_kMAD": int(count_simul) / (float(revenu_principal) / 1000 + 1),
            "ratio_simul_recents": int(count_simul_mois_n_1) / (int(count_simul) + 1),
            "score_fragilite": int(solde_nb_negatif) * (float(solde_std) / max(abs(float(solde_moyen)), 1)),
            "solde_acceleration": 0.0,
        }
        shap_simul = _build_simul_shap(client_row)
        _run_agent_only(client_row, proba_ml, shap_simul)


def _run_full_pipeline(client_row):
    """Exécute Features → CatBoost/LGBM → SHAP → Agent IA avec indicateur de progression."""
    steps = [
        (20,  "Pipeline ML : feature engineering (91 jours → 55 signaux)..."),
        (55,  "Pipeline ML : CatBoost/LGBM scoring + SHAP..."),
        (70,  "Agent IA : Analyse risque (règles AWB)..."),
        (82,  "Agent IA : Calcul de l'offre commerciale..."),
        (93,  "Agent IA : Narration Mistral 7B..."),
        (100, "✅ Analyse terminée"),
    ]

    progress = st.progress(0, text=steps[0][1])
    try:
        progress.progress(steps[0][0], text=steps[0][1])
        scoring = st.session_state.scoring_runner.score_client(client_row)

        for pct, txt in steps[1:4]:
            progress.progress(pct, text=txt)

        fiche = st.session_state.agent.run(
            scoring["client_row"],
            scoring["proba"],
            scoring["top_5_shap"],
        )
        progress.progress(100, text=steps[-1][1])
        fiche["_simulation"] = False
        st.session_state.fiche = fiche
        st.session_state.history.append(fiche["metadata"]["id_client"])
        st.rerun()

    except Exception as e:
        progress.empty()
        st.error(f"**Erreur** : {e}")
        with st.expander("Détails techniques"):
            st.code(traceback.format_exc())


def _run_agent_only(client_row, proba_ml, shap_simul):
    """Exécute uniquement l'Agent IA (sans pipeline ML)."""
    steps = [
        (30,  "Agent IA : Enrichissement du profil..."),
        (55,  "Agent IA : Analyse risque (règles AWB)..."),
        (75,  "Agent IA : Calcul de l'offre commerciale..."),
        (92,  "Agent IA : Narration Mistral 7B..."),
        (100, "✅ Simulation terminée"),
    ]

    progress = st.progress(0, text=steps[0][1])
    try:
        for pct, txt in steps[:-1]:
            progress.progress(pct, text=txt)

        fiche = st.session_state.agent.run(client_row, proba_ml, shap_simul)
        progress.progress(100, text=steps[-1][1])
        fiche["_simulation"] = True
        st.session_state.fiche = fiche
        st.session_state.history.append("SIM")
        st.rerun()

    except Exception as e:
        progress.empty()
        st.error(f"**Erreur** : {e}")
        with st.expander("Détails techniques"):
            st.code(traceback.format_exc())


def _render_fiche_screen(fiche):
    """Affiche la fiche complète avec barre d'actions."""
    # Actions toolbar
    col_b, col_pdf, col_json, col_space = st.columns([2, 1.2, 1, 3])
    with col_b:
        if st.button("← Nouvelle analyse", use_container_width=True):
            st.session_state.fiche = None
            st.rerun()
    with col_pdf:
        try:
            pdf_bytes = generate_pdf(fiche)
            ext = "pdf"
        except Exception:
            pdf_bytes = json.dumps(fiche, ensure_ascii=False, indent=2, default=str).encode()
            ext = "txt"
        st.download_button(
            "📄 Télécharger PDF",
            data=pdf_bytes,
            file_name=f"fiche_{fiche['metadata']['id_client']}.{ext}",
            mime=f"application/{ext}",
            use_container_width=True,
        )
    with col_json:
        st.download_button(
            "{ } JSON",
            data=json.dumps(fiche, ensure_ascii=False, indent=2, default=str).encode(),
            file_name=f"fiche_{fiche['metadata']['id_client']}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    display_fiche(fiche)


def _build_simul_shap(row):
    """SHAP synthétique pour la saisie manuelle."""
    te = row["taux_endettement"]
    rv = row["revenu_principal"]
    sm = row["solde_moyen"]
    nb = row["solde_nb_negatif"]
    cs = row["count_simul"]
    return [
        {"rank": 1, "feature": "revenu_principal", "feature_label": "Revenu principal",
         "feature_value": rv, "shap_value": 0.13 if rv >= 5000 else -0.09,
         "direction": "+" if rv >= 5000 else "-", "contribution_pct": 30.0},
        {"rank": 2, "feature": "taux_endettement", "feature_label": "Taux d'endettement",
         "feature_value": te, "shap_value": -0.09 if te > 0.3 else 0.07,
         "direction": "-" if te > 0.3 else "+", "contribution_pct": 22.0},
        {"rank": 3, "feature": "solde_moyen", "feature_label": "Solde moyen 3 mois",
         "feature_value": sm, "shap_value": 0.08 if sm >= 3000 else -0.05,
         "direction": "+" if sm >= 3000 else "-", "contribution_pct": 18.0},
        {"rank": 4, "feature": "count_simul", "feature_label": "Nombre de simulations",
         "feature_value": float(cs), "shap_value": round(0.05 * min(cs, 6) / 6, 4),
         "direction": "+", "contribution_pct": 15.0},
        {"rank": 5, "feature": "solde_nb_negatif", "feature_label": "Jours de découvert",
         "feature_value": float(nb), "shap_value": round(-0.04 * min(nb, 10) / 10, 4),
         "direction": "-" if nb > 0 else "+", "contribution_pct": 10.0},
    ]
