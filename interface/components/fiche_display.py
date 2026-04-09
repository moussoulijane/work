"""
Affichage de la fiche client — design v2.
4 onglets : Dashboard · Agent IA · Offre Commerciale · Conformité
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ═══ Constantes ═══════════════════════════════════════════════
NOTE_COLORS = {"A": "#10B981", "B": "#34D399", "C": "#FBBF24", "D": "#FB923C", "E": "#EF4444"}
NOTE_GRADIENTS = {
    "A": ("135deg", "#10B981", "#059669"),
    "B": ("135deg", "#34D399", "#10B981"),
    "C": ("135deg", "#FBBF24", "#F59E0B"),
    "D": ("135deg", "#FB923C", "#EA580C"),
    "E": ("135deg", "#F87171", "#EF4444"),
}
ZONE_ICONS = {
    "STAR": "⭐", "CROISSANCE": "📈", "DORMANT": "💤",
    "PRUDENCE": "⚠️", "FIDELISATION": "🤝", "EXCLUSION": "🚫",
}
DIM_LABELS = {
    "capacite_remboursement": ("Capacité remboursement", 30, "💰"),
    "stabilite_financiere":   ("Stabilité financière",   25, "📊"),
    "profil_revenu":          ("Profil revenu",           20, "💼"),
    "historique_bancaire":    ("Historique bancaire",     15, "🏦"),
    "signal_appetence":       ("Signal appétence",        10, "📡"),
}
REGLEMENTATION = {
    "taux_endettement": {
        "titre": "Taux d'endettement — BAM",
        "reference": "Circulaire BAM n°19/G/2002",
        "plafond": 0.40,
        "seuil_vigilance": 0.33,
        "detail": (
            "Bank Al-Maghrib fixe le taux d'endettement maximal à <b>40 %</b> du revenu net "
            "mensuel pour les crédits aux particuliers. Le seuil de vigilance interne AWB est à "
            "<b>33 %</b> : au-delà, une analyse approfondie de la capacité de remboursement est requise."
        ),
    },
    "assurance": {
        "titre": "Assurance décès-invalidité",
        "reference": "Pratiques prudentielles BAM",
        "detail": (
            "Pour toute durée supérieure à <b>60 mois</b>, la souscription d'une assurance "
            "décès-invalidité est obligatoire conformément aux pratiques prudentielles en vigueur au Maroc."
        ),
    },
    "protection": {
        "titre": "Protection du consommateur",
        "reference": "Loi n°31-08, art. 102-120",
        "detail": (
            "Le chargé de clientèle doit informer le client sur le <b>TEG</b>, le coût total du crédit, "
            "les conditions de remboursement anticipé et les garanties associées, avant toute signature."
        ),
    },
    "devoir_conseil": {
        "titre": "Devoir de conseil",
        "reference": "Charte éthique AWB / orientations BAM",
        "detail": (
            "L'établissement est tenu de s'assurer que le crédit proposé est <b>adapté à la situation "
            "financière réelle du client</b>. En cas de doute sur la capacité de remboursement, "
            "une instruction complémentaire est requise avant décision."
        ),
    },
    "teg": {
        "titre": "TEG et transparence tarifaire",
        "reference": "Circulaire BAM n°9/G/2014",
        "detail": (
            "Le Taux Effectif Global (TEG) doit être communiqué par écrit avant toute offre contractuelle. "
            "Il inclut le taux nominal, les frais de dossier et l'assurance le cas échéant."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════
def display_fiche(fiche):
    profile = fiche["profil"]
    risk    = fiche["analyse_risque"]
    offers  = fiche["offres"]
    narr    = fiche["narration"]
    is_sim  = fiche.get("_simulation", False)

    # ── Hero header ──
    _hero(fiche, profile, risk, is_sim)

    # ── 4 onglets ──
    t1, t2, t3, t4 = st.tabs([
        "📊  Dashboard",
        "🤖  Agent IA",
        "💰  Offre Commerciale",
        "⚖️  Conformité & Réglementation",
    ])

    with t1: _tab_dashboard(profile, risk)
    with t2: _tab_agent_ia(narr, risk, profile)
    with t3: _tab_offer(offers, risk, profile)
    with t4: _tab_compliance(offers, profile, risk)


# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
def _hero(fiche, profile, risk, is_sim):
    note      = risk["note"]
    nc        = NOTE_COLORS.get(note, "#64748B")
    g         = NOTE_GRADIENTS.get(note, ("135deg", "#64748B", "#475569"))
    decision  = risk["decision"]
    zone      = profile["zone_risque"]
    zi        = ZONE_ICONS.get(zone, "?")
    meta      = fiche["metadata"]
    a         = profile["appetence"]
    sim_badge = '<span style="background:#FEF3C7;color:#92400E;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;margin-left:8px">🔬 SIMULATION</span>' if is_sim else ""

    dc = {"APPROUVE": ("#D1FAE5", "#065F46", "#A7F3D0"),
          "INSTRUCTION": ("#FEF3C7", "#92400E", "#FDE68A"),
          "REFUS": ("#FEE2E2", "#991B1B", "#FCA5A5")}.get(decision, ("#F1F5F9", "#475569", "#CBD5E1"))

    st.markdown(f"""
    <div style='background:white;border-radius:14px;padding:24px 28px;
                box-shadow:0 2px 16px rgba(0,0,0,0.07);border:1px solid #E2E8F0;
                margin-bottom:20px'>
        <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px'>

            <!-- Identité -->
            <div style='display:flex;align-items:center;gap:20px'>
                <div style='background:linear-gradient({g[0]},{g[1]},{g[2]});width:60px;height:60px;
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-size:1.6rem;font-weight:900;color:white;box-shadow:0 4px 12px rgba(0,0,0,0.2)'>
                    {note}
                </div>
                <div>
                    <div style='font-size:1.3rem;font-weight:800;color:#1C2333'>
                        Client #{meta["id_client"]}{sim_badge}
                    </div>
                    <div style='font-size:0.85rem;color:#64748B;margin-top:2px'>
                        {profile["signaletique"]["type_revenu"]} · {profile["signaletique"]["segment"]} · {profile["signaletique"]["age"]} ans
                    </div>
                    <div style='font-size:0.75rem;color:#94A3B8;margin-top:2px'>
                        Généré le {meta["date_generation"][:16].replace("T"," à ")}
                    </div>
                </div>
            </div>

            <!-- KPIs -->
            <div style='display:flex;gap:16px;flex-wrap:wrap'>
                <div style='text-align:center'>
                    <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.08em;color:#94A3B8'>Revenu</div>
                    <div style='font-size:1.1rem;font-weight:800;color:#1C2333'>
                        {profile["signaletique"]["revenu_principal"]:,.0f} <span style='font-size:0.75rem;color:#64748B'>MAD</span>
                    </div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.08em;color:#94A3B8'>Score</div>
                    <div style='font-size:1.1rem;font-weight:800;color:{nc}'>{risk["score"]}/100</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.08em;color:#94A3B8'>Appétence</div>
                    <div style='font-size:1.1rem;font-weight:800;color:#3B82F6'>{a["score"]:.0%}</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.08em;color:#94A3B8'>Zone</div>
                    <div style='font-size:0.9rem;font-weight:700;color:#1C2333'>{zi} {zone}</div>
                </div>
            </div>

            <!-- Décision -->
            <div style='background:{dc[0]};border:1px solid {dc[2]};border-radius:12px;
                        padding:14px 22px;text-align:center;min-width:120px'>
                <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.08em;color:{dc[1]}'>Décision</div>
                <div style='font-size:1.1rem;font-weight:900;color:{dc[1]};margin-top:2px'>{decision}</div>
                <div style='font-size:0.75rem;color:{dc[1]};opacity:0.8;margin-top:1px'>Note {note} — {risk.get("label","")}</div>
            </div>

        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 1 : DASHBOARD
# ═══════════════════════════════════════════════════════════════
def _tab_dashboard(profile, risk):
    sv = profile["solvabilite"]
    c  = profile["comportement"]
    a  = profile["appetence"]

    # ── Row 1 : 4 métriques ──
    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, "Taux d'endettement", f"{sv['taux_endettement_actuel']:.1%}",
                 "✅ Conforme BAM" if sv["taux_endettement_actuel"] <= 0.40 else "🚫 Dépasse le plafond",
                 "#10B981" if sv["taux_endettement_actuel"] <= 0.33 else "#F59E0B" if sv["taux_endettement_actuel"] <= 0.40 else "#EF4444")
    _metric_card(m2, "Capacité résiduelle", f"{sv['capacite_mensuelle_residuelle']:,.0f} MAD",
                 "Après mensualités actuelles",
                 "#10B981" if sv["capacite_mensuelle_residuelle"] >= 2000 else "#F59E0B" if sv["capacite_mensuelle_residuelle"] >= 1000 else "#EF4444")
    _metric_card(m3, "Découverts 3 mois", f"{c['nb_decouverts_3m']} j",
                 "Jours de solde négatif",
                 "#10B981" if c["nb_decouverts_3m"] == 0 else "#F59E0B" if c["nb_decouverts_3m"] <= 5 else "#EF4444")
    _metric_card(m4, "Score appétence ML", f"{a['score']:.0%}",
                 a["classe"],
                 "#3B82F6" if a["score"] >= 0.7 else "#F59E0B" if a["score"] >= 0.4 else "#94A3B8")

    st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        # ── Radar chart risque ──
        st.markdown("<div class='card-header' style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:8px'>Profil risque — 5 dimensions</div>", unsafe_allow_html=True)
        _radar_chart(risk)

    with col_right:
        # ── Score gauge ──
        st.markdown("<div class='card-header' style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:8px'>Score global</div>", unsafe_allow_html=True)
        _gauge_chart(risk["score"], NOTE_COLORS.get(risk["note"], "#64748B"))

        # ── Cohérence ML × règles ──
        coherence = _assess_coherence(a["score"], risk)
        cl = {"OK": "coherence-ok", "WARN": "coherence-warn", "ERR": "coherence-err"}.get(coherence["level"], "coherence-ok")
        st.markdown(f"""
        <div class='{cl} coherence-box' style='margin-top:10px'>
            <b>{coherence["message"]}</b>
            <ul style='margin:6px 0 0 0;padding-left:16px;font-size:0.82rem'>
                {"".join(f"<li>{d}</li>" for d in coherence["details"])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

    # ── SHAP bars ──
    st.markdown("<div class='card-header' style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:8px'>Facteurs explicatifs (SHAP)</div>", unsafe_allow_html=True)
    _shap_bars(profile["top_5_shap"])

    # ── Comportement solde ──
    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)
    _comportement_cards(profile)


# ═══════════════════════════════════════════════════════════════
# TAB 2 : AGENT IA
# ═══════════════════════════════════════════════════════════════
def _tab_agent_ia(narr, risk, profile):
    source  = narr.get("source", "fallback_template")
    is_llm  = source == "llm"
    src_badge = (
        '<span class="llm-source-badge-llm">✦ Mistral 7B</span>' if is_llm
        else '<span class="llm-source-badge-template">○ Template fallback</span>'
    )

    # ── Header LLM ──
    st.markdown(f"""
    <div class="llm-header">
        <div style='font-size:1.3rem'>🤖</div>
        <div style='flex:1'>
            <div style='font-weight:700;font-size:0.95rem'>Agent IA — Narration générée</div>
            <div style='font-size:0.75rem;color:#94A3B8;margin-top:1px'>
                Pipeline : Enrichissement → Risque → Offre → {src_badge}
            </div>
        </div>
        <span class="llm-model-badge">mistral:7b-instruct</span>
    </div>
    <div class="llm-body">
    """, unsafe_allow_html=True)

    # ── Résumé exécutif ──
    st.markdown(f"""
        <div class='llm-field-label'>Résumé exécutif</div>
        <div class='llm-resume'>{narr.get("resume_executif", "—")}</div>
    """, unsafe_allow_html=True)

    # ── Argumentation ──
    st.markdown(f"""
        <div class='llm-field-label'>Argumentation commerciale</div>
        <div class='llm-field-content'>{narr.get("argumentation_commerciale", "—")}</div>
    """, unsafe_allow_html=True)

    # ── Justification taux ──
    if narr.get("justification_taux") and narr["justification_taux"] != "Non applicable (refus).":
        st.markdown(f"""
            <div class='llm-field-label'>Justification du taux</div>
            <div class='llm-field-content' style='color:#1E40AF;border-color:#BFDBFE;background:#EFF6FF'>
                💡 {narr["justification_taux"]}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Points forts / attention ──
    col_pf, col_pa = st.columns(2)
    with col_pf:
        st.markdown("<div class='card-header' style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:8px'>✅ Points forts</div>", unsafe_allow_html=True)
        pf_list = risk.get("points_forts", [])
        if pf_list:
            for pf in pf_list:
                st.markdown(f"<div style='background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;padding:8px 12px;margin:4px 0;font-size:0.85rem;color:#14532D'>✓ {pf}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#94A3B8;font-size:0.85rem'>Aucun point fort identifié</div>", unsafe_allow_html=True)

    with col_pa:
        st.markdown("<div class='card-header' style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:8px'>⚠️ Points d'attention</div>", unsafe_allow_html=True)
        pa_list = risk.get("points_attention", [])
        if pa_list:
            for pa in pa_list:
                st.markdown(f"<div style='background:#FFFBEB;border:1px solid #FCD34D;border-radius:8px;padding:8px 12px;margin:4px 0;font-size:0.85rem;color:#78350F'>⚠ {pa}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#94A3B8;font-size:0.85rem'>Aucun point d'attention</div>", unsafe_allow_html=True)

    if risk.get("red_flags"):
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        for rf in risk["red_flags"]:
            st.markdown(f"<div style='background:#FEF2F2;border:1px solid #FCA5A5;border-radius:8px;padding:8px 12px;margin:4px 0;font-size:0.85rem;color:#991B1B;font-weight:600'>🚫 {rf}</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin:20px 0'></div>", unsafe_allow_html=True)

    # ── Points de vigilance ──
    pdv = narr.get("points_de_vigilance", [])
    if pdv:
        st.markdown("<div class='card-header' style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:8px'>📌 Points de vigilance pour l'appel</div>", unsafe_allow_html=True)
        for pv in pdv:
            st.markdown(f"<div class='vigilance-item'>📍 {pv}</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin:20px 0'></div>", unsafe_allow_html=True)

    # ── Script d'appel ──
    st.markdown("<div class='card-header' style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:8px'>📞 Script d'appel recommandé</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='llm-script'>\"{narr.get('script_appel', '—')}\"</div>", unsafe_allow_html=True)

    if not is_llm:
        st.markdown("""
        <div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 14px;
                    margin-top:12px;font-size:0.8rem;color:#64748B'>
            ℹ️ <b>Narration template</b> — Mistral 7B indisponible.
            Lancez <code>ollama serve && ollama pull mistral</code> pour activer la narration IA.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 : OFFRE COMMERCIALE
# ═══════════════════════════════════════════════════════════════
def _tab_offer(offers, risk, profile):
    if risk["decision"] == "REFUS":
        _refus_block(risk, offers)
        return

    op = offers["offre_principale"]
    oc = offers["offres_alternatives"]["confort"]
    oe = offers["offres_alternatives"]["economie"]

    # ── 3 cartes offres ──
    c1, c2, c3 = st.columns(3)
    _offer_card(c1, op, "principal", "Recommandée")
    _offer_card(c2, oc, "confort",   "Confort")
    _offer_card(c3, oe, "economie",  "Économie")

    st.markdown("<div style='margin:20px 0'></div>", unsafe_allow_html=True)

    # ── Comparatif visuel ──
    st.markdown("<div class='card-header' style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:8px'>Comparatif des 3 offres</div>", unsafe_allow_html=True)
    _offers_comparison_chart(op, oc, oe)

    # ── Simulation mensualité ──
    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)
    _mensualite_impact(op, profile)


def _offer_card(col, offer, style, label):
    is_main = style == "principal"
    border  = "#C8102E" if is_main else "#E2E8F0"
    hdr_bg  = "#C8102E" if is_main else ("#DBEAFE" if style == "confort" else "#D1FAE5")
    hdr_col = "white" if is_main else ("#1E40AF" if style == "confort" else "#065F46")
    icons   = {"principal": "★  RECOMMANDÉE", "confort": "🛋️  CONFORT", "economie": "💎  ÉCONOMIE"}

    rows = [
        ("Montant", f"{offer['montant']:,} MAD"),
        ("Durée", f"{offer['duree_mois']} mois"),
        ("Taux annuel", f"{offer['taux_annuel']:.2%}"),
        ("Mensualité", f"{offer['mensualite']:,.0f} MAD"),
        ("+ Assurance", f"{offer['assurance_mensuelle']:.0f} MAD"),
        ("Coût total", f"{offer['cout_total_credit']:,.0f} MAD"),
        ("TEG", f"{offer['teg']:.2%}"),
        ("Frais dossier", f"{offer['frais_dossier']} MAD"),
    ]
    rows_html = "".join(
        f"<div class='offer-row'><span class='offer-row-label'>{r}</span>"
        f"<span class='offer-row-value'>{v}</span></div>"
        for r, v in rows
    )

    with col:
        st.markdown(f"""
        <div class='offer-card {"principal" if is_main else ""}' style='border-color:{border}'>
            <div class='offer-card-header {style}' style='background:{hdr_bg};color:{hdr_col}'>
                {icons[style]}
            </div>
            <div class='offer-card-body'>
                <div class='offer-montant'>{offer['montant']:,}</div>
                <div class='offer-detail'>MAD — {offer['duree_mois']} mois</div>
                <div style='margin-top:12px'>{rows_html}</div>
                <div style='margin-top:10px;padding:8px;background:#F8FAFC;border-radius:6px;
                            font-size:0.78rem;color:#64748B;font-style:italic'>
                    {offer.get("argument","")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _offers_comparison_chart(op, oc, oe):
    fig = go.Figure()
    cats = ["Mensualité (MAD)", "Coût total (MAD)", "Taux (%)"]
    for offer, name, color in [
        (op, "Recommandée", "#C8102E"),
        (oc, "Confort",     "#3B82F6"),
        (oe, "Économie",    "#10B981"),
    ]:
        fig.add_trace(go.Bar(
            name=name,
            x=cats,
            y=[offer["mensualite"], offer["cout_total_credit"] / 10, offer["taux_annuel"] * 1000],
            marker_color=color,
            text=[f"{offer['mensualite']:.0f}", f"{offer['cout_total_credit']:,.0f}", f"{offer['taux_annuel']:.2%}"],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=280, margin=dict(t=20, b=20, l=10, r=10),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
        font=dict(size=11),
        yaxis=dict(showticklabels=False, showgrid=False),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def _mensualite_impact(op, profile):
    sv   = profile["solvabilite"]
    rv   = profile["signaletique"]["revenu_principal"]
    mens = op["mensualite_totale"]
    mens_act = sv["mensualites_actuelles"]
    apres = rv - mens_act - mens
    avant = rv - mens_act

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Revenu", "Mensualités actuelles", "Nouvelle mensualité", "Reste"],
        y=[rv, -mens_act, -mens, 0],
        text=[f"{rv:,.0f}", f"-{mens_act:,.0f}", f"-{mens:,.0f}", f"{apres:,.0f}"],
        textposition="outside",
        connector={"line": {"color": "#E2E8F0"}},
        increasing={"marker": {"color": "#10B981"}},
        decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#3B82F6"}},
    ))
    fig.update_layout(
        title=dict(text="Impact sur le budget mensuel (MAD)", font=dict(size=12), x=0),
        height=280, margin=dict(t=40, b=20, l=10, r=10),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=11),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def _refus_block(risk, offers):
    st.markdown(f"""
    <div style='background:#FEF2F2;border:1px solid #FCA5A5;border-radius:12px;
                padding:20px 24px;margin-bottom:16px'>
        <div style='font-size:1.1rem;font-weight:800;color:#991B1B;margin-bottom:6px'>
            🚫 Demande refusée
        </div>
        <div style='font-size:0.9rem;color:#7F1D1D'>{risk.get("motif_refus","—")}</div>
    </div>
    """, unsafe_allow_html=True)

    if offers and offers.get("rebond"):
        r = offers["rebond"]
        st.markdown(f"""
        <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;
                    padding:20px 24px'>
            <div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#1E40AF;margin-bottom:6px'>
                → Rebond produit recommandé
            </div>
            <div style='font-size:1rem;font-weight:700;color:#1E3A8A'>{r.get("produit","—")}</div>
            <div style='font-size:0.88rem;color:#1E40AF;margin-top:4px'>{r.get("argument","—")}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 : CONFORMITÉ
# ═══════════════════════════════════════════════════════════════
def _tab_compliance(offers, profile, risk):
    sv = profile["solvabilite"]
    te = sv["taux_endettement_actuel"]

    col_check, col_refs = st.columns([1, 1.4])

    with col_check:
        st.markdown("""
        <div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.08em;color:#64748B;margin-bottom:12px'>
            Checklist de conformité
        </div>""", unsafe_allow_html=True)

        # Taux d'endettement
        r = REGLEMENTATION["taux_endettement"]
        _compliance_item(
            r["titre"], r["reference"], r["detail"],
            "ok" if te <= r["plafond"] else "err",
            f"Taux actuel : {te:.1%} / Plafond : {r['plafond']:.0%}"
        )

        # Assurance
        if offers.get("offre_principale"):
            duree = offers["offre_principale"]["duree_mois"]
            ra = REGLEMENTATION["assurance"]
            _compliance_item(
                ra["titre"], ra["reference"], ra["detail"],
                "warn" if duree > 60 else "ok",
                f"Durée : {duree} mois {'→ assurance obligatoire' if duree > 60 else '→ assurance optionnelle'}"
            )

        # Protection consommateur
        rp = REGLEMENTATION["protection"]
        _compliance_item(rp["titre"], rp["reference"], rp["detail"], "info",
                         "À mentionner lors de l'appel client")

        # Devoir de conseil
        rd = REGLEMENTATION["devoir_conseil"]
        _compliance_item(rd["titre"], rd["reference"], rd["detail"],
                         "warn" if risk["decision"] == "INSTRUCTION" else "ok",
                         "Instruction requise" if risk["decision"] == "INSTRUCTION" else "Profil adapté")

        # TEG
        rt = REGLEMENTATION["teg"]
        _compliance_item(rt["titre"], rt["reference"], rt["detail"], "info",
                         "Document précontractuel obligatoire")

    with col_refs:
        st.markdown("""
        <div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.08em;color:#64748B;margin-bottom:12px'>
            Indicateurs réglementaires clés
        </div>""", unsafe_allow_html=True)

        _reg_gauge_chart(te)

        st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

        # Tableau récapitulatif
        c = profile["comportement"]
        items = [
            ("Taux d'endettement", f"{te:.1%}", te <= 0.40, f"Plafond BAM : 40%"),
            ("Jours de découvert", f"{c['nb_decouverts_3m']} j", c["nb_decouverts_3m"] <= 15, "Seuil exclusion : 15 j"),
            ("Âge client", f"{profile['signaletique']['age']} ans", 21 <= profile["signaletique"]["age"] <= 70, "Cible AWB : 21–70 ans"),
            ("Capacité résiduelle", f"{sv['capacite_mensuelle_residuelle']:,.0f} MAD", sv["capacite_mensuelle_residuelle"] > 0, "Doit être > 0"),
        ]
        rows_html = "".join(
            f"""<div style='display:flex;align-items:center;justify-content:space-between;
                            padding:8px 12px;border-bottom:1px solid #E2E8F0;font-size:0.85rem'>
                    <div>
                        <span style='color:#1C2333;font-weight:500'>{lbl}</span>
                        <span style='color:#94A3B8;font-size:0.78rem;margin-left:8px'>{note}</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:8px'>
                        <span style='font-weight:700;color:#1C2333'>{val}</span>
                        <span style='font-size:0.85rem'>{'✅' if ok else '⚠️'}</span>
                    </div>
                </div>"""
            for lbl, val, ok, note in items
        )
        st.markdown(f"""
        <div style='background:white;border:1px solid #E2E8F0;border-radius:10px;overflow:hidden'>
            {rows_html}
        </div>""", unsafe_allow_html=True)


def _compliance_item(title, reference, detail, level, status_text):
    colors = {
        "ok":   ("#F0FDF4", "#86EFAC", "#14532D", "✅"),
        "warn": ("#FFFBEB", "#FCD34D", "#78350F", "⚠️"),
        "err":  ("#FEF2F2", "#FCA5A5", "#991B1B", "🚫"),
        "info": ("#EFF6FF", "#BFDBFE", "#1E3A8A", "ℹ️"),
    }
    bg, border, txt, icon = colors.get(level, colors["info"])
    st.markdown(f"""
    <div class='reg-card reg-{level if level in ("conforme","vigilance","nonconforme","info") else "info"}'
         style='background:{bg};border-color:{border};border-left-color:{border};margin-bottom:10px;border-radius:10px;border:1px solid {border}'>
        <div class='reg-card-title' style='color:{txt}'>
            {icon} {title}
            <span style='font-size:0.7rem;font-weight:400;color:{txt};opacity:0.7;margin-left:4px'>— {reference}</span>
        </div>
        <div class='reg-card-text' style='color:{txt};opacity:0.85'>{detail}</div>
        <div style='margin-top:6px;font-size:0.78rem;font-weight:600;color:{txt};
                    background:rgba(255,255,255,0.5);padding:4px 8px;border-radius:6px;
                    display:inline-block'>{status_text}</div>
    </div>
    """, unsafe_allow_html=True)


def _reg_gauge_chart(te):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=te * 100,
        title={"text": "Taux d'endettement (%)", "font": {"size": 13}},
        delta={"reference": 33, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 60], "tickwidth": 1, "tickcolor": "#E2E8F0"},
            "bar": {"color": "#10B981" if te <= 0.33 else "#F59E0B" if te <= 0.40 else "#EF4444", "thickness": 0.3},
            "bgcolor": "white",
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 33],  "color": "#D1FAE5"},
                {"range": [33, 40], "color": "#FEF3C7"},
                {"range": [40, 60], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "#C8102E", "width": 3},
                "thickness": 0.75,
                "value": 40,
            },
        },
    ))
    fig.update_layout(
        height=200, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="white", font=dict(size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# COMPOSANTS PARTAGÉS
# ═══════════════════════════════════════════════════════════════
def _metric_card(col, label, value, sub, color):
    with col:
        st.markdown(f"""
        <div style='background:white;border-radius:10px;padding:14px 16px;
                    border:1px solid #E2E8F0;box-shadow:0 2px 8px rgba(0,0,0,0.05);
                    border-top:3px solid {color}'>
            <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#94A3B8;margin-bottom:4px'>{label}</div>
            <div style='font-size:1.5rem;font-weight:800;color:#1C2333;line-height:1.1'>{value}</div>
            <div style='font-size:0.75rem;color:{color};margin-top:3px;font-weight:500'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)


def _radar_chart(risk):
    dims  = risk.get("dimensions", {})
    if not dims:
        st.info("Données de dimensions non disponibles.")
        return

    labels = [DIM_LABELS[k][0] for k in DIM_LABELS if k in dims]
    values = [dims.get(k, 0) for k in DIM_LABELS if k in dims]
    maxes  = [DIM_LABELS[k][1] for k in DIM_LABELS if k in dims]
    pcts   = [v / m * 100 for v, m in zip(values, maxes)]

    labels_closed = labels + [labels[0]]
    pcts_closed   = pcts   + [pcts[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pcts_closed, theta=labels_closed, fill="toself",
        fillcolor="rgba(200,16,46,0.12)", line=dict(color="#C8102E", width=2),
        name="Score",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[100] * (len(labels) + 1), theta=labels_closed, fill="toself",
        fillcolor="rgba(226,232,240,0.3)", line=dict(color="#E2E8F0", width=1),
        name="Max", showlegend=False,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        showlegend=False, height=280,
        margin=dict(t=20, b=20, l=30, r=30),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def _gauge_chart(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 32, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#E2E8F0"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0,  40], "color": "#FEE2E2"},
                {"range": [40, 55], "color": "#FEF3C7"},
                {"range": [55, 70], "color": "#FEF9C3"},
                {"range": [70, 85], "color": "#D1FAE5"},
                {"range": [85, 100], "color": "#A7F3D0"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(
        height=200, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="white", font=dict(size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


def _shap_bars(top_5):
    if not top_5:
        st.info("Pas de données SHAP.")
        return

    st.markdown("<div style='background:white;border-radius:10px;border:1px solid #E2E8F0;padding:16px 20px'>", unsafe_allow_html=True)
    max_abs = max(abs(item.get("shap_value", 0)) for item in top_5) or 1

    for item in top_5:
        val   = item.get("shap_value", 0)
        label = item.get("feature_label", item.get("feature", "?"))
        pct   = abs(val) / max_abs * 100
        color = "#10B981" if val > 0 else "#EF4444"
        sign  = "+" if val > 0 else ""
        fv    = item.get("feature_value")
        fv_str = f"= {fv:,.2f}" if fv is not None else ""

        st.markdown(f"""
        <div class='shap-row'>
            <div class='shap-label'>
                <b>{label}</b>
                <span style='color:#94A3B8;font-size:0.78rem;margin-left:6px'>{fv_str}</span>
            </div>
            <div class='shap-bar-wrap'>
                <div class='shap-bar-fill' style='width:{pct:.0f}%;background:{color}'></div>
            </div>
            <div class='shap-value' style='color:{color}'>{sign}{val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem;color:#94A3B8;margin-top:4px'>Valeurs SHAP : positif = favorise l'appétence · négatif = freine</div>", unsafe_allow_html=True)


def _comportement_cards(profile):
    c  = profile["comportement"]
    sv = profile["solvabilite"]
    a  = profile["appetence"]

    items = [
        ("Solde moyen", f"{c['solde_moyen']:,.0f} MAD", "#3B82F6"),
        ("Tendance", f"{c['tendance_compte']:+.0f} MAD/j", "#10B981" if c["tendance_compte"] >= 0 else "#EF4444"),
        ("Ratio épargne", f"{sv['ratio_epargne']:.1f}x revenu", "#8B5CF6"),
        ("Simulations", f"{a['count_simul']} (dont {a['count_simul_recent']} récentes)", "#F59E0B"),
        ("Profil", "🟢 Sain" if c["profil_sain"] else "🔴 Fragile", "#10B981" if c["profil_sain"] else "#EF4444"),
        ("Zone risque", f"{ZONE_ICONS.get(profile['zone_risque'],'?')} {profile['zone_risque']}", "#64748B"),
    ]

    cols = st.columns(len(items))
    for col, (label, value, color) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div style='background:white;border-radius:8px;padding:10px 12px;
                        border:1px solid #E2E8F0;text-align:center'>
                <div style='font-size:0.68rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.07em;color:#94A3B8'>{label}</div>
                <div style='font-size:0.9rem;font-weight:700;color:{color};margin-top:3px'>{value}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══ Utilitaires ════════════════════════════════════════════════
def _assess_coherence(proba, risk):
    decision = risk["decision"]
    note     = risk["note"]

    if proba >= 0.7 and decision == "APPROUVE":
        return {"level": "OK",
                "message": "✅ Cohérence optimale",
                "details": [f"Score ML {proba:.0%} aligné avec la note {note}",
                             "Signal d'appétence confirmé par le profil financier"]}
    if proba < 0.4 and decision == "REFUS":
        return {"level": "OK",
                "message": "✅ Double signal défavorable",
                "details": [f"Score ML {proba:.0%} confirme le refus",
                             "Cohérence ML + règles métier"]}
    if proba >= 0.6 and decision == "REFUS":
        return {"level": "WARN",
                "message": "⚠️ Score ML positif — blocage réglementaire",
                "details": [f"Appétence forte ({proba:.0%}) mais contrainte réglementaire",
                             "Réévaluation possible après régularisation du profil",
                             "→ Orienter vers un plan d'assainissement"]}
    if proba < 0.4 and decision in ("APPROUVE", "INSTRUCTION"):
        return {"level": "WARN",
                "message": "⚠️ Profil solide — appétence faible",
                "details": [f"Note {note} mais score ML {proba:.0%}",
                             "Client non demandeur → approche proactive nécessaire",
                             "→ Argumenter le besoin, ne pas forcer"]}
    return {"level": "OK",
            "message": f"✅ Analyse cohérente — Note {note}",
            "details": [f"Score global : {risk['score']}/100",
                        f"Appétence : {proba:.0%}"]}
