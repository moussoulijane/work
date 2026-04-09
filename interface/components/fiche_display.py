"""
Composants Streamlit pour afficher la fiche client complète.
"""
import streamlit as st


ZONE_COLORS = {
    'STAR':        ('#FFD700', '⭐'),
    'CROISSANCE':  ('#4CAF50', '📈'),
    'DORMANT':     ('#2196F3', '💤'),
    'PRUDENCE':    ('#FF9800', '⚠️'),
    'FIDELISATION':('#9C27B0', '🤝'),
    'EXCLUSION':   ('#F44336', '🚫'),
}

DECISION_COLORS = {
    'APPROUVE':    '#4CAF50',
    'INSTRUCTION': '#FF9800',
    'REFUS':       '#F44336',
}

NOTE_COLORS = {
    'A': '#4CAF50', 'B': '#8BC34A',
    'C': '#FFC107', 'D': '#FF9800', 'E': '#F44336',
}


def display_header(fiche: dict):
    """Bandeau haut avec id_client, zone et décision."""
    meta    = fiche['meta']
    profil  = fiche['profil']
    risque  = fiche['risque']
    zone    = profil['zone_risque']
    color_z, icon_z = ZONE_COLORS.get(zone, ('#607D8B', '?'))
    color_d = DECISION_COLORS.get(risque['decision'], '#607D8B')
    color_n = NOTE_COLORS.get(risque['note'], '#607D8B')

    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    with col1:
        st.markdown(f"### Client `{meta['id_client']}`")
        st.caption(f"Généré le {meta['generated_at'][:19].replace('T', ' à ')}")
    with col2:
        st.markdown(
            f"<div style='background:{color_z};padding:8px;border-radius:6px;"
            f"text-align:center;color:white;font-weight:bold'>"
            f"{icon_z} {zone}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div style='background:{color_d};padding:8px;border-radius:6px;"
            f"text-align:center;color:white;font-weight:bold'>"
            f"{risque['decision']}</div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div style='background:{color_n};padding:8px;border-radius:6px;"
            f"text-align:center;color:white;font-weight:bold;font-size:1.4em'>"
            f"Note {risque['note']} — {risque['score']}/100</div>",
            unsafe_allow_html=True,
        )


def display_profil(fiche: dict):
    """Bloc profil client et indicateurs clés."""
    p  = fiche['profil']
    s  = p['signaletique']
    sv = p['solvabilite']
    c  = p['comportement']
    a  = p['appetence']

    with st.expander("Profil client", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Signalétique**")
            st.write(f"- Âge : **{s['age']} ans**")
            st.write(f"- Revenu : **{s['revenu_principal']:,.0f} MAD**")
            st.write(f"- Type : **{s['type_revenu']}**")
            st.write(f"- Segment : **{s['segment']}**")

        with col2:
            st.markdown("**Solvabilité**")
            te_pct = sv['taux_endettement_actuel'] * 100
            color_te = 'green' if te_pct < 25 else 'orange' if te_pct < 35 else 'red'
            st.markdown(
                f"- Endettement : <span style='color:{color_te};font-weight:bold'>"
                f"{te_pct:.1f}%</span>",
                unsafe_allow_html=True,
            )
            st.write(f"- Capacité résid. : **{sv['capacite_mensuelle_residuelle']:,.0f} MAD**")
            st.write(f"- Ratio épargne : **{sv['ratio_epargne']:.1f}x**")

        with col3:
            st.markdown("**Comportement compte**")
            st.write(f"- Solde moyen : **{c['solde_moyen']:,.0f} MAD**")
            st.write(f"- Tendance : **{c['tendance_compte']:+.0f} MAD/j**")
            st.write(f"- Découverts 3M : **{c['nb_decouverts_3m']} j**")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Score d'appétence",
                f"{a['score']:.0%}",
                delta=a['classe'],
            )
        with col_b:
            st.metric(
                "Simulations crédit",
                f"{a['count_simul']}",
                delta=f"+{a['count_simul_recent']} ce mois",
            )


def display_risk_analysis(fiche: dict):
    """Bloc analyse risque : score radar, points forts/attention."""
    risque = fiche['risque']

    with st.expander("Analyse risque", expanded=True):
        if risque['dimensions']:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Score par dimension**")
                dims = risque['dimensions']
                dim_labels = {
                    'capacite_remboursement': 'Capacité remb. (/30)',
                    'stabilite_financiere': 'Stabilité fin. (/25)',
                    'profil_revenu': 'Profil revenu (/20)',
                    'historique_bancaire': 'Historique banc. (/15)',
                    'signal_appetence': 'Signal appétence (/10)',
                }
                for key, label in dim_labels.items():
                    val = dims.get(key, 0)
                    max_val = int(label.split('/')[1].rstrip(')'))
                    pct = val / max_val if max_val > 0 else 0
                    color = '#4CAF50' if pct > 0.7 else '#FF9800' if pct > 0.4 else '#F44336'
                    st.markdown(
                        f"{label} : **{val:.0f}**  "
                        f"<div style='background:#eee;border-radius:4px;height:8px'>"
                        f"<div style='background:{color};width:{pct*100:.0f}%;height:8px;"
                        f"border-radius:4px'></div></div>",
                        unsafe_allow_html=True,
                    )
            with col2:
                st.metric("Score total", f"{risque['score']}/100")
                st.markdown(f"**Note** : {risque['note']} — *{risque.get('label', '')}*")

        col_a, col_b = st.columns(2)
        with col_a:
            if risque['points_forts']:
                st.success("**Points forts**\n\n" + "\n".join(f"✅ {p}" for p in risque['points_forts']))
        with col_b:
            if risque['points_attention']:
                st.warning("**Points d'attention**\n\n" + "\n".join(f"⚠️ {p}" for p in risque['points_attention']))

        if risque.get('red_flags'):
            st.error("**Motif de refus**\n\n" + "\n".join(f"🚫 {r}" for r in risque['red_flags']))


def display_offers(fiche: dict):
    """Bloc offres commerciales (ou rebond si refus)."""
    offres = fiche['offres']

    with st.expander("Offre commerciale", expanded=True):
        if offres['offre_principale'] is None:
            # Refus → rebond
            rebond = offres.get('rebond', {})
            st.error(f"**Refus crédit consommation**")
            if rebond:
                st.info(
                    f"**Rebond produit**\n\n"
                    f"- Produit proposé : **{rebond.get('produit', '–')}**\n"
                    f"- Argument : {rebond.get('argument', '–')}"
                )
            return

        op = offres['offre_principale']
        oc = offres['offres_alternatives']['confort']
        oe = offres['offres_alternatives']['economie']

        col1, col2, col3 = st.columns(3)

        def _offer_card(col, offer, highlight=False):
            bg = '#E8F5E9' if highlight else '#F5F5F5'
            with col:
                st.markdown(
                    f"<div style='background:{bg};padding:12px;border-radius:8px;"
                    f"border:{\"2px solid #4CAF50\" if highlight else \"1px solid #ddd\"}'>"
                    f"<b>{offer['type']}</b><br/>"
                    f"<span style='font-size:1.4em;font-weight:bold'>{offer['montant']:,} MAD</span><br/>"
                    f"sur {offer['duree_mois']} mois<br/>"
                    f"Mensualité : <b>{offer['mensualite_totale']:,.0f} MAD</b><br/>"
                    f"Taux : {offer['taux_annuel']:.2%}<br/>"
                    f"Coût total : {offer['cout_total_credit']:,.0f} MAD<br/>"
                    f"<small style='color:#666'>{offer['argument']}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        _offer_card(col1, op, highlight=True)
        _offer_card(col2, oc)
        _offer_card(col3, oe)


def display_narration(fiche: dict):
    """Bloc narration LLM avec script appel."""
    narration = fiche['narration']
    source = narration.get('source', 'unknown')
    source_label = "IA (Mistral)" if source == 'llm' else "Template"

    with st.expander(f"Fiche argumentée — source : {source_label}", expanded=True):
        st.markdown("#### Résumé exécutif")
        st.info(narration.get('resume_executif', '–'))

        st.markdown("#### Argumentation commerciale")
        st.markdown(narration.get('argumentation_commerciale', '–'))

        st.markdown("#### Justification du taux")
        st.markdown(narration.get('justification_taux', '–'))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Points de vigilance")
            for point in narration.get('points_de_vigilance', []):
                st.markdown(f"⚠️ {point}")

        with col2:
            st.markdown("#### Script d'appel")
            st.text_area(
                "À dire au client :",
                value=narration.get('script_appel', ''),
                height=150,
                disabled=False,
                key="script_appel_text",
            )


def display_shap(fiche: dict):
    """Bloc top-5 features SHAP."""
    top5 = fiche['profil'].get('top_5_shap', [])
    if not top5:
        return

    with st.expander("Facteurs clés (analyse explicative)", expanded=False):
        st.markdown("Les 5 caractéristiques ayant le plus influencé le score :")
        for item in top5:
            sign = "+" if item['shap_value'] > 0 else ""
            color = '#4CAF50' if item['shap_value'] > 0 else '#F44336'
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 8px;border-left:4px solid {color};margin:4px 0'>"
                f"<span>{item['label']}</span>"
                f"<span style='color:{color};font-weight:bold'>"
                f"{sign}{item['shap_value']:.4f}</span></div>",
                unsafe_allow_html=True,
            )
