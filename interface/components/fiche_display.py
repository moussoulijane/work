"""
Composants d'affichage Streamlit pour la fiche client.
Zones : validation modèle, offre, recommandation réglementaire, script.
"""
import streamlit as st


# ─── Palettes ───
NOTE_COLORS = {'A': '#28a745', 'B': '#5cb85c', 'C': '#f0ad4e', 'D': '#ff7043', 'E': '#d9534f'}
ZONE_ICONS  = {'STAR': '⭐', 'CROISSANCE': '📈', 'DORMANT': '💤',
               'PRUDENCE': '⚠️', 'FIDELISATION': '🤝', 'EXCLUSION': '🚫'}

# ─── Références réglementaires Bank Al-Maghrib ───
REGLEMENTATION = {
    'taux_endettement': {
        'seuil': 0.40,
        'texte': "Bank Al-Maghrib (Circulaire n°19/G/2002) fixe le taux d'endettement maximal à 40 % "
                 "du revenu net mensuel pour les crédits aux particuliers.",
        'seuil_vigilance': 0.33,
        'texte_vigilance': "Le seuil de vigilance interne AWB est à 33 %. Au-delà, "
                           "une analyse approfondie de la capacité de remboursement est requise.",
    },
    'duree_credit': {
        'seuil_assurance': 60,
        'texte': "Pour toute durée supérieure à 60 mois, l'assurance décès-invalidité est "
                 "obligatoire conformément aux pratiques prudentielles en vigueur au Maroc.",
    },
    'protection_consommateur': {
        'texte': "La loi n°31-08 sur la protection du consommateur (art. 102-120) impose "
                 "l'information préalable sur le TEG, le coût total du crédit et les "
                 "conditions de remboursement anticipé.",
    },
    'devoir_conseil': {
        'texte': "La charte éthique AWB et les orientations de Bank Al-Maghrib imposent "
                 "un devoir de conseil : l'établissement doit s'assurer que le crédit "
                 "proposé est adapté à la situation financière réelle du client.",
    },
}


def display_fiche(fiche):
    """Affiche une fiche complète avec validation modèle + recommandation réglementaire."""

    profile = fiche['profil']
    risk    = fiche['analyse_risque']
    offers  = fiche['offres']
    narr    = fiche['narration']
    is_sim  = fiche.get('_simulation', False)

    # ─── En-tête client ───
    _display_client_header(fiche, profile, risk, is_sim)
    st.divider()

    # ─── Validation Modèle (nouveau bloc) ───
    display_model_validation(fiche, profile, risk)
    st.divider()

    # ─── Offre ou Refus ───
    if risk['decision'] == 'REFUS':
        display_refus_block(risk, offers)
    else:
        display_offer_block(offers)
    st.divider()

    # ─── Recommandation avec vue réglementaire ───
    display_recommendation(narr, risk, profile, offers)
    st.divider()

    # ─── Script d'appel ───
    display_script_block(narr)

    # ─── Détails techniques collapsible ───
    with st.expander("🔬 Détails techniques — Top 5 SHAP"):
        display_shap_details(profile['top_5_shap'])


def _display_client_header(fiche, profile, risk, is_sim):
    """Bandeau identité + note."""
    meta = fiche['metadata']
    s    = profile['signaletique']
    note = risk['note']
    nc   = NOTE_COLORS.get(note, '#607D8B')
    zi   = ZONE_ICONS.get(profile['zone_risque'], '?')

    col1, col2 = st.columns([3, 1])
    with col1:
        badge = " 🔬 *Simulation*" if is_sim else ""
        st.subheader(f"👤 Client #{meta['id_client']}{badge}")
        c = st.columns(4)
        c[0].metric("Âge", f"{s['age']} ans")
        c[1].metric("Type revenu", s['type_revenu'])
        c[2].metric("Segment", s['segment'])
        c[3].metric("Revenu", f"{s['revenu_principal']:,.0f} MAD")

    with col2:
        st.subheader("📊 Note risque")
        st.markdown(
            f'<div class="note-{note.lower()}">'
            f'Note {note} — {risk.get("label","")}<br>Score : {risk["score"]}/100'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption(
            f"Zone : {zi} **{profile['zone_risque']}** | "
            f"Appétence : **{profile['appetence']['score']:.0%}** "
            f"({profile['appetence']['classe']})"
        )


def display_model_validation(fiche, profile, risk):
    """
    Bloc 'Validation du Modèle' :
    - Score ML (probabilité) avec jauge visuelle
    - Niveau de confiance (fort / modéré / faible)
    - Score risque sur 100 avec détail des 5 dimensions
    - Cohérence modèle × règles métier
    """
    st.subheader("🤖 Validation du Modèle")

    proba  = profile['appetence']['score']
    classe = profile['appetence']['classe']
    note   = risk['note']
    score  = risk['score']

    col_ml, col_risk, col_coherence = st.columns(3)

    # ── Score ML ──
    with col_ml:
        st.markdown("**Score d'appétence (ML)**")
        # Jauge colorée
        color_ml = '#28a745' if proba >= 0.7 else '#f0ad4e' if proba >= 0.4 else '#d9534f'
        label_ml = 'Forte appétence' if proba >= 0.7 else 'Appétence modérée' if proba >= 0.4 else 'Faible appétence'
        st.markdown(
            f"<div style='font-size:2em;font-weight:bold;color:{color_ml}'>"
            f"{proba:.0%}</div>"
            f"<div style='background:#eee;border-radius:8px;height:12px;margin:4px 0'>"
            f"<div style='background:{color_ml};width:{proba*100:.0f}%;height:12px;border-radius:8px'></div></div>"
            f"<small style='color:{color_ml}'>{label_ml}</small>",
            unsafe_allow_html=True,
        )
        # Niveau de confiance
        intensite = profile['appetence']['intensite_digitale']
        if intensite >= 5:
            conf = "🟢 Confiance forte"
            conf_detail = f"{intensite} interactions digitales récentes"
        elif intensite >= 2:
            conf = "🟡 Confiance modérée"
            conf_detail = f"{intensite} interactions digitales"
        else:
            conf = "🔴 Confiance faible"
            conf_detail = "Peu de signaux comportementaux"
        st.caption(f"Confiance : **{conf}**")
        st.caption(conf_detail)

    # ── Score risque ──
    with col_risk:
        st.markdown("**Score risque métier (/100)**")
        nc = NOTE_COLORS.get(note, '#607D8B')
        st.markdown(
            f"<div style='font-size:2em;font-weight:bold;color:{nc}'>"
            f"{score}/100</div>"
            f"<div style='background:#eee;border-radius:8px;height:12px;margin:4px 0'>"
            f"<div style='background:{nc};width:{score}%;height:12px;border-radius:8px'></div></div>"
            f"<small style='color:{nc}'>Note {note} — {risk.get('label','')}</small>",
            unsafe_allow_html=True,
        )
        if risk.get('dimensions'):
            dims = risk['dimensions']
            for k, v in dims.items():
                label = k.replace('_', ' ').title()
                st.caption(f"• {label} : **{v:.0f}** pts")

    # ── Cohérence ML × Règles ──
    with col_coherence:
        st.markdown("**Cohérence Modèle × Règles**")
        coherence = _assess_coherence(proba, risk)
        color_c = '#28a745' if coherence['level'] == 'OK' else '#f0ad4e' if coherence['level'] == 'WARN' else '#d9534f'
        st.markdown(
            f"<div style='padding:10px;background:{color_c}20;border-left:4px solid {color_c};"
            f"border-radius:4px'>{coherence['message']}</div>",
            unsafe_allow_html=True,
        )
        for detail in coherence['details']:
            st.caption(f"→ {detail}")

    # ── Indicateurs clés réglementaires ──
    st.markdown("---")
    sv = profile['solvabilite']
    c  = profile['comportement']
    te = sv['taux_endettement_actuel']
    seuil_bam = REGLEMENTATION['taux_endettement']['seuil']
    seuil_vig = REGLEMENTATION['taux_endettement']['seuil_vigilance']

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        color_te = '#28a745' if te <= seuil_vig else '#f0ad4e' if te <= seuil_bam else '#d9534f'
        statut_te = "✅ En dessous du seuil" if te <= seuil_vig else "⚠️ Zone de vigilance" if te <= seuil_bam else "🚫 Au-dessus du plafond"
        st.markdown(
            f"<div style='background:{color_te}15;padding:10px;border-radius:8px;"
            f"border-left:3px solid {color_te}'>"
            f"<b>Taux d'endettement</b><br>"
            f"<span style='font-size:1.5em;color:{color_te}'>{te:.1%}</span><br>"
            f"<small>{statut_te}<br>Plafond BAM : {seuil_bam:.0%}</small></div>",
            unsafe_allow_html=True,
        )
    with col_r2:
        cap_res = sv['capacite_mensuelle_residuelle']
        color_cap = '#28a745' if cap_res >= 2000 else '#f0ad4e' if cap_res >= 1000 else '#d9534f'
        st.markdown(
            f"<div style='background:{color_cap}15;padding:10px;border-radius:8px;"
            f"border-left:3px solid {color_cap}'>"
            f"<b>Capacité résiduelle</b><br>"
            f"<span style='font-size:1.5em;color:{color_cap}'>{cap_res:,.0f} MAD</span><br>"
            f"<small>Après mensualité proposée</small></div>",
            unsafe_allow_html=True,
        )
    with col_r3:
        decouv = c['nb_decouverts_3m']
        color_d = '#28a745' if decouv == 0 else '#f0ad4e' if decouv <= 5 else '#d9534f'
        st.markdown(
            f"<div style='background:{color_d}15;padding:10px;border-radius:8px;"
            f"border-left:3px solid {color_d}'>"
            f"<b>Incidents de paiement</b><br>"
            f"<span style='font-size:1.5em;color:{color_d}'>{decouv} j</span><br>"
            f"<small>Jours de découvert sur 3 mois<br>Seuil alerte : 15 j</small></div>",
            unsafe_allow_html=True,
        )


def display_offer_block(offers):
    """Affiche l'offre principale + alternatives."""
    st.subheader("💰 Offre commerciale")

    op = offers['offre_principale']
    st.markdown('<div class="offer-box">', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Montant", f"{op['montant']:,} MAD")
    cols[1].metric("Durée", f"{op['duree_mois']} mois")
    cols[2].metric("Taux", f"{op['taux_annuel']:.2%}")
    cols[3].metric("Mensualité", f"{op['mensualite']:,.0f} MAD")

    st.caption(
        f"Coût total du crédit : **{op['cout_total_credit']:,.0f} MAD** | "
        f"TEG : **{op['teg']:.2%}** | "
        f"Frais de dossier : {op['frais_dossier']} MAD | "
        f"Assurance : {op['assurance_mensuelle']:.0f} MAD/mois"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Voir les 2 variantes"):
        oc = offers['offres_alternatives']['confort']
        oe = offers['offres_alternatives']['economie']
        col_c, col_e = st.columns(2)
        with col_c:
            st.markdown("**🛋️ Offre CONFORT** — mensualité allégée")
            st.write(f"{oc['montant']:,} MAD sur {oc['duree_mois']} mois")
            st.write(f"Mensualité : **{oc['mensualite']:,.0f} MAD** | Taux : {oc['taux_annuel']:.2%}")
        with col_e:
            st.markdown("**💎 Offre ÉCONOMIE** — remboursement rapide")
            st.write(f"{oe['montant']:,} MAD sur {oe['duree_mois']} mois")
            st.write(f"Mensualité : **{oe['mensualite']:,.0f} MAD** | Taux : {oe['taux_annuel']:.2%}")


def display_refus_block(risk, offers):
    """Affiche le bloc refus + rebond."""
    st.subheader("❌ Refus")
    st.error(f"**Motif** : {risk['motif_refus']}")
    if offers and offers.get('rebond'):
        rebond = offers['rebond']
        st.info(f"**Rebond produit recommandé** : {rebond['produit']}\n\n{rebond['argument']}")


def display_recommendation(narr, risk, profile, offers):
    """
    Bloc recommandation avec :
    - Résumé exécutif
    - Argumentation commerciale
    - Justification réglementaire (Bank Al-Maghrib, loi 31-08, devoir de conseil)
    - Points forts / attention
    """
    st.subheader("📋 Recommandation & Argumentation")

    # Résumé exécutif
    st.info(f"**Résumé exécutif** : {narr['resume_executif']}")

    col_arg, col_reg = st.columns([3, 2])

    with col_arg:
        st.markdown("#### Argumentation commerciale")
        st.markdown(narr['argumentation_commerciale'])

        if narr.get('justification_taux'):
            st.markdown(
                f"<div style='background:#f8f9fa;padding:10px;border-radius:6px;"
                f"border-left:3px solid #C8102E;margin-top:8px'>"
                f"<b>Justification du taux</b><br>{narr['justification_taux']}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col_pf, col_pa = st.columns(2)
        with col_pf:
            st.markdown("**✅ Points forts**")
            for pf in risk.get('points_forts', []):
                st.markdown(f'<span class="point-fort">• {pf}</span>', unsafe_allow_html=True)
        with col_pa:
            st.markdown("**⚠️ Points d'attention**")
            for pa in risk.get('points_attention', []):
                st.markdown(f'<span class="point-attention">• {pa}</span>', unsafe_allow_html=True)
        if risk.get('red_flags'):
            st.markdown("**🚨 Red flags**")
            for rf in risk['red_flags']:
                st.markdown(f'<span class="red-flag">• {rf}</span>', unsafe_allow_html=True)

    with col_reg:
        st.markdown("#### Vue réglementaire")
        sv = profile['solvabilite']
        te = sv['taux_endettement_actuel']

        # Conformité taux d'endettement
        regle_te = REGLEMENTATION['taux_endettement']
        color_te = '#28a745' if te <= regle_te['seuil_vigilance'] else '#f0ad4e' if te <= regle_te['seuil'] else '#d9534f'
        statut_conf = "✅ Conforme" if te <= regle_te['seuil'] else "🚫 Non conforme"
        st.markdown(
            f"<div style='background:#f8f9fa;padding:10px;border-radius:6px;margin-bottom:8px'>"
            f"<b>Taux d'endettement</b> <span style='color:{color_te}'>{statut_conf}</span><br>"
            f"<small style='color:#555'>{regle_te['texte']}</small></div>",
            unsafe_allow_html=True,
        )

        # Obligation d'assurance
        if offers.get('offre_principale') and offers['offre_principale']:
            duree = offers['offre_principale']['duree_mois']
            regle_ass = REGLEMENTATION['duree_credit']
            if duree > regle_ass['seuil_assurance']:
                st.markdown(
                    f"<div style='background:#fff3cd;padding:10px;border-radius:6px;margin-bottom:8px'>"
                    f"<b>⚠️ Assurance obligatoire</b> (durée {duree} mois > {regle_ass['seuil_assurance']} mois)<br>"
                    f"<small style='color:#555'>{regle_ass['texte']}</small></div>",
                    unsafe_allow_html=True,
                )

        # Protection consommateur
        st.markdown(
            f"<div style='background:#e8f4f8;padding:10px;border-radius:6px;margin-bottom:8px'>"
            f"<b>📜 Protection du consommateur</b><br>"
            f"<small style='color:#555'>{REGLEMENTATION['protection_consommateur']['texte']}</small></div>",
            unsafe_allow_html=True,
        )

        # Devoir de conseil
        st.markdown(
            f"<div style='background:#f0f8e8;padding:10px;border-radius:6px'>"
            f"<b>⚖️ Devoir de conseil AWB</b><br>"
            f"<small style='color:#555'>{REGLEMENTATION['devoir_conseil']['texte']}</small></div>",
            unsafe_allow_html=True,
        )


def display_script_block(narr):
    """Affiche le script d'appel + vigilances."""
    st.subheader("📞 Script d'appel")

    st.markdown(
        f'<div class="script-box">{narr["script_appel"]}</div>',
        unsafe_allow_html=True,
    )

    if narr.get('points_de_vigilance'):
        st.markdown("**Vigilances pour l'appel** :")
        for pv in narr['points_de_vigilance']:
            st.markdown(f"• {pv}")

    if narr.get('source') in ('fallback_template', 'fallback_refusal'):
        st.caption("ℹ️ Narration générée par template (Mistral indisponible ou désactivé)")


def display_shap_details(top_5_shap):
    """Affiche le top 5 SHAP avec barres visuelles."""
    import pandas as pd

    if not top_5_shap:
        st.info("Pas de données SHAP disponibles pour cette analyse.")
        return

    df = pd.DataFrame(top_5_shap)
    if 'rank' in df.columns and 'feature_label' in df.columns:
        df_display = df[['rank', 'feature_label', 'feature_value',
                          'direction', 'contribution_pct']].copy()
        df_display.columns = ['Rang', 'Variable', 'Valeur', 'Impact', 'Contribution %']
        df_display['Contribution %'] = df_display['Contribution %'].round(1)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        # Format alternatif (feature + shap_value)
        for item in top_5_shap:
            label = item.get('feature_label', item.get('feature', '?'))
            val   = item.get('shap_value', 0)
            sign  = "+" if val > 0 else ""
            color = '#28a745' if val > 0 else '#d9534f'
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:4px 8px;"
                f"border-left:4px solid {color};margin:3px 0'>"
                f"<span>{label}</span>"
                f"<span style='color:{color};font-weight:bold'>{sign}{val:.4f}</span></div>",
                unsafe_allow_html=True,
            )


# ─── Utilitaires ───

def _assess_coherence(proba, risk):
    """Évalue la cohérence entre le score ML et la décision règles métier."""
    decision = risk['decision']
    note     = risk['note']

    # Appétence forte + décision favorable → cohérent
    if proba >= 0.7 and decision == 'APPROUVE':
        return {
            'level': 'OK',
            'message': '✅ Cohérence modèle × règles : optimale',
            'details': [
                f"Score ML {proba:.0%} aligné avec la note {note}",
                "L'appétence forte est confirmée par le profil financier sain",
            ]
        }
    # Appétence faible + refus → cohérent
    if proba < 0.4 and decision == 'REFUS':
        return {
            'level': 'OK',
            'message': '✅ Cohérence modèle × règles : bonne',
            'details': [
                f"Score ML {proba:.0%} confirme le refus",
                "Double signal défavorable (ML + règles métier)",
            ]
        }
    # Appétence forte mais refus réglementaire → signal à noter
    if proba >= 0.6 and decision == 'REFUS':
        return {
            'level': 'WARN',
            'message': '⚠️ Signal ML positif mais blocage réglementaire',
            'details': [
                f"Score ML élevé ({proba:.0%}) mais contrainte réglementaire",
                "Réévaluation possible après amélioration du profil",
                "Proposer un plan d'assainissement financier",
            ]
        }
    # Appétence faible mais approbation → profil solide sans intention
    if proba < 0.4 and decision in ('APPROUVE', 'INSTRUCTION'):
        return {
            'level': 'WARN',
            'message': '⚠️ Profil solide mais appétence faible',
            'details': [
                f"Profil financier noté {note} mais score ML {proba:.0%}",
                "Approche commerciale proactive recommandée",
                "Client potentiellement non demandeur : argumenter le besoin",
            ]
        }
    # Cas standard
    return {
        'level': 'OK',
        'message': f'✅ Profil évalué — Note {note}, Appétence {proba:.0%}',
        'details': [
            "Analyse combinée ML + règles métier effectuée",
            f"Score risque : {risk['score']}/100",
        ]
    }
