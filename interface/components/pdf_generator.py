"""
Génération de la fiche client en PDF via ReportLab.
Retourne les bytes du PDF pour téléchargement Streamlit.
"""
from io import BytesIO
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ─── Couleurs AWB ───
AWB_RED   = colors.HexColor('#C8102E')
AWB_GRAY  = colors.HexColor('#F5F5F5')
AWB_DARK  = colors.HexColor('#333333')
GREEN     = colors.HexColor('#4CAF50')
ORANGE    = colors.HexColor('#FF9800')
RED_LIGHT = colors.HexColor('#F44336')

DECISION_COLORS_RL = {
    'APPROUVE':    GREEN,
    'INSTRUCTION': ORANGE,
    'REFUS':       RED_LIGHT,
}


def generate_pdf(fiche: dict) -> bytes:
    """
    Génère un PDF de la fiche client.

    Returns:
        bytes du PDF, ou PDF minimaliste si ReportLab absent.
    """
    if not REPORTLAB_AVAILABLE:
        return _fallback_text_pdf(fiche)

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )

    styles  = getSampleStyleSheet()
    story   = []
    meta    = fiche['meta']
    profil  = fiche['profil']
    risque  = fiche['risque']
    offres  = fiche['offres']
    narr    = fiche['narration']
    s       = profil['signaletique']
    sv      = profil['solvabilite']
    c       = profil['comportement']
    a       = profil['appetence']

    # ── Styles ──
    h1 = ParagraphStyle('h1', parent=styles['Heading1'],
                        textColor=AWB_RED, fontSize=16, spaceAfter=4)
    h2 = ParagraphStyle('h2', parent=styles['Heading2'],
                        textColor=AWB_DARK, fontSize=12, spaceAfter=2)
    body = ParagraphStyle('body', parent=styles['Normal'],
                          fontSize=9, leading=13)
    caption = ParagraphStyle('caption', parent=styles['Normal'],
                             fontSize=8, textColor=colors.gray)

    # ── En-tête ──
    decision_color = DECISION_COLORS_RL.get(risque['decision'], AWB_DARK)
    story.append(Paragraph("Attijariwafa Bank — Fiche Crédit Consommation", h1))
    story.append(HRFlowable(width='100%', color=AWB_RED, thickness=2))
    story.append(Spacer(1, 6))

    header_data = [
        [
            Paragraph(f"<b>Client #{meta['id_client']}</b>", body),
            Paragraph(f"Zone : <b>{profil['zone_risque']}</b>", body),
            Paragraph(
                f"<font color='#{_hex(decision_color)}'><b>{risque['decision']}</b></font>",
                body
            ),
            Paragraph(f"Note <b>{risque['note']}</b> — {risque['score']}/100", body),
        ]
    ]
    header_table = Table(header_data, colWidths=[4*cm, 4*cm, 4*cm, 4.5*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), AWB_GRAY),
        ('BOX',        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('INNERGRID',  (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Généré le {meta['generated_at'][:19].replace('T', ' à ')} | "
        f"Source : {narr.get('source', '–')}",
        caption,
    ))
    story.append(Spacer(1, 10))

    # ── Résumé ──
    story.append(Paragraph("Résumé exécutif", h2))
    story.append(Paragraph(narr.get('resume_executif', '–'), body))
    story.append(Spacer(1, 8))

    # ── Profil ──
    story.append(Paragraph("Profil client", h2))
    profil_data = [
        ['Signalétique', 'Solvabilité', 'Comportement'],
        [
            f"{s['age']} ans, {s['type_revenu']}\n{s['segment']}\n{s['revenu_principal']:,.0f} MAD",
            f"Endettement : {sv['taux_endettement_actuel']:.1%}\n"
            f"Capacité : {sv['capacite_mensuelle_residuelle']:,.0f} MAD\n"
            f"Épargne : {sv['ratio_epargne']:.1f}x",
            f"Solde moy. : {c['solde_moyen']:,.0f} MAD\n"
            f"Tendance : {c['tendance_compte']:+.0f} MAD/j\n"
            f"Découverts : {c['nb_decouverts_3m']} j",
        ],
    ]
    profil_table = Table(profil_data, colWidths=[5.4*cm, 5.4*cm, 5.6*cm])
    profil_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), AWB_RED),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND',    (0, 1), (-1, -1), AWB_GRAY),
        ('BOX',           (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('INNERGRID',     (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ('FONTSIZE',      (0, 0), (-1, -1), 8),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(profil_table)
    story.append(Spacer(1, 8))

    # ── Offre ──
    story.append(Paragraph("Offre commerciale", h2))
    if offres['offre_principale']:
        op = offres['offre_principale']
        offer_data = [
            ['Montant', 'Durée', 'Taux', 'Mensualité', 'Coût total', 'TEG'],
            [
                f"{op['montant']:,} MAD",
                f"{op['duree_mois']} mois",
                f"{op['taux_annuel']:.2%}",
                f"{op['mensualite_totale']:,.0f} MAD",
                f"{op['cout_total_credit']:,.0f} MAD",
                f"{op['teg']:.2%}",
            ],
        ]
        offer_table = Table(offer_data, colWidths=[2.8*cm]*6)
        offer_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), AWB_RED),
            ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
            ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND',    (0, 1), (-1, -1), colors.white),
            ('BOX',           (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('INNERGRID',     (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ('FONTSIZE',      (0, 0), (-1, -1), 8),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ]))
        story.append(offer_table)
    else:
        rebond = offres.get('rebond', {})
        story.append(Paragraph(
            f"Refus — Rebond : {rebond.get('produit', '–')}", body
        ))
    story.append(Spacer(1, 8))

    # ── Argumentation ──
    story.append(Paragraph("Argumentation commerciale", h2))
    story.append(Paragraph(narr.get('argumentation_commerciale', '–'), body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Justification du taux", h2))
    story.append(Paragraph(narr.get('justification_taux', '–'), body))
    story.append(Spacer(1, 6))

    # ── Points de vigilance ──
    story.append(Paragraph("Points de vigilance", h2))
    for point in narr.get('points_de_vigilance', []):
        story.append(Paragraph(f"• {point}", body))
    story.append(Spacer(1, 6))

    # ── Script d'appel ──
    story.append(Paragraph("Script d'appel", h2))
    story.append(Paragraph(narr.get('script_appel', '–'), body))

    doc.build(story)
    return buf.getvalue()


def _hex(color) -> str:
    """Convertit une couleur ReportLab en hex sans '#'."""
    try:
        r, g, b = int(color.red * 255), int(color.green * 255), int(color.blue * 255)
        return f"{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "333333"


def _fallback_text_pdf(fiche: dict) -> bytes:
    """PDF minimaliste en texte si ReportLab n'est pas installé."""
    lines = [
        "ATTIJARIWAFA BANK — FICHE CRÉDIT CONSOMMATION",
        "=" * 50,
        f"Client : {fiche['meta']['id_client']}",
        f"Date   : {fiche['meta']['generated_at'][:19]}",
        f"Décision : {fiche['risque']['decision']} (Note {fiche['risque']['note']}, {fiche['risque']['score']}/100)",
        "",
        "RÉSUMÉ",
        fiche['narration'].get('resume_executif', '–'),
        "",
        "ARGUMENTATION",
        fiche['narration'].get('argumentation_commerciale', '–'),
        "",
        "SCRIPT D'APPEL",
        fiche['narration'].get('script_appel', '–'),
    ]
    content = "\n".join(lines).encode('utf-8')
    # Retourner comme bytes bruts (pas un vrai PDF mais téléchargeable)
    return content
