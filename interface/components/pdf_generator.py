"""
Génération de la fiche en PDF via ReportLab.
Layout : 1-2 pages A4, charte AWB.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from datetime import datetime


AWB_RED = HexColor('#C8102E')
AWB_DARK = HexColor('#2C3E50')
LIGHT_GRAY = HexColor('#F8F9FA')


def generate_pdf(fiche):
    """
    Génère une fiche PDF.
    
    Args:
        fiche: dict — fiche complète
    
    Returns:
        bytes — contenu PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        leftMargin=1.5*cm, rightMargin=1.5*cm
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Styles custom
    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'],
        fontSize=16, textColor=AWB_RED, alignment=TA_CENTER,
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Heading2'],
        fontSize=12, textColor=AWB_DARK, spaceAfter=8, spaceBefore=12
    )
    normal = styles['Normal']
    
    # ── En-tête ──
    story.append(Paragraph(
        "ATTIJARIWAFA BANK<br/>Fiche d'analyse crédit consommation",
        title_style
    ))
    
    story.append(Paragraph(
        f"Client #{fiche['metadata']['id_client']} — "
        f"Généré le {datetime.fromisoformat(fiche['metadata']['date_generation']).strftime('%d/%m/%Y %H:%M')}",
        ParagraphStyle('meta', parent=normal, alignment=TA_CENTER, fontSize=9)
    ))
    story.append(Spacer(1, 0.5*cm))
    
    # ── Bloc Identité ──
    story.append(Paragraph("1. PROFIL CLIENT", subtitle_style))
    
    p = fiche['profil']
    s = p['signaletique']
    sv = p['solvabilite']
    c = p['comportement']
    
    identity_data = [
        ['Âge', f"{s['age']} ans", 'Type revenu', s['type_revenu']],
        ['Segment', s['segment'], 'Revenu', f"{s['revenu_principal']:,.0f} MAD"],
        ['Taux endettement', f"{sv['taux_endettement_actuel']:.0%}", 
         'Capacité résiduelle', f"{sv['capacite_mensuelle_residuelle']:,.0f} MAD"],
        ['Solde moyen', f"{c['solde_moyen']:,.0f} MAD",
         'Découverts 3M', f"{c['nb_decouverts_3m']} jours"],
    ]
    
    t = Table(identity_data, colWidths=[3.5*cm, 4.5*cm, 3.5*cm, 4.5*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
        ('BACKGROUND', (2, 0), (2, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#DDDDDD')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))
    
    # ── Bloc Scoring ──
    story.append(Paragraph("2. ANALYSE RISQUE", subtitle_style))
    
    risk = fiche['analyse_risque']
    
    # Pastille note
    note_color = {'A': '#28a745', 'B': '#5cb85c', 'C': '#f0ad4e',
                  'D': '#ff7043', 'E': '#d9534f'}.get(risk['note'], '#666')
    
    scoring_data = [[
        f"NOTE {risk['note']}",
        f"Score : {risk['score']}/100",
        f"Décision : {risk['decision']}",
        f"Zone : {p['zone_risque']}",
    ]]
    t = Table(scoring_data, colWidths=[3*cm, 4*cm, 4.5*cm, 4.5*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), 14),
        ('FONTSIZE', (1, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (0, 0), HexColor(note_color)),
        ('TEXTCOLOR', (0, 0), (0, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#DDDDDD')),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    
    # Points forts / attention
    if risk.get('points_forts'):
        story.append(Paragraph("<b>Points forts :</b>", normal))
        for pf in risk['points_forts']:
            story.append(Paragraph(f"• {pf}", normal))
    
    if risk.get('points_attention'):
        story.append(Paragraph("<b>Points d'attention :</b>", normal))
        for pa in risk['points_attention']:
            story.append(Paragraph(f"• {pa}", normal))
    
    story.append(Spacer(1, 0.4*cm))
    
    # ── Bloc Offre ──
    offers = fiche['offres']
    
    if risk['decision'] != 'REFUS' and offers.get('offre_principale'):
        story.append(Paragraph("3. OFFRE COMMERCIALE", subtitle_style))
        
        op = offers['offre_principale']
        offer_data = [
            ['Montant', f"{op['montant']:,} MAD", 'Durée', f"{op['duree_mois']} mois"],
            ['Taux annuel', f"{op['taux_annuel']:.2%}", 'TEG', f"{op['teg']:.2%}"],
            ['Mensualité', f"{op['mensualite']:,.0f} MAD", 
             'Coût total', f"{op['cout_total_credit']:,.0f} MAD"],
        ]
        t = Table(offer_data, colWidths=[3.5*cm, 4.5*cm, 3.5*cm, 4.5*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (0, -1), AWB_RED),
            ('TEXTCOLOR', (0, 0), (0, -1), white),
            ('BACKGROUND', (2, 0), (2, -1), LIGHT_GRAY),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#DDDDDD')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
    else:
        # Refus + rebond
        story.append(Paragraph("3. DÉCISION", subtitle_style))
        story.append(Paragraph(
            f"<b>REFUS</b> — {risk.get('motif_refus', 'Critères non remplis')}",
            normal
        ))
        if offers and offers.get('rebond'):
            rebond = offers['rebond']
            story.append(Paragraph(
                f"<b>Rebond produit</b> : {rebond['produit']}<br/>{rebond['argument']}",
                normal
            ))
    
    story.append(Spacer(1, 0.4*cm))
    
    # ── Bloc Argumentation ──
    story.append(Paragraph("4. ARGUMENTATION", subtitle_style))
    
    narr = fiche['narration']
    story.append(Paragraph(f"<b>Résumé :</b> {narr['resume_executif']}", normal))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(narr['argumentation_commerciale'], normal))
    story.append(Spacer(1, 0.2*cm))
    
    if narr.get('justification_taux'):
        story.append(Paragraph(
            f"<i>{narr['justification_taux']}</i>",
            ParagraphStyle('italic', parent=normal, textColor=HexColor('#555'))
        ))
    
    story.append(Spacer(1, 0.3*cm))
    
    # ── Bloc Script ──
    story.append(Paragraph("5. SCRIPT D'APPEL", subtitle_style))
    
    script_table = Table(
        [[Paragraph(narr['script_appel'], normal)]],
        colWidths=[16*cm]
    )
    script_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#FFF3CD')),
        ('BOX', (0, 0), (-1, -1), 1, HexColor('#FFC107')),
        ('PADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(script_table)
    
    if narr.get('points_de_vigilance'):
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("<b>Vigilances :</b>", normal))
        for pv in narr['points_de_vigilance']:
            story.append(Paragraph(f"• {pv}", normal))
    
    # ── Pied de page ──
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "<i>Document généré automatiquement — À valider par le conseiller</i>",
        ParagraphStyle('footer', parent=normal, fontSize=8,
                      textColor=HexColor('#999'), alignment=TA_CENTER)
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

