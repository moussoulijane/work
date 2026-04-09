"""
Templates de secours si le LLM échoue.
Ces templates remplissent des trous dans un texte pré-rédigé.
Jamais aussi bons que le LLM, mais garantissent 100% de succès.
"""


def build_template_narration(profile, risk, offers):
    """Narration template pour les cas approuvés."""
    s = profile['signaletique']
    sv = profile['solvabilite']
    c = profile['comportement']
    a = profile['appetence']
    op = offers['offre_principale']
    
    profil_court = f"{s['type_revenu'].lower()} de {s['age']} ans du segment {s['segment']}"
    
    return {
        'resume_executif': (
            f"Client {profil_court} avec un score d'appétence de {a['score']:.0%} "
            f"et une note risque {risk['note']}. "
            f"Offre de {op['montant']:,} MAD recommandée à {op['taux_annuel']:.2%}."
        ),
        
        'argumentation_commerciale': (
            f"Ce client présente un revenu de {s['revenu_principal']:.0f} MAD avec "
            f"un taux d'endettement actuel de {sv['taux_endettement_actuel']:.0%}. "
            f"Sa capacité mensuelle résiduelle de {sv['capacite_mensuelle_residuelle']:.0f} MAD "
            f"permet d'absorber confortablement la mensualité proposée de "
            f"{op['mensualite']:.0f} MAD. "
            f"Sur les 3 derniers mois, son compte montre {c['nb_decouverts_3m']} jour(s) "
            f"de découvert et une tendance de {c['tendance_compte']:.0f} MAD/jour. "
            f"L'offre de {op['montant']:,} MAD sur {op['duree_mois']} mois correspond "
            f"à un niveau adapté à son profil financier."
        ),
        
        'justification_taux': (
            f"Le taux de {op['taux_annuel']:.2%} correspond à la note {risk['note']} "
            f"du client selon la grille tarifaire AWB, avec les ajustements liés à "
            f"son segment {s['segment']} et sa durée de {op['duree_mois']} mois."
        ),
        
        'points_de_vigilance': [
            "Vérifier l'absence d'autre projet crédit en cours",
            f"Proposer l'assurance décès-invalidité ({op['assurance_mensuelle']:.0f} MAD/mois)",
            "Mentionner la possibilité de remboursement anticipé",
        ],
        
        'script_appel': (
            f"Bonjour, je vous contacte de la part d'Attijariwafa Bank. "
            f"Suite à votre intérêt pour nos solutions de crédit, nous avons préparé "
            f"une offre personnalisée de {op['montant']:,} MAD sur {op['duree_mois']} mois, "
            f"soit une mensualité de {op['mensualite']:.0f} MAD. "
            f"Auriez-vous quelques minutes pour en discuter ?"
        ),
        
        'source': 'fallback_template',
    }


def build_refusal_narration(profile, risk):
    """Narration pour les refus (n'utilise jamais le LLM)."""
    s = profile['signaletique']
    motif = risk.get('motif_refus', 'Critères non remplis')
    
    return {
        'resume_executif': (
            f"Refus de la demande de crédit pour le client {s['type_revenu'].lower()} "
            f"de {s['age']} ans. Motif : {motif}."
        ),
        
        'argumentation_commerciale': (
            f"Après analyse du dossier, le client ne répond pas aux critères "
            f"d'attribution du crédit consommation AWB. {motif}. "
            f"Une solution alternative est proposée via le rebond produit."
        ),
        
        'justification_taux': "Non applicable (refus).",
        
        'points_de_vigilance': [
            "Annoncer le refus avec tact et respect",
            "Présenter le rebond produit comme une opportunité",
            "Ne pas insister si le client rejette l'alternative",
        ],
        
        'script_appel': (
            f"Bonjour, je vous contacte de la part d'Attijariwafa Bank. "
            f"Nous avons étudié votre demande et, pour le moment, nous ne pouvons "
            f"pas donner suite favorablement. Cependant, nous avons identifié une "
            f"solution alternative qui pourrait correspondre à votre situation. "
            f"Puis-je vous la présenter ?"
        ),
        
        'source': 'fallback_refusal',
    }

