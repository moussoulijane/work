"""
Couche 2 : Expert Risque.

Applique les règles métier AWB (chargées depuis YAML) pour :
1. Vérifier les règles d'exclusion
2. Calculer le score dimensionnel sur 100 points
3. Attribuer une note A/B/C/D/E
4. Générer les motifs de décision

Aucun LLM ici — tout est déterministe et auditable.
"""
import yaml


class RiskExpert:

    def __init__(self, rules_path='agent_config/business_rules.yaml'):
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)

    def evaluate(self, profile_enriched):
        """
        Returns:
            dict avec decision, note, score, dimensions, points_forts,
            points_attention, red_flags, motif_refus, rebond_key
        """
        exclusion = self._check_exclusions(profile_enriched)
        if exclusion['refused']:
            return {
                'decision': 'REFUS',
                'note': 'E',
                'score': 0,
                'label': 'Refus',
                'dimensions': {},
                'points_forts': [],
                'points_attention': [],
                'red_flags': [exclusion['motif']],
                'motif_refus': exclusion['motif'],
                'rebond_key': exclusion['rebond'],
            }

        dimensions = self._compute_dimensions(profile_enriched)
        total_score = sum(dimensions.values())
        note, decision, label = self._get_rating(total_score)
        points_forts, points_attention = self._generate_motifs(profile_enriched, dimensions)

        return {
            'decision': decision,
            'note': note,
            'score': int(total_score),
            'label': label,
            'dimensions': dimensions,
            'points_forts': points_forts,
            'points_attention': points_attention,
            'red_flags': [],
            'motif_refus': None,
            'rebond_key': None,
        }

    def _check_exclusions(self, p):
        s = p['signaletique']
        sv = p['solvabilite']
        c = p['comportement']

        if s['revenu_principal'] < 3000:
            return {'refused': True,
                    'motif': 'Revenu inférieur au seuil minimum AWB (3000 MAD)',
                    'rebond': 'revenu_insuffisant'}
        if sv['taux_endettement_actuel'] > 0.40:
            return {'refused': True,
                    'motif': "Taux d'endettement au-delà du plafond (40%)",
                    'rebond': 'endettement_sature'}
        if c['nb_decouverts_3m'] > 15:
            return {'refused': True,
                    'motif': 'Plus de 15 jours de découvert sur 3 mois',
                    'rebond': 'incidents_paiement'}
        if s['age'] < 21 or s['age'] > 70:
            return {'refused': True,
                    'motif': f"Âge {s['age']} hors cible (21-70 ans)",
                    'rebond': 'age_hors_cible'}
        if s['type_revenu'] == 'AUTRE' and s['revenu_principal'] < 5000:
            return {'refused': True,
                    'motif': 'Type de revenu non standard avec niveau faible',
                    'rebond': 'profil_non_verifiable'}
        if c['solde_min'] < -2 * s['revenu_principal']:
            return {'refused': True,
                    'motif': 'Découvert structurel important',
                    'rebond': 'incidents_paiement'}

        return {'refused': False, 'motif': None, 'rebond': None}

    def _compute_dimensions(self, p):
        sv = p['solvabilite']
        c = p['comportement']
        s = p['signaletique']
        a = p['appetence']

        # D1 Capacité remboursement (30)
        te = sv['taux_endettement_actuel']
        d1_te = 15 if te < 0.20 else 10 if te < 0.30 else 5 if te < 0.35 else 0
        cr = sv['capacite_mensuelle_residuelle']
        d1_cr = 15 if cr > 3000 else 10 if cr > 2000 else 5 if cr > 1000 else 0
        d1 = d1_te + d1_cr

        # D2 Stabilité financière (25)
        d2_stab = min(10, 10 * c['stabilite_revenu_proxy'])
        d2_tend = 10 if c['tendance_compte'] > 0 else 5 if c['tendance_compte'] == 0 else 0
        d2_dec = 5 if c['nb_decouverts_3m'] == 0 else 3 if c['nb_decouverts_3m'] <= 2 else 0
        d2 = d2_stab + d2_tend + d2_dec

        # D3 Profil revenu (20)
        d3_map = {'SALARIE': 20, 'RETRAITE': 18, 'PROFESSION_LIBERALE': 15,
                  'COMMERCANT': 12, 'AUTRE': 8}
        d3 = d3_map.get(s['type_revenu'], 8)

        # D4 Historique bancaire (15)
        re = sv['ratio_epargne']
        d4_re = 10 if re > 2 else 7 if re > 1 else 4 if re > 0.5 else 0
        d4_sm = 5 if c['solde_max'] > 50000 else 3 if c['solde_max'] > 20000 else 0
        d4 = d4_re + d4_sm

        # D5 Signal appétence (10)
        d5 = min(10, 10 * a['score'])

        return {
            'capacite_remboursement': round(d1, 1),
            'stabilite_financiere': round(d2, 1),
            'profil_revenu': round(d3, 1),
            'historique_bancaire': round(d4, 1),
            'signal_appetence': round(d5, 1),
        }

    def _get_rating(self, score):
        if score >= 85: return 'A', 'APPROUVE', 'Profil premium'
        if score >= 70: return 'B', 'APPROUVE', 'Bon profil'
        if score >= 55: return 'C', 'APPROUVE', 'Profil acceptable'
        if score >= 40: return 'D', 'INSTRUCTION', 'Profil à instruire'
        return 'E', 'REFUS', 'Refus'

    def _generate_motifs(self, p, dimensions):
        s = p['signaletique']
        sv = p['solvabilite']
        c = p['comportement']

        points_forts = []
        points_attention = []

        if sv['taux_endettement_actuel'] < 0.20:
            points_forts.append(
                f"Taux d'endettement très faible ({sv['taux_endettement_actuel']:.0%})"
            )
        elif sv['taux_endettement_actuel'] > 0.30:
            points_attention.append(
                f"Taux d'endettement à {sv['taux_endettement_actuel']:.0%} "
                f"(proche du seuil de vigilance 30%)"
            )

        if sv['capacite_mensuelle_residuelle'] > 3000:
            points_forts.append(
                f"Capacité résiduelle confortable : "
                f"{sv['capacite_mensuelle_residuelle']:.0f} MAD/mois"
            )
        elif sv['capacite_mensuelle_residuelle'] < 1500:
            points_attention.append(
                f"Capacité résiduelle limitée : "
                f"{sv['capacite_mensuelle_residuelle']:.0f} MAD/mois"
            )

        if c['nb_decouverts_3m'] == 0:
            points_forts.append("Aucun découvert sur les 3 derniers mois")
        elif c['nb_decouverts_3m'] > 5:
            points_attention.append(
                f"{c['nb_decouverts_3m']} jours de découvert sur 3 mois"
            )

        if c['tendance_compte'] > 20:
            points_forts.append(
                f"Tendance d'épargne positive (+{c['tendance_compte']:.0f} MAD/jour)"
            )
        elif c['tendance_compte'] < -20:
            points_attention.append(
                f"Tendance de compte négative ({c['tendance_compte']:.0f} MAD/jour)"
            )

        if s['type_revenu'] == 'SALARIE':
            points_forts.append("Revenu stable de salarié")
        elif s['type_revenu'] == 'COMMERCANT':
            points_attention.append("Revenu variable de commerçant")

        if sv['ratio_epargne'] > 2:
            points_forts.append(
                f"Ratio d'épargne excellent ({sv['ratio_epargne']:.1f} mois de revenu)"
            )
        elif sv['ratio_epargne'] < 0.5:
            points_attention.append(
                f"Ratio d'épargne faible ({sv['ratio_epargne']:.1f} mois de revenu)"
            )

        return points_forts, points_attention
