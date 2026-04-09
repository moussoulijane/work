"""
Couche 3 : Expert Commercial.

Transforme la décision risque en offre commerciale concrète :
- Montant maximum (3 contraintes, prendre le min)
- Durée adaptée
- Taux selon grille (note + durée + segment + bonifications)
- Mensualité, coût total, TEG
- 3 variantes : principale, confort, économie
- Rebond produit si refus
"""
import yaml
import math


class CommercialExpert:

    def __init__(self, grid_path='agent_config/pricing_grid.yaml'):
        with open(grid_path, 'r', encoding='utf-8') as f:
            self.grid = yaml.safe_load(f)
        from agent.config_agent import REBOND_PRODUITS
        self.rebonds = REBOND_PRODUITS

    def build_offers(self, profile_enriched, risk_decision):
        """
        Returns:
            dict avec offre_principale, offres_alternatives, rebond
        """
        if risk_decision['decision'] == 'REFUS':
            rebond_key = risk_decision.get('rebond_key', 'profil_non_verifiable')
            return {
                'offre_principale': None,
                'offres_alternatives': None,
                'rebond': self.rebonds.get(rebond_key, {
                    'produit': 'Rendez-vous en agence',
                    'argument': "Discuter d'autres solutions adaptées"
                })
            }

        note = risk_decision['note']
        segment = profile_enriched['signaletique']['segment']
        revenu = profile_enriched['signaletique']['revenu_principal']
        mensualite_max = (
            revenu * self.grid['ratio_mensualite_max']
            - profile_enriched['solvabilite']['mensualites_actuelles']
        )

        plafond_note = self.grid['plafonds'][note]
        montant_max_note = plafond_note['montant_max']
        duree_max_note = plafond_note['duree_max']

        mult_segment = self.grid['multiplicateurs_segment'].get(segment, 8)
        montant_max_segment = revenu * mult_segment

        taux_base_60 = self.grid['taux_base'][note]
        montant_max_capacite = self._pv_from_payment(
            max(mensualite_max, 0), taux_base_60 / 12, 60
        )

        montant_max = max(0, min(montant_max_note, montant_max_segment, montant_max_capacite))

        offre_principale = self._build_single_offer(
            montant_max * 0.70, 48, note, segment, profile_enriched, 'PRINCIPALE'
        )
        offre_confort = self._build_single_offer(
            montant_max * 0.60, duree_max_note, note, segment, profile_enriched, 'CONFORT'
        )
        offre_economie = self._build_single_offer(
            montant_max * 0.80, 36, note, segment, profile_enriched, 'ECONOMIE'
        )

        return {
            'offre_principale': offre_principale,
            'offres_alternatives': {
                'confort': offre_confort,
                'economie': offre_economie,
            },
            'rebond': None,
        }

    def _build_single_offer(self, montant_brut, duree, note, segment, profile, type_offre):
        montant = math.floor(montant_brut / 1000) * 1000
        montant = max(5000, montant)

        taux = self._compute_rate(note, duree, segment, profile)
        mensualite = self._compute_payment(montant, taux / 12, duree)
        cout_total = mensualite * duree - montant

        assurance_mensuelle = 0.0
        if duree > self.grid['assurance']['obligatoire_si_duree']:
            assurance_mensuelle = montant * self.grid['assurance']['taux_mensuel']

        frais = self.grid['frais_dossier_forfait']
        teg = self._compute_teg(montant, mensualite + assurance_mensuelle, duree, frais)

        arguments = {
            'PRINCIPALE': "Offre recommandée — équilibre optimal entre mensualité et coût",
            'CONFORT': "Mensualité allégée pour préserver votre budget",
            'ECONOMIE': "Meilleur coût total — remboursement rapide",
        }

        return {
            'type': type_offre,
            'montant': int(montant),
            'duree_mois': int(duree),
            'taux_annuel': round(taux, 4),
            'mensualite': round(mensualite, 2),
            'assurance_mensuelle': round(assurance_mensuelle, 2),
            'mensualite_totale': round(mensualite + assurance_mensuelle, 2),
            'cout_total_credit': round(cout_total, 2),
            'teg': round(teg, 4),
            'frais_dossier': frais,
            'argument': arguments[type_offre],
        }

    def _compute_rate(self, note, duree, segment, profile):
        taux = self.grid['taux_base'][note]
        maj = self.grid['majorations']

        if 60 < duree <= 72:
            taux += maj['duree_60_72']
        elif duree > 72:
            taux += maj['duree_60_72'] + maj['duree_sup_72']

        if segment == 'PREMIUM':
            taux += maj['segment_premium']
        elif segment == 'PRIVE':
            taux += maj['segment_prive']

        if profile['raw_features'].get('mensualite_immo', 0) > 0:
            taux += maj['client_fidele']

        if profile['comportement']['tendance_compte'] < -20:
            taux += maj['tendance_negative']

        if note == 'A' and profile['solvabilite']['ratio_epargne'] > 3:
            taux += maj['profil_tres_sain']

        taux = max(self.grid['taux_plancher'], min(self.grid['taux_plafond'], taux))
        return taux

    def _compute_payment(self, P, r, n):
        if r == 0:
            return P / n
        return P * r / (1 - (1 + r) ** -n)

    def _pv_from_payment(self, pmt, r, n):
        if r == 0 or pmt <= 0:
            return 0.0
        return pmt * (1 - (1 + r) ** -n) / r

    def _compute_teg(self, montant, mensualite, duree, frais):
        if montant <= 0:
            return 0.0
        total_paye = mensualite * duree + frais
        cout = total_paye - montant
        return (cout / montant) * (12 / duree) * 1.1
