"""
Couche 1 : Enrichissement du profil client.

Transforme la sortie brute du pipeline ML en un profil enrichi avec
des indicateurs métier calculés. Aucune IA ici — que du déterministe.

Input  : dict avec les 52 features + proba + SHAP top 5
Output : dict avec profil enrichi (55+ champs)
"""
import numpy as np
import pandas as pd


class ProfileEnricher:
    """
    Enrichit un profil client avec des indicateurs métier dérivés.
    
    Les indicateurs calculés sont utilisés par :
    - L'expert risque (couche 2) pour appliquer les règles
    - L'expert commercial (couche 3) pour calibrer l'offre
    - Le LLM (couche 4) pour contextualiser la narration
    """
    
    def __init__(self):
        pass
    
    def enrich(self, client_row, proba, top_5_shap, lstm_shap_aggregated):
        """
        Args:
            client_row: pd.Series ou dict avec les 52 features + id_client
            proba: float — probabilité CatBoost
            top_5_shap: list de dicts — top 5 features SHAP du client
            lstm_shap_aggregated: float — SHAP agrégé des 32 dims LSTM
        
        Returns:
            dict — profil enrichi avec ~55 champs
        """
        # Extraire les données brutes (robuste dict ou Series)
        def get(key, default=0):
            if isinstance(client_row, dict):
                return client_row.get(key, default)
            return client_row.get(key, default) if key in client_row else default
        
        revenu = float(get('revenu_principal', 0))
        mensualite_actif = float(get('total_mensualite_actif', 0))
        mensualite_immo = float(get('mensualite_immo', 0))
        mensualite_totale = mensualite_actif + mensualite_immo
        
        # ── Indicateurs de solvabilité ──
        taux_endettement = mensualite_totale / max(revenu, 1)
        capacite_residuelle = max(0, revenu * 0.40 - mensualite_totale)
        ratio_epargne = float(get('solde_moyen', 0)) / max(revenu, 1)
        
        # ── Indicateurs comportementaux ──
        solde_moyen = float(get('solde_moyen', 0))
        solde_std = float(get('solde_std', 0))
        stabilite_revenu_proxy = 1 - (solde_std / max(abs(solde_moyen), 1))
        stabilite_revenu_proxy = max(0, min(1, stabilite_revenu_proxy))
        
        tendance = float(get('solde_tendance', 0))
        nb_decouverts = int(get('solde_nb_negatif', 0))
        
        profil_sain = (nb_decouverts <= 2) and (tendance >= 0)
        profil_fragile = (nb_decouverts >= 5) or (ratio_epargne < 0.3)
        
        # ── Indicateurs d'appétence ──
        count_simul = int(get('count_simul', 0))
        count_simul_recent = int(get('count_simul_mois_n_1', 0))
        intensite_digitale = count_simul + 2 * count_simul_recent
        
        if proba > 0.70:
            appetence_class = 'FORTE'
        elif proba > 0.40:
            appetence_class = 'MOYENNE'
        else:
            appetence_class = 'FAIBLE'
        
        # ── Zone de risque ──
        zone = self._determine_zone(
            appetence_class, profil_sain, profil_fragile, revenu
        )
        
        # ── Assembler le profil enrichi ──
        return {
            'id_client': get('id_client'),
            
            'signaletique': {
                'age': int(get('age', 0)),
                'type_revenu': str(get('type_revenu', 'INCONNU')),
                'segment': str(get('segment', 'MASS')),
                'revenu_principal': revenu,
            },
            
            'solvabilite': {
                'taux_endettement_actuel': round(taux_endettement, 4),
                'capacite_mensuelle_residuelle': round(capacite_residuelle, 2),
                'mensualites_actuelles': round(mensualite_totale, 2),
                'ratio_epargne': round(ratio_epargne, 2),
            },
            
            'comportement': {
                'solde_moyen': round(solde_moyen, 2),
                'solde_min': float(get('solde_min', 0)),
                'solde_max': float(get('solde_max', 0)),
                'nb_decouverts_3m': nb_decouverts,
                'stabilite_revenu_proxy': round(stabilite_revenu_proxy, 4),
                'tendance_compte': round(tendance, 2),
                'profil_sain': profil_sain,
                'profil_fragile': profil_fragile,
            },
            
            'appetence': {
                'score': round(float(proba), 4),
                'classe': appetence_class,
                'count_simul': count_simul,
                'count_simul_recent': count_simul_recent,
                'intensite_digitale': intensite_digitale,
                'profil_temporel_lstm': round(float(lstm_shap_aggregated), 4),
            },
            
            'zone_risque': zone,
            
            'top_5_shap': top_5_shap,
            
            # Données brutes pour traçabilité
            'raw_features': dict(client_row) if isinstance(client_row, dict) 
                            else client_row.to_dict(),
        }
    
    def _determine_zone(self, appetence_class, profil_sain, profil_fragile, revenu):
        """Détermine la zone de risque selon la grille 6 zones."""
        if appetence_class == 'FORTE':
            if profil_sain and revenu >= 7000:
                return 'STAR'
            elif profil_sain:
                return 'CROISSANCE'
            else:
                return 'PRUDENCE'
        elif appetence_class == 'MOYENNE':
            if profil_sain:
                return 'DORMANT'
            else:
                return 'PRUDENCE'
        else:  # FAIBLE
            if profil_sain:
                return 'FIDELISATION'
            else:
                return 'EXCLUSION'

