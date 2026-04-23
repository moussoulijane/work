"""
Orchestrateur de l'agent IA complet.

Enchaîne les 4 couches :
1. Enricher → profil enrichi
2. RiskExpert → décision + note
3. CommercialExpert → offres
4. LLMNarrator → narration

Utilisé par l'interface Streamlit.
"""
from datetime import datetime
import json
import os

from agent.enricher import ProfileEnricher
from agent.risk_expert import RiskExpert
from agent.commercial_expert import CommercialExpert
from agent.llm_narrator import LLMNarrator
from agent.config_agent import LLM_CONFIG, AGENT_PATHS


class AgentOrchestrator:
    """
    Point d'entrée unique pour l'agent IA.
    
    Usage :
        orch = AgentOrchestrator()
        fiche = orch.run(client_row, proba, top_5_shap)
    """
    
    def __init__(self):
        self.enricher = ProfileEnricher()
        self.risk_expert = RiskExpert(AGENT_PATHS['business_rules'])
        self.commercial_expert = CommercialExpert(AGENT_PATHS['pricing_grid'])
        self.narrator = LLMNarrator(LLM_CONFIG)
    
    def run(self, client_row, proba, top_5_shap):
        """
        Exécute les 4 couches et retourne la fiche complète.

        Args:
            client_row: pd.Series ou dict avec les 55 features + id_client
            proba: float — probabilité calibrée (ensemble CatBoost + LGBM)
            top_5_shap: list[dict] — top 5 features SHAP

        Returns:
            dict — fiche complète prête à afficher/sauvegarder
        """
        # Couche 1 : enrichissement
        profile = self.enricher.enrich(client_row, proba, top_5_shap)
        
        # Couche 2 : expert risque
        risk_decision = self.risk_expert.evaluate(profile)
        
        # Couche 3 : expert commercial
        offers = self.commercial_expert.build_offers(profile, risk_decision)
        
        # Couche 4 : narration LLM
        narration = self.narrator.generate(profile, risk_decision, offers)
        
        # Assembler la fiche finale
        fiche = {
            'metadata': {
                'id_client': profile['id_client'],
                'date_generation': datetime.now().isoformat(),
                'version_agent': '1.0',
                'version_pipeline': 'catboost_lgbm_ensemble_v2',
            },
            'profil': profile,
            'analyse_risque': risk_decision,
            'offres': offers,
            'narration': narration,
        }
        
        # Sauvegarder
        self._save_fiche(fiche)
        
        return fiche
    
    def _save_fiche(self, fiche):
        """Sauvegarde la fiche en JSON dans outputs/fiches/."""
        os.makedirs(AGENT_PATHS['fiches_output'], exist_ok=True)
        
        id_client = fiche['metadata']['id_client']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        path = os.path.join(
            AGENT_PATHS['fiches_output'],
            f'fiche_client_{id_client}_{timestamp}.json'
        )
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(fiche, f, ensure_ascii=False, indent=2, default=str)

