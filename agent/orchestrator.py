"""
Orchestrateur agent IA — pipeline end-to-end pour un seul client.

Enchaîne les 4 couches :
  1. ProfileEnricher    — indicateurs métier dérivés
  2. RiskExpert         — score + décision + motifs
  3. CommercialExpert   — offres tarifées
  4. LLMNarrator        — texte rédigé (avec fallback)

Puis sauvegarde la fiche JSON dans outputs/fiches/.

Usage :
    from agent.orchestrator import AgentOrchestrator
    orch = AgentOrchestrator()
    fiche = orch.run(client_row, proba, top_5_shap, lstm_shap_agg)
"""
import os
import json
import logging
from datetime import datetime

from agent.config_agent import AGENT_PATHS, LLM_CONFIG
from agent.enricher import ProfileEnricher
from agent.risk_expert import RiskExpert
from agent.commercial_expert import CommercialExpert
from agent.llm_narrator import LLMNarrator

logger = logging.getLogger(__name__)


class AgentOrchestrator:

    def __init__(self):
        self.enricher   = ProfileEnricher()
        self.risk       = RiskExpert(AGENT_PATHS['business_rules'])
        self.commercial = CommercialExpert(AGENT_PATHS['pricing_grid'])
        self.narrator   = LLMNarrator(LLM_CONFIG)

    def run(
        self,
        client_row,
        proba: float,
        top_5_shap: list,
        lstm_shap_aggregated: float = 0.0,
        save: bool = True,
    ) -> dict:
        """
        Pipeline complet pour un client.

        Args:
            client_row           : pd.Series ou dict — features du client
            proba                : float — probabilité CatBoost
            top_5_shap           : list[dict] — top 5 features SHAP
            lstm_shap_aggregated : float — SHAP agrégé LSTM
            save                 : bool — sauvegarder la fiche JSON

        Returns:
            dict — fiche complète (profil + risque + offres + narration)
        """
        # ── Couche 1 : Enrichissement ──
        profile = self.enricher.enrich(client_row, proba, top_5_shap, lstm_shap_aggregated)
        logger.info(f"[Agent] Client {profile['id_client']} — zone {profile['zone_risque']}")

        # ── Couche 2 : Risque ──
        risk_result = self.risk.evaluate(profile)
        logger.info(
            f"[Agent] Décision : {risk_result['decision']} "
            f"(note {risk_result['note']}, score {risk_result['score']})"
        )

        # ── Couche 3 : Commercial ──
        offers = self.commercial.build_offers(profile, risk_result)

        # ── Couche 4 : Narration ──
        narration = self.narrator.generate(profile, risk_result, offers)
        logger.info(f"[Agent] Narration source : {narration.get('source', '?')}")

        # ── Assemblage de la fiche ──
        fiche = {
            'meta': {
                'id_client': profile['id_client'],
                'generated_at': datetime.now().isoformat(),
                'narration_source': narration.get('source', 'unknown'),
            },
            'profil': profile,
            'risque': risk_result,
            'offres': offers,
            'narration': narration,
        }

        if save:
            self._save_fiche(fiche)

        return fiche

    def _save_fiche(self, fiche: dict):
        """Sauvegarde la fiche JSON dans outputs/fiches/."""
        out_dir = AGENT_PATHS['fiches_output']
        os.makedirs(out_dir, exist_ok=True)

        id_client = fiche['meta']['id_client']
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(out_dir, f"fiche_{id_client}_{ts}.json")

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(fiche, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"[Agent] Fiche sauvegardée → {path}")
        return path
