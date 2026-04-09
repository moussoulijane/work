"""
Enchaîne les 4 couches agent pour un client.
Appelé depuis l'interface Streamlit après scoring_runner.
"""
import streamlit as st
from agent.orchestrator import AgentOrchestrator


@st.cache_resource(show_spinner=False)
def get_orchestrator():
    """Singleton — l'orchestrateur est créé une seule fois."""
    return AgentOrchestrator()


def run_agent(client_row, proba: float, top_5_shap: list,
              lstm_shap_aggregated: float = 0.0) -> dict:
    """
    Lance le pipeline agent (4 couches) sur un client scoré.

    Returns:
        dict fiche complète (profil + risque + offres + narration)
    """
    orch = get_orchestrator()
    return orch.run(
        client_row=client_row,
        proba=proba,
        top_5_shap=top_5_shap,
        lstm_shap_aggregated=lstm_shap_aggregated,
        save=True,
    )
