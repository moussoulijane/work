"""
Configuration spécifique à l'agent IA.
Importe la config de base et ajoute les paramètres agent.
"""
from config import *  # Import tout du config.py existant

# ========== LLM LOCAL (Ollama + Mistral) ==========
LLM_CONFIG = {
    'provider': 'ollama',
    'base_url': 'http://localhost:11434',
    'model': 'mistral',
    'temperature': 0.3,
    'max_tokens': 800,
    'timeout': 60,
    'retries': 3,
    'retry_temp_increment': 0.1,
}

# ========== CHEMINS AGENT ==========
AGENT_PATHS = {
    'business_rules': 'agent_config/business_rules.yaml',
    'pricing_grid': 'agent_config/pricing_grid.yaml',
    'blacklist_words': 'agent_config/blacklist_words.yaml',
    'prompt_system': 'prompts/narrator_system.txt',
    'prompt_template': 'prompts/narrator_user_template.txt',
    'fiches_output': 'outputs/fiches',
}

# ========== VALIDATION FICHES ==========
VALIDATION_CONFIG = {
    'max_resume_chars': 250,
    'max_argumentation_chars': 800,
    'max_script_chars': 600,
    'required_keys': [
        'resume_executif',
        'argumentation_commerciale',
        'justification_taux',
        'points_de_vigilance',
        'script_appel'
    ],
    'min_language_fr_ratio': 0.8,
}

# ========== INTERFACE STREAMLIT ==========
UI_CONFIG = {
    'default_excel_path': 'data/raw/clients.xlsx',
    'page_title': 'AWB — Agent Crédit Conso',
    'page_icon': '🏦',
    'layout': 'wide',
    'theme_color': '#C8102E',
}

# ========== REBOND PRODUIT (pour les refus) ==========
REBOND_PRODUITS = {
    'revenu_insuffisant': {
        'produit': 'Compte épargne + découvert autorisé',
        'argument': 'Construire une base financière avant de solliciter un crédit'
    },
    'endettement_sature': {
        'produit': 'Rachat de crédits',
        'argument': 'Regrouper les crédits existants pour réduire la mensualité globale'
    },
    'incidents_paiement': {
        'produit': 'Accompagnement bancaire + micro-crédit encadré',
        'argument': 'Restaurer la santé financière progressivement'
    },
    'age_hors_cible': {
        'produit': 'Assurance décès-invalidité ou produit retraite',
        'argument': "Solutions adaptées à votre profil d'âge"
    },
    'profil_non_verifiable': {
        'produit': 'Rendez-vous en agence pour régularisation',
        'argument': 'Compléter votre dossier pour accéder à nos services crédit'
    },
}
