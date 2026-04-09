"""
Couche 4 : Narration LLM via Mistral 7B local (Ollama).

IMPORTANT : le LLM ne prend AUCUNE décision. Il reçoit :
- Le profil enrichi
- La décision risque (déjà prise)
- L'offre commerciale (déjà calculée)

Et il produit UNIQUEMENT du texte rédigé selon un format JSON strict.

Si le LLM échoue (Ollama down, JSON invalide, mot banni) → fallback template.
"""
import requests
import json
import os
from pathlib import Path


class LLMNarrator:
    
    def __init__(self, config):
        self.config = config
        self.base_url = config['base_url']
        self.model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.timeout = config['timeout']
        self.retries = config['retries']
        
        # Charger les prompts
        self.system_prompt = self._load_prompt('prompts/narrator_system.txt')
        self.user_template = self._load_prompt('prompts/narrator_user_template.txt')
    
    def _load_prompt(self, path):
        if not os.path.exists(path):
            return self._default_prompt(path)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _default_prompt(self, path):
        """Fallback si le fichier prompt n'existe pas."""
        if 'system' in path:
            return (
                "Tu es un expert conseiller crédit chez Attijariwafa Bank avec "
                "15 ans d'expérience. Tu rédiges des fiches argumentées pour les "
                "chargés de clientèle. Tu ne prends AUCUNE décision — tu argumentes "
                "des décisions déjà prises par l'analyse risque et la grille tarifaire. "
                "Tu écris en français professionnel, factuel, orienté bénéfice client. "
                "Tu ne fais jamais d'exagération commerciale."
            )
        return "{profile_block}\
\
{risk_block}\
\
{offer_block}\
\
{format_block}"
    
    def generate(self, profile_enriched, risk_decision, commercial_offers):
        """
        Génère la narration via Mistral.
        
        Returns:
            dict avec :
                - resume_executif
                - argumentation_commerciale
                - justification_taux
                - points_de_vigilance
                - script_appel
                - source: 'llm' | 'fallback_template'
        """
        # Cas refus → narration refus (pas besoin de LLM compliqué)
        if risk_decision['decision'] == 'REFUS':
            return self._generate_refusal_narration(profile_enriched, risk_decision)
        
        # Construire le prompt
        user_prompt = self._build_user_prompt(
            profile_enriched, risk_decision, commercial_offers
        )
        
        # Appeler Mistral avec retries
        for attempt in range(self.retries):
            try:
                response = self._call_ollama(user_prompt, attempt)
                narration = self._parse_response(response)
                
                # Valider
                from agent.validator import NarrationValidator
                validator = NarrationValidator()
                if validator.validate(narration, commercial_offers):
                    narration['source'] = 'llm'
                    return narration
            except Exception as e:
                print(f"   ⚠️ Tentative {attempt+1} échouée : {e}")
        
        # Fallback template
        return self._fallback_template(profile_enriched, risk_decision, commercial_offers)
    
    def _build_user_prompt(self, profile, risk, offers):
        """Construit le prompt user avec toutes les données injectées."""
        s = profile['signaletique']
        sv = profile['solvabilite']
        c = profile['comportement']
        a = profile['appetence']
        
        profile_block = f"""
PROFIL CLIENT #{profile['id_client']}
─────────────────────────
Signalétique : {s['age']} ans, {s['type_revenu']}, segment {s['segment']}
Revenu : {s['revenu_principal']:.0f} MAD/mois

Comportement bancaire (3 derniers mois) :
- Solde moyen : {c['solde_moyen']:.0f} MAD
- Tendance : {c['tendance_compte']:.0f} MAD/jour
- Découverts : {c['nb_decouverts_3m']} jours
- Stabilité revenu : {c['stabilite_revenu_proxy']:.0%}

Engagements actuels :
- Taux d'endettement : {sv['taux_endettement_actuel']:.0%}
- Capacité résiduelle : {sv['capacite_mensuelle_residuelle']:.0f} MAD/mois
- Ratio d'épargne : {sv['ratio_epargne']:.1f} mois de revenu

Activité digitale :
- {a['count_simul']} simulations crédit
- Dont {a['count_simul_recent']} le mois dernier
- Score d'appétence : {a['score']:.0%} ({a['classe']})
"""
        
        risk_block = f"""
DÉCISION RISQUE : {risk['decision']} (Note {risk['note']}, score {risk['score']}/100)

Points forts :
{chr(10).join(f'- {pf}' for pf in risk['points_forts'])}

Points d'attention :
{chr(10).join(f'- {pa}' for pa in risk['points_attention'])}
"""
        
        op = offers['offre_principale']
        oc = offers['offres_alternatives']['confort']
        oe = offers['offres_alternatives']['economie']
        
        offer_block = f"""
OFFRE PRINCIPALE À ARGUMENTER :
- Montant : {op['montant']:,} MAD
- Durée : {op['duree_mois']} mois
- Taux : {op['taux_annuel']:.2%}
- Mensualité : {op['mensualite']:.0f} MAD
- Coût total : {op['cout_total_credit']:.0f} MAD
- TEG : {op['teg']:.2%}

Offres alternatives (mentionner brièvement) :
- CONFORT : {oc['montant']:,} MAD / {oc['duree_mois']} mois / {oc['mensualite']:.0f} MAD/mois
- ECONOMIE : {oe['montant']:,} MAD / {oe['duree_mois']} mois / {oe['mensualite']:.0f} MAD/mois
"""
        
        format_block = """
PRODUIS UNIQUEMENT un JSON valide avec cette structure exacte :

{
  "resume_executif": "2 phrases max résumant profil et recommandation",
  "argumentation_commerciale": "4-6 phrases justifiant l'offre en lien avec le profil",
  "justification_taux": "1-2 phrases expliquant pourquoi ce taux",
  "points_de_vigilance": ["point 1", "point 2", "point 3"],
  "script_appel": "3-4 phrases à dire au client pour introduire l'offre"
}

RÈGLES :
- Pas d'exagération ("exceptionnel", "unique", "opportunité rare")
- Pas de jargon technique (SHAP, embedding, scale_pos_weight)
- Mentionner les chiffres réels
- Français professionnel, phrases courtes
- Ton respectueux
"""
        
        return profile_block + risk_block + offer_block + format_block
    
    def _call_ollama(self, user_prompt, attempt):
        """Appel HTTP à Ollama."""
        temp = self.temperature + attempt * 0.1
        
        payload = {
            'model': self.model,
            'prompt': user_prompt,
            'system': self.system_prompt,
            'temperature': temp,
            'max_tokens': self.max_tokens,
            'stream': False,
            'format': 'json',
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()['response']
    
    def _parse_response(self, response_text):
        """Parse le JSON retourné par le LLM."""
        # Nettoyer
        text = response_text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        text = text.strip()
        
        return json.loads(text)
    
    def _fallback_template(self, profile, risk, offers):
        """Template de secours si le LLM échoue."""
        from agent.fallback_templates import build_template_narration
        return build_template_narration(profile, risk, offers)
    
    def _generate_refusal_narration(self, profile, risk):
        """Narration spécifique aux refus (ne passe pas par le LLM)."""
        from agent.fallback_templates import build_refusal_narration
        return build_refusal_narration(profile, risk)

