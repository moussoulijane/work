"""
Couche 4 : Narration LLM via Mistral 7B local (Ollama).

IMPORTANT : le LLM ne prend AUCUNE décision. Il reçoit le profil enrichi,
la décision risque et l'offre commerciale déjà calculés, et produit
UNIQUEMENT du texte en JSON strict.

Si Ollama est down, JSON invalide ou mot banni → fallback template.
"""
import requests
import json
import os


class LLMNarrator:

    def __init__(self, config):
        self.config = config
        self.base_url = config['base_url']
        self.model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.timeout = config['timeout']
        self.retries = config['retries']

        self.system_prompt = self._load_prompt('prompts/narrator_system.txt')
        self.user_template = self._load_prompt('prompts/narrator_user_template.txt')

    def _load_prompt(self, path):
        if not os.path.exists(path):
            return self._default_prompt(path)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _default_prompt(self, path):
        if 'system' in path:
            return (
                "Tu es un expert conseiller crédit chez Attijariwafa Bank avec "
                "15 ans d'expérience. Tu rédiges des fiches argumentées pour les "
                "chargés de clientèle. Tu ne prends AUCUNE décision — tu argumentes "
                "des décisions déjà prises par l'analyse risque et la grille tarifaire. "
                "Tu écris en français professionnel, factuel, orienté bénéfice client. "
                "Tu ne fais jamais d'exagération commerciale."
            )
        return "{profile_block}\n\n{risk_block}\n\n{offer_block}\n\n{format_block}"

    def generate(self, profile_enriched, risk_decision, commercial_offers):
        """
        Returns:
            dict avec resume_executif, argumentation_commerciale,
            justification_taux, points_de_vigilance, script_appel, source
        """
        if risk_decision['decision'] == 'REFUS':
            return self._generate_refusal_narration(profile_enriched, risk_decision)

        user_prompt = self._build_user_prompt(profile_enriched, risk_decision, commercial_offers)

        for attempt in range(self.retries):
            try:
                response_text = self._call_ollama(user_prompt, attempt)
                narration = self._parse_response(response_text)

                from agent.validator import NarrationValidator
                validator = NarrationValidator()
                if validator.validate(narration, commercial_offers):
                    narration['source'] = 'llm'
                    return narration
            except Exception as e:
                print(f"   Tentative LLM {attempt + 1}/{self.retries} échouée : {e}")

        return self._fallback_template(profile_enriched, risk_decision, commercial_offers)

    def _build_user_prompt(self, profile, risk, offers):
        s = profile['signaletique']
        sv = profile['solvabilite']
        c = profile['comportement']
        a = profile['appetence']

        profile_block = (
            f"PROFIL CLIENT #{profile['id_client']}\n"
            f"{'─' * 40}\n"
            f"Signalétique : {s['age']} ans, {s['type_revenu']}, segment {s['segment']}\n"
            f"Revenu : {s['revenu_principal']:.0f} MAD/mois\n\n"
            f"Comportement bancaire (3 derniers mois) :\n"
            f"- Solde moyen : {c['solde_moyen']:.0f} MAD\n"
            f"- Tendance : {c['tendance_compte']:.0f} MAD/jour\n"
            f"- Découverts : {c['nb_decouverts_3m']} jours\n"
            f"- Stabilité revenu : {c['stabilite_revenu_proxy']:.0%}\n\n"
            f"Engagements actuels :\n"
            f"- Taux d'endettement : {sv['taux_endettement_actuel']:.0%}\n"
            f"- Capacité résiduelle : {sv['capacite_mensuelle_residuelle']:.0f} MAD/mois\n"
            f"- Ratio d'épargne : {sv['ratio_epargne']:.1f} mois de revenu\n\n"
            f"Activité digitale :\n"
            f"- {a['count_simul']} simulations crédit\n"
            f"- Dont {a['count_simul_recent']} le mois dernier\n"
            f"- Score d'appétence : {a['score']:.0%} ({a['classe']})\n"
        )

        pf_lines = '\n'.join(f'- {x}' for x in risk['points_forts']) or '- Aucun'
        pa_lines = '\n'.join(f'- {x}' for x in risk['points_attention']) or '- Aucun'
        risk_block = (
            f"DÉCISION RISQUE : {risk['decision']} (Note {risk['note']}, score {risk['score']}/100)\n\n"
            f"Points forts :\n{pf_lines}\n\n"
            f"Points d'attention :\n{pa_lines}\n"
        )

        op = offers['offre_principale']
        oc = offers['offres_alternatives']['confort']
        oe = offers['offres_alternatives']['economie']
        offer_block = (
            f"OFFRE PRINCIPALE À ARGUMENTER :\n"
            f"- Montant : {op['montant']:,} MAD\n"
            f"- Durée : {op['duree_mois']} mois\n"
            f"- Taux : {op['taux_annuel']:.2%}\n"
            f"- Mensualité : {op['mensualite']:.0f} MAD\n"
            f"- Coût total : {op['cout_total_credit']:.0f} MAD\n"
            f"- TEG : {op['teg']:.2%}\n\n"
            f"Offres alternatives (mentionner brièvement) :\n"
            f"- CONFORT : {oc['montant']:,} MAD / {oc['duree_mois']} mois / {oc['mensualite']:.0f} MAD/mois\n"
            f"- ECONOMIE : {oe['montant']:,} MAD / {oe['duree_mois']} mois / {oe['mensualite']:.0f} MAD/mois\n"
        )

        format_block = (
            'PRODUIS UNIQUEMENT un JSON valide avec cette structure exacte :\n\n'
            '{\n'
            '  "resume_executif": "2 phrases max résumant profil et recommandation",\n'
            '  "argumentation_commerciale": "4-6 phrases justifiant l\'offre",\n'
            '  "justification_taux": "1-2 phrases expliquant le taux",\n'
            '  "points_de_vigilance": ["point 1", "point 2", "point 3"],\n'
            '  "script_appel": "3-4 phrases à dire au client"\n'
            '}\n\n'
            'RÈGLES :\n'
            '- Pas d\'exagération ("exceptionnel", "unique", "opportunité rare")\n'
            '- Pas de jargon technique (SHAP, embedding, LSTM, modèle)\n'
            '- Mentionner les chiffres réels\n'
            '- Français professionnel, phrases courtes\n'
            '- Ton respectueux\n'
        )

        return profile_block + '\n' + risk_block + '\n' + offer_block + '\n' + format_block

    def _call_ollama(self, user_prompt, attempt):
        temp = self.temperature + attempt * self.config.get('retry_temp_increment', 0.1)
        payload = {
            'model': self.model,
            'prompt': user_prompt,
            'system': self.system_prompt,
            'temperature': temp,
            'num_predict': self.max_tokens,
            'stream': False,
            'format': 'json',
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()['response']

    def _parse_response(self, response_text):
        text = response_text.strip()
        # Strip markdown code fences if present
        if text.startswith('```'):
            parts = text.split('```')
            text = parts[1] if len(parts) > 1 else text
            if text.lower().startswith('json'):
                text = text[4:]
        return json.loads(text.strip())

    def _fallback_template(self, profile, risk, offers):
        from agent.fallback_templates import build_template_narration
        return build_template_narration(profile, risk, offers)

    def _generate_refusal_narration(self, profile, risk):
        from agent.fallback_templates import build_refusal_narration
        return build_refusal_narration(profile, risk)
