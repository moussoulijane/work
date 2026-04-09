"""
Validation des narrations produites par le LLM.
Si une narration échoue → fallback template.
"""
import re
import yaml


class NarrationValidator:

    def __init__(self, blacklist_path='agent_config/blacklist_words.yaml'):
        try:
            with open(blacklist_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.blacklist = [w.lower() for w in data.get('banned_words', [])]
        except FileNotFoundError:
            self.blacklist = [
                'exceptionnel', 'unique', 'extraordinaire', 'garantie',
                'sans risque', '100%', 'opportunité rare', 'incroyable',
                'shap', 'embedding', 'lstm', 'catboost', 'machine learning',
            ]

        from agent.config_agent import VALIDATION_CONFIG
        self.config = VALIDATION_CONFIG

    def validate(self, narration: dict, commercial_offers: dict) -> bool:
        """
        Retourne True si la narration est valide, False sinon.
        """
        # 1. Clés requises
        for key in self.config['required_keys']:
            if key not in narration:
                return False
            if not narration[key]:
                return False

        # 2. Longueurs max
        checks = [
            ('resume_executif', self.config['max_resume_chars']),
            ('argumentation_commerciale', self.config['max_argumentation_chars']),
            ('script_appel', self.config['max_script_chars']),
        ]
        for key, max_len in checks:
            if key in narration and isinstance(narration[key], str):
                if len(narration[key]) > max_len:
                    return False

        # 3. Mots bannis
        for key in ['resume_executif', 'argumentation_commerciale', 'script_appel']:
            text = narration.get(key, '')
            if isinstance(text, str):
                text_lower = text.lower()
                for word in self.blacklist:
                    if word in text_lower:
                        return False

        # 4. Cohérence chiffres — si offre disponible, le montant doit apparaître
        if commercial_offers and commercial_offers.get('offre_principale'):
            op = commercial_offers['offre_principale']
            montant_str = str(op['montant'])
            # Au moins un des textes clés doit mentionner le montant
            texts = [
                narration.get('resume_executif', ''),
                narration.get('argumentation_commerciale', ''),
                narration.get('script_appel', ''),
            ]
            # On accepte si montant présent dans au moins un champ (sans espace)
            montant_found = any(
                montant_str.replace(',', '') in t.replace(' ', '').replace(',', '')
                for t in texts
            )
            if not montant_found:
                return False

        # 5. points_de_vigilance doit être une liste non vide
        pdv = narration.get('points_de_vigilance', [])
        if not isinstance(pdv, list) or len(pdv) == 0:
            return False

        return True

    def sanitize(self, narration: dict) -> dict:
        """
        Nettoie une narration partiellement valide :
        tronque les textes trop longs, retire les mots bannis.
        """
        result = dict(narration)
        checks = [
            ('resume_executif', self.config['max_resume_chars']),
            ('argumentation_commerciale', self.config['max_argumentation_chars']),
            ('script_appel', self.config['max_script_chars']),
        ]
        for key, max_len in checks:
            if key in result and isinstance(result[key], str):
                if len(result[key]) > max_len:
                    result[key] = result[key][:max_len - 3] + '...'
        return result
