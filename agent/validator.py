"""
Validation des narrations produites par le LLM.
Si une narration échoue la validation → fallback template.
"""
import re
import yaml


class NarrationValidator:
    
    def __init__(self, blacklist_path='agent_config/blacklist_words.yaml'):
        try:
            with open(blacklist_path, 'r', encoding='utf-8') as f:
                self.blacklist = yaml.safe_load(f).get('banned_words', [])
        except FileNotFoundError:
            self.blacklist = [
                'exceptionnel', 'unique', 'extraordinaire', 'garantie',
                'sans risque', '100%', 'opportunité rare', 'incroyable'
            ]
        
        from agent.config_agent import VALIDATION_CONFIG
        self.config = VALIDATION_CONFIG
    
    def validate(self, narration, offers):
        """
        Returns True si la narration passe toutes les validations.
        """
        # 1. Structure
        for key in self.config['required_keys']:
            if key not in narration:
                print(f"   ❌ Clé manquante : {key}")
                return False
        
        # 2. Longueurs
        if len(narration['resume_executif']) > self.config['max_resume_chars']:
            print("   ❌ Résumé trop long")
            return False
        
        if len(narration['argumentation_commerciale']) > self.config['max_argumentation_chars']:
            print("   ❌ Argumentation trop longue")
            return False
        
        if len(narration['script_appel']) > self.config['max_script_chars']:
            print("   ❌ Script trop long")
            return False
        
        # 3. Mots bannis
        full_text = ' '.join([
            narration['resume_executif'],
            narration['argumentation_commerciale'],
            narration['justification_taux'],
            narration['script_appel'],
        ]).lower()
        
        for word in self.blacklist:
            if word.lower() in full_text:
                print(f"   ❌ Mot banni détecté : {word}")
                return False
        
        # 4. Cohérence chiffres (le montant doit apparaître quelque part)
        if offers and offers.get('offre_principale'):
            montant = offers['offre_principale']['montant']
            # Chercher le montant dans le texte (avec ou sans séparateur)
            montant_str = str(montant)
            montant_sep = f"{montant:,}".replace(',', ' ').replace(',', '.')
            if montant_str not in full_text and montant_sep not in full_text:
                # Tolérance : accepter aussi "85000" vs "85 000" vs "85,000"
                montant_clean = re.sub(r'[^\\d]', '', full_text)
                if str(montant) not in montant_clean:
                    print(f"   ⚠️ Montant {montant} non trouvé dans la narration")
                    # Warning mais pas bloquant
        
        # 5. Points de vigilance doit être une liste
        if not isinstance(narration['points_de_vigilance'], list):
            print("   ❌ points_de_vigilance doit être une liste")
            return False
        
        return True

