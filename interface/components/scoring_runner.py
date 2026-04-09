"""
Exécute le pipeline ML sur UN SEUL client.
Réutilise les fonctions du dossier src/ (pas de duplication).
"""
import pandas as pd
import numpy as np
import torch
import os

from src.preprocessing import preprocess
from src.feature_engineering import add_balance_features
from src.sequence_builder import SequenceBuilder
from src.lstm_model import LSTMEncoder, LSTMTrainer
from src.feature_merger import FeatureMerger
from src.shap_engine import SHAPEngine
from catboost import CatBoostClassifier
from catboost import Pool

from config import (
    FEATURE_COLS, CAT_FEATURES, LSTM_CONFIG, LSTM_EMBEDDING_COLS,
    FEATURE_LABELS, revenu_treshold
)


class ScoringRunner:
    """
    Enchaîne le pipeline ML pour un seul client.
    Charge les modèles une seule fois (cache).
    """
    
    def __init__(self):
        self._check_artifacts()
        self.seq_builder = SequenceBuilder("models/lstm_scaler.pkl")
        self.lstm_model = self._load_lstm()
        self.lstm_trainer = LSTMTrainer(LSTM_CONFIG)
        self.merger = FeatureMerger(LSTM_CONFIG['embedding_dim'])
        self.model_low = self._load_catboost('low')
        self.model_high = self._load_catboost('high')
    
    def _check_artifacts(self):
        required = [
            "models/lstm_scaler.pkl",
            "models/lstm_encoder.pt",
            f"models/{revenu_treshold}_catboost_low.cbm",
            f"models/{revenu_treshold}_catboost_high.cbm",
        ]
        for f in required:
            if not os.path.exists(f):
                raise FileNotFoundError(
                    f"❌ {f} manquant. Lance d'abord : python main.py train"
                )
    
    def _load_lstm(self):
        model = LSTMEncoder(
            input_size=LSTM_CONFIG['input_size'],
            hidden_size=LSTM_CONFIG['hidden_size'],
            num_layers=LSTM_CONFIG['num_layers'],
            dropout=LSTM_CONFIG['dropout'],
            bidirectional=LSTM_CONFIG['bidirectional'],
            embedding_dim=LSTM_CONFIG['embedding_dim'],
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(
            "models/lstm_encoder.pt", map_location=device, weights_only=True
        ))
        model.to(device).eval()
        return model
    
    def _load_catboost(self, segment):
        model = CatBoostClassifier()
        model.load_model(f"models/{revenu_treshold}_catboost_{segment}.cbm")
        return model
    
    def score_client(self, client_row):
        """
        Score un client unique.
        
        Args:
            client_row: pd.Series avec toutes les features brutes
        
        Returns:
            dict avec proba, segment, top_5_shap, lstm_shap_aggregated
        """
        # Convertir en DataFrame 1 ligne pour réutiliser les fonctions existantes
        df = pd.DataFrame([client_row])
        
        # Preprocessing (même fonctions que le pipeline train/infer)
        df = preprocess(df)
        df = add_balance_features(df)
        
        # Séquences LSTM (transform seulement, charge le scaler)
        sequences, seq_ids = self.seq_builder.transform(df)
        
        # Extraction embeddings
        embeddings = self.lstm_trainer.extract_embeddings(
            self.lstm_model, sequences, batch_size=1
        )
        
        # Fusion features
        df_final = self.merger.merge(df, embeddings, seq_ids)
        
        # Sélectionner les 52 features
        available = [f for f in FEATURE_COLS if f in df_final.columns]
        X = df_final[available].copy()
        
        # Déterminer le segment
        revenu = float(df_final['revenu_principal'].iloc[0])
        if revenu <= revenu_treshold:
            model = self.model_low
            segment_model = 'LOW'
        else:
            model = self.model_high
            segment_model = 'HIGH'
        
        # Prédire
        proba = float(model.predict_proba(X)[0, 1])
        prediction = int(proba >= 0.5)
        
        # SHAP
        cat_idx = [available.index(c) for c in CAT_FEATURES if c in available]
        pool = Pool(X, cat_features=cat_idx)
        raw_shap = model.get_feature_importance(data=pool, type='ShapValues')
        shap_values = raw_shap[0, :-1]  # (52,) pour ce client
        base_value = float(raw_shap[0, -1])
        
        # Agréger les 32 dims LSTM
        lstm_indices = [i for i, n in enumerate(available) if n in LSTM_EMBEDDING_COLS]
        non_lstm_indices = [i for i, n in enumerate(available) if n not in LSTM_EMBEDDING_COLS]
        
        shap_non_lstm = shap_values[non_lstm_indices]
        shap_lstm = shap_values[lstm_indices]
        lstm_magnitude = np.sum(np.abs(shap_lstm))
        lstm_signed_sum = np.sum(shap_lstm)
        lstm_aggregated = lstm_magnitude * np.sign(lstm_signed_sum)
        
        names_non_lstm = [available[i] for i in non_lstm_indices]
        
        # Construire la liste réduite + score LSTM agrégé
        reduced_shap = list(zip(names_non_lstm, shap_non_lstm.tolist()))
        reduced_shap.append(('lstm_embedding', float(lstm_aggregated)))
        
        # Top 5 par |SHAP|
        reduced_shap.sort(key=lambda x: abs(x[1]), reverse=True)
        top_5 = reduced_shap[:5]
        
        # Formatter le top 5
        total_abs = sum(abs(s) for _, s in reduced_shap)
        top_5_formatted = []
        for rank, (feat, shap_val) in enumerate(top_5, 1):
            feat_value = df_final[feat].iloc[0] if feat in df_final.columns else None
            top_5_formatted.append({
                'rank': rank,
                'feature': feat,
                'feature_label': FEATURE_LABELS.get(feat, feat),
                'feature_value': float(feat_value) if feat_value is not None else None,
                'shap_value': float(shap_val),
                'direction': '+' if shap_val > 0 else '-',
                'contribution_pct': abs(shap_val) / total_abs * 100 if total_abs > 0 else 0,
            })
        
        # Enrichir client_row avec les indicateurs calculés
        enriched_row = df_final.iloc[0].to_dict()
        
        return {
            'client_row': enriched_row,
            'proba': proba,
            'prediction': prediction,
            'segment_model': segment_model,
            'top_5_shap': top_5_formatted,
            'lstm_shap_aggregated': float(lstm_aggregated),
            'base_value': base_value,
        }

