"""
Exécute le pipeline ML complet sur un seul client.

Réutilise EXACTEMENT les mêmes étapes que main.py inference :
  preprocess → add_balance_features → add_advanced_features
  → SequenceBuilder.transform → LSTMEncoder.extract_embeddings
  → FeatureMerger → CatBoostTrainer.predict → SHAPEngine

Retourne : proba, top_5_shap, lstm_shap_aggregated
"""
import os
import numpy as np
import pandas as pd
import torch
import streamlit as st

from src.preprocessing import preprocess
from src.feature_engineering import add_balance_features, add_advanced_features
from src.sequence_builder import SequenceBuilder
from src.lstm_model import LSTMTrainer, build_lstm_encoder
from src.feature_merger import FeatureMerger
from src.catboost_trainer import _prepare_X
from src.shap_engine import SHAPEngine
from catboost import CatBoostClassifier
from config import (
    FEATURE_COLS, CAT_FEATURES, LSTM_CONFIG, LSTM_EMBEDDING_COLS,
    revenu_treshold, SHAP_CONFIG, FEATURE_LABELS,
)


@st.cache_resource(show_spinner="Chargement des modèles ML...")
def load_models(models_dir: str = "models"):
    """Charge les 4 artefacts ML une seule fois (cache Streamlit)."""
    required = {
        'scaler':  os.path.join(models_dir, 'lstm_scaler.pkl'),
        'encoder': os.path.join(models_dir, 'lstm_encoder.pt'),
        'low':     os.path.join(models_dir, f'{revenu_treshold}_catboost_low.cbm'),
        'high':    os.path.join(models_dir, f'{revenu_treshold}_catboost_high.cbm'),
    }
    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artefact '{name}' introuvable : {path}\n"
                "Lance d'abord : python main.py train"
            )

    # LSTM
    seq_builder = SequenceBuilder(scaler_path=required['scaler'])
    device = torch.device('cpu')
    encoder = build_lstm_encoder(LSTM_CONFIG)
    encoder.load_state_dict(torch.load(required['encoder'], map_location=device, weights_only=True))
    encoder.eval()

    # CatBoost
    cb_low  = CatBoostClassifier(); cb_low.load_model(required['low'])
    cb_high = CatBoostClassifier(); cb_high.load_model(required['high'])

    return seq_builder, encoder, cb_low, cb_high


def run_pipeline(client_row: pd.Series, models_dir: str = "models") -> dict:
    """
    Exécute le pipeline ML complet sur une ligne client.

    Returns:
        dict avec proba, top_5_shap, lstm_shap_aggregated
    """
    seq_builder, encoder, cb_low, cb_high = load_models(models_dir)

    # ── 1. Construire un DataFrame d'un client ──
    df = pd.DataFrame([client_row])

    # ── 2. Preprocessing (même pipeline que main.py) ──
    df = preprocess(df)
    df = add_balance_features(df)
    df = add_advanced_features(df)

    # ── 3. LSTM : séquences → embeddings ──
    sequences, ids = seq_builder.transform(df)

    trainer = LSTMTrainer(config=LSTM_CONFIG)
    embeddings = trainer.extract_embeddings(encoder, sequences, batch_size=1)

    # ── 4. Fusion features ──
    merger = FeatureMerger(embedding_dim=LSTM_CONFIG['embedding_dim'])
    df_final = merger.merge(df, embeddings, ids)

    # ── 5. Sélection segment ──
    revenu = float(df_final['revenu_principal'].iloc[0]) if 'revenu_principal' in df_final.columns else 0
    model = cb_high if revenu >= revenu_treshold else cb_low

    # ── 6. Prédiction ──
    X = _prepare_X(df_final, FEATURE_COLS, CAT_FEATURES)
    proba = float(model.predict_proba(X)[0, 1])

    # Calibration si disponible
    seg = 'high' if revenu >= revenu_treshold else 'low'
    cal_path = os.path.join(models_dir, f'calibrator_{seg}.pkl')
    if os.path.exists(cal_path):
        from src.calibration import ProbabilityCalibrator
        cal = ProbabilityCalibrator.load(cal_path)
        proba = float(cal.transform(np.array([proba]))[0])

    # ── 7. SHAP ──
    shap_engine = SHAPEngine(
        feature_cols=FEATURE_COLS,
        cat_features=CAT_FEATURES,
        lstm_embedding_cols=LSTM_EMBEDDING_COLS,
        feature_labels=FEATURE_LABELS,
        top_k=SHAP_CONFIG['top_k'],
    )
    shap_values = shap_engine.compute_shap(model, X)  # (1, n_features)

    # Top-5 SHAP pour ce client
    available_cols = [c for c in FEATURE_COLS if c in X.columns]
    top_5 = _extract_top5_shap(shap_values[0], available_cols, FEATURE_LABELS)

    # LSTM SHAP agrégé
    lstm_cols = [c for c in available_cols if c.startswith('lstm_emb_')]
    lstm_shap_agg = 0.0
    if lstm_cols and SHAP_CONFIG.get('aggregate_lstm_shap', True):
        lstm_idx = [available_cols.index(c) for c in lstm_cols if c in available_cols]
        if lstm_idx:
            lstm_shap_vals = shap_values[0, lstm_idx]
            lstm_shap_agg = float(np.sum(np.abs(lstm_shap_vals)) * np.sign(np.sum(lstm_shap_vals)))

    return {
        'proba': proba,
        'top_5_shap': top_5,
        'lstm_shap_aggregated': lstm_shap_agg,
        'df_final': df_final,
    }


def _extract_top5_shap(shap_row: np.ndarray, feature_cols: list, labels: dict) -> list:
    """Extrait les top-5 features par valeur absolue SHAP."""
    pairs = sorted(
        zip(feature_cols, shap_row),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    result = []
    for feat, val in pairs[:5]:
        result.append({
            'feature': feat,
            'label': labels.get(feat, feat),
            'shap_value': round(float(val), 4),
            'direction': '+' if val > 0 else '-',
        })
    return result
