"""
Étape 02 — Entraînement du LSTM encoder.
Charge modeling_base.parquet, fit_transform les séquences,
entraîne le LSTM, sauvegarde lstm_encoder.pt + lstm_scaler.pkl.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.sequence_builder import SequenceBuilder
from src.lstm_model import LSTMTrainer
from config import LSTM_CONFIG

if __name__ == "__main__":
    print("=== Étape 02 : Entraînement LSTM ===\n")

    df = pd.read_parquet("data/processed/modeling_base.parquet")
    print(f"Base chargée : {df.shape}")

    # Séquences
    seq_builder = SequenceBuilder("models/lstm_scaler.pkl")
    sequences, seq_ids = seq_builder.fit_transform(df)
    print(f"Séquences : {sequences.shape}")

    # Targets alignées
    id_to_target = df.set_index('id_client')['target'].to_dict()
    targets      = np.array([id_to_target[i] for i in seq_ids])
    print(f"Positifs : {targets.sum():.0f} / {len(targets)}")

    # Entraînement
    trainer = LSTMTrainer(LSTM_CONFIG, save_dir="models")
    model, history = trainer.train(sequences, targets)

    print(f"\nHistorique (val_auc) :")
    for i, auc in enumerate(history['val_auc']):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Epoch {i+1:3d} : {auc:.4f}")

    print(f"\n✅ Modèle sauvegardé → models/lstm_encoder.pt")
