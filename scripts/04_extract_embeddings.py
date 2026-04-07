"""
Étape 04 — Extraction des embeddings LSTM + fusion avec features statiques.
Produit final_train.parquet (52 features).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
from src.sequence_builder import SequenceBuilder
from src.lstm_model import LSTMEncoder, LSTMTrainer
from src.feature_merger import FeatureMerger
from config import LSTM_CONFIG

if __name__ == "__main__":
    print("=== Étape 04 : Extraction embeddings + fusion ===\n")

    df = pd.read_parquet("data/processed/modeling_base.parquet")
    print(f"Base : {df.shape}")

    # Charger séquences (scaler déjà fitté)
    seq_builder = SequenceBuilder("models/lstm_scaler.pkl")
    sequences, seq_ids = seq_builder.transform(df)
    print(f"Séquences : {sequences.shape}")

    # Charger LSTM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = LSTMEncoder(**{
        k: LSTM_CONFIG[k]
        for k in ['input_size', 'hidden_size', 'num_layers',
                  'dropout', 'bidirectional', 'embedding_dim']
    })
    lstm_model.load_state_dict(
        torch.load("models/lstm_encoder.pt", map_location=device, weights_only=True)
    )
    lstm_model.to(device).eval()
    print(f"LSTM chargé sur {device}")

    # Extraire embeddings
    trainer    = LSTMTrainer(LSTM_CONFIG, save_dir="models")
    embeddings = trainer.extract_embeddings(lstm_model, sequences)
    print(f"Embeddings : {embeddings.shape}")

    # Fusion
    merger   = FeatureMerger(LSTM_CONFIG['embedding_dim'])
    df_final = merger.merge(df, embeddings, seq_ids)
    print(f"Après fusion : {df_final.shape}")

    # Sauvegarder embeddings séparément
    from config import LSTM_EMBEDDING_COLS
    df_emb = pd.DataFrame(embeddings, columns=LSTM_EMBEDDING_COLS)
    df_emb['id_client'] = seq_ids
    df_emb.to_parquet("data/processed/embeddings_train.parquet", index=False)

    df_final.to_parquet("data/processed/final_train.parquet", index=False)
    print(f"\n✅ Sauvegardé → data/processed/final_train.parquet  ({df_final.shape})")
    print(f"   Embeddings  → data/processed/embeddings_train.parquet")
