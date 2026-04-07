"""
Étape 03 — Optimisation Optuna des hyperparamètres LSTM.
Usage : python scripts/03_optimize_lstm.py [--n_trials 50]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import pandas as pd
import joblib
from src.sequence_builder import SequenceBuilder
from src.lstm_optimizer import optimize_lstm, build_config_from_optuna
from src.lstm_model import LSTMTrainer
from config import LSTM_CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--retrain", action="store_true",
                        help="Ré-entraîner avec les meilleurs params après optim")
    args = parser.parse_args()

    print(f"=== Étape 03 : Optimisation LSTM ({args.n_trials} trials) ===\n")

    df = pd.read_parquet("data/processed/modeling_base.parquet")

    seq_builder = SequenceBuilder("models/lstm_scaler.pkl")
    if os.path.exists("models/lstm_scaler.pkl"):
        sequences, seq_ids = seq_builder.transform(df)
    else:
        sequences, seq_ids = seq_builder.fit_transform(df)

    id_to_target = df.set_index('id_client')['target'].to_dict()
    targets      = np.array([id_to_target[i] for i in seq_ids])

    # Optimisation
    study, best_params = optimize_lstm(
        sequences, targets, n_trials=args.n_trials,
        save_path="models/optuna/lstm_study.pkl",
    )

    # Sauvegarder les résultats CSV
    import pandas as pd
    os.makedirs("outputs/optimization", exist_ok=True)
    df_results = study.trials_dataframe()
    df_results.to_csv("outputs/optimization/lstm_optuna_results.csv", index=False)
    print(f"Résultats Optuna → outputs/optimization/lstm_optuna_results.csv")

    if args.retrain:
        print(f"\n  Ré-entraînement avec les meilleurs hyperparamètres...")
        best_config = build_config_from_optuna(best_params)
        trainer     = LSTMTrainer(best_config, save_dir="models")
        id_to_target = df.set_index('id_client')['target'].to_dict()
        targets      = np.array([id_to_target[i] for i in seq_ids])
        trainer.train(sequences, targets)
        print(f"  ✅ Modèle optimisé sauvegardé → models/lstm_encoder.pt")
