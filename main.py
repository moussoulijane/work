"""
Orchestrateur principal.

Usage :
    python main.py train                  # Entraînement complet
    python main.py train --force_rebuild  # Ignorer les caches Parquet
    python main.py infer                  # Inférence sur inference_data.csv
    python main.py infer --data file.csv  # Inférence sur un fichier custom
    python main.py infer --skip_shap      # Sans SHAP (plus rapide)
"""
import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

from config import (
    TRAIN_FILES, INFER_FILES, LSTM_CONFIG, FEATURE_COLS, CAT_FEATURES,
    MODEL_PARAMS, SHAP_CONFIG, LSTM_EMBEDDING_COLS, FEATURE_LABELS,
    revenu_treshold,
)
from src.data_loading import load_base, merge_common
from src.preprocessing import preprocess
from src.feature_engineering import add_balance_features
from src.sequence_builder import SequenceBuilder
from src.lstm_model import LSTMEncoder, LSTMTrainer
from src.feature_merger import FeatureMerger
from src.catboost_trainer import CatBoostTrainer
from src.metrics import ModelEvaluator
from src.shap_engine import SHAPEngine
from src.error_analysis import ErrorAnalyzer


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

REQUIRED_ARTIFACTS = [
    "models/lstm_scaler.pkl",
    "models/lstm_encoder.pt",
    f"models/{revenu_treshold}_catboost_low.cbm",
    f"models/{revenu_treshold}_catboost_high.cbm",
]


def check_artifacts():
    missing = [f for f in REQUIRED_ARTIFACTS if not os.path.exists(f)]
    if missing:
        for f in missing:
            print(f"  ❌ Manquant : {f}")
        raise FileNotFoundError(
            "Artefacts manquants. Lance d'abord : python main.py train"
        )


def build_base(files: list[str], cache_path: str, force_rebuild: bool) -> pd.DataFrame:
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"  Cache trouvé → {cache_path}")
        return pd.read_parquet(cache_path)
    df = load_base(files)
    df = merge_common(df)
    df = preprocess(df)
    df = add_balance_features(df)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Base sauvegardée → {cache_path}")
    return df


# ─────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────

def run_train(args):
    print("\n" + "═" * 60)
    print("  MODE TRAIN")
    print("═" * 60)

    # 1-4. Données
    df = build_base(TRAIN_FILES, "data/processed/modeling_base.parquet", args.force_rebuild)
    assert 'target' in df.columns, "Colonne 'target' absente — vérifier les CSV train"
    print(f"  Base : {len(df):,} clients  |  positifs : {int(df['target'].sum()):,}")

    # 5-6. SequenceBuilder.fit_transform
    print("\n  ── LSTM : SequenceBuilder ──")
    seq_builder        = SequenceBuilder("models/lstm_scaler.pkl")
    sequences, seq_ids = seq_builder.fit_transform(df)

    # Aligner les targets avec les ids séquencés
    id_to_target = df.set_index('id_client')['target'].to_dict()
    targets      = np.array([id_to_target[i] for i in seq_ids])

    # 7. LSTM train
    print("\n  ── LSTM : Entraînement ──")
    lstm_trainer       = LSTMTrainer(LSTM_CONFIG, save_dir="models")
    lstm_model, history = lstm_trainer.train(sequences, targets)

    # 8. Extract embeddings
    print("\n  ── LSTM : Extraction embeddings ──")
    embeddings = lstm_trainer.extract_embeddings(lstm_model, sequences)
    print(f"  Embeddings : {embeddings.shape}")

    # 9. Feature merger
    print("\n  ── Fusion features (52) ──")
    merger   = FeatureMerger(LSTM_CONFIG['embedding_dim'])
    df_final = merger.merge(df, embeddings, seq_ids)
    df_final.to_parquet("data/processed/final_train.parquet", index=False)
    print(f"  final_train.parquet : {df_final.shape}")

    # 10. CatBoost train
    print("\n  ── CatBoost : Entraînement ──")
    cb_trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    model_low, model_high, results_low, results_high = cb_trainer.train(df_final)

    # 11. Métriques
    print("\n  ── Évaluation ──")
    evaluator = ModelEvaluator("outputs/metrics")
    m_low  = evaluator.evaluate(
        results_low['y_true'],  results_low['y_proba'],  "catboost_low_hybrid"
    )
    m_high = evaluator.evaluate(
        results_high['y_true'], results_high['y_proba'], "catboost_high_hybrid"
    )
    evaluator.compare([m_low, m_high])

    # 12. SHAP
    print("\n  ── SHAP ──")
    shap_engine = SHAPEngine(
        FEATURE_COLS, CAT_FEATURES, LSTM_EMBEDDING_COLS,
        FEATURE_LABELS, SHAP_CONFIG['top_k'],
    )
    shap_engine.run(df_final, model_low, model_high, "outputs/shap")

    # 13. Error analysis (sur le segment LOW pour illustration)
    print("\n  ── Analyse erreurs (segment LOW) ──")
    df_low_eval = df_final[df_final['revenu_principal'] <= revenu_treshold].copy()
    if len(df_low_eval) > 0 and 'target' in df_low_eval.columns:
        analyzer = ErrorAnalyzer()
        analyzer.analyze(
            df_low_eval,
            results_low['y_true'],
            results_low['y_proba'],
            output_dir="outputs/metrics",
        )

    print("\n" + "═" * 60)
    print("  ✅ TRAIN TERMINÉ")
    print("═" * 60)


# ─────────────────────────────────────────────────────────
# INFER
# ─────────────────────────────────────────────────────────

def run_infer(args):
    print("\n" + "═" * 60)
    print("  MODE INFER")
    print("═" * 60)

    # 1. Vérifier artefacts
    check_artifacts()
    print("  Artefacts OK")

    # 2-5. Données
    files = [args.data] if args.data else INFER_FILES
    df    = build_base(files, "data/processed/inference_base.parquet", args.force_rebuild)
    print(f"  Base inférence : {len(df):,} clients")

    # 6-7. SequenceBuilder.transform (charge scaler, pas de fit)
    print("\n  ── LSTM : SequenceBuilder ──")
    seq_builder        = SequenceBuilder("models/lstm_scaler.pkl")
    sequences, seq_ids = seq_builder.transform(df)

    # 8. Charger LSTM encoder + encode
    print("\n  ── LSTM : Chargement + encodage ──")
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = LSTMEncoder(**{
        k: LSTM_CONFIG[k]
        for k in ['input_size', 'hidden_size', 'num_layers',
                  'dropout', 'bidirectional', 'embedding_dim']
    })
    lstm_model.load_state_dict(
        torch.load("models/lstm_encoder.pt", map_location=device, weights_only=True)
    )
    lstm_model.to(device).eval()

    lstm_trainer = LSTMTrainer(LSTM_CONFIG, save_dir="models")
    embeddings   = lstm_trainer.extract_embeddings(lstm_model, sequences)
    print(f"  Embeddings : {embeddings.shape}")

    # 9. Feature merger
    print("\n  ── Fusion features (52) ──")
    merger   = FeatureMerger(LSTM_CONFIG['embedding_dim'])
    df_final = merger.merge(df, embeddings, seq_ids)
    df_final.to_parquet("data/processed/final_infer.parquet", index=False)
    print(f"  final_infer.parquet : {df_final.shape}")

    # 10. CatBoost predict (split SIMPLE par revenu)
    print("\n  ── CatBoost : Prédictions ──")
    cb_trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    df_results = cb_trainer.predict(df_final, model_dir="models")
    print(f"\n  Résultats : {len(df_results):,} clients")
    print(f"  Positifs prédits : {int(df_results['prediction'].sum()):,}")

    os.makedirs("outputs", exist_ok=True)
    df_results.to_parquet("outputs/inference_results.parquet", index=False)
    print(f"  Sauvegardé → outputs/inference_results.parquet")

    # 11. Si target disponible : évaluation
    if 'target' in df_final.columns:
        print("\n  ── Évaluation (target disponible) ──")
        df_merged_eval = df_results.merge(
            df_final[['id_client', 'target']], on='id_client', how='left'
        )
        evaluator = ModelEvaluator("outputs/metrics")
        evaluator.evaluate(
            df_merged_eval['target'].values,
            df_merged_eval['proba'].values,
            "infer_evaluation",
        )

    # 12. SHAP
    if not args.skip_shap:
        print("\n  ── SHAP ──")
        from catboost import CatBoostClassifier
        model_low  = CatBoostClassifier()
        model_high = CatBoostClassifier()
        model_low.load_model(f"models/{revenu_treshold}_catboost_low.cbm")
        model_high.load_model(f"models/{revenu_treshold}_catboost_high.cbm")

        shap_engine = SHAPEngine(
            FEATURE_COLS, CAT_FEATURES, LSTM_EMBEDDING_COLS,
            FEATURE_LABELS, SHAP_CONFIG['top_k'],
        )
        shap_engine.run(df_final, model_low, model_high, "outputs/shap")

    print("\n" + "═" * 60)
    print("  ✅ INFER TERMINÉ")
    print("═" * 60)


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modèle d'appétence crédit conso (LSTM + CatBoost)"
    )
    parser.add_argument("mode", choices=["train", "infer"],
                        help="Mode d'exécution")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Ignorer les caches Parquet et reconstruire")
    parser.add_argument("--data", default=None,
                        help="Fichier CSV custom pour l'inférence")
    parser.add_argument("--skip_shap", action="store_true",
                        help="Désactiver le calcul SHAP (infer uniquement)")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    else:
        run_infer(args)
