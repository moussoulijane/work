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
from src.feature_engineering import add_balance_features, add_advanced_features
from src.sequence_builder import SequenceBuilder
from src.lstm_model import build_lstm_encoder, LSTMEncoder, LSTMTrainer
from src.feature_merger import FeatureMerger
from src.catboost_trainer import CatBoostTrainer
from src.calibration import ProbabilityCalibrator
from src.metrics import ModelEvaluator
from src.threshold_optimizer import optimize_threshold
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
    df = add_advanced_features(df)   # 6 features avancées
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

    # 7. LSTM train (LSTMEncoderWithAttention si use_attention=True dans config)
    print("\n  ── LSTM : Entraînement"
          + (" (avec attention)" if LSTM_CONFIG.get('use_attention') else "") + " ──")
    lstm_trainer       = LSTMTrainer(LSTM_CONFIG, save_dir="models")
    lstm_model, history = lstm_trainer.train(sequences, targets)

    # 8. Extract embeddings
    print("\n  ── LSTM : Extraction embeddings ──")
    embeddings = lstm_trainer.extract_embeddings(lstm_model, sequences)
    print(f"  Embeddings : {embeddings.shape}")

    # 9. Feature merger
    n_features = len(FEATURE_COLS)
    print(f"\n  ── Fusion features ({n_features}) ──")
    merger   = FeatureMerger(LSTM_CONFIG['embedding_dim'])
    df_final = merger.merge(df, embeddings, seq_ids)
    df_final.to_parquet("data/processed/final_train.parquet", index=False)
    print(f"  final_train.parquet : {df_final.shape}")

    # 10. CatBoost train (avec calibration isotonique intégrée)
    use_two_stage = getattr(args, 'two_stage', False)
    print(f"\n  ── CatBoost : Entraînement"
          + (" (two-stage)" if use_two_stage else "") + " ──")
    cb_trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    if use_two_stage:
        model_low, model_high, results_low, results_high = cb_trainer.train_two_stage(
            df_final, calibrate=True
        )
        # two_stage retourne tuples (s1, s2, threshold) pour models
        model_low_pred  = model_low[1]   # stage-2 pour SHAP
        model_high_pred = model_high[1]
    else:
        model_low, model_high, results_low, results_high = cb_trainer.train(
            df_final, calibrate=True
        )
        model_low_pred  = model_low
        model_high_pred = model_high

    # 11. Métriques + seuil optimisé
    print("\n  ── Évaluation + seuil optimal ──")
    evaluator = ModelEvaluator("outputs/metrics")

    all_metrics = []
    for seg_name, results in [("catboost_low_hybrid",  results_low),
                               ("catboost_high_hybrid", results_high)]:
        # Seuil F2 (rappel)
        t_f2, _, _ = optimize_threshold(results['y_true'], results['y_proba'], strategy='f2')
        # Seuil précision-cible 10%
        t_pt, _, _ = optimize_threshold(results['y_true'], results['y_proba'],
                                        strategy='precision_target', min_precision=0.10)
        m_f2 = evaluator.evaluate(results['y_true'], results['y_proba'],
                                  f"{seg_name}_f2",    threshold=t_f2)
        m_pt = evaluator.evaluate(results['y_true'], results['y_proba'],
                                  f"{seg_name}_prec10", threshold=t_pt)
        all_metrics.extend([m_f2, m_pt])
        print(f"  {seg_name} | seuil F2={t_f2:.3f}  seuil prec10={t_pt:.3f}")

    evaluator.compare(all_metrics)

    # 12. SHAP
    print("\n  ── SHAP ──")
    shap_engine = SHAPEngine(
        FEATURE_COLS, CAT_FEATURES, LSTM_EMBEDDING_COLS,
        FEATURE_LABELS, SHAP_CONFIG['top_k'],
    )
    shap_engine.run(df_final, model_low_pred, model_high_pred, "outputs/shap")

    # 13. Error analysis
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
    lstm_model = build_lstm_encoder(LSTM_CONFIG)
    lstm_model.load_state_dict(
        torch.load("models/lstm_encoder.pt", map_location=device)
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

    # 10. CatBoost predict (split SIMPLE par revenu, avec calibration si dispo)
    use_two_stage = getattr(args, 'two_stage', False)
    print(f"\n  ── CatBoost : Prédictions"
          + (" (two-stage)" if use_two_stage else "") + " ──")
    cb_trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    df_results = cb_trainer.predict(
        df_final, model_dir="models",
        use_two_stage=use_two_stage, use_calibration=True,
    )
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
    parser.add_argument("--two_stage", action="store_true",
                        help="Entraînement / inférence two-stage (meilleure précision)")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    else:
        run_infer(args)
