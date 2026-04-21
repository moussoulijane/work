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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

import json

from config import (
    TRAIN_FILES, INFER_FILES, FEATURE_COLS, CAT_FEATURES,
    MODEL_PARAMS, SHAP_CONFIG, LSTM_EMBEDDING_COLS, FEATURE_LABELS,
    revenu_treshold,
)
from src.data_loading import load_base, merge_common
from src.preprocessing import preprocess
from src.feature_engineering import (
    add_balance_features, add_advanced_features,
    add_temporal_features, add_appetite_signals,
)
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
    f"models/{revenu_treshold}_catboost_low.cbm",
    f"models/{revenu_treshold}_catboost_high.cbm",
    "models/thresholds.json",
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
    df = add_advanced_features(df)
    df = add_temporal_features(df)   # 14 features temporelles (remplace LSTM)
    df = add_appetite_signals(df)    # 10 signaux d'appétence (interactions + minima)
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

    # 1. Données
    df = build_base(TRAIN_FILES, "data/processed/modeling_base.parquet", args.force_rebuild)
    assert 'target' in df.columns, "Colonne 'target' absente — vérifier les CSV train"
    n_pos = int(df['target'].sum())
    print(f"  Base : {len(df):,} clients  |  positifs : {n_pos:,}  "
          f"({n_pos / len(df):.2%})")

    # 2. CatBoost train (calibration isotonique intégrée)
    use_two_stage = getattr(args, 'two_stage', False)
    print(f"\n  ── CatBoost : Entraînement"
          + (" (two-stage)" if use_two_stage else "") + " ──")
    cb_trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    if use_two_stage:
        model_low, model_high, results_low, results_high = cb_trainer.train_two_stage(
            df, calibrate=True
        )
        model_low_pred  = model_low[1]
        model_high_pred = model_high[1]
    else:
        model_low, model_high, results_low, results_high = cb_trainer.train(
            df, calibrate=True
        )
        model_low_pred  = model_low
        model_high_pred = model_high

    # 3. Métriques + seuils optimisés → sauvegarde JSON
    print("\n  ── Évaluation + seuils optimaux ──")
    evaluator       = ModelEvaluator("outputs/metrics")
    saved_thresholds = {}
    all_metrics     = []

    for seg_name, seg_key, results in [
        ("catboost_low_hybrid",  "LOW",  results_low),
        ("catboost_high_hybrid", "HIGH", results_high),
    ]:
        t_f2, _, _ = optimize_threshold(
            results['y_true'], results['y_proba'], strategy='f2')
        t_pt, _, _ = optimize_threshold(
            results['y_true'], results['y_proba'],
            strategy='precision_target', min_precision=0.10)

        saved_thresholds[seg_key] = {'f2': round(t_f2, 4), 'precision_target': round(t_pt, 4)}

        m_f2 = evaluator.evaluate(results['y_true'], results['y_proba'],
                                  f"{seg_name}_f2",     threshold=t_f2)
        m_pt = evaluator.evaluate(results['y_true'], results['y_proba'],
                                  f"{seg_name}_prec10", threshold=t_pt)
        all_metrics.extend([m_f2, m_pt])
        print(f"  {seg_name} | seuil F2={t_f2:.3f}  seuil prec10={t_pt:.3f}")

    os.makedirs("models", exist_ok=True)
    with open("models/thresholds.json", 'w') as f:
        json.dump(saved_thresholds, f, indent=2)
    print(f"\n  Seuils sauvegardés → models/thresholds.json")
    print(f"  {json.dumps(saved_thresholds, indent=4)}")

    evaluator.compare(all_metrics)

    # 4. SHAP
    print("\n  ── SHAP ──")
    shap_engine = SHAPEngine(
        FEATURE_COLS, CAT_FEATURES, LSTM_EMBEDDING_COLS,
        FEATURE_LABELS, SHAP_CONFIG['top_k'],
    )
    shap_engine.run(df, model_low_pred, model_high_pred, "outputs/shap")

    # 5. Error analysis
    print("\n  ── Analyse erreurs (segment LOW) ──")
    df_low_eval = df[df['revenu_principal'] <= revenu_treshold].copy()
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

    # 2. Données
    files = [args.data] if args.data else INFER_FILES
    df    = build_base(files, "data/processed/inference_base.parquet", args.force_rebuild)
    print(f"  Base inférence : {len(df):,} clients")

    # 3. CatBoost predict (seuils optimisés chargés automatiquement)
    use_two_stage = getattr(args, 'two_stage', False)
    print(f"\n  ── CatBoost : Prédictions"
          + (" (two-stage)" if use_two_stage else "") + " ──")
    cb_trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    df_results = cb_trainer.predict(
        df, model_dir="models",
        use_two_stage=use_two_stage, use_calibration=True,
    )
    print(f"\n  Résultats : {len(df_results):,} clients")
    print(f"  Positifs prédits : {int(df_results['prediction'].sum()):,}")

    os.makedirs("outputs", exist_ok=True)
    df_results.to_parquet("outputs/inference_results.parquet", index=False)
    print(f"  Sauvegardé → outputs/inference_results.parquet")

    # 4. Si target disponible : évaluation avec seuil F2 pondéré
    if 'target' in df.columns:
        print("\n  ── Évaluation (target disponible) ──")
        thresholds_data = {}
        if os.path.exists("models/thresholds.json"):
            with open("models/thresholds.json") as f:
                thresholds_data = json.load(f)
        # Seuil représentatif : moyenne des seuils F2 LOW et HIGH
        t_low  = thresholds_data.get('LOW',  {}).get('f2', 0.5)
        t_high = thresholds_data.get('HIGH', {}).get('f2', 0.5)
        n_low  = (df['revenu_principal'] <= revenu_treshold).sum()
        n_high = len(df) - n_low
        eval_threshold = (t_low * n_low + t_high * n_high) / len(df)

        df_merged_eval = df_results.merge(
            df[['id_client', 'target']], on='id_client', how='left'
        )
        evaluator = ModelEvaluator("outputs/metrics")
        evaluator.evaluate(
            df_merged_eval['target'].values,
            df_merged_eval['proba'].values,
            "infer_evaluation",
            threshold=eval_threshold,
        )

    # 5. SHAP
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
        shap_engine.run(df, model_low, model_high, "outputs/shap")

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
