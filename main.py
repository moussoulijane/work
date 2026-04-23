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
    revenu_treshold, N_FOLDS,
)
from src.data_loading import load_base, merge_common
from src.preprocessing import preprocess
from src.feature_engineering import (
    add_balance_features, add_advanced_features,
    add_temporal_features, add_appetite_signals, add_credit_context_features,
    add_interaction_features,
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
    df = add_temporal_features(df)        # 14 features temporelles
    df = add_appetite_signals(df)         # 10 signaux d'appétence
    df = add_credit_context_features(df)  # 5 features contexte crédit
    df = add_interaction_features(df)     # 6 features d'interaction inter-variables
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

    # 2. CatBoost train — K-Fold OOF par défaut (meilleure AUC + calibration fiable)
    use_two_stage = getattr(args, 'two_stage', False)
    cb_trainer    = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)

    if use_two_stage:
        print(f"\n  ── CatBoost : Entraînement two-stage ──")
        model_low_cb, model_high_cb, results_low_cb, results_high_cb = (
            cb_trainer.train_two_stage(df, calibrate=True)
        )
        model_low_shap  = model_low_cb[1]
        model_high_shap = model_high_cb[1]
    else:
        print(f"\n  ── CatBoost : K-Fold OOF ({N_FOLDS} folds) ──")
        model_low_cb, model_high_cb, results_low_cb, results_high_cb = (
            cb_trainer.train_kfold(df, n_splits=N_FOLDS, calibrate=True)
        )
        model_low_shap  = model_low_cb   # fold_0 pour SHAP
        model_high_shap = model_high_cb

    # 3. LightGBM train K-Fold (ensemble avec CatBoost)
    lgbm_results_low  = results_low_cb
    lgbm_results_high = results_high_cb
    lgbm_available    = False
    try:
        from src.lgbm_trainer import LGBMTrainer
        print(f"\n  ── LightGBM : K-Fold OOF ({N_FOLDS} folds) ──")
        lgbm_trainer = LGBMTrainer(FEATURE_COLS, CAT_FEATURES, revenu_treshold)
        _, _, lgbm_results_low, lgbm_results_high = lgbm_trainer.train_kfold(
            df, n_splits=N_FOLDS, calibrate=True
        )
        lgbm_available = True
        print("  LightGBM K-Fold entraîné ✓")
    except Exception as e:
        print(f"  LightGBM ignoré ({e})")

    # 4. Ensemble : moyenne CatBoost + LightGBM
    def _ensemble(r_cb, r_lgbm, use_lgbm):
        if not use_lgbm:
            return r_cb
        return {
            'y_true':  r_cb['y_true'],
            'y_proba': (r_cb['y_proba'] + r_lgbm['y_proba']) / 2.0,
        }

    results_low  = _ensemble(results_low_cb,  lgbm_results_low,  lgbm_available)
    results_high = _ensemble(results_high_cb, lgbm_results_high, lgbm_available)
    suffix_label = (
        f"ensemble K-Fold CatBoost+LGBM ({N_FOLDS} folds)" if lgbm_available
        else f"CatBoost K-Fold OOF ({N_FOLDS} folds)"
    )
    print(f"\n  Modèle final : {suffix_label}")

    # 5. Métriques + seuils optimisés → sauvegarde JSON
    print("\n  ── Évaluation + seuils optimaux ──")
    evaluator        = ModelEvaluator("outputs/metrics")
    saved_thresholds = {}
    all_metrics      = []

    for seg_name, seg_key, results in [
        ("low_hybrid",  "LOW",  results_low),
        ("high_hybrid", "HIGH", results_high),
    ]:
        t_f2, _, _    = optimize_threshold(results['y_true'], results['y_proba'], strategy='f2')
        t_youden, _,_ = optimize_threshold(results['y_true'], results['y_proba'], strategy='youden')
        t_pt, _, _    = optimize_threshold(
            results['y_true'], results['y_proba'],
            strategy='precision_target', min_precision=0.10)

        saved_thresholds[seg_key] = {
            'f2':               round(t_f2, 4),
            'youden':           round(t_youden, 4),
            'precision_target': round(t_pt, 4),
        }

        m_f2 = evaluator.evaluate(results['y_true'], results['y_proba'],
                                  f"{seg_name}_f2",     threshold=t_f2)
        m_yd = evaluator.evaluate(results['y_true'], results['y_proba'],
                                  f"{seg_name}_youden", threshold=t_youden)
        m_pt = evaluator.evaluate(results['y_true'], results['y_proba'],
                                  f"{seg_name}_prec10", threshold=t_pt)
        all_metrics.extend([m_f2, m_yd, m_pt])
        print(f"  {seg_key} | F2={t_f2:.3f}  Youden={t_youden:.3f}  prec10={t_pt:.3f}")

    os.makedirs("models", exist_ok=True)
    with open("models/thresholds.json", 'w') as f:
        json.dump(saved_thresholds, f, indent=2)
    print(f"\n  Seuils sauvegardés → models/thresholds.json")
    evaluator.compare(all_metrics)

    # 6. SHAP (sur modèle CatBoost uniquement — TreeSHAP)
    print("\n  ── SHAP ──")
    shap_engine = SHAPEngine(
        FEATURE_COLS, CAT_FEATURES, LSTM_EMBEDDING_COLS,
        FEATURE_LABELS, SHAP_CONFIG['top_k'],
    )
    shap_engine.run(df, model_low_shap, model_high_shap, "outputs/shap")

    # 5. Error analysis (avec seuil optimisé)
    print("\n  ── Analyse erreurs (segment LOW) ──")
    df_low_eval = df[df['revenu_principal'] <= revenu_treshold].copy()
    if len(df_low_eval) > 0 and 'target' in df_low_eval.columns:
        t_low_f2 = saved_thresholds.get('LOW', {}).get('f2', 0.5)
        analyzer = ErrorAnalyzer()
        analyzer.analyze(
            df_low_eval,
            results_low['y_true'],
            results_low['y_proba'],
            threshold=t_low_f2,
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

    # 3. CatBoost predict
    use_two_stage = getattr(args, 'two_stage', False)
    print(f"\n  ── CatBoost : Prédictions"
          + (" (two-stage)" if use_two_stage else "") + " ──")
    cb_trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    df_results = cb_trainer.predict(
        df, model_dir="models",
        use_two_stage=use_two_stage, use_calibration=True,
    )

    # 4. LightGBM predict + fusion des probas (si modèles disponibles)
    try:
        from src.lgbm_trainer import LGBMTrainer
        lgbm_low_path  = f"models/{revenu_treshold}_lgbm_low.txt"
        lgbm_high_path = f"models/{revenu_treshold}_lgbm_high.txt"
        if os.path.exists(lgbm_low_path) and os.path.exists(lgbm_high_path):
            print("\n  ── LightGBM : Prédictions + ensemble ──")
            lgbm_trainer  = LGBMTrainer(FEATURE_COLS, CAT_FEATURES, revenu_treshold)
            lgbm_proba_map = lgbm_trainer.predict(df, model_dir="models", use_calibration=True)
            # Moyenne des probas CatBoost et LightGBM
            df_results['proba'] = df_results.apply(
                lambda r: (r['proba'] + lgbm_proba_map.get(r['id_client'], r['proba'])) / 2.0,
                axis=1,
            )
            print("  Ensemble CatBoost + LightGBM ✓")
    except Exception as e:
        logger.warning(f"Ensemble LightGBM ignoré : {e}")

    # Re-appliquer les seuils après fusion
    if os.path.exists("models/thresholds.json"):
        with open("models/thresholds.json") as _f:
            _th = json.load(_f)
        df_results['prediction'] = df_results.apply(
            lambda r: int(r['proba'] >= _th.get(r['segment_model'], {}).get('f2', 0.5)),
            axis=1,
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
