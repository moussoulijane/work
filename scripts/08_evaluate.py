"""
Étape 08 — Évaluation complète avec seuil optimisé (F2 + precision_target).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from src.catboost_trainer import CatBoostTrainer, _prepare_X
from src.threshold_optimizer import optimize_threshold
from src.metrics import ModelEvaluator
from src.error_analysis import ErrorAnalyzer
from config import FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="f2",
                        choices=["f1", "f2", "profit", "youden", "precision_target"])
    parser.add_argument("--min_precision", type=float, default=0.10)
    args = parser.parse_args()

    print(f"=== Étape 08 : Évaluation (stratégie={args.strategy}) ===\n")

    df      = pd.read_parquet("data/processed/final_train.parquet")
    trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    df_low, df_high = trainer.split_data(df, mode='train')

    evaluator   = ModelEvaluator("outputs/metrics")
    all_metrics = []

    for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
        print(f"\n{'─'*50}\n  Segment {name}\n{'─'*50}")

        # _prepare_X → élimine ValueError
        X = _prepare_X(df_seg, FEATURE_COLS, CAT_FEATURES)
        y = df_seg['target'].reset_index(drop=True)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        model    = CatBoostClassifier()
        cbm_path = f"models/{revenu_treshold}_catboost_{name.lower()}.cbm"
        model.load_model(cbm_path)

        y_proba = model.predict_proba(X_te)[:, 1]
        y_true  = y_te.values

        # Calibration si disponible
        cal_path = f"models/calibrator_{name.lower()}.pkl"
        if os.path.exists(cal_path):
            from src.calibration import ProbabilityCalibrator
            cal     = ProbabilityCalibrator.load(cal_path)
            y_proba = cal.transform(y_proba)
            print(f"  Calibration chargée → {cal_path}")

        # Seuil 0.5
        m_default = evaluator.evaluate(
            y_true, y_proba, f"catboost_{name.lower()}_seuil05", threshold=0.5
        )

        # Seuil optimisé selon la stratégie demandée
        kw    = {'min_precision': args.min_precision} if args.strategy == 'precision_target' else {}
        opt_t, _, _ = optimize_threshold(y_true, y_proba, strategy=args.strategy, **kw)
        m_opt = evaluator.evaluate(
            y_true, y_proba, f"catboost_{name.lower()}_seuilopt", threshold=opt_t
        )

        # Seuil precision_target 10% (toujours calculé pour comparaison)
        t_pt, _, _ = optimize_threshold(y_true, y_proba,
                                        strategy='precision_target', min_precision=0.10)
        m_pt = evaluator.evaluate(
            y_true, y_proba, f"catboost_{name.lower()}_prec10", threshold=t_pt
        )

        all_metrics.extend([m_default, m_opt, m_pt])
        print(f"\n  seuil_default=0.5  |  seuil_{args.strategy}={opt_t:.3f}"
              f"  |  seuil_prec10={t_pt:.3f}")

        # Analyse des erreurs
        df_te = X_te.copy()
        df_te['target'] = y_true
        if 'id_client' in df_seg.columns:
            df_te['id_client'] = df_seg.iloc[y_te.index]['id_client'].values
        analyzer = ErrorAnalyzer()
        analyzer.analyze(df_te, y_true, y_proba, threshold=opt_t)

    evaluator.compare(all_metrics)
    print(f"\n✅ Évaluation terminée → outputs/metrics/")
