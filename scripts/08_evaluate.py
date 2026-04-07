"""
Étape 08 — Évaluation complète avec seuil optimisé.
Charge final_train.parquet, ré-évalue avec seuil F2 optimal, compare modèles.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from src.catboost_trainer import CatBoostTrainer
from src.threshold_optimizer import optimize_threshold
from src.metrics import ModelEvaluator
from src.error_analysis import ErrorAnalyzer
from config import FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="f2",
                        choices=["f1", "f2", "profit", "youden"])
    args = parser.parse_args()

    print(f"=== Étape 08 : Évaluation complète (stratégie seuil = {args.strategy}) ===\n")

    df       = pd.read_parquet("data/processed/final_train.parquet")
    trainer  = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    df_low, df_high = trainer.split_data(df, mode='train')

    evaluator = ModelEvaluator("outputs/metrics")
    all_metrics = []

    for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
        print(f"\n{'─'*50}\n  Segment {name}\n{'─'*50}")

        available = [f for f in FEATURE_COLS if f in df_seg.columns]
        X = df_seg[available].copy()
        y = df_seg['target'].copy()
        for c in CAT_FEATURES:
            if c in X.columns:
                X[c] = X[c].fillna('INCONNU').astype(str)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        model = CatBoostClassifier()
        cbm_path = f"models/{revenu_treshold}_catboost_{name.lower()}.cbm"
        model.load_model(cbm_path)

        y_proba = model.predict_proba(X_te)[:, 1]
        y_true  = y_te.values

        # Seuil 0.5 (baseline)
        m_default = evaluator.evaluate(
            y_true, y_proba,
            model_name=f"catboost_{name.lower()}_seuil05",
            threshold=0.5,
        )

        # Seuil optimisé
        opt_t, _, _ = optimize_threshold(y_true, y_proba, strategy=args.strategy)
        m_optimal = evaluator.evaluate(
            y_true, y_proba,
            model_name=f"catboost_{name.lower()}_seuilopt",
            threshold=opt_t,
        )

        all_metrics.extend([m_default, m_optimal])

        # Analyse des erreurs
        print(f"\n  Analyse des erreurs — {name} (seuil={opt_t:.3f})")
        df_te_with_feats = df_seg.iloc[X_te.index].copy() if hasattr(X_te, 'index') else df_seg.head(len(X_te)).copy()
        analyzer = ErrorAnalyzer()
        analyzer.analyze(df_te_with_feats, y_true, y_proba, threshold=opt_t)

    evaluator.compare(all_metrics)
    print(f"\n✅ Évaluation terminée → outputs/metrics/")
