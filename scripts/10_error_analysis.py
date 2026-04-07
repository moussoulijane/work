"""
Étape 10 — Analyse des erreurs (FP / FN / TP / TN).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from src.catboost_trainer import CatBoostTrainer
from src.threshold_optimizer import optimize_threshold
from src.error_analysis import ErrorAnalyzer
from config import FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="Seuil fixe (défaut : F2 optimal)")
    args = parser.parse_args()

    print("=== Étape 10 : Analyse des erreurs ===\n")

    df      = pd.read_parquet("data/processed/final_train.parquet")
    trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    df_low, df_high = trainer.split_data(df, mode='train')

    analyzer = ErrorAnalyzer()

    for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
        print(f"\n{'═'*55}\n  Segment {name}\n{'═'*55}")

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
        model.load_model(f"models/{revenu_treshold}_catboost_{name.lower()}.cbm")

        y_proba = model.predict_proba(X_te)[:, 1]
        y_true  = y_te.values

        threshold = args.threshold
        if threshold is None:
            threshold, _, _ = optimize_threshold(y_true, y_proba, strategy='f2')
            print(f"  Seuil F2 optimal : {threshold:.3f}")

        # Reconstruire df_te avec features originales
        df_te = df_seg.iloc[y_te.index].copy() if hasattr(y_te, 'index') else df_seg.head(len(y_te)).copy()

        analyzer.analyze(
            df_te, y_true, y_proba,
            threshold=threshold,
            output_dir="outputs/metrics",
        )

    print(f"\n✅ Analyse des erreurs → outputs/metrics/error_*.csv")
