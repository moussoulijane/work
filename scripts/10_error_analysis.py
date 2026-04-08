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
from src.catboost_trainer import CatBoostTrainer, _prepare_X
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

        # _prepare_X → élimine ValueError "could not convert string to float"
        X = _prepare_X(df_seg, FEATURE_COLS, CAT_FEATURES)
        y = df_seg['target'].reset_index(drop=True)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        model = CatBoostClassifier()
        model.load_model(f"models/{revenu_treshold}_catboost_{name.lower()}.cbm")

        y_proba = model.predict_proba(X_te)[:, 1]
        y_true  = y_te.values

        # Calibration si disponible
        cal_path = f"models/calibrator_{name.lower()}.pkl"
        if os.path.exists(cal_path):
            from src.calibration import ProbabilityCalibrator
            cal     = ProbabilityCalibrator.load(cal_path)
            y_proba = cal.transform(y_proba)

        threshold = args.threshold
        if threshold is None:
            threshold, _, _ = optimize_threshold(y_true, y_proba, strategy='f2')
            print(f"  Seuil F2 optimal : {threshold:.3f}")

        # Reconstruire df_te avec id_client si disponible
        df_te = X_te.copy()
        df_te['target'] = y_true
        if 'id_client' in df_seg.columns:
            df_te['id_client'] = df_seg.iloc[y_te.index]['id_client'].values

        analyzer.analyze(
            df_te, y_true, y_proba,
            threshold=threshold,
            output_dir="outputs/metrics",
        )

    print(f"\n✅ Analyse des erreurs → outputs/metrics/error_*.csv")
