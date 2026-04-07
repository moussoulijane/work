"""
Étape 09 — Analyse SHAP complète.
Charge les modèles et final_train.parquet, calcule SHAP agrégé + top K.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from catboost import CatBoostClassifier
from src.shap_engine import SHAPEngine
from config import (FEATURE_COLS, CAT_FEATURES, LSTM_EMBEDDING_COLS,
                    FEATURE_LABELS, SHAP_CONFIG, revenu_treshold)

if __name__ == "__main__":
    print("=== Étape 09 : Analyse SHAP ===\n")

    df = pd.read_parquet("data/processed/final_train.parquet")
    print(f"Base : {df.shape}")

    model_low  = CatBoostClassifier()
    model_high = CatBoostClassifier()
    model_low.load_model(f"models/{revenu_treshold}_catboost_low.cbm")
    model_high.load_model(f"models/{revenu_treshold}_catboost_high.cbm")
    print("Modèles chargés")

    shap_engine = SHAPEngine(
        feature_cols        = FEATURE_COLS,
        cat_features        = CAT_FEATURES,
        lstm_embedding_cols = LSTM_EMBEDDING_COLS,
        feature_labels      = FEATURE_LABELS,
        top_k               = SHAP_CONFIG['top_k'],
    )
    shap_engine.run(df, model_low, model_high, output_dir="outputs/shap")

    print("\n✅ SHAP terminé → outputs/shap/")
    print("   shap_aggregated.csv")
    print("   shap_complete.parquet")
    print("   shap_topk.parquet")
    print("   shap_summary_low.png")
    print("   shap_summary_high.png")
