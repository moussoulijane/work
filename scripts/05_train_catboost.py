"""
Étape 05 — Entraînement CatBoost (LOW + HIGH).
Charge final_train.parquet, split asymétrique, entraîne, sauvegarde .cbm.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.catboost_trainer import CatBoostTrainer
from src.metrics import ModelEvaluator
from config import FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold

if __name__ == "__main__":
    print("=== Étape 05 : Entraînement CatBoost ===\n")

    df = pd.read_parquet("data/processed/final_train.parquet")
    print(f"Base finale : {df.shape}")
    print(f"Positifs : {int(df['target'].sum()):,} / {len(df):,}")

    trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    model_low, model_high, res_low, res_high = trainer.train(df, save_dir="models", calibrate=True)

    # Métriques
    evaluator = ModelEvaluator("outputs/metrics")
    m_low  = evaluator.evaluate(res_low['y_true'],  res_low['y_proba'],  "catboost_low_hybrid")
    m_high = evaluator.evaluate(res_high['y_true'], res_high['y_proba'], "catboost_high_hybrid")
    evaluator.compare([m_low, m_high])

    print(f"\n✅ Modèles sauvegardés → models/{revenu_treshold}_catboost_*.cbm")
