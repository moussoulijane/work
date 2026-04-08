"""
Étape 01 — Préparation des données.
Charge les 3 CSV train, merge demographics + financials,
préprocesse, calcule les features de solde, sauvegarde modeling_base.parquet.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.data_loading import load_base, merge_common
from src.preprocessing import preprocess
from src.feature_engineering import add_balance_features, add_advanced_features
from config import TRAIN_FILES

if __name__ == "__main__":
    print("=== Étape 01 : Préparation des données ===\n")

    df = load_base(TRAIN_FILES)
    print(f"Base brute : {df.shape}")

    df = merge_common(df)
    print(f"Après merge : {df.shape}")

    df = preprocess(df)
    print(f"Après preprocess : {df.shape}")

    df = add_balance_features(df)
    df = add_advanced_features(df)
    print(f"Après feature engineering : {df.shape}")

    if 'target' in df.columns:
        print(f"\nDistribution target :\n{df['target'].value_counts()}")
        print(f"Taux positifs : {df['target'].mean():.3%}")

    os.makedirs("data/processed", exist_ok=True)
    out = "data/processed/modeling_base.parquet"
    df.to_parquet(out, index=False)
    print(f"\n✅ Sauvegardé → {out}")
