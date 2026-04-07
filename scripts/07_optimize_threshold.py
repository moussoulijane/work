"""
Étape 07 — Optimisation du seuil de décision.
Teste 4 stratégies (f1, f2, profit, youden) sur les sets d'évaluation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from src.catboost_trainer import CatBoostTrainer
from src.threshold_optimizer import optimize_threshold
from config import FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold

if __name__ == "__main__":
    print("=== Étape 07 : Optimisation du seuil de décision ===\n")

    df = pd.read_parquet("data/processed/final_train.parquet")

    trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    df_low, df_high = trainer.split_data(df, mode='train')

    os.makedirs("outputs/optimization", exist_ok=True)

    all_results = []

    for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
        print(f"\n{'═'*50}\n  Segment {name}\n{'═'*50}")

        available = [f for f in FEATURE_COLS if f in df_seg.columns]
        X = df_seg[available].copy()
        y = df_seg['target'].copy()
        for c in CAT_FEATURES:
            if c in X.columns:
                X[c] = X[c].fillna('INCONNU').astype(str)

        # Utiliser un split pour avoir y_true / y_proba
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        model = CatBoostClassifier()
        cbm_path = f"models/{revenu_treshold}_catboost_{name.lower()}.cbm"
        if not os.path.exists(cbm_path):
            print(f"  ⚠️  Modèle {cbm_path} absent — entraînement rapide...")
            n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
            p = {**MODEL_PARAMS, 'scale_pos_weight': n_neg / n_pos, 'verbose': 0}
            cat_idx = [X_tr.columns.tolist().index(c) for c in CAT_FEATURES if c in X_tr.columns]
            model = CatBoostClassifier(**p)
            model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_te, y_te), verbose=0)
        else:
            model.load_model(cbm_path)

        y_proba = model.predict_proba(X_te)[:, 1]
        y_true  = y_te.values

        for strategy in ['f1', 'f2', 'profit', 'youden']:
            opt_t, metrics_at_opt, df_all = optimize_threshold(
                y_true, y_proba, strategy=strategy
            )
            all_results.append({
                'segment':   name,
                'strategy':  strategy,
                'threshold': opt_t,
                **{k: v for k, v in metrics_at_opt.items()
                   if k in ('precision', 'recall', 'f1', 'f2', 'profit')},
            })
            df_all.to_csv(
                f"outputs/optimization/threshold_{name.lower()}_{strategy}.csv",
                index=False,
            )

    pd.DataFrame(all_results).to_csv(
        "outputs/optimization/threshold_summary.csv", index=False
    )
    print(f"\n✅ Résultats → outputs/optimization/threshold_summary.csv")
    print(pd.DataFrame(all_results).to_string(index=False))
