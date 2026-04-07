"""
Étape 06 — Optimisation Optuna CatBoost.
Usage :
    python scripts/06_optimize_catboost.py --segment LOW  --n_trials 100
    python scripts/06_optimize_catboost.py --segment HIGH --n_trials 100
    python scripts/06_optimize_catboost.py --segment ALL  --n_trials 100
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import pandas as pd
from src.catboost_trainer import CatBoostTrainer
from src.catboost_optimizer import optimize_catboost, build_model_params_from_optuna
from config import FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment",  choices=["LOW", "HIGH", "ALL"], default="ALL")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--retrain",  action="store_true",
                        help="Ré-entraîner avec les meilleurs params après optim")
    args = parser.parse_args()

    print(f"=== Étape 06 : Optimisation CatBoost (segment={args.segment}, "
          f"{args.n_trials} trials) ===\n")

    df = pd.read_parquet("data/processed/final_train.parquet")

    base_trainer = CatBoostTrainer(FEATURE_COLS, CAT_FEATURES, MODEL_PARAMS, revenu_treshold)
    df_low, df_high = base_trainer.split_data(df, mode='train')

    segments_to_run = []
    if args.segment in ("LOW",  "ALL"): segments_to_run.append(("LOW",  df_low))
    if args.segment in ("HIGH", "ALL"): segments_to_run.append(("HIGH", df_high))

    os.makedirs("outputs/optimization", exist_ok=True)
    best_params_all = {}

    for name, df_seg in segments_to_run:
        print(f"\n{'─'*50}\n  Segment {name} — {len(df_seg):,} clients\n{'─'*50}")
        available = [f for f in FEATURE_COLS if f in df_seg.columns]
        X = df_seg[available].copy()
        y = df_seg['target'].copy()

        for c in CAT_FEATURES:
            if c in X.columns:
                X[c] = X[c].fillna('INCONNU').astype(str)

        study, best_params = optimize_catboost(
            X, y, CAT_FEATURES,
            n_trials=args.n_trials,
            segment_name=name,
            save_dir="models/optuna",
        )

        df_res = study.trials_dataframe()
        df_res.to_csv(
            f"outputs/optimization/catboost_{name.lower()}_optuna_results.csv",
            index=False,
        )
        best_params_all[name] = best_params

    if args.retrain and best_params_all:
        print(f"\n  Ré-entraînement avec les meilleurs hyperparamètres...")
        # Utiliser les params du segment LOW s'ils existent, sinon HIGH
        best = best_params_all.get("LOW", best_params_all.get("HIGH", {}))
        if best:
            optimized_params = build_model_params_from_optuna(best)
            trainer = CatBoostTrainer(
                FEATURE_COLS, CAT_FEATURES, optimized_params, revenu_treshold
            )
            trainer.train(df, save_dir="models")
            print(f"  ✅ Modèles optimisés sauvegardés → models/{revenu_treshold}_catboost_*.cbm")
