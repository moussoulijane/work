"""
Pipeline complet — exécute toutes les étapes séquentiellement.
Usage :
    python scripts/run_all.py
    python scripts/run_all.py --skip_optuna    # Sans les optimisations Optuna
    python scripts/run_all.py --n_lstm_trials 20 --n_cb_trials 50
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import subprocess
import time

STEPS = [
    ("01", "Préparation des données",         ["python", "scripts/01_prepare_data.py"]),
    ("02", "Entraînement LSTM",               ["python", "scripts/02_train_lstm.py"]),
    ("04", "Extraction embeddings",           ["python", "scripts/04_extract_embeddings.py"]),
    ("05", "Entraînement CatBoost",           ["python", "scripts/05_train_catboost.py"]),
    ("07", "Optimisation seuil",              ["python", "scripts/07_optimize_threshold.py"]),
    ("08", "Évaluation complète",             ["python", "scripts/08_evaluate.py"]),
    ("09", "Analyse SHAP",                    ["python", "scripts/09_shap_analysis.py"]),
    ("10", "Analyse des erreurs",             ["python", "scripts/10_error_analysis.py"]),
]

OPTUNA_STEPS = [
    ("03", "Optimisation LSTM (Optuna)",      None),   # cmd built dynamically
    ("06", "Optimisation CatBoost (Optuna)",  None),
]


def run_step(step_id: str, desc: str, cmd: list[str]) -> bool:
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE {step_id} — {desc}")
    print(f"{'═'*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ❌ Étape {step_id} ÉCHOUÉE (code={result.returncode})")
        return False
    print(f"\n  ✅ Étape {step_id} terminée en {elapsed:.0f}s")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_optuna",    action="store_true")
    parser.add_argument("--n_lstm_trials",  type=int, default=30)
    parser.add_argument("--n_cb_trials",    type=int, default=50)
    args = parser.parse_args()

    steps = list(STEPS)
    if not args.skip_optuna:
        steps.insert(2, ("03", "Optimisation LSTM",
                         ["python", "scripts/03_optimize_lstm.py",
                          f"--n_trials={args.n_lstm_trials}", "--retrain"]))
        steps.insert(5, ("06", "Optimisation CatBoost",
                         ["python", "scripts/06_optimize_catboost.py",
                          "--segment=ALL",
                          f"--n_trials={args.n_cb_trials}", "--retrain"]))

    print(f"\n{'═'*60}")
    print(f"  PIPELINE COMPLET — {len(steps)} étapes")
    if args.skip_optuna:
        print(f"  (Optimisation Optuna désactivée)")
    print(f"{'═'*60}")

    t_start = time.time()
    for step_id, desc, cmd in steps:
        ok = run_step(step_id, desc, cmd)
        if not ok:
            print(f"\n  Pipeline arrêté à l'étape {step_id}")
            sys.exit(1)

    total = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  ✅ PIPELINE TERMINÉ en {total/60:.1f} minutes")
    print(f"{'═'*60}")
