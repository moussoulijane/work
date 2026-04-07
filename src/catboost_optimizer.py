"""
Optimisation des hyperparamètres CatBoost via Optuna.
3-fold CV stratifiée par trial. iterations=3000 + early_stopping_rounds=50.
"""
import os
import joblib
import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)


def optimize_catboost(
    X,
    y,
    cat_features: list[str],
    n_trials: int = 100,
    segment_name: str = "",
    save_dir: str = "models/optuna",
) -> tuple:
    """
    Optuna pour CatBoost — 3-fold CV stratifiée, objectif = mean(AUC).

    Args:
        X            : pd.DataFrame — features
        y            : pd.Series — target
        cat_features : liste des noms de features catégorielles
        n_trials     : nombre de trials
        segment_name : "LOW" ou "HIGH" (pour le nom de l'étude)
        save_dir     : répertoire de sauvegarde de l'étude

    Returns:
        study      : optuna.Study
        best_params: dict
    """
    os.makedirs(save_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'depth':              trial.suggest_int('depth', 4, 10),
            'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations':         3000,
            'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count':       trial.suggest_categorical('border_count', [32, 64, 128, 254]),
            'subsample':          trial.suggest_float('subsample', 0.6, 1.0),
            'min_data_in_leaf':   trial.suggest_int('min_data_in_leaf', 1, 50),
            'random_strength':    trial.suggest_float('random_strength', 0.0, 10.0),
            'loss_function':      'Logloss',
            'eval_metric':        'AUC',
            'random_seed':        42,
            'verbose':            0,
            'early_stopping_rounds': 50,
            'task_type':          'CPU',
            'bootstrap_type':     'Bernoulli',
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_aucs = []

        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            # Cat features par indices (CatBoost attend des entiers)
            col_list = X_tr.columns.tolist()
            cat_idx  = [col_list.index(c) for c in cat_features if c in col_list]

            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            if n_pos == 0:
                return 0.5
            params['scale_pos_weight'] = n_neg / n_pos

            # Nettoyage cat features
            X_tr  = X_tr.copy()
            X_val = X_val.copy()
            for c in cat_features:
                if c in X_tr.columns:
                    X_tr[c]  = X_tr[c].fillna('INCONNU').astype(str)
                    X_val[c] = X_val[c].fillna('INCONNU').astype(str)

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                cat_features=cat_idx,
                eval_set=(X_val, y_val),
                verbose=0,
            )
            y_proba = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, y_proba))

        return float(np.mean(fold_aucs))

    study = optuna.create_study(
        direction='maximize',
        study_name=f'catboost_{segment_name}',
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n   ✅ Best CatBoost {segment_name}:")
    print(f"   AUC (3-fold CV) : {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"      {k:25s} : {v}")

    save_path = os.path.join(save_dir, f"catboost_{segment_name.lower()}_study.pkl")
    joblib.dump(study, save_path)
    print(f"   Étude sauvegardée → {save_path}")

    return study, study.best_params


def build_model_params_from_optuna(best_params: dict) -> dict:
    """Construit un MODEL_PARAMS complet depuis les best_params Optuna."""
    return {
        'depth':              best_params['depth'],
        'learning_rate':      best_params['learning_rate'],
        'iterations':         3000,
        'l2_leaf_reg':        best_params['l2_leaf_reg'],
        'border_count':       best_params['border_count'],
        'subsample':          best_params['subsample'],
        'min_data_in_leaf':   best_params['min_data_in_leaf'],
        'random_strength':    best_params['random_strength'],
        'loss_function':      'Logloss',
        'eval_metric':        'AUC',
        'random_seed':        42,
        'verbose':            100,
        'early_stopping_rounds': 50,
        'task_type':          'CPU',
        'bootstrap_type':     'Bernoulli',
    }
