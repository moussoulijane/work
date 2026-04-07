"""
Optimisation des hyperparamètres LSTM via Optuna.
Pruner MedianPruner — arrête tôt les trials sous la médiane.
Score = max(val_auc) sur le meilleur epoch.
"""
import os
import joblib
import numpy as np
import optuna
from optuna.pruners import MedianPruner
import logging

logger = logging.getLogger(__name__)


def optimize_lstm(
    sequences,
    targets,
    n_trials: int = 50,
    save_path: str = "models/optuna/lstm_study.pkl",
) -> tuple:
    """
    Lance l'optimisation Optuna pour le LSTM.

    Args:
        sequences : torch.Tensor (n, 91, 1)
        targets   : np.ndarray (n,)
        n_trials  : nombre de trials Optuna
        save_path : chemin de sauvegarde de l'étude

    Returns:
        study      : optuna.Study
        best_params: dict des meilleurs hyperparamètres
    """
    from src.lstm_model import LSTMTrainer

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        config = {
            'input_size':    1,
            'sequence_length': 91,
            'hidden_size':   trial.suggest_categorical('hidden_size',   [32, 64, 128, 256]),
            'num_layers':    trial.suggest_int('num_layers',            1, 3),
            'dropout':       trial.suggest_float('dropout',             0.0, 0.5, step=0.1),
            'embedding_dim': trial.suggest_categorical('embedding_dim', [16, 32, 64]),
            'learning_rate': trial.suggest_float('learning_rate',       1e-4, 1e-2, log=True),
            'batch_size':    trial.suggest_categorical('batch_size',    [64, 128, 256, 512]),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
            'epochs':        30,   # réduit pour l'optim
            'patience':      7,
        }
        save_dir = f"models/optuna/trial_{trial.number}"
        trainer = LSTMTrainer(config, save_dir=save_dir)
        _, history = trainer.train(sequences, targets)
        best_auc = float(max(history['val_auc']))

        # Signaler les intermédiaires pour le pruner
        for step, auc in enumerate(history['val_auc']):
            trial.report(auc, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_auc

    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name='lstm_optimization',
    )
    # Silencer les logs Optuna sauf WARNING
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n   ✅ Best trial : #{study.best_trial.number}")
    print(f"   Best AUC     : {study.best_value:.4f}")
    print(f"   Best params  :")
    for k, v in study.best_params.items():
        print(f"      {k:20s} : {v}")

    joblib.dump(study, save_path)
    print(f"   Étude sauvegardée → {save_path}")

    return study, study.best_params


def build_config_from_optuna(best_params: dict) -> dict:
    """Construit un LSTM_CONFIG complet depuis les best_params Optuna."""
    return {
        'input_size':     1,
        'sequence_length': 91,
        'hidden_size':    best_params['hidden_size'],
        'num_layers':     best_params['num_layers'],
        'dropout':        best_params['dropout'],
        'embedding_dim':  best_params['embedding_dim'],
        'learning_rate':  best_params['learning_rate'],
        'batch_size':     best_params['batch_size'],
        'bidirectional':  best_params['bidirectional'],
        'epochs':         50,    # full training après optim
        'patience':       10,
    }
