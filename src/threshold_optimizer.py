"""
Optimisation du seuil de décision.

Stratégies classiques : f1, f2, profit, youden.
Stratégie précision-cible : precision_target (recommandée pour campagnes ciblées).

Pour l'appétence crédit :
  - Campagne large   → f2  (rappel prioritaire, seuil ~0.25-0.40)
  - Campagne ciblée  → precision_target(min_precision=0.10-0.15)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
import logging

logger = logging.getLogger(__name__)

_STRATEGIES = ('f1', 'f2', 'profit', 'youden', 'precision_target')


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = 'f2',
    gain_tp: float = 4800,
    cost_fp: float = 50,
    cost_fn: float = 2400,
    min_precision: float = 0.10,   # utilisé uniquement pour strategy='precision_target'
) -> tuple[float, dict, pd.DataFrame]:
    """
    Trouve le seuil optimal selon la stratégie choisie.

    Args:
        y_true         : array 0/1
        y_proba        : array de probas [0, 1]
        strategy       : 'f1' | 'f2' | 'profit' | 'youden' | 'precision_target'
        gain_tp        : gain par vrai positif (MAD)
        cost_fp        : coût par faux positif (MAD)
        cost_fn        : coût par faux négatif (MAD)
        min_precision  : précision minimale cible (strategy='precision_target' seulement)

    Returns:
        optimal_threshold : float
        metrics_at_opt    : dict des métriques au seuil optimal
        df_all            : DataFrame de toutes les métriques par seuil
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy doit être parmi {_STRATEGIES}, reçu : {strategy}")

    thresholds = np.linspace(0.01, 0.99, 1000)
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    records = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1          = (2 * precision * recall / (precision + recall)
                       if (precision + recall) > 0 else 0.0)
        f2          = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        profit      = tp * gain_tp - fp * cost_fp - fn * cost_fn
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden      = recall + specificity - 1.0

        records.append({
            'threshold':             float(t),
            'precision':             precision,
            'recall':                recall,
            'f1':                    f1,
            'f2':                    f2,
            'profit':                float(profit),
            'youden':                youden,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'n_predicted_positive':  tp + fp,
        })

    df = pd.DataFrame(records)

    if strategy == 'precision_target':
        # Seuil le plus bas qui garantit precision >= min_precision
        # → maximise le recall sous contrainte de précision
        eligible = df[df['precision'] >= min_precision]
        if eligible.empty:
            logger.warning(
                f"Aucun seuil n'atteint precision >= {min_precision:.2f} — "
                "retour au seuil F2"
            )
            best_idx = int(df['f2'].idxmax())
        else:
            best_idx = int(eligible['recall'].idxmax())
    else:
        best_idx = int(df[strategy].idxmax())

    optimal_row       = df.iloc[best_idx]
    optimal_threshold = float(optimal_row['threshold'])

    # ── Affichage ──
    print(f"\n  Seuil optimal ({strategy}) : {optimal_threshold:.3f}")
    if strategy == 'precision_target':
        print(f"  Contrainte précision ≥ {min_precision:.2%}")
    print(f"\n  {'Seuil':>8} {'Précision':>10} {'Rappel':>8} "
          f"{'F1':>6} {'F2':>6} {'Profit':>10} {'N prédit+':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*10} {'─'*10}")

    shown = set()
    for t in [0.3, 0.4, 0.5, optimal_threshold, 0.7]:
        t_r = round(t, 3)
        if t_r in shown:
            continue
        shown.add(t_r)
        row    = df.iloc[(df['threshold'] - t).abs().idxmin()]
        marker = " ← optimal" if abs(row['threshold'] - optimal_threshold) < 0.002 else ""
        print(f"  {row['threshold']:8.3f} {row['precision']:10.4f} {row['recall']:8.4f} "
              f"{row['f1']:6.4f} {row['f2']:6.4f} {row['profit']:10.0f} "
              f"{int(row['n_predicted_positive']):10d}{marker}")

    return optimal_threshold, optimal_row.to_dict(), df
