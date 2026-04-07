"""
Optimisation du seuil de décision.

4 stratégies : f1, f2, profit, youden.
Pour l'appétence crédit : recommander F2 (favorise le rappel).
Le seuil optimal sera typiquement ~0.25-0.45 (pas 0.5).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
import logging

logger = logging.getLogger(__name__)


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = 'f2',
    gain_tp: float = 4800,
    cost_fp: float = 50,
    cost_fn: float = 2400,
) -> tuple[float, dict, pd.DataFrame]:
    """
    Trouve le seuil optimal selon la stratégie choisie.

    Args:
        y_true    : array 0/1
        y_proba   : array de probas [0, 1]
        strategy  : 'f1', 'f2', 'profit', 'youden'
        gain_tp   : gain par vrai positif (MAD) — marge crédit conso
        cost_fp   : coût par faux positif (MAD) — temps chargé CC
        cost_fn   : coût par faux négatif (MAD) — manque à gagner

    Returns:
        optimal_threshold : float
        metrics_at_opt    : dict des métriques au seuil optimal
        df_all            : DataFrame de toutes les métriques par seuil
    """
    if strategy not in ('f1', 'f2', 'profit', 'youden'):
        raise ValueError(f"strategy doit être parmi f1/f2/profit/youden, reçu : {strategy}")

    thresholds = np.linspace(0.01, 0.99, 1000)
    y_true = np.asarray(y_true)
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
            'threshold':  float(t),
            'precision':  precision,
            'recall':     recall,
            'f1':         f1,
            'f2':         f2,
            'profit':     float(profit),
            'youden':     youden,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'n_predicted_positive': tp + fp,
        })

    df = pd.DataFrame(records)
    best_idx = int(df[strategy].idxmax())
    optimal_row = df.iloc[best_idx]
    optimal_threshold = float(optimal_row['threshold'])

    # Affichage comparatif
    print(f"\n  Seuil optimal ({strategy}) : {optimal_threshold:.3f}")
    print(f"\n  {'Seuil':>8} {'Précision':>10} {'Rappel':>8} "
          f"{'F1':>6} {'F2':>6} {'Profit':>10} {'N prédit+':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*10} {'─'*10}")

    key_thresholds = [0.3, 0.4, 0.5, optimal_threshold, 0.7]
    shown = set()
    for t in key_thresholds:
        t_rounded = round(t, 3)
        if t_rounded in shown:
            continue
        shown.add(t_rounded)
        row = df.iloc[(df['threshold'] - t).abs().idxmin()]
        marker = " ← optimal" if abs(row['threshold'] - optimal_threshold) < 0.002 else ""
        print(f"  {row['threshold']:8.3f} {row['precision']:10.4f} {row['recall']:8.4f} "
              f"{row['f1']:6.4f} {row['f2']:6.4f} {row['profit']:10.0f} "
              f"{int(row['n_predicted_positive']):10d}{marker}")

    return optimal_threshold, optimal_row.to_dict(), df
