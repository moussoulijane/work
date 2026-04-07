"""
Évaluation complète du modèle : 10+ métriques + 5 plots PNG.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss, f1_score, fbeta_score,
    accuracy_score, log_loss, confusion_matrix,
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:

    def __init__(self, output_dir: str = "outputs/metrics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # Evaluate
    # ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "model",
        threshold: float = 0.5,
    ) -> dict:
        """
        Évaluation complète : métriques scalaires + 5 plots.
        Returns: dict de toutes les métriques.
        """
        y_true  = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        y_pred  = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_proba)
        auc_roc = roc_auc_score(y_true, y_proba)

        metrics = {
            'model':             model_name,
            'threshold':         threshold,
            'n_samples':         len(y_true),
            'n_positive':        int(y_true.sum()),
            'n_negative':        int((1 - y_true).sum()),
            'prevalence':        float(y_true.mean()),
            'auc_roc':           auc_roc,
            'gini':              2 * auc_roc - 1,
            'ks_statistic':      float(np.max(tpr_curve - fpr_curve)),
            'average_precision': average_precision_score(y_true, y_proba),
            'brier_score':       brier_score_loss(y_true, y_proba),
            'log_loss':          log_loss(y_true, y_proba),
            'accuracy':          accuracy_score(y_true, y_pred),
            'f1':                f1_score(y_true, y_pred, zero_division=0),
            'f2':                fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            'precision':         tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall':            tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity':       tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'fpr':               fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }

        # Affichage
        print(f"\n{'═'*57}")
        print(f"  ÉVALUATION : {model_name}  (seuil={threshold})")
        print(f"{'═'*57}")
        for k in ['auc_roc', 'gini', 'ks_statistic', 'average_precision',
                  'brier_score', 'f1', 'f2', 'precision', 'recall', 'specificity']:
            print(f"  {k:22s} : {metrics[k]:.4f}")
        print(f"  {'─'*55}")
        print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
        print(f"{'═'*57}")

        # Plots
        self._plot_roc(y_true, y_proba, model_name)
        self._plot_pr(y_true, y_proba, model_name)
        self._plot_calibration(y_true, y_proba, model_name)
        self._plot_lift(y_true, y_proba, model_name)
        self._plot_score_dist(y_true, y_proba, model_name)

        # Sauvegarde CSV
        pd.DataFrame([metrics]).to_csv(
            os.path.join(self.output_dir, f"{model_name}_metrics.csv"), index=False
        )

        return metrics

    # ─────────────────────────────────────────────────────────
    # Compare
    # ─────────────────────────────────────────────────────────

    def compare(self, metrics_list: list[dict]) -> pd.DataFrame:
        """Tableau comparatif de plusieurs modèles."""
        df = pd.DataFrame(metrics_list)
        cols = ['model', 'auc_roc', 'gini', 'ks_statistic', 'average_precision',
                'f1', 'f2', 'precision', 'recall', 'brier_score']
        df_display = df[[c for c in cols if c in df.columns]]

        print(f"\n  {'─'*70}")
        print(f"  COMPARAISON DES MODÈLES")
        print(f"  {'─'*70}")
        print(df_display.to_string(index=False))

        path = os.path.join(self.output_dir, "model_comparison.csv")
        df_display.to_csv(path, index=False)
        print(f"\n  Comparaison sauvegardée → {path}")
        return df_display

    # ─────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────

    def _save_fig(self, fig, name: str):
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Plot sauvegardé → {path}")

    def _plot_roc(self, y_true, y_proba, name):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Aléatoire')
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title(f'Courbe ROC — {name}')
        ax.legend()
        ax.grid(alpha=0.3)
        self._save_fig(fig, f"{name}_roc.png")

    def _plot_pr(self, y_true, y_proba, name):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        baseline = float(y_true.mean())
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, lw=2, label=f'AP = {ap:.4f}')
        ax.axhline(baseline, color='r', linestyle='--', alpha=0.5,
                   label=f'Baseline = {baseline:.4f}')
        ax.set_xlabel('Rappel')
        ax.set_ylabel('Précision')
        ax.set_title(f'Courbe Précision-Rappel — {name}')
        ax.legend()
        ax.grid(alpha=0.3)
        self._save_fig(fig, f"{name}_pr_curve.png")

    def _plot_calibration(self, y_true, y_proba, name):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(prob_pred, prob_true, 'o-', lw=2, label='Modèle')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Calibration parfaite')
        ax.set_xlabel('Probabilité prédite (moyenne par bin)')
        ax.set_ylabel('Fraction de positifs réels')
        ax.set_title(f'Courbe de Calibration — {name}')
        ax.legend()
        ax.grid(alpha=0.3)
        self._save_fig(fig, f"{name}_calibration.png")

    def _plot_lift(self, y_true, y_proba, name):
        sorted_idx = np.argsort(y_proba)[::-1]
        y_sorted   = y_true[sorted_idx]
        cum_pos    = np.cumsum(y_sorted)
        total_pos  = y_true.sum()
        pct_pop    = np.arange(1, len(y_true) + 1) / len(y_true)
        pct_cap    = cum_pos / total_pos

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(pct_pop * 100, pct_cap * 100, lw=2, label='Modèle')
        ax.plot([0, 100], [0, 100], 'k--', lw=1, label='Aléatoire')
        ax.set_xlabel('% de la population contactée')
        ax.set_ylabel('% des souscripteurs capturés')
        ax.set_title(f'Courbe de Gains Cumulatifs — {name}')
        ax.legend()
        ax.grid(alpha=0.3)

        for pct in [10, 20, 30]:
            idx      = min(int(len(y_true) * pct / 100), len(pct_cap) - 1)
            captured = pct_cap[idx] * 100
            lift     = captured / pct if pct > 0 else 0
            ax.annotate(
                f'{pct}% → {captured:.0f}%\n(lift {lift:.1f}×)',
                xy=(pct, captured),
                xytext=(pct + 8, max(captured - 12, 5)),
                fontsize=8,
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
            )

        self._save_fig(fig, f"{name}_lift.png")

    def _plot_score_dist(self, y_true, y_proba, name):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(y_proba[y_true == 0], bins=50, alpha=0.6,
                label='Non-souscripteurs (0)', density=True)
        ax.hist(y_proba[y_true == 1], bins=50, alpha=0.6,
                label='Souscripteurs (1)', density=True)
        ax.set_xlabel('Score de probabilité')
        ax.set_ylabel('Densité')
        ax.set_title(f'Distribution des scores — {name}')
        ax.legend()
        ax.grid(alpha=0.3)
        self._save_fig(fig, f"{name}_score_dist.png")
