"""
Calibration post-hoc des probabilités via régression isotonique.

Pourquoi calibrer :
  - CatBoost avec scale_pos_weight produit des probas mal calibrées sur les
    classes très déséquilibrées (brier_score élevé sur segment LOW).
  - La régression isotonique corrige la courbe de calibration sans changer
    le ranking (AUC inchangé) mais améliore le Brier score et rend le
    choix de seuil plus robuste.

Usage :
    cal = ProbabilityCalibrator()
    cal.fit(y_true_holdout, y_proba_raw_holdout)   # fit sur hold-out (30% eval set)
    cal.save("models/calibrator_low.pkl")

    y_proba_cal = cal.transform(y_proba_raw)        # infer
    cal2 = ProbabilityCalibrator.load("models/calibrator_low.pkl")
"""
import os
import numpy as np
import joblib
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:

    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self._fitted    = False

    def fit(self, y_true: np.ndarray, y_proba_raw: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit sur un hold-out set (JAMAIS sur les données d'entraînement CatBoost).
        Typiquement : le 30% eval set issu du split CatBoostTrainer.
        """
        y_true      = np.asarray(y_true,      dtype=float)
        y_proba_raw = np.asarray(y_proba_raw, dtype=float)
        self.calibrator.fit(y_proba_raw, y_true)
        self._fitted = True

        # Log amélioration Brier
        from sklearn.metrics import brier_score_loss
        brier_before = brier_score_loss(y_true, y_proba_raw)
        brier_after  = brier_score_loss(y_true, self.transform(y_proba_raw))
        logger.info(
            f"Calibration : Brier {brier_before:.4f} → {brier_after:.4f} "
            f"(Δ={brier_after - brier_before:+.4f})"
        )
        print(f"  Calibration : Brier {brier_before:.4f} → {brier_after:.4f}")
        return self

    def transform(self, y_proba_raw: np.ndarray) -> np.ndarray:
        """Calibre les probabilités brutes."""
        if not self._fitted:
            raise RuntimeError("ProbabilityCalibrator non fitté — appeler .fit() d'abord")
        return self.calibrator.predict(np.asarray(y_proba_raw, dtype=float))

    def fit_transform(self, y_true: np.ndarray, y_proba_raw: np.ndarray) -> np.ndarray:
        return self.fit(y_true, y_proba_raw).transform(y_proba_raw)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Calibrateur sauvegardé → {path}")

    @classmethod
    def load(cls, path: str) -> 'ProbabilityCalibrator':
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Calibrateur introuvable : {path}. Lance d'abord : python main.py train"
            )
        return joblib.load(path)

    def plot_calibration_comparison(
        self, y_true, y_proba_raw, model_name: str = "model", output_dir: str = "outputs/metrics"
    ):
        """Compare la courbe de calibration avant / après correction."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        y_proba_cal = self.transform(y_proba_raw)
        prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_proba_raw, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_proba_cal, n_bins=10)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(prob_pred_raw, prob_true_raw, 'o--', lw=1.5, alpha=0.7, label='Avant calibration')
        ax.plot(prob_pred_cal, prob_true_cal, 's-',  lw=2,   label='Après calibration (isotonique)')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Calibration parfaite')
        ax.set_xlabel('Probabilité prédite (moyenne par bin)')
        ax.set_ylabel('Fraction de positifs réels')
        ax.set_title(f'Calibration avant/après — {model_name}')
        ax.legend()
        ax.grid(alpha=0.3)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{model_name}_calibration_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Plot calibration → {path}")
