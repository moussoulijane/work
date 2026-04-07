"""Tests unitaires pour ModelEvaluator et ThresholdOptimizer."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import pandas as pd
from src.metrics import ModelEvaluator
from src.threshold_optimizer import optimize_threshold


@pytest.fixture
def y_true_proba():
    """Données de test avec déséquilibre 5%."""
    rng = np.random.default_rng(42)
    n   = 500
    y_true  = np.zeros(n, dtype=int)
    y_true[:25] = 1  # 5% positifs
    # Probas corrélées avec y_true (modèle semi-décent)
    y_proba = rng.beta(2, 5, size=n)
    y_proba[y_true == 1] = rng.beta(5, 2, size=25)
    y_proba = np.clip(y_proba, 0.01, 0.99)
    return y_true, y_proba


class TestModelEvaluator:

    def test_evaluate_returns_dict_with_all_keys(self, y_true_proba, tmp_path):
        y_true, y_proba = y_true_proba
        ev = ModelEvaluator(output_dir=str(tmp_path))
        metrics = ev.evaluate(y_true, y_proba, model_name="test")
        required_keys = [
            'auc_roc', 'gini', 'ks_statistic', 'average_precision',
            'brier_score', 'f1', 'f2', 'precision', 'recall', 'specificity',
            'tp', 'fp', 'fn', 'tn',
        ]
        for k in required_keys:
            assert k in metrics, f"Clé manquante : {k}"

    def test_gini_equals_2_auc_minus_1(self, y_true_proba, tmp_path):
        y_true, y_proba = y_true_proba
        ev = ModelEvaluator(output_dir=str(tmp_path))
        metrics = ev.evaluate(y_true, y_proba, model_name="gini_test")
        assert metrics['gini'] == pytest.approx(2 * metrics['auc_roc'] - 1, abs=1e-9)

    def test_auc_above_random(self, y_true_proba, tmp_path):
        y_true, y_proba = y_true_proba
        ev = ModelEvaluator(output_dir=str(tmp_path))
        metrics = ev.evaluate(y_true, y_proba, model_name="auc_test")
        assert metrics['auc_roc'] > 0.5

    def test_plots_created(self, y_true_proba, tmp_path):
        y_true, y_proba = y_true_proba
        ev = ModelEvaluator(output_dir=str(tmp_path))
        ev.evaluate(y_true, y_proba, model_name="plot_test")
        expected_plots = [
            "plot_test_roc.png", "plot_test_pr_curve.png",
            "plot_test_calibration.png", "plot_test_lift.png",
            "plot_test_score_dist.png",
        ]
        for plot in expected_plots:
            assert os.path.exists(os.path.join(str(tmp_path), plot)), f"Plot manquant : {plot}"

    def test_csv_created(self, y_true_proba, tmp_path):
        y_true, y_proba = y_true_proba
        ev = ModelEvaluator(output_dir=str(tmp_path))
        ev.evaluate(y_true, y_proba, model_name="csv_test")
        assert os.path.exists(os.path.join(str(tmp_path), "csv_test_metrics.csv"))

    def test_cm_consistency(self, y_true_proba, tmp_path):
        """TP + FP + FN + TN doit égaler n_samples."""
        y_true, y_proba = y_true_proba
        ev = ModelEvaluator(output_dir=str(tmp_path))
        m = ev.evaluate(y_true, y_proba, model_name="cm_test")
        assert m['tp'] + m['fp'] + m['fn'] + m['tn'] == len(y_true)

    def test_compare_saves_csv(self, y_true_proba, tmp_path):
        y_true, y_proba = y_true_proba
        ev = ModelEvaluator(output_dir=str(tmp_path))
        m1 = ev.evaluate(y_true, y_proba, model_name="m1")
        m2 = ev.evaluate(y_true, y_proba, model_name="m2", threshold=0.3)
        ev.compare([m1, m2])
        assert os.path.exists(os.path.join(str(tmp_path), "model_comparison.csv"))


class TestThresholdOptimizer:

    def test_returns_three_values(self, y_true_proba):
        y_true, y_proba = y_true_proba
        result = optimize_threshold(y_true, y_proba, strategy='f2')
        assert len(result) == 3

    def test_threshold_in_range(self, y_true_proba):
        y_true, y_proba = y_true_proba
        opt_t, _, _ = optimize_threshold(y_true, y_proba, strategy='f2')
        assert 0.01 <= opt_t <= 0.99

    @pytest.mark.parametrize("strategy", ["f1", "f2", "profit", "youden"])
    def test_all_strategies(self, y_true_proba, strategy):
        y_true, y_proba = y_true_proba
        opt_t, metrics, df_all = optimize_threshold(y_true, y_proba, strategy=strategy)
        assert isinstance(opt_t, float)
        assert len(df_all) == 1000

    def test_f2_favors_recall(self, y_true_proba):
        """Le seuil F2 doit produire un recall >= celui du seuil 0.5."""
        y_true, y_proba = y_true_proba
        opt_t_f2, m_f2, _ = optimize_threshold(y_true, y_proba, strategy='f2')
        opt_t_f1, m_f1, _ = optimize_threshold(y_true, y_proba, strategy='f1')
        # F2 favorise le rappel → seuil plus bas → recall plus élevé
        assert m_f2['recall'] >= m_f1['recall'] - 0.01  # tolérance numérique

    def test_invalid_strategy_raises(self, y_true_proba):
        y_true, y_proba = y_true_proba
        with pytest.raises(ValueError):
            optimize_threshold(y_true, y_proba, strategy='invalid')
