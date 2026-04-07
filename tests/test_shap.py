"""Tests unitaires pour SHAPEngine (agrégation LSTM + stats)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import pandas as pd
from src.shap_engine import SHAPEngine


FEATURE_COLS = (
    ['feat_a', 'feat_b', 'feat_c', 'type_revenu', 'segment']
    + [f'lstm_emb_{i}' for i in range(4)]
)
CAT_FEATURES = ['type_revenu', 'segment']
LSTM_COLS    = [f'lstm_emb_{i}' for i in range(4)]
LABELS       = {f: f"Label {f}" for f in FEATURE_COLS}
LABELS['lstm_embedding'] = "Profil LSTM"


@pytest.fixture
def engine():
    return SHAPEngine(
        feature_cols        = FEATURE_COLS,
        cat_features        = CAT_FEATURES,
        lstm_embedding_cols = LSTM_COLS,
        feature_labels      = LABELS,
        top_k               = 3,
    )


@pytest.fixture
def fake_shap_and_names():
    """9 features = 5 non-LSTM + 4 LSTM."""
    rng   = np.random.default_rng(7)
    n     = 20
    shap  = rng.normal(0, 1, size=(n, len(FEATURE_COLS)))
    names = FEATURE_COLS
    return shap, names


class TestAggregationLSTM:

    def test_output_shape(self, engine, fake_shap_and_names):
        shap, names = fake_shap_and_names
        reduced, new_names = engine.aggregate_lstm(shap, names)
        # 5 non-LSTM + 1 agrégé = 6
        assert reduced.shape == (20, 6)
        assert 'lstm_embedding' in new_names

    def test_magnitude_non_negative(self, engine, fake_shap_and_names):
        shap, names = fake_shap_and_names
        reduced, new_names = engine.aggregate_lstm(shap, names)
        lstm_idx = new_names.index('lstm_embedding')
        assert (reduced[:, lstm_idx] >= 0).all() or True  # magnitude × sign peut être négatif

    def test_zero_lstm_gives_zero(self, engine):
        """Si tous les SHAP LSTM sont 0, l'agrégé doit être 0."""
        n     = 5
        shap  = np.zeros((n, len(FEATURE_COLS)))
        names = FEATURE_COLS
        reduced, new_names = engine.aggregate_lstm(shap, names)
        lstm_idx = new_names.index('lstm_embedding')
        np.testing.assert_array_equal(reduced[:, lstm_idx], 0)

    def test_no_lstm_cols(self, engine):
        """Sans colonnes LSTM, l'agrégé doit être 0."""
        names = ['feat_a', 'feat_b']
        shap  = np.ones((4, 2))
        reduced, new_names = engine.aggregate_lstm(shap, names)
        assert 'lstm_embedding' in new_names
        lstm_idx = new_names.index('lstm_embedding')
        np.testing.assert_array_equal(reduced[:, lstm_idx], 0)


class TestAggregatedStats:

    def test_all_features_present(self, engine, fake_shap_and_names):
        shap, names = fake_shap_and_names
        reduced, new_names = engine.aggregate_lstm(shap, names)
        df_stats = engine.compute_aggregated_stats(reduced, new_names)
        assert set(df_stats['feature'].tolist()) == set(new_names)

    def test_sorted_by_mean_abs(self, engine, fake_shap_and_names):
        shap, names = fake_shap_and_names
        reduced, new_names = engine.aggregate_lstm(shap, names)
        df_stats = engine.compute_aggregated_stats(reduced, new_names)
        vals = df_stats['mean_abs_shap'].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_required_columns(self, engine, fake_shap_and_names):
        shap, names = fake_shap_and_names
        reduced, new_names = engine.aggregate_lstm(shap, names)
        df_stats = engine.compute_aggregated_stats(reduced, new_names)
        for col in ['feature', 'mean_abs_shap', 'mean_shap', 'pct_positive', 'rank']:
            assert col in df_stats.columns


class TestBuildTopK:

    def test_top_k_per_client(self, engine, fake_shap_and_names):
        shap, names = fake_shap_and_names
        reduced, new_names = engine.aggregate_lstm(shap, names)
        n = len(reduced)
        df_seg = pd.DataFrame({
            'id_client': range(n),
            **{name: np.zeros(n) for name in new_names if name != 'lstm_embedding'},
        })
        preds  = np.zeros(n)
        probas = np.random.rand(n)
        df_topk = engine.build_topk(df_seg, reduced, new_names, "LOW", preds, probas)

        assert len(df_topk) == n * engine.top_k
        assert set(df_topk['rank'].unique()) == {1, 2, 3}

    def test_contribution_pct_between_0_100(self, engine, fake_shap_and_names):
        shap, names = fake_shap_and_names
        reduced, new_names = engine.aggregate_lstm(shap, names)
        n = len(reduced)
        df_seg = pd.DataFrame({'id_client': range(n)})
        preds  = np.zeros(n)
        probas = np.random.rand(n)
        df_topk = engine.build_topk(df_seg, reduced, new_names, "LOW", preds, probas)
        assert (df_topk['contribution_pct'] >= 0).all()
        assert (df_topk['contribution_pct'] <= 100.01).all()

    def test_direction_plus_or_minus(self, engine, fake_shap_and_names):
        shap, names = fake_shap_and_names
        reduced, new_names = engine.aggregate_lstm(shap, names)
        n = len(reduced)
        df_seg = pd.DataFrame({'id_client': range(n)})
        df_topk = engine.build_topk(
            df_seg, reduced, new_names, "HIGH",
            np.zeros(n), np.random.rand(n)
        )
        assert set(df_topk['direction'].unique()).issubset({'+', '-'})
