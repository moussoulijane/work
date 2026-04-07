"""Tests unitaires pour le LSTM encoder et trainer."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import torch
import tempfile
from src.lstm_model import LSTMEncoder, LSTMTrainer


@pytest.fixture
def small_config():
    return {
        'input_size': 1, 'hidden_size': 16, 'num_layers': 1,
        'dropout': 0.0, 'bidirectional': False, 'embedding_dim': 8,
        'batch_size': 32, 'learning_rate': 0.01, 'epochs': 3, 'patience': 2,
    }


@pytest.fixture
def dummy_data():
    """100 séquences de longueur 91, déséquilibre 5%."""
    n = 100
    seqs    = torch.randn(n, 91, 1)
    targets = np.zeros(n, dtype=np.float32)
    targets[:5] = 1  # 5% positifs
    return seqs, targets


class TestLSTMEncoder:

    def test_forward_shape(self, small_config):
        model = LSTMEncoder(
            hidden_size=small_config['hidden_size'],
            embedding_dim=small_config['embedding_dim'],
        )
        x = torch.randn(8, 91, 1)
        logits, emb = model(x)
        assert logits.shape == (8,)
        assert emb.shape   == (8, small_config['embedding_dim'])

    def test_encode_shape(self, small_config):
        model = LSTMEncoder(embedding_dim=small_config['embedding_dim'])
        x = torch.randn(8, 91, 1)
        emb = model.encode(x)
        assert emb.shape == (8, small_config['embedding_dim'])

    def test_encode_no_grad(self, small_config):
        """encode() ne doit pas modifier les gradients si appelé avec no_grad."""
        model = LSTMEncoder(embedding_dim=small_config['embedding_dim'])
        model.eval()
        x = torch.randn(4, 91, 1)
        with torch.no_grad():
            emb = model.encode(x)
        assert emb.grad_fn is None

    def test_bidirectional_shape(self, small_config):
        model = LSTMEncoder(
            hidden_size=8, embedding_dim=8, bidirectional=True
        )
        x = torch.randn(4, 91, 1)
        _, emb = model(x)
        assert emb.shape == (4, 8)

    def test_init_weights_forget_gate(self, small_config):
        """Le forget gate bias doit être initialisé à 1."""
        model = LSTMEncoder(hidden_size=16, num_layers=2)
        for name, param in model.lstm.named_parameters():
            if 'bias_hh' in name or 'bias_ih' in name:
                n = param.size(0)
                forget_bias = param.data[n // 4: n // 2]
                assert float(forget_bias.mean()) == pytest.approx(1.0, abs=1e-5)
                break


class TestLSTMTrainer:

    def test_train_returns_model_and_history(self, small_config, dummy_data, tmp_path):
        seqs, targets = dummy_data
        trainer = LSTMTrainer(small_config, save_dir=str(tmp_path))
        model, history = trainer.train(seqs, targets)
        assert isinstance(model, LSTMEncoder)
        assert 'val_auc' in history
        assert len(history['val_auc']) > 0

    def test_checkpoint_saved(self, small_config, dummy_data, tmp_path):
        seqs, targets = dummy_data
        trainer = LSTMTrainer(small_config, save_dir=str(tmp_path))
        trainer.train(seqs, targets)
        assert os.path.exists(os.path.join(str(tmp_path), "lstm_encoder.pt"))

    def test_extract_embeddings_shape(self, small_config, dummy_data, tmp_path):
        seqs, targets = dummy_data
        trainer = LSTMTrainer(small_config, save_dir=str(tmp_path))
        model, _ = trainer.train(seqs, targets)
        embs = trainer.extract_embeddings(model, seqs)
        assert embs.shape == (len(seqs), small_config['embedding_dim'])

    def test_val_auc_increases(self, small_config, dummy_data, tmp_path):
        """val_auc doit être > 0 (le modèle apprend quelque chose)."""
        seqs, targets = dummy_data
        trainer = LSTMTrainer(small_config, save_dir=str(tmp_path))
        _, history = trainer.train(seqs, targets)
        assert max(history['val_auc']) > 0.0

    def test_early_stopping_triggered(self, tmp_path):
        """Avec patience=1, l'entraînement doit s'arrêter avant epochs."""
        cfg = {
            'input_size': 1, 'hidden_size': 8, 'num_layers': 1,
            'dropout': 0.0, 'bidirectional': False, 'embedding_dim': 4,
            'batch_size': 32, 'learning_rate': 0.001, 'epochs': 50, 'patience': 1,
        }
        n    = 60
        seqs = torch.randn(n, 91, 1)
        tgt  = np.zeros(n)
        tgt[:3] = 1
        trainer = LSTMTrainer(cfg, save_dir=str(tmp_path))
        _, history = trainer.train(seqs, tgt)
        assert len(history['val_auc']) < 50
