"""
Transforme jour_1..jour_91 en tenseurs PyTorch normalisés.
"""
import os
import numpy as np
import torch
import joblib
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

JOUR_COLS = [f'jour_{i}' for i in range(1, 92)]


class SequenceBuilder:
    def __init__(self, scaler_path: str = "models/lstm_scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler: StandardScaler | None = None

    def _get_jour_cols(self, df):
        cols = sorted([c for c in df.columns if c.startswith('jour_')],
                      key=lambda x: int(x.split('_')[1]))
        assert len(cols) == 91, f"Attendu 91 colonnes jour_*, trouvé {len(cols)}"
        return cols

    def fit_transform(self, df):
        """
        MODE TRAIN : fit StandardScaler sur les 91 jours, transform, save scaler.
        Returns:
            sequences : torch.FloatTensor (n, 91, 1)
            ids       : np.ndarray (n,) — id_client dans le même ordre
        """
        jour_cols = self._get_jour_cols(df)
        raw = df[jour_cols].values.astype(np.float32)  # (n, 91)

        self.scaler = StandardScaler()
        normalized = self.scaler.fit_transform(raw)  # fit sur tout le dataset

        os.makedirs(os.path.dirname(self.scaler_path) or ".", exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler LSTM sauvegardé → {self.scaler_path}")

        sequences = torch.FloatTensor(normalized).unsqueeze(-1)  # (n, 91, 1)
        ids = df['id_client'].values
        logger.info(f"Séquences train : {sequences.shape}")
        return sequences, ids

    def transform(self, df):
        """
        MODE INFER : charge le scaler depuis disk, transform seulement (pas de fit).
        Returns:
            sequences : torch.FloatTensor (n, 91, 1)
            ids       : np.ndarray (n,)
        """
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Scaler LSTM introuvable : {self.scaler_path}. "
                "Lance d'abord : python main.py train"
            )
        jour_cols = self._get_jour_cols(df)
        raw = df[jour_cols].values.astype(np.float32)

        self.scaler = joblib.load(self.scaler_path)
        normalized = self.scaler.transform(raw)

        sequences = torch.FloatTensor(normalized).unsqueeze(-1)
        ids = df['id_client'].values
        logger.info(f"Séquences infer : {sequences.shape}")
        return sequences, ids
