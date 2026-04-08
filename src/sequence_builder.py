"""
Transforme les colonnes jour_* (format numérique ou date) en tenseurs PyTorch normalisés.
"""
import os
import numpy as np
import pandas as pd
import torch
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from src.jour_utils import get_jour_cols, to_float_array as _to_float_array

logger = logging.getLogger(__name__)


class SequenceBuilder:
    def __init__(self, scaler_path: str = "models/lstm_scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler: StandardScaler | None = None

    def _get_jour_cols(self, df):
        cols = get_jour_cols(df)
        if not cols:
            raise ValueError("Aucune colonne jour_* trouvée dans le DataFrame.")
        return cols

    def fit_transform(self, df):
        """
        MODE TRAIN : fit StandardScaler sur les 91 jours, transform, save scaler.
        Returns:
            sequences : torch.FloatTensor (n, 91, 1)
            ids       : np.ndarray (n,) — id_client dans le même ordre
        """
        jour_cols = self._get_jour_cols(df)
        raw = _to_float_array(df, jour_cols)   # gère virgule française + NaN

        self.scaler = StandardScaler()
        normalized = self.scaler.fit_transform(raw)

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
        raw = _to_float_array(df, jour_cols)   # gère virgule française + NaN

        self.scaler = joblib.load(self.scaler_path)
        normalized = self.scaler.transform(raw)

        sequences = torch.FloatTensor(normalized).unsqueeze(-1)
        ids = df['id_client'].values
        logger.info(f"Séquences infer : {sequences.shape}")
        return sequences, ids
