"""
Transforme jour_1..jour_91 en tenseurs PyTorch normalisés.
"""
import os
import numpy as np
import pandas as pd
import torch
import joblib
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

JOUR_COLS = [f'jour_{i}' for i in range(1, 92)]


def _to_float_array(df: pd.DataFrame, jour_cols: list) -> np.ndarray:
    """
    Convertit les colonnes jour_* en np.float32, quelle que soit leur forme :
      - déjà float/int  → cast direct
      - str avec virgule française ('2515,61') → remplacement virgule→point puis cast
      - NaN / vide → 0.0

    C'est la seule source du ValueError "could not convert string to float :
    '2515,61'" : le cache parquet peut conserver les colonnes en object dtype
    si elles n'ont pas été converties avant sauvegarde.
    """
    subset = df[jour_cols]

    # Cas rapide : toutes les colonnes sont déjà numériques
    if all(pd.api.types.is_numeric_dtype(subset[c]) for c in jour_cols):
        arr = subset.values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0)

    # Cas général : colonnes mixtes ou object → conversion colonne par colonne
    arrays = []
    for col in jour_cols:
        s = subset[col]
        if pd.api.types.is_numeric_dtype(s):
            arrays.append(pd.to_numeric(s, errors='coerce').fillna(0.0).values)
        else:
            # Virgule française : supprimer espaces + points milliers, virgule → point
            arrays.append(
                s.astype(str)
                 .str.replace(' ', '', regex=False)
                 .str.replace('.', '', regex=False)   # séparateur milliers
                 .str.replace(',', '.', regex=False)  # décimal français
                 .replace({'nan': '0', 'None': '0', '': '0'})
                 .astype(float)
                 .values
            )
    result = np.column_stack(arrays).astype(np.float32)
    return np.nan_to_num(result, nan=0.0)


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
