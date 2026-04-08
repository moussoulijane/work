"""
Utilitaire partagé pour détecter et trier les colonnes de soldes quotidiens.

Les colonnes peuvent être nommées de deux façons selon la source :
  - Format numérique : jour_1, jour_2, ..., jour_91
  - Format date     : jour_01/10/2024, jour_02/10/2024, ...

Toutes les fonctions de ce module acceptent les deux formats.
"""
import re
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_jour_cols(df: pd.DataFrame) -> list[str]:
    """
    Détecte et trie les colonnes qui représentent des soldes quotidiens.
    Accepte indifféremment :
      - jour_1, jour_2, ... jour_91  (format numérique)
      - jour_01/10/2024, ...         (format date DD/MM/YYYY)
      - Tout autre suffixe après 'jour_'
    """
    cols = [c for c in df.columns if c.startswith('jour_')]
    if not cols:
        return []

    def sort_key(col: str):
        suffix = col[len('jour_'):]
        # Essai 1 : entier pur (jour_1, jour_91)
        try:
            return (0, int(suffix), datetime.min)
        except ValueError:
            pass
        # Essai 2 : date DD/MM/YYYY (jour_01/10/2024)
        for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d_%m_%Y'):
            try:
                return (1, 0, datetime.strptime(suffix, fmt))
            except ValueError:
                pass
        # Fallback alphabétique
        return (2, 0, datetime.min)

    return sorted(cols, key=sort_key)


def to_float_array(df: pd.DataFrame, jour_cols: list) -> np.ndarray:
    """
    Convertit les colonnes jour_* en np.float32, quelle que soit leur forme :
      - Déjà float/int  → cast direct + nan_to_num(0)
      - str '2515,61'   → virgule→point + cast
      - str '1 234.50'  → supprime espace, cast
      - NaN / vide      → 0.0

    Traite chaque colonne individuellement selon son dtype pour éviter
    de supprimer les points sur des colonnes déjà float.
    """
    subset = df[jour_cols]

    # Cas rapide : toutes numériques
    if all(pd.api.types.is_numeric_dtype(subset[c]) for c in jour_cols):
        return np.nan_to_num(subset.values.astype(np.float32), nan=0.0)

    # Cas général : colonne par colonne
    arrays = []
    for col in jour_cols:
        s = subset[col]
        if pd.api.types.is_numeric_dtype(s):
            arrays.append(
                pd.to_numeric(s, errors='coerce').fillna(0.0).values.astype(np.float32)
            )
        else:
            # Virgule française : '2 515,61' → '2515.61'
            converted = (
                s.astype(str)
                 .str.strip()
                 .str.replace(r'\s', '', regex=True)    # espaces
                 .str.replace('.', '', regex=False)      # séparateur milliers
                 .str.replace(',', '.', regex=False)     # décimal FR
                 .replace({'nan': '0', 'None': '0', 'none': '0', '': '0'})
            )
            arrays.append(
                pd.to_numeric(converted, errors='coerce').fillna(0.0).values.astype(np.float32)
            )

    return np.nan_to_num(np.column_stack(arrays), nan=0.0)
