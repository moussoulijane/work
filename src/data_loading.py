"""
Chargement et fusion des CSV sources.
"""
import os
import pandas as pd
import logging
from src.jour_utils import get_jour_cols

logger = logging.getLogger(__name__)


def _parse_french_float(series: pd.Series) -> pd.Series:
    """Convertit les flottants au format virgule française (1.234,56 → 1234.56)."""
    if series.dtype == object:
        return (
            series.astype(str)
            .str.replace(' ', '', regex=False)
            .str.replace('.', '', regex=False)  # séparateur milliers
            .str.replace(',', '.', regex=False)  # séparateur décimal
            .replace('nan', float('nan'))
            .astype(float)
        )
    return series.astype(float)


def load_base(files: list[str], sep: str = ";") -> pd.DataFrame:
    """
    Charge et concatène les CSV de base (train ou inference).
    Convertit les colonnes jour_* du format virgule française.
    """
    dfs = []
    for path in files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier source introuvable : {path}")
        df = pd.read_csv(path, sep=sep, low_memory=False)
        dfs.append(df)
        logger.info(f"  Chargé {path} → {len(df):,} lignes")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Base concaténée : {len(df):,} lignes")

    # Convertir jour_* en float (supporte jour_1..91 ET jour_DD/MM/YYYY)
    jour_present = get_jour_cols(df)
    for col in jour_present:
        df[col] = _parse_french_float(df[col])

    # Convertir colonnes financières à virgule française si présentes
    for col in ['revenu_principal', 'mensualite_immo',
                'total_mensualite_actif', 'duree_restante_ponderee']:
        if col in df.columns:
            df[col] = _parse_french_float(df[col])

    return df


def merge_common(df: pd.DataFrame, common_files: dict = None) -> pd.DataFrame:
    """
    Enrichit le DataFrame avec les fichiers communs (demographics, financials).
    Utilise un LEFT JOIN sur id_client.
    """
    if common_files is None:
        from config import COMMON_FILES
        common_files = COMMON_FILES

    for name, (path, sep) in common_files.items():
        if not os.path.exists(path):
            logger.warning(f"Fichier d'enrichissement absent : {path}")
            continue
        df_extra = pd.read_csv(path, sep=sep, low_memory=False)
        # Convertir colonnes financières si présentes
        for col in ['revenu_principal', 'mensualite_immo',
                    'total_mensualite_actif', 'duree_restante_ponderee']:
            if col in df_extra.columns:
                df_extra[col] = _parse_french_float(df_extra[col])
        # Éviter les doublons de colonnes
        cols_to_add = [c for c in df_extra.columns
                       if c == 'id_client' or c not in df.columns]
        df_extra = df_extra[cols_to_add]
        df = df.merge(df_extra, on='id_client', how='left')
        logger.info(f"  Merge {name} ({path}) → {len(df):,} lignes, {len(df.columns)} colonnes")

    return df
