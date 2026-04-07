"""
Preprocessing : nettoyage, imputation, encodage.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

FILLNA_ZERO_COLS = [
    'count_simul', 'count_simul_mois_n_1', 'age',
    'mensualite_immo', 'total_mensualite_actif', 'duree_restante_ponderee',
]

JOUR_COLS = [f'jour_{i}' for i in range(1, 92)]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de nettoyage :
    1. Fillna(0) sur colonnes numériques à NaN métier
    2. Imputation médiane sur revenu_principal si NaN
    3. Fillna(0) sur séquences jour_*
    4. Feature dérivée : total_mensualite_conso_immo + taux_endettement
    5. Valeurs aberrantes : clip revenu/age
    """
    df = df.copy()

    # 1. Fillna(0) colonnes connues
    for col in FILLNA_ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 2. Imputation revenu_principal (médiane si NaN)
    if 'revenu_principal' in df.columns:
        median_rev = df['revenu_principal'].median()
        df['revenu_principal'] = df['revenu_principal'].fillna(median_rev)

    # 3. Séquences : fillna(0) = solde nul
    jour_present = [c for c in JOUR_COLS if c in df.columns]
    if jour_present:
        df[jour_present] = df[jour_present].fillna(0)

    # 4. Features dérivées
    if 'total_mensualite_actif' in df.columns and 'mensualite_immo' in df.columns:
        df['total_mensualite_conso_immo'] = (
            df['total_mensualite_actif'] + df['mensualite_immo']
        )
    elif 'total_mensualite_actif' in df.columns:
        df['total_mensualite_conso_immo'] = df['total_mensualite_actif']
    else:
        df['total_mensualite_conso_immo'] = 0.0

    if 'revenu_principal' in df.columns:
        df['taux_endettement'] = np.where(
            df['revenu_principal'] > 0,
            df['total_mensualite_conso_immo'] / df['revenu_principal'],
            0.0,
        )

    # 5. Clip valeurs aberrantes
    if 'age' in df.columns:
        df['age'] = df['age'].clip(lower=18, upper=100)
    if 'revenu_principal' in df.columns:
        df['revenu_principal'] = df['revenu_principal'].clip(lower=0)

    # 6. Normaliser type_revenu / segment (strip + upper)
    for col in ['type_revenu', 'segment']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace({'NAN': 'INCONNU', 'NONE': 'INCONNU'})

    logger.info(f"Preprocessing terminé → {len(df):,} lignes, {len(df.columns)} colonnes")
    return df
