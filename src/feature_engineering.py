"""
Feature engineering : 9 statistiques de solde sur 91 jours.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

JOUR_COLS = [f'jour_{i}' for i in range(1, 92)]


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule 9 features statistiques à partir des soldes jour_1..jour_91 :
    - solde_moyen           : moyenne des 91 jours
    - solde_min             : minimum
    - solde_max             : maximum
    - solde_std             : écart-type
    - solde_volatilite      : coefficient de variation (std/|mean|)
    - solde_nb_negatif      : nombre de jours avec solde < 0
    - solde_dernier_jour    : solde du jour 91
    - solde_variation_moy   : variation journalière moyenne (diff(1).mean())
    - solde_tendance        : coefficient directeur d'une régression linéaire
    """
    df = df.copy()
    jour_present = [c for c in JOUR_COLS if c in df.columns]

    if not jour_present:
        logger.warning("Aucune colonne jour_* trouvée — features solde mises à 0")
        for col in ['solde_moyen', 'solde_min', 'solde_max', 'solde_std',
                    'solde_volatilite', 'solde_nb_negatif', 'solde_dernier_jour',
                    'solde_variation_moy', 'solde_tendance']:
            df[col] = 0.0
        return df

    balances = df[jour_present].values.astype(np.float32)  # (n, 91)

    df['solde_moyen']       = balances.mean(axis=1)
    df['solde_min']         = balances.min(axis=1)
    df['solde_max']         = balances.max(axis=1)
    df['solde_std']         = balances.std(axis=1)
    df['solde_volatilite']  = np.where(
        np.abs(df['solde_moyen']) > 1e-6,
        df['solde_std'] / np.abs(df['solde_moyen']),
        0.0,
    )
    df['solde_nb_negatif']  = (balances < 0).sum(axis=1).astype(float)
    df['solde_dernier_jour'] = balances[:, -1]

    # Variation journalière moyenne
    diffs = np.diff(balances, axis=1)          # (n, 90)
    df['solde_variation_moy'] = diffs.mean(axis=1)

    # Tendance (pente de la régression linéaire sur 91 points)
    x = np.arange(len(jour_present), dtype=np.float32)
    x_centered = x - x.mean()
    x_sq_sum = (x_centered ** 2).sum()
    # pente = sum((xi - x_mean) * yi) / sum((xi - x_mean)^2)
    df['solde_tendance'] = (
        (balances * x_centered).sum(axis=1) / x_sq_sum
    )

    logger.info(f"Balance features ajoutées → {len(df):,} lignes")
    return df
