"""
Feature engineering :
  - 9 statistiques de solde sur 91 jours
  - 6 features avancées métier (intention, capacité, fragilité)
"""
import pandas as pd
import numpy as np
import logging
from src.jour_utils import get_jour_cols, to_float_array

logger = logging.getLogger(__name__)


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
    jour_present = get_jour_cols(df)

    if not jour_present:
        logger.warning("Aucune colonne jour_* trouvée — features solde mises à 0")
        for col in ['solde_moyen', 'solde_min', 'solde_max', 'solde_std',
                    'solde_volatilite', 'solde_nb_negatif', 'solde_dernier_jour',
                    'solde_variation_moy', 'solde_tendance']:
            df[col] = 0.0
        return df

    balances = to_float_array(df, jour_present)  # (n, N) float32, virgule FR gérée

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


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    6 features avancées pour améliorer la précision (surtout segment LOW) :

    - simul_par_kMAD       : nb_simul / (revenu/1000 + 1) — intention pondérée capacité
    - marge_mensuelle      : revenu - total_mensualites — argent disponible
    - capacite_credit_supp : max(0, revenu×0.33 - mensualites) — règle 33% banque
    - ratio_simul_recents  : simul_mois_n1 / (simul_total + 1) — activité récente
    - score_fragilite      : nb_negatif × volatilite — risque de découvert
    - solde_acceleration   : var_moy 2ème moitié - var_moy 1ère moitié des 91 jours
                             positif = accélération, négatif = décélération

    IMPORTANT : doit être appelée APRÈS add_balance_features() (utilise
    solde_nb_negatif, solde_volatilite) ET pendant que les colonnes jour_* sont
    encore présentes (pour solde_acceleration).
    """
    df = df.copy()

    # ── solde_acceleration : accélération de la tendance de solde ──
    jour_present = get_jour_cols(df)
    if jour_present and len(jour_present) > 2:
        balances = df[jour_present].values.astype(np.float32)  # (n, 91)
        diffs    = np.diff(balances, axis=1)                    # (n, 90)
        mid      = diffs.shape[1] // 2
        df['solde_acceleration'] = (
            diffs[:, mid:].mean(axis=1) - diffs[:, :mid].mean(axis=1)
        )
    else:
        df['solde_acceleration'] = 0.0

    # ── Features basées sur colonnes existantes ──
    rev  = df['revenu_principal'].clip(lower=0) if 'revenu_principal' in df.columns else 0
    mens = df['total_mensualite_conso_immo'] if 'total_mensualite_conso_immo' in df.columns else 0

    # Simulations pondérées par la capacité financière
    df['simul_par_kMAD'] = (
        df['count_simul'] / (rev / 1000 + 1)
        if 'count_simul' in df.columns else 0.0
    )

    # Marge mensuelle brute
    df['marge_mensuelle'] = rev - mens

    # Capacité de crédit supplémentaire selon la règle des 33%
    df['capacite_credit_supp'] = (rev * 0.33 - mens).clip(lower=0)

    # Proportion des simulations récentes (signal d'intention récente)
    if 'count_simul_mois_n_1' in df.columns and 'count_simul' in df.columns:
        df['ratio_simul_recents'] = (
            df['count_simul_mois_n_1'] / (df['count_simul'] + 1)
        )
    else:
        df['ratio_simul_recents'] = 0.0

    # Score de fragilité : plus un client est souvent en négatif ET volatile, plus il est fragile
    if 'solde_nb_negatif' in df.columns and 'solde_volatilite' in df.columns:
        df['score_fragilite'] = df['solde_nb_negatif'] * df['solde_volatilite']
    else:
        df['score_fragilite'] = 0.0

    logger.info(f"Advanced features ajoutées → {len(df):,} lignes")
    return df
