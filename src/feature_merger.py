"""
Fusionne les embeddings LSTM (32 dims) avec les features statiques (11 + 9 stats).
LEFT JOIN sur id_client — 52 features au total.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureMerger:

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim  = embedding_dim
        self.embedding_cols = [f'lstm_emb_{i}' for i in range(embedding_dim)]

    def merge(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_ids: np.ndarray,
    ) -> pd.DataFrame:
        """
        Fusionne df (statiques + stats) avec les embeddings LSTM.

        Args:
            df            : DataFrame source avec id_client
            embeddings    : np.ndarray (n, embedding_dim)
            embedding_ids : np.ndarray (n,) — id_client dans le même ordre

        Returns:
            df_final avec 52 features (11 statiques + 9 stats + 32 LSTM)
        """
        assert embeddings.shape[1] == self.embedding_dim, (
            f"Embeddings shape {embeddings.shape} ≠ embedding_dim {self.embedding_dim}"
        )

        df_emb = pd.DataFrame(embeddings, columns=self.embedding_cols)
        df_emb['id_client'] = embedding_ids

        df_final = df.merge(df_emb, on='id_client', how='left')

        n_missing = df_final[self.embedding_cols[0]].isna().sum()
        if n_missing > 0:
            logger.warning(f"{n_missing} clients sans embedding LSTM → rempli par 0")
            df_final[self.embedding_cols] = df_final[self.embedding_cols].fillna(0.0)

        logger.info(
            f"Feature merger : {len(df_final):,} clients, "
            f"{len(self.embedding_cols)} emb + features statiques"
        )
        return df_final
