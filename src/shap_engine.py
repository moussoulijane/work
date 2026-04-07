"""
SHAP via TreeSHAP natif CatBoost + agrégation des 32 dims LSTM en 1 score.

Piège critique : get_feature_importance(type='ShapValues') retourne (n, n_features+1)
— la DERNIÈRE colonne est le biais → toujours supprimer avant usage.

3 fichiers produits :
  - shap_aggregated.csv    : importance globale par feature
  - shap_complete.parquet  : 1 ligne/client, 1 col SHAP/feature
  - shap_topk.parquet      : top K features par client (format long)
"""
import os
import numpy as np
import pandas as pd
from catboost import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class SHAPEngine:

    def __init__(
        self,
        feature_cols: list[str],
        cat_features: list[str],
        lstm_embedding_cols: list[str],
        feature_labels: dict,
        top_k: int = 5,
    ):
        self.feature_cols        = feature_cols
        self.cat_features        = cat_features
        self.lstm_embedding_cols = set(lstm_embedding_cols)
        self.feature_labels      = feature_labels
        self.top_k               = top_k

    # ─────────────────────────────────────────────────────────
    # SHAP brut
    # ─────────────────────────────────────────────────────────

    def compute_shap(self, model, X: pd.DataFrame):
        """
        Calcule les valeurs SHAP via TreeSHAP CatBoost.

        Returns:
            shap_values : np.ndarray (n, n_features)  — biais supprimé
            base_value  : float — valeur de base (biais SHAP)
            feature_names : list[str] — colonnes dans le même ordre que X
        """
        available   = [f for f in self.feature_cols if f in X.columns]
        X_clean     = X[available].copy()

        for c in self.cat_features:
            if c in X_clean.columns:
                X_clean[c] = X_clean[c].fillna('INCONNU').astype(str)

        cat_idx = [available.index(c) for c in self.cat_features if c in available]
        pool    = Pool(X_clean, cat_features=cat_idx)

        raw_shap   = model.get_feature_importance(data=pool, type='ShapValues')
        # raw_shap shape : (n, n_features + 1) — dernière col = biais
        shap_values = raw_shap[:, :-1]
        base_value  = float(raw_shap[0, -1])

        return shap_values, base_value, available

    # ─────────────────────────────────────────────────────────
    # Agrégation LSTM
    # ─────────────────────────────────────────────────────────

    def aggregate_lstm(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
    ):
        """
        Réduit les 32 dims LSTM en 1 score agrégé :
          magnitude = sum(|shap_lstm_i|)
          direction = signe(sum(shap_lstm_i))
          score     = magnitude × direction

        Returns:
            shap_reduced  : np.ndarray (n, 21)
            names_reduced : list[str]  (20 non-LSTM + 'lstm_embedding')
        """
        lstm_idx     = [i for i, n in enumerate(feature_names) if n in self.lstm_embedding_cols]
        non_lstm_idx = [i for i, n in enumerate(feature_names) if n not in self.lstm_embedding_cols]

        shap_non_lstm  = shap_values[:, non_lstm_idx]
        names_non_lstm = [feature_names[i] for i in non_lstm_idx]

        if lstm_idx:
            shap_lstm   = shap_values[:, lstm_idx]            # (n, 32)
            magnitude   = np.sum(np.abs(shap_lstm), axis=1, keepdims=True)
            signed_sum  = np.sum(shap_lstm, axis=1, keepdims=True)
            lstm_agg    = magnitude * np.sign(signed_sum)     # (n, 1)
        else:
            lstm_agg    = np.zeros((len(shap_values), 1))
            logger.warning("Aucune colonne LSTM trouvée dans les features SHAP")

        shap_reduced  = np.hstack([shap_non_lstm, lstm_agg])
        names_reduced = names_non_lstm + ['lstm_embedding']
        return shap_reduced, names_reduced

    # ─────────────────────────────────────────────────────────
    # Stats agrégées
    # ─────────────────────────────────────────────────────────

    def compute_aggregated_stats(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        segment: str = "",
    ) -> pd.DataFrame:
        """Stats globales par feature : mean_abs, mean, std, pct_pos, pct_neg..."""
        records = []
        for i, name in enumerate(feature_names):
            col = shap_values[:, i]
            records.append({
                'feature':       name,
                'feature_label': self.feature_labels.get(name, name),
                'segment':       segment,
                'mean_abs_shap': float(np.mean(np.abs(col))),
                'mean_shap':     float(np.mean(col)),
                'std_shap':      float(np.std(col)),
                'pct_positive':  float((col > 0).mean() * 100),
                'pct_negative':  float((col < 0).mean() * 100),
                'max_shap':      float(np.max(col)),
                'min_shap':      float(np.min(col)),
            })
        df = (
            pd.DataFrame(records)
            .sort_values('mean_abs_shap', ascending=False)
            .reset_index(drop=True)
        )
        df['rank'] = range(1, len(df) + 1)
        return df

    # ─────────────────────────────────────────────────────────
    # Top K par client
    # ─────────────────────────────────────────────────────────

    def build_topk(
        self,
        df_segment: pd.DataFrame,
        shap_values: np.ndarray,
        feature_names: list[str],
        segment_name: str,
        predictions: np.ndarray,
        probas: np.ndarray,
    ) -> pd.DataFrame:
        """Top K features par client — format long."""
        records = []
        ids = df_segment['id_client'].values

        for i in range(len(df_segment)):
            client_shap = shap_values[i]
            top_idx     = np.argsort(np.abs(client_shap))[::-1][:self.top_k]
            total_abs   = float(np.sum(np.abs(client_shap)))

            for rank, fi in enumerate(top_idx, 1):
                fname  = feature_names[fi]
                shap_v = float(client_shap[fi])
                fval   = (df_segment[fname].values[i]
                          if fname in df_segment.columns else None)
                records.append({
                    'id_client':         ids[i],
                    'segment_model':     segment_name,
                    'prediction':        int(predictions[i]),
                    'probability':       float(probas[i]),
                    'rank':              rank,
                    'feature':           fname,
                    'feature_label':     self.feature_labels.get(fname, fname),
                    'feature_value':     fval,
                    'shap_value':        shap_v,
                    'direction':         '+' if shap_v > 0 else '-',
                    'contribution_pct':  (abs(shap_v) / total_abs * 100
                                         if total_abs > 0 else 0.0),
                })
        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────
    # Plot summary
    # ─────────────────────────────────────────────────────────

    def _plot_summary(
        self,
        df_stats: pd.DataFrame,
        segment: str,
        output_dir: str,
        top_n: int = 15,
    ):
        df_top = df_stats.head(top_n).sort_values('mean_abs_shap')
        labels = df_top['feature_label'].tolist()
        values = df_top['mean_abs_shap'].tolist()

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
        bars = ax.barh(labels, values, color='steelblue', alpha=0.8)
        ax.set_xlabel('|SHAP| moyen')
        ax.set_title(f'Importance SHAP — Segment {segment} (top {top_n})')
        ax.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8)

        fig.tight_layout()
        path = os.path.join(output_dir, f"shap_summary_{segment.lower()}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Plot SHAP sauvegardé → {path}")

    # ─────────────────────────────────────────────────────────
    # Pipeline complet
    # ─────────────────────────────────────────────────────────

    def run(
        self,
        df_final: pd.DataFrame,
        model_low,
        model_high,
        output_dir: str = "outputs/shap",
    ):
        """
        Pipeline SHAP complet pour LOW + HIGH.
        Produit : shap_aggregated.csv, shap_complete.parquet, shap_topk.parquet
        """
        os.makedirs(output_dir, exist_ok=True)

        from config import revenu_treshold
        df_low  = df_final[df_final['revenu_principal'] <= revenu_treshold].copy()
        df_high = df_final[df_final['revenu_principal'] >  revenu_treshold].copy()

        all_stats   = []
        all_complete = []
        all_topk    = []

        for segment, df_seg, model in [
            ("LOW",  df_low,  model_low),
            ("HIGH", df_high, model_high),
        ]:
            if len(df_seg) == 0:
                logger.warning(f"Segment {segment} vide — SHAP ignoré")
                continue

            print(f"\n  SHAP segment {segment} — {len(df_seg):,} clients")

            # 1. SHAP brut
            shap_values, base_value, feat_names = self.compute_shap(model, df_seg)

            # 2. Agrégation LSTM
            shap_reduced, names_reduced = self.aggregate_lstm(shap_values, feat_names)

            # 3. Stats globales
            df_stats = self.compute_aggregated_stats(shap_reduced, names_reduced, segment)
            all_stats.append(df_stats)
            self._plot_summary(df_stats, segment, output_dir)

            # 4. Matrice complète (shap par feature, avant agrégation)
            df_complete = pd.DataFrame(
                shap_values,
                columns=[f'shap_{n}' for n in feat_names],
            )
            df_complete['id_client']    = df_seg['id_client'].values
            df_complete['segment_model'] = segment
            all_complete.append(df_complete)

            # 5. Top K
            X_clean   = df_seg[[f for f in self.feature_cols if f in df_seg.columns]].copy()
            preds     = model.predict(X_clean)
            probas    = model.predict_proba(X_clean)[:, 1]
            df_topk   = self.build_topk(
                df_seg, shap_reduced, names_reduced, segment, preds, probas
            )
            all_topk.append(df_topk)

            print(f"  Base SHAP : {base_value:.4f}")

        # ── Sauvegarde ──
        if all_stats:
            pd.concat(all_stats, ignore_index=True).to_csv(
                os.path.join(output_dir, "shap_aggregated.csv"), index=False
            )
        if all_complete:
            pd.concat(all_complete, ignore_index=True).to_parquet(
                os.path.join(output_dir, "shap_complete.parquet"), index=False
            )
        if all_topk:
            df_topk = pd.concat(all_topk, ignore_index=True)
            # feature_value est mixed (float pour numériques, str pour catégorielles)
            # → cast en str pour que pyarrow puisse sérialiser sans erreur
            df_topk['feature_value'] = df_topk['feature_value'].astype(str)
            df_topk.to_parquet(
                os.path.join(output_dir, "shap_topk.parquet"), index=False
            )

        print(f"\n  SHAP sauvegardé → {output_dir}/")
        print(f"    shap_aggregated.csv")
        print(f"    shap_complete.parquet")
        print(f"    shap_topk.parquet")
