"""
CatBoost trainer avec split paramétré TRAIN/INFER.

Points critiques :
  - split_data(mode='train') : ASYMÉTRIQUE — souscripteurs dans LOW et HIGH
  - split_data(mode='infer') : SIMPLE par revenu
  - Cat features : ['type_revenu', 'segment'] — indices numériques dans Pool
  - scale_pos_weight = n_neg / n_pos calculé automatiquement
  - Sauvegarde .cbm + feature_importance.csv par segment
"""
import os
import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class CatBoostTrainer:

    def __init__(
        self,
        feature_cols: list[str],
        cat_features: list[str],
        model_params: dict,
        revenu_threshold: int = 7000,
    ):
        self.feature_cols     = feature_cols
        self.cat_features     = cat_features
        self.model_params     = model_params
        self.threshold        = revenu_threshold

    # ─────────────────────────────────────────────────────────
    # Split
    # ─────────────────────────────────────────────────────────

    def split_data(self, df: pd.DataFrame, mode: str = 'train'):
        """
        mode='train' : split ASYMÉTRIQUE (positifs dans les deux segments).
        mode='infer' : split SIMPLE par revenu uniquement.
        """
        if mode == 'train':
            assert 'target' in df.columns, "Mode train requiert la colonne 'target'"
            df_pos      = df[df['target'] == 1]
            df_neg_low  = df[(df['target'] == 0) & (df['revenu_principal'] <= self.threshold)]
            df_neg_high = df[(df['target'] == 0) & (df['revenu_principal'] >  self.threshold)]
            df_low  = pd.concat([df_pos, df_neg_low],  ignore_index=True)
            df_high = pd.concat([df_pos, df_neg_high], ignore_index=True)
            return df_low, df_high
        else:
            df_low  = df[df['revenu_principal'] <= self.threshold].copy()
            df_high = df[df['revenu_principal'] >  self.threshold].copy()
            return df_low, df_high

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _get_available_features(self, df: pd.DataFrame) -> list[str]:
        return [f for f in self.feature_cols if f in df.columns]

    def _cat_indices(self, columns: list[str]) -> list[int]:
        return [columns.index(c) for c in self.cat_features if c in columns]

    # ─────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, save_dir: str = "models"):
        """
        MODE TRAIN : split asymétrique → 70/30 stratifié par segment → fit → save.

        Returns:
            model_low, model_high : CatBoostClassifier entraînés
            results_low, results_high : dict(y_true, y_proba, y_pred) sur eval set
        """
        os.makedirs(save_dir, exist_ok=True)
        df_low, df_high = self.split_data(df, mode='train')

        models  = {}
        results = {}

        for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
            logger.info(f"\n{'='*50}\n  Segment {name} — {len(df_seg):,} clients\n{'='*50}")
            print(f"\n  Segment {name} — {len(df_seg):,} clients")

            available = self._get_available_features(df_seg)
            cat_idx   = self._cat_indices(available)

            X = df_seg[available].copy()
            y = df_seg['target'].copy()

            # Remplacer les NaN dans les cat features par 'INCONNU'
            for c in self.cat_features:
                if c in X.columns:
                    X[c] = X[c].fillna('INCONNU').astype(str)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )

            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            spw   = n_neg / n_pos
            print(f"  train={len(y_tr):,}  eval={len(y_te):,}  "
                  f"positifs={n_pos}  scale_pos_weight={spw:.1f}")

            params = {**self.model_params, 'scale_pos_weight': spw}
            model  = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                cat_features=cat_idx,
                eval_set=(X_te, y_te),
            )

            y_proba = model.predict_proba(X_te)[:, 1]
            y_pred  = model.predict(X_te)
            auc     = roc_auc_score(y_te, y_proba)
            print(f"  AUC {name} (eval set) : {auc:.4f}")

            # Sauvegarde modèle
            cbm_path = os.path.join(save_dir, f"{self.threshold}_catboost_{name.lower()}.cbm")
            model.save_model(cbm_path)
            logger.info(f"  Modèle sauvegardé → {cbm_path}")

            # Importance features
            fi = model.get_feature_importance(prettified=True)
            fi.to_csv(
                os.path.join(save_dir, f"feature_importance_{name.lower()}.csv"),
                index=False,
            )

            models[name]  = model
            results[name] = {
                'y_true':  y_te.values,
                'y_proba': y_proba,
                'y_pred':  y_pred,
            }

        return models['LOW'], models['HIGH'], results['LOW'], results['HIGH']

    # ─────────────────────────────────────────────────────────
    # Predict (infer)
    # ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, model_dir: str = "models") -> pd.DataFrame:
        """
        MODE INFER : split SIMPLE par revenu → load .cbm → predict.

        Returns:
            DataFrame avec colonnes : id_client, segment_model, prediction, proba
        """
        df_low, df_high = self.split_data(df, mode='infer')
        results_dfs = []

        for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
            if len(df_seg) == 0:
                logger.warning(f"Segment {name} vide — ignoré")
                continue

            cbm_path = os.path.join(model_dir, f"{self.threshold}_catboost_{name.lower()}.cbm")
            if not os.path.exists(cbm_path):
                raise FileNotFoundError(
                    f"Modèle {cbm_path} introuvable. Lance d'abord : python main.py train"
                )

            model = CatBoostClassifier()
            model.load_model(cbm_path)

            available = self._get_available_features(df_seg)
            X = df_seg[available].copy()
            for c in self.cat_features:
                if c in X.columns:
                    X[c] = X[c].fillna('INCONNU').astype(str)

            predictions = model.predict(X)
            probas      = model.predict_proba(X)[:, 1]

            out = pd.DataFrame({
                'id_client':     df_seg['id_client'].values,
                'segment_model': name,
                'prediction':    predictions,
                'proba':         probas,
            })
            results_dfs.append(out)
            print(f"  Segment {name} : {len(out):,} prédictions, "
                  f"{int(predictions.sum())} positifs prédits")

        if not results_dfs:
            raise ValueError("Aucune prédiction produite — vérifier les données d'inférence")

        return pd.concat(results_dfs, ignore_index=True)
