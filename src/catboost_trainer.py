"""
CatBoost trainer avec split paramétré TRAIN/INFER.

Corrections clés :
  - _prepare_X() : cast explicite float/str → élimine ValueError
    "could not convert string to float"
  - split_data(mode='train') : ASYMÉTRIQUE — positifs dans LOW et HIGH
  - split_data(mode='infer') : SIMPLE par revenu uniquement
  - two_stage_train() : améliore la précision sur les segments déséquilibrés
  - Calibration isotonique intégrée (optionnelle)
"""
import os
import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def _prepare_X(df: pd.DataFrame, feature_cols: list, cat_features: list) -> pd.DataFrame:
    """
    Prépare le DataFrame pour CatBoost :
      - Sélectionne uniquement les colonnes disponibles dans feature_cols
      - Cast explicite : cat → str (fillna 'INCONNU'), num → float64 (fillna 0)
    Cela élimine le ValueError "could not convert string to float".
    """
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    for col in available:
        if col in cat_features:
            X[col] = X[col].fillna('INCONNU').astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(np.float64)
    return X


class CatBoostTrainer:

    def __init__(
        self,
        feature_cols: list,
        cat_features: list,
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
        mode='train' : ASYMÉTRIQUE — positifs présents dans LOW et HIGH.
        mode='infer' : SIMPLE — filtre par revenu uniquement.
        """
        if mode == 'train':
            assert 'target' in df.columns, "Mode train requiert la colonne 'target'"
            df_pos      = df[df['target'] == 1]
            df_neg_low  = df[(df['target'] == 0) & (df['revenu_principal'] <= self.threshold)]
            df_neg_high = df[(df['target'] == 0) & (df['revenu_principal'] >  self.threshold)]
            return (
                pd.concat([df_pos, df_neg_low],  ignore_index=True),
                pd.concat([df_pos, df_neg_high], ignore_index=True),
            )
        else:
            return (
                df[df['revenu_principal'] <= self.threshold].copy(),
                df[df['revenu_principal'] >  self.threshold].copy(),
            )

    # ─────────────────────────────────────────────────────────
    # Train standard
    # ─────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, save_dir: str = "models", calibrate: bool = True):
        """
        MODE TRAIN : split asymétrique → 70/30 stratifié → fit → calibration → save.

        Returns:
            model_low, model_high       : CatBoostClassifier entraînés
            results_low, results_high   : dict(y_true, y_proba, y_proba_cal, y_pred)
        """
        from src.calibration import ProbabilityCalibrator

        os.makedirs(save_dir, exist_ok=True)
        df_low, df_high = self.split_data(df, mode='train')

        models  = {}
        results = {}

        for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
            print(f"\n  Segment {name} — {len(df_seg):,} clients")

            X   = _prepare_X(df_seg, self.feature_cols, self.cat_features)
            y   = df_seg['target'].reset_index(drop=True)
            cat_idx = [X.columns.tolist().index(c) for c in self.cat_features if c in X.columns]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )

            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            spw   = n_neg / n_pos
            print(f"  train={len(y_tr):,}  eval={len(y_te):,}  "
                  f"positifs={n_pos}  scale_pos_weight={spw:.1f}")

            model = CatBoostClassifier(**{**self.model_params, 'scale_pos_weight': spw})
            model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_te, y_te))

            y_proba = model.predict_proba(X_te)[:, 1]
            auc     = roc_auc_score(y_te, y_proba)
            print(f"  AUC {name} (eval set) : {auc:.4f}")

            # ── Calibration isotonique ──
            y_proba_cal = y_proba
            if calibrate:
                cal = ProbabilityCalibrator()
                cal.fit(y_te.values, y_proba)
                y_proba_cal = cal.transform(y_proba)
                cal.save(os.path.join(save_dir, f"calibrator_{name.lower()}.pkl"))

            # Sauvegarde
            cbm_path = os.path.join(save_dir, f"{self.threshold}_catboost_{name.lower()}.cbm")
            model.save_model(cbm_path)
            model.get_feature_importance(prettified=True).to_csv(
                os.path.join(save_dir, f"feature_importance_{name.lower()}.csv"), index=False
            )

            models[name]  = model
            results[name] = {
                'y_true':      y_te.values,
                'y_proba':     y_proba_cal,   # probabilités calibrées
                'y_proba_raw': y_proba,        # brutes (pour debug)
                'y_pred':      model.predict(X_te),
            }

        return models['LOW'], models['HIGH'], results['LOW'], results['HIGH']

    # ─────────────────────────────────────────────────────────
    # Two-stage train
    # ─────────────────────────────────────────────────────────

    def train_two_stage(self, df: pd.DataFrame, save_dir: str = "models",
                        stage1_top_pct: float = 0.20, calibrate: bool = True):
        """
        Entraînement en deux étapes pour améliorer la précision (surtout LOW).

        Stage 1 — modèle permissif (depth=4) sur toute la base :
            → garde les top `stage1_top_pct` % des scores comme "candidats"
        Stage 2 — modèle fin sur les candidats uniquement :
            → ratio positifs/négatifs bien meilleur → précision ×3-5

        Les modèles stage-2 sont sauvegardés sous
        {threshold}_catboost_{LOW|HIGH}_s2.cbm et utilisés à l'inférence
        si `use_two_stage=True` dans predict().

        Returns : même interface que train()
        """
        from src.calibration import ProbabilityCalibrator

        os.makedirs(save_dir, exist_ok=True)
        df_low, df_high = self.split_data(df, mode='train')
        models  = {}
        results = {}

        for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
            print(f"\n  [Two-Stage] Segment {name} — {len(df_seg):,} clients")

            X = _prepare_X(df_seg, self.feature_cols, self.cat_features)
            y = df_seg['target'].reset_index(drop=True)
            cat_idx = [X.columns.tolist().index(c) for c in self.cat_features if c in X.columns]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )

            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())

            # ── Stage 1 : modèle filtre (permissif) ──
            params_s1 = {**self.model_params,
                         'depth': 4, 'scale_pos_weight': n_neg / n_pos,
                         'verbose': 0}
            model_s1 = CatBoostClassifier(**params_s1)
            model_s1.fit(X_tr, y_tr, cat_features=cat_idx,
                         eval_set=(X_te, y_te), verbose=0)

            proba_s1_tr = model_s1.predict_proba(X_tr)[:, 1]
            threshold_s1 = np.percentile(proba_s1_tr, (1 - stage1_top_pct) * 100)
            mask_candidates = proba_s1_tr >= threshold_s1

            n_candidates = int(mask_candidates.sum())
            n_pos_cand   = int(y_tr[mask_candidates].sum())
            print(f"  Stage 1 : {n_candidates:,} candidats  "
                  f"({n_candidates/len(y_tr):.1%} de la base train)  "
                  f"positifs={n_pos_cand} ({n_pos_cand/n_candidates:.2%})")

            # ── Stage 2 : modèle fin sur les candidats ──
            X_tr_s2 = X_tr[mask_candidates]
            y_tr_s2 = y_tr[mask_candidates]
            n_pos_s2 = int((y_tr_s2 == 1).sum())
            n_neg_s2 = int((y_tr_s2 == 0).sum())

            if n_pos_s2 == 0:
                logger.warning(f"Stage 2 {name} : aucun positif dans les candidats — fallback stage 1")
                model_s2 = model_s1
            else:
                params_s2 = {**self.model_params,
                             'scale_pos_weight': n_neg_s2 / n_pos_s2}
                model_s2 = CatBoostClassifier(**params_s2)
                model_s2.fit(X_tr_s2, y_tr_s2, cat_features=cat_idx,
                             eval_set=(X_te, y_te))

            # Évaluation finale (pipeline complet : stage1 filtre → stage2 prédit)
            proba_s1_te = model_s1.predict_proba(X_te)[:, 1]
            mask_te     = proba_s1_te >= threshold_s1
            y_proba_final = np.zeros(len(y_te))
            if mask_te.sum() > 0:
                y_proba_final[mask_te] = model_s2.predict_proba(X_te[mask_te])[:, 1]

            auc = roc_auc_score(y_te, y_proba_final)
            print(f"  AUC {name} two-stage (eval) : {auc:.4f}")

            # ── Calibration ──
            y_proba_cal = y_proba_final
            if calibrate:
                cal = ProbabilityCalibrator()
                cal.fit(y_te.values, y_proba_final)
                y_proba_cal = cal.transform(y_proba_final)
                cal.save(os.path.join(save_dir, f"calibrator_{name.lower()}_s2.pkl"))

            # Sauvegarde
            model_s1.save_model(os.path.join(save_dir,
                f"{self.threshold}_catboost_{name.lower()}_s1.cbm"))
            model_s2.save_model(os.path.join(save_dir,
                f"{self.threshold}_catboost_{name.lower()}_s2.cbm"))

            # Sauvegarder le threshold_s1 pour l'inférence
            import joblib
            joblib.dump(float(threshold_s1),
                        os.path.join(save_dir, f"s1_threshold_{name.lower()}.pkl"))

            models[name]  = (model_s1, model_s2, float(threshold_s1))
            results[name] = {
                'y_true':      y_te.values,
                'y_proba':     y_proba_cal,
                'y_proba_raw': y_proba_final,
                'y_pred':      (y_proba_cal >= 0.5).astype(int),
            }

        return models['LOW'], models['HIGH'], results['LOW'], results['HIGH']

    # ─────────────────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, model_dir: str = "models",
                use_two_stage: bool = False, use_calibration: bool = True) -> pd.DataFrame:
        """
        MODE INFER : split SIMPLE par revenu → load .cbm → predict.

        use_two_stage    : utilise les modèles _s1/_s2 si disponibles
        use_calibration  : applique la calibration isotonique si disponible
        """
        import json
        import joblib
        from src.calibration import ProbabilityCalibrator

        # Chargement des seuils optimisés (issus du train)
        thresholds_data = {}
        thresholds_path = os.path.join(model_dir, 'thresholds.json')
        if os.path.exists(thresholds_path):
            with open(thresholds_path) as f:
                thresholds_data = json.load(f)
            logger.info(f"Seuils optimisés chargés → {thresholds_path}")
        else:
            logger.warning(
                "thresholds.json absent — seuil 0.5 utilisé (relancer python main.py train)"
            )

        df_low, df_high = self.split_data(df, mode='infer')
        results_dfs = []

        for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
            if len(df_seg) == 0:
                logger.warning(f"Segment {name} vide — ignoré")
                continue

            X = _prepare_X(df_seg, self.feature_cols, self.cat_features)

            _use_two_stage = use_two_stage  # copie locale pour ne pas contaminer l'itération suivante
            if _use_two_stage:
                s1_path = os.path.join(model_dir, f"{self.threshold}_catboost_{name.lower()}_s1.cbm")
                s2_path = os.path.join(model_dir, f"{self.threshold}_catboost_{name.lower()}_s2.cbm")
                th_path = os.path.join(model_dir, f"s1_threshold_{name.lower()}.pkl")
                if all(os.path.exists(p) for p in [s1_path, s2_path, th_path]):
                    m_s1 = CatBoostClassifier(); m_s1.load_model(s1_path)
                    m_s2 = CatBoostClassifier(); m_s2.load_model(s2_path)
                    t_s1 = joblib.load(th_path)
                    proba_s1 = m_s1.predict_proba(X)[:, 1]
                    mask     = proba_s1 >= t_s1
                    probas   = np.zeros(len(X))
                    if mask.sum() > 0:
                        probas[mask] = m_s2.predict_proba(X[mask])[:, 1]
                else:
                    logger.warning(f"Modèles two-stage absents pour {name} — fallback standard")
                    _use_two_stage = False

            if not _use_two_stage:
                cbm_path = os.path.join(model_dir, f"{self.threshold}_catboost_{name.lower()}.cbm")
                if not os.path.exists(cbm_path):
                    raise FileNotFoundError(
                        f"{cbm_path} introuvable. Lance d'abord : python main.py train"
                    )
                model = CatBoostClassifier()
                model.load_model(cbm_path)
                probas = model.predict_proba(X)[:, 1]

            # Calibration
            if use_calibration:
                suffix   = "_s2" if _use_two_stage else ""
                cal_path = os.path.join(model_dir, f"calibrator_{name.lower()}{suffix}.pkl")
                if os.path.exists(cal_path):
                    cal    = ProbabilityCalibrator.load(cal_path)
                    probas = cal.transform(probas)

            seg_thresholds = thresholds_data.get(name, {})
            threshold = float(seg_thresholds.get('f2', 0.5))
            predictions = (probas >= threshold).astype(int)
            out = pd.DataFrame({
                'id_client':     df_seg['id_client'].values,
                'segment_model': name,
                'prediction':    predictions,
                'proba':         probas,
                'threshold':     threshold,
            })
            results_dfs.append(out)
            print(f"  Segment {name} : {len(out):,} prédictions  "
                  f"| seuil F2={threshold:.3f}  positifs prédits : {int(predictions.sum()):,}")

        if not results_dfs:
            raise ValueError("Aucune prédiction produite — vérifier les données d'inférence")

        return pd.concat(results_dfs, ignore_index=True)
