"""
LightGBM trainer — même interface que CatBoostTrainer.train().
Utilisé en ensemble avec CatBoost pour améliorer le lift.
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.catboost_trainer import _prepare_X

logger = logging.getLogger(__name__)

LGBM_PARAMS = {
    'objective':        'binary',
    'metric':           'auc',
    'n_estimators':     2000,
    'learning_rate':    0.03,
    'num_leaves':       63,
    'max_depth':        -1,
    'min_child_samples': 30,
    'subsample':        0.8,
    'subsample_freq':   1,
    'colsample_bytree': 0.7,
    'reg_alpha':        0.1,
    'reg_lambda':       5.0,
    'random_state':     42,
    'n_jobs':           -1,
    'verbose':          -1,
}


class LGBMTrainer:

    def __init__(self, feature_cols: list, cat_features: list, revenu_threshold: int = 7000):
        self.feature_cols = feature_cols
        self.cat_features = cat_features
        self.threshold    = revenu_threshold

    def split_data(self, df: pd.DataFrame):
        """Split asymétrique identique à CatBoostTrainer."""
        df_pos      = df[df['target'] == 1]
        df_neg_low  = df[(df['target'] == 0) & (df['revenu_principal'] <= self.threshold)]
        df_neg_high = df[(df['target'] == 0) & (df['revenu_principal'] >  self.threshold)]
        return (
            pd.concat([df_pos, df_neg_low],  ignore_index=True),
            pd.concat([df_pos, df_neg_high], ignore_index=True),
        )

    def train(self, df: pd.DataFrame, save_dir: str = "models",
              calibrate: bool = True) -> tuple:
        """
        Entraîne deux modèles LightGBM (LOW / HIGH).
        Retourne la même structure que CatBoostTrainer.train().
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM non installé — pip install lightgbm"
            )
        from src.calibration import ProbabilityCalibrator

        os.makedirs(save_dir, exist_ok=True)
        df_low, df_high = self.split_data(df)

        models  = {}
        results = {}

        for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
            print(f"\n  [LGBM] Segment {name} — {len(df_seg):,} clients")

            X = _prepare_X(df_seg, self.feature_cols, self.cat_features)
            y = df_seg['target'].reset_index(drop=True)

            # LightGBM gère les catégorielles nativement via category dtype
            for col in self.cat_features:
                if col in X.columns:
                    X[col] = X[col].astype('category')

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )

            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            spw   = n_neg / n_pos
            print(f"  train={len(y_tr):,}  eval={len(y_te):,}  "
                  f"positifs={n_pos}  ratio={n_neg/n_pos:.0f}:1")

            params = {**LGBM_PARAMS, 'scale_pos_weight': spw}
            model  = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=200),
                ],
            )

            y_proba = model.predict_proba(X_te)[:, 1]
            auc     = roc_auc_score(y_te, y_proba)
            print(f"  AUC {name} LGBM (eval) : {auc:.4f}  "
                  f"(best iter={model.best_iteration_})")

            # Calibration
            y_proba_cal = y_proba
            if calibrate:
                cal = ProbabilityCalibrator()
                cal.fit(y_te.values, y_proba)
                y_proba_cal = cal.transform(y_proba)
                cal.save(os.path.join(save_dir, f"calibrator_lgbm_{name.lower()}.pkl"))

            # Sauvegarde
            model_path = os.path.join(save_dir, f"{self.threshold}_lgbm_{name.lower()}.txt")
            model.booster_.save_model(model_path)

            pd.DataFrame({
                'feature':    model.booster_.feature_name(),
                'importance': model.booster_.feature_importance(importance_type='gain'),
            }).sort_values('importance', ascending=False).to_csv(
                os.path.join(save_dir, f"feature_importance_lgbm_{name.lower()}.csv"),
                index=False,
            )

            models[name]  = model
            results[name] = {
                'y_true':      y_te.values,
                'y_proba':     y_proba_cal,
                'y_proba_raw': y_proba,
            }

        return models['LOW'], models['HIGH'], results['LOW'], results['HIGH']

    def predict(self, df: pd.DataFrame, model_dir: str = "models",
                use_calibration: bool = True) -> dict[str, np.ndarray]:
        """
        Retourne un dict {id_client → proba} par segment.
        Utilisé dans le pipeline d'ensemble.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM non installé — pip install lightgbm")
        from src.calibration import ProbabilityCalibrator

        df_low  = df[df['revenu_principal'] <= self.threshold].copy()
        df_high = df[df['revenu_principal'] >  self.threshold].copy()

        all_ids    = []
        all_probas = []

        for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
            if len(df_seg) == 0:
                continue

            X = _prepare_X(df_seg, self.feature_cols, self.cat_features)
            for col in self.cat_features:
                if col in X.columns:
                    X[col] = X[col].astype('category')

            model_path = os.path.join(model_dir, f"{self.threshold}_lgbm_{name.lower()}.txt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"{model_path} introuvable — relancer python main.py train"
                )
            booster = lgb.Booster(model_file=model_path)
            probas  = booster.predict(X)

            if use_calibration:
                cal_path = os.path.join(model_dir, f"calibrator_lgbm_{name.lower()}.pkl")
                if os.path.exists(cal_path):
                    cal    = ProbabilityCalibrator.load(cal_path)
                    probas = cal.transform(probas)

            all_ids.append(df_seg['id_client'].values)
            all_probas.append(probas)

        ids    = np.concatenate(all_ids)
        probas = np.concatenate(all_probas)
        return dict(zip(ids, probas))
