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

    def train_kfold(self, df: pd.DataFrame, n_splits: int = 5,
                    save_dir: str = "models", calibrate: bool = True) -> tuple:
        """
        K-Fold stratifié LightGBM avec OOF predictions.
        Même logique que CatBoostTrainer.train_kfold().

        Sauvegarde : {thr}_lgbm_{seg}_fold{k}.txt + calibrator_lgbm_{seg}.pkl
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM non installé — pip install lightgbm")
        import json
        from sklearn.model_selection import StratifiedKFold
        from src.calibration import ProbabilityCalibrator

        os.makedirs(save_dir, exist_ok=True)
        df_low, df_high = self.split_data(df)

        fold_models = {}
        results     = {}

        for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
            print(f"\n  [LGBM K-Fold] Segment {name} — {len(df_seg):,} clients  ({n_splits} folds)")

            X = _prepare_X(df_seg, self.feature_cols, self.cat_features)
            y = df_seg['target'].reset_index(drop=True)
            for col in self.cat_features:
                if col in X.columns:
                    X[col] = X[col].astype('category')

            n_pos_total = int((y == 1).sum())
            n_neg_total = int((y == 0).sum())
            print(f"  total={len(y):,}  positifs={n_pos_total}  ratio={n_neg_total/n_pos_total:.0f}:1")

            skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            oof_proba = np.zeros(len(y), dtype=np.float64)
            auc_folds = []
            first_booster = None

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

                n_pos = int((y_tr == 1).sum())
                n_neg = int((y_tr == 0).sum())
                spw   = n_neg / max(n_pos, 1)

                params = {**LGBM_PARAMS, 'scale_pos_weight': spw}
                model  = lgb.LGBMClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100, verbose=False),
                        lgb.log_evaluation(period=500),
                    ],
                )

                fold_proba        = model.predict_proba(X_va)[:, 1]
                oof_proba[va_idx] = fold_proba
                fold_auc          = roc_auc_score(y_va, fold_proba)
                auc_folds.append(fold_auc)
                print(f"    Fold {fold+1}/{n_splits}  AUC={fold_auc:.4f}  "
                      f"best_iter={model.best_iteration_}")

                fold_path = os.path.join(save_dir, f"{self.threshold}_lgbm_{name.lower()}_fold{fold}.txt")
                model.booster_.save_model(fold_path)

                if first_booster is None:
                    first_booster = model.booster_
                    pd.DataFrame({
                        'feature':    model.booster_.feature_name(),
                        'importance': model.booster_.feature_importance(importance_type='gain'),
                    }).sort_values('importance', ascending=False).to_csv(
                        os.path.join(save_dir, f"feature_importance_lgbm_{name.lower()}.csv"),
                        index=False,
                    )

            oof_auc = roc_auc_score(y, oof_proba)
            print(f"  AUC OOF {name} LGBM : {oof_auc:.4f}  "
                  f"(folds: {np.mean(auc_folds):.4f} ± {np.std(auc_folds):.4f})")

            oof_proba_cal = oof_proba
            if calibrate:
                cal = ProbabilityCalibrator()
                cal.fit(y.values, oof_proba)
                oof_proba_cal = cal.transform(oof_proba)
                cal.save(os.path.join(save_dir, f"calibrator_lgbm_{name.lower()}.pkl"))

            # Copie du fold_0 comme modèle canonique
            canon_path = os.path.join(save_dir, f"{self.threshold}_lgbm_{name.lower()}.txt")
            first_booster.save_model(canon_path)

            meta = {'n_splits': n_splits, 'oof_auc': round(oof_auc, 6)}
            with open(os.path.join(save_dir, f"kfold_meta_lgbm_{name.lower()}.json"), 'w') as f:
                json.dump(meta, f, indent=2)

            fold_models[name] = first_booster
            results[name] = {
                'y_true':      y.values,
                'y_proba':     oof_proba_cal,
                'y_proba_raw': oof_proba,
            }

        return fold_models['LOW'], fold_models['HIGH'], results['LOW'], results['HIGH']

    def train(self, df: pd.DataFrame, save_dir: str = "models",
              calibrate: bool = True) -> tuple:
        """Split 70/30 unique — conservé pour compatibilité. Préférer train_kfold()."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM non installé — pip install lightgbm")
        from src.calibration import ProbabilityCalibrator

        os.makedirs(save_dir, exist_ok=True)
        df_low, df_high = self.split_data(df)

        models  = {}
        results = {}

        for name, df_seg in [("LOW", df_low), ("HIGH", df_high)]:
            print(f"\n  [LGBM] Segment {name} — {len(df_seg):,} clients")

            X = _prepare_X(df_seg, self.feature_cols, self.cat_features)
            y = df_seg['target'].reset_index(drop=True)
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
            print(f"  AUC {name} LGBM (eval) : {auc:.4f}  best_iter={model.best_iteration_}")

            y_proba_cal = y_proba
            if calibrate:
                cal = ProbabilityCalibrator()
                cal.fit(y_te.values, y_proba)
                y_proba_cal = cal.transform(y_proba)
                cal.save(os.path.join(save_dir, f"calibrator_lgbm_{name.lower()}.pkl"))

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
        Détecte automatiquement les modèles K-Fold et les moyenne.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM non installé — pip install lightgbm")
        import json
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

            # Détection K-Fold
            meta_path = os.path.join(model_dir, f"kfold_meta_lgbm_{name.lower()}.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    n_splits = json.load(f)['n_splits']
                fold_probas = []
                for k in range(n_splits):
                    fp = os.path.join(model_dir, f"{self.threshold}_lgbm_{name.lower()}_fold{k}.txt")
                    if os.path.exists(fp):
                        fold_probas.append(lgb.Booster(model_file=fp).predict(X))
                probas = np.mean(fold_probas, axis=0) if fold_probas else None
            else:
                model_path = os.path.join(model_dir, f"{self.threshold}_lgbm_{name.lower()}.txt")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"{model_path} introuvable — relancer python main.py train"
                    )
                probas = lgb.Booster(model_file=model_path).predict(X)

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
