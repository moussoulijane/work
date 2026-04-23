"""
Exécute le pipeline ML sur UN SEUL client.
Réutilise les fonctions du dossier src/ (pas de duplication).
"""
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from config import (
    CAT_FEATURES, FEATURE_COLS, FEATURE_LABELS, revenu_treshold,
)
from src.calibration import ProbabilityCalibrator
from src.feature_engineering import (
    add_advanced_features, add_appetite_signals,
    add_balance_features, add_credit_context_features, add_temporal_features,
)
from src.preprocessing import preprocess


class ScoringRunner:
    """
    Enchaîne le pipeline ML pour un seul client.
    Charge les modèles une seule fois (cache).
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._check_artifacts()
        self.model_low  = self._load_catboost('low')
        self.model_high = self._load_catboost('high')
        self.thresholds = self._load_thresholds()
        self.cal_low    = self._load_calibrator('low')
        self.cal_high   = self._load_calibrator('high')

    def _check_artifacts(self):
        required = [
            os.path.join(self.model_dir, f"{revenu_treshold}_catboost_low.cbm"),
            os.path.join(self.model_dir, f"{revenu_treshold}_catboost_high.cbm"),
        ]
        for f in required:
            if not os.path.exists(f):
                raise FileNotFoundError(
                    f"❌ {f} manquant. Lance d'abord : python main.py train"
                )

    def _load_catboost(self, segment: str) -> CatBoostClassifier:
        model = CatBoostClassifier()
        model.load_model(
            os.path.join(self.model_dir, f"{revenu_treshold}_catboost_{segment}.cbm")
        )
        return model

    def _load_thresholds(self) -> dict:
        path = os.path.join(self.model_dir, "thresholds.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _load_calibrator(self, segment: str):
        path = os.path.join(self.model_dir, f"calibrator_{segment}.pkl")
        if os.path.exists(path):
            return ProbabilityCalibrator.load(path)
        return None

    def score_client(self, client_row: pd.Series) -> dict:
        """
        Score un client unique.

        Args:
            client_row: pd.Series avec toutes les features brutes

        Returns:
            dict avec proba, prediction, segment, top_5_shap
        """
        df = pd.DataFrame([client_row])

        # Pipeline feature engineering (identique à build_base dans main.py)
        df = preprocess(df)
        df = add_balance_features(df)
        df = add_advanced_features(df)
        df = add_temporal_features(df)
        df = add_appetite_signals(df)
        df = add_credit_context_features(df)

        # Déterminer le segment et choisir le modèle
        revenu = float(df['revenu_principal'].iloc[0])
        if revenu <= revenu_treshold:
            model        = self.model_low
            segment_name = 'LOW'
            calibrator   = self.cal_low
        else:
            model        = self.model_high
            segment_name = 'HIGH'
            calibrator   = self.cal_high

        # Préparer X (uniquement les features disponibles dans FEATURE_COLS)
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].copy()
        for col in available:
            if col in CAT_FEATURES:
                X[col] = X[col].fillna('INCONNU').astype(str)
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(float)

        # Prédire
        cat_idx = [available.index(c) for c in CAT_FEATURES if c in available]
        proba_raw = float(model.predict_proba(X)[0, 1])

        # Calibration
        proba = calibrator.transform(np.array([proba_raw]))[0] if calibrator else proba_raw

        # Seuil F2 optimisé (ou 0.5 par défaut)
        threshold = float(self.thresholds.get(segment_name, {}).get('f2', 0.5))
        prediction = int(proba >= threshold)

        # SHAP
        pool = Pool(X, cat_features=cat_idx)
        raw_shap   = model.get_feature_importance(data=pool, type='ShapValues')
        shap_vals  = raw_shap[0, :-1]   # (n_features,)
        base_value = float(raw_shap[0, -1])

        # Top 5 features par |SHAP|
        shap_pairs = list(zip(available, shap_vals.tolist()))
        shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_5 = shap_pairs[:5]

        total_abs = sum(abs(s) for _, s in shap_pairs)
        top_5_formatted = []
        for rank, (feat, shap_val) in enumerate(top_5, 1):
            feat_value = df[feat].iloc[0] if feat in df.columns else None
            top_5_formatted.append({
                'rank':             rank,
                'feature':          feat,
                'feature_label':    FEATURE_LABELS.get(feat, feat),
                'feature_value':    float(feat_value) if feat_value is not None else None,
                'shap_value':       float(shap_val),
                'direction':        '+' if shap_val > 0 else '-',
                'contribution_pct': abs(shap_val) / total_abs * 100 if total_abs > 0 else 0,
            })

        return {
            'client_row':    df.iloc[0].to_dict(),
            'proba':         proba,
            'proba_raw':     proba_raw,
            'prediction':    prediction,
            'threshold':     threshold,
            'segment_model': segment_name,
            'top_5_shap':    top_5_formatted,
            'base_value':    base_value,
        }
