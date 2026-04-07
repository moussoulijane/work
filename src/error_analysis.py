"""
Analyse des erreurs du modèle :
  - Profils moyens par quadrant (TP / FP / FN / TN)
  - Top 10 prédictions surprenantes (FP confiants, FN confiants)
  - AUC par sous-groupe : type_revenu, segment, tranche_age
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)


class ErrorAnalyzer:

    def analyze(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        output_dir: str = "outputs/metrics",
    ) -> dict:
        """
        Args:
            df        : DataFrame avec features (doit contenir id_client)
            y_true    : array 0/1
            y_proba   : array de probas [0, 1]
            threshold : seuil de décision

        Returns:
            dict(profiles, fp_confident, fn_confident, subgroup_aucs)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        y_true  = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        y_pred  = (y_proba >= threshold).astype(int)

        df = df.copy().reset_index(drop=True)
        df['_pred']     = y_pred
        df['_true']     = y_true
        df['_proba']    = y_proba
        df['_quadrant'] = np.where(
            (df['_pred'] == 1) & (df['_true'] == 1), 'TP',
            np.where(
                (df['_pred'] == 1) & (df['_true'] == 0), 'FP',
                np.where(
                    (df['_pred'] == 0) & (df['_true'] == 1), 'FN', 'TN'
                )
            )
        )

        counts = df['_quadrant'].value_counts()
        print(f"\n  Quadrants : {counts.to_dict()}")

        # ── Profils moyens par quadrant ──
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if not c.startswith('_') and not c.startswith('lstm_emb_')
        ]
        profiles = df.groupby('_quadrant')[numeric_cols].mean().T
        print(f"\n  Profils moyens par quadrant (features clés) :")
        key_cols = [c for c in [
            'age', 'revenu_principal', 'taux_endettement', 'count_simul',
            'solde_moyen', 'solde_nb_negatif', 'solde_tendance',
        ] if c in profiles.index]
        if key_cols:
            print(profiles.loc[key_cols].to_string())

        profiles.to_csv(f"{output_dir}/error_profiles.csv")

        # ── Top 10 surprenants ──
        fp_confident = (
            df[df['_quadrant'] == 'FP']
            .nlargest(10, '_proba')[['id_client', '_proba'] + key_cols]
            if 'id_client' in df.columns
            else df[df['_quadrant'] == 'FP'].nlargest(10, '_proba')
        )
        fn_confident = (
            df[df['_quadrant'] == 'FN']
            .nsmallest(10, '_proba')[['id_client', '_proba'] + key_cols]
            if 'id_client' in df.columns
            else df[df['_quadrant'] == 'FN'].nsmallest(10, '_proba')
        )

        print(f"\n  Top 10 FP confiants (score élevé mais négatif réel) :")
        print(fp_confident.to_string(index=False))
        print(f"\n  Top 10 FN confiants (score bas mais positif réel) :")
        print(fn_confident.to_string(index=False))

        fp_confident.to_csv(f"{output_dir}/error_fp_confident.csv", index=False)
        fn_confident.to_csv(f"{output_dir}/error_fn_confident.csv", index=False)

        # ── AUC par sous-groupe ──
        subgroup_aucs = {}

        for col in ['type_revenu', 'segment']:
            if col not in df.columns:
                continue
            print(f"\n  AUC par {col} :")
            col_aucs = {}
            for val in sorted(df[col].dropna().unique()):
                sub = df[df[col] == val]
                if len(sub) < 20 or sub['_true'].nunique() < 2:
                    continue
                auc = roc_auc_score(sub['_true'], sub['_proba'])
                col_aucs[str(val)] = round(auc, 4)
                print(f"    {str(val):30s} : {auc:.4f}  (n={len(sub):,})")
            subgroup_aucs[col] = col_aucs

        # Tranches d'âge
        if 'age' in df.columns:
            df['_age_bin'] = pd.cut(
                df['age'], bins=[0, 30, 45, 60, 100],
                labels=['18-30', '30-45', '45-60', '60+']
            )
            print(f"\n  AUC par tranche d'âge :")
            age_aucs = {}
            for val in ['18-30', '30-45', '45-60', '60+']:
                sub = df[df['_age_bin'] == val]
                if len(sub) < 20 or sub['_true'].nunique() < 2:
                    continue
                auc = roc_auc_score(sub['_true'], sub['_proba'])
                age_aucs[str(val)] = round(auc, 4)
                print(f"    {str(val):10s} : {auc:.4f}  (n={len(sub):,})")
            subgroup_aucs['age_bin'] = age_aucs

        # Sauvegarder les AUC sous-groupes
        rows = []
        for group, vals in subgroup_aucs.items():
            for val, auc in vals.items():
                rows.append({'group': group, 'value': val, 'auc': auc})
        if rows:
            pd.DataFrame(rows).to_csv(
                f"{output_dir}/error_subgroup_aucs.csv", index=False
            )

        # Identifier les sous-groupes faibles (AUC < 0.65)
        weak_groups = [
            f"{g}={v} (AUC={a})"
            for g, vals in subgroup_aucs.items()
            for v, a in vals.items()
            if a < 0.65
        ]
        if weak_groups:
            print(f"\n  ⚠️  Sous-groupes faibles (AUC < 0.65) : {', '.join(weak_groups)}")
        else:
            print(f"\n  ✅ Pas de sous-groupe faible détecté (AUC < 0.65)")

        return {
            'profiles':      profiles,
            'fp_confident':  fp_confident,
            'fn_confident':  fn_confident,
            'subgroup_aucs': subgroup_aucs,
        }
