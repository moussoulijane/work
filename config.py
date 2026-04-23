TRAIN_FILES = [
    "data/raw/train_base_part1.csv",
    "data/raw/train_base_part2.csv",
    "data/raw/train_base_part3.csv",
]
INFER_FILES = ["data/raw/inference_data.csv"]
COMMON_FILES = {
    'demographics': ("data/raw/client_demographics.csv", ";"),
    'financials':   ("data/raw/client_financials.csv",   ";"),
}

revenu_treshold = 7000

# ── Segmentation 3 niveaux (amélioration précision LOW) ──
REVENUE_THRESHOLDS = [4000, 7000]   # VERYLOW ≤4000 / MID 4000-7000 / HIGH >7000
SEGMENT_NAMES_3   = ['VERYLOW', 'MID', 'HIGH']

STATIC_FEATURE_COLS = [
    'count_simul', 'count_simul_mois_n_1', 'age',
    'mensualite_immo', 'total_mensualite_actif', 'duree_restante_ponderee',
    'revenu_principal', 'type_revenu', 'segment',
    'total_mensualite_conso_immo', 'taux_endettement',
]

BALANCE_STAT_COLS = [
    'solde_moyen', 'solde_min', 'solde_max', 'solde_std',
    'solde_volatilite', 'solde_nb_negatif', 'solde_dernier_jour',
    'solde_variation_moy', 'solde_tendance',
]

# ── 6 nouvelles features avancées ──
ADVANCED_FEATURE_COLS = [
    'simul_par_kMAD',       # intention × capacité financière
    'marge_mensuelle',      # revenu - mensualités (MAD)
    'capacite_credit_supp', # marge théorique règle 33%
    'ratio_simul_recents',  # proportion simulations récentes
    'score_fragilite',      # nb_négatif × volatilité
    'solde_acceleration',   # accélération du solde (2ème dérivée)
]

CAT_FEATURES = ['type_revenu', 'segment']

TEMPORAL_FEATURE_COLS = [
    'solde_moy_m1', 'solde_moy_m2', 'solde_moy_m3', 'ratio_solde_recent',
    'solde_p10', 'solde_p25', 'solde_p75', 'solde_p90',
    'max_consecutif_negatif', 'nb_zero_crossings', 'pct_jours_positifs',
    'solde_debut_periode', 'solde_fin_periode', 'ratio_fin_vs_debut',
]

APPETITE_SIGNAL_COLS = [
    'simul_recent_flag',        # a simulé le mois dernier (0/1)
    'simul_croissant',          # simulation_n1 / simulation_total
    'simul_x_capacite',         # count_simul × capacite_credit_supp (veut ET peut)
    'solde_bon_et_simule',      # pct_jours_positifs × count_simul
    'amplitude_solde',          # solde_max - solde_min
    'solde_min_m1',             # creux mois 1
    'solde_min_m2',             # creux mois 2
    'solde_min_m3',             # creux mois 3
    'is_stable_last_month',     # aucun jour négatif au 3ème mois
    'ratio_endettement_vs_cap', # taux_endettement / capacite_résiduelle
]

CREDIT_CONTEXT_COLS = [
    'credit_presque_fini',   # durée restante < 12 mois
    'credit_en_cours',       # a une mensualité active
    'marge_33pct_seuil',     # marge normalisée règle 33%
    'age_prime_credit',      # 28-55 ans (peak crédit)
    'score_appetence',       # score synthétique pondéré
]

INTERACTION_FEATURE_COLS = [
    'ratio_epargne_revenu',      # solde_moyen / revenu (liquidité normalisée)
    'marge_relative',            # marge_mensuelle / revenu (% revenu disponible)
    'solde_vs_mensualite',       # solde_moyen / mensualités (buffer de liquidité)
    'intensite_simul_ponderee',  # (simul + 3×simul_n1) / (revenu/kMAD + 1)
    'fragilite_relative',        # score_fragilite / (revenu/kMAD + 1)
    'stabilite_x_capacite',      # pct_jours_positifs × (1 - taux_endettement)
]

# 11 statiques + 9 stats + 6 avancées + 14 temporelles + 10 signaux + 5 contexte + 6 interactions = 61 features
FEATURE_COLS = (STATIC_FEATURE_COLS + BALANCE_STAT_COLS + ADVANCED_FEATURE_COLS
                + TEMPORAL_FEATURE_COLS + APPETITE_SIGNAL_COLS + CREDIT_CONTEXT_COLS
                + INTERACTION_FEATURE_COLS)

# Conservé pour compatibilité avec les anciens artefacts LSTM
LSTM_EMBEDDING_COLS = []
LSTM_CONFIG = {}

MODEL_PARAMS = {
    'iterations':            3000,
    'learning_rate':         0.02,   # LR réduit → convergence plus fine, meilleure généralisation
    'grow_policy':           'Lossguide',  # leaf-wise (comme LGBM) : meilleur AUC sur données déséquilibrées
    'max_leaves':            63,           # remplace depth avec Lossguide
    'l2_leaf_reg':           3,
    'min_data_in_leaf':      20,
    'loss_function':         'Logloss',
    'eval_metric':           'AUC',
    'random_seed':           42,
    'verbose':               500,
    'early_stopping_rounds': 150,
    'task_type':             'CPU',
    'bootstrap_type':        'Bernoulli',
    'subsample':             0.8,
    'colsample_bylevel':     0.7,
    'random_strength':       1.5,    # exploration accrue → réduit l'overfitting
    'border_count':          254,    # bins fins pour features numériques
    'one_hot_max_size':      5,      # one-hot pour catégorielles à peu de modalités
}

N_FOLDS = 5   # K-Fold stratifié — entraîne N modèles par segment, moyenne à l'inférence

SHAP_CONFIG = {
    'top_k':               5,
    'aggregate_lstm_shap': True,
}

FEATURE_LABELS = {
    'count_simul':                 'Nombre de simulations crédit',
    'count_simul_mois_n_1':        'Simulations crédit mois dernier',
    'age':                         'Âge du client',
    'mensualite_immo':             'Mensualité crédit immobilier',
    'total_mensualite_actif':      'Total mensualités actives',
    'duree_restante_ponderee':     'Durée restante pondérée (mois)',
    'revenu_principal':            'Revenu principal',
    'type_revenu':                 'Type de revenu',
    'segment':                     'Segment client',
    'total_mensualite_conso_immo': 'Total mensualités conso + immo',
    'taux_endettement':            "Taux d'endettement",
    'solde_moyen':                 'Solde moyen 3 mois',
    'solde_min':                   'Solde minimum',
    'solde_max':                   'Solde maximum',
    'solde_std':                   'Écart-type du solde',
    'solde_volatilite':            'Volatilité du solde',
    'solde_nb_negatif':            'Jours en découvert',
    'solde_dernier_jour':          'Solde dernier jour',
    'solde_variation_moy':         'Variation journalière moyenne',
    'solde_tendance':              'Tendance du solde',
    # Features avancées
    'simul_par_kMAD':              'Simulations / k MAD revenu',
    'marge_mensuelle':             'Marge mensuelle (revenu - mensualités)',
    'capacite_credit_supp':        'Capacité crédit supplémentaire (règle 33%)',
    'ratio_simul_recents':         'Ratio simulations récentes',
    'score_fragilite':             'Score de fragilité financière',
    'solde_acceleration':          'Accélération du solde',
    # Features temporelles
    'solde_moy_m1':               'Solde moyen mois 1',
    'solde_moy_m2':               'Solde moyen mois 2',
    'solde_moy_m3':               'Solde moyen mois 3',
    'ratio_solde_recent':         'Ratio solde récent / historique',
    'solde_p10':                  'Solde percentile 10',
    'solde_p25':                  'Solde percentile 25',
    'solde_p75':                  'Solde percentile 75',
    'solde_p90':                  'Solde percentile 90',
    'max_consecutif_negatif':     'Max jours consécutifs en découvert',
    'nb_zero_crossings':          'Passages par zéro du solde',
    'pct_jours_positifs':         '% jours avec solde positif',
    'solde_debut_periode':        'Solde moyen début de période',
    'solde_fin_periode':          'Solde moyen fin de période',
    'ratio_fin_vs_debut':         'Ratio fin / début de période',
    # Signaux d'appétence
    'simul_recent_flag':          'A simulé le mois dernier',
    'simul_croissant':            'Accélération des simulations',
    'simul_x_capacite':           'Simulations × capacité crédit',
    'solde_bon_et_simule':        'Solde sain ET simulation',
    'amplitude_solde':            'Amplitude du solde (max - min)',
    'solde_min_m1':               'Creux de solde mois 1',
    'solde_min_m2':               'Creux de solde mois 2',
    'solde_min_m3':               'Creux de solde mois 3',
    'is_stable_last_month':       'Aucun découvert dernier mois',
    'ratio_endettement_vs_cap':   'Ratio endettement / capacité résiduelle',
    # Contexte crédit
    'credit_presque_fini':        'Crédit presque remboursé (< 12 mois)',
    'credit_en_cours':            'A un crédit actif',
    'marge_33pct_seuil':          'Marge règle des 33%',
    'age_prime_credit':           'Âge prime crédit (28-55 ans)',
    'score_appetence':            'Score synthétique d\'appétence',
    # Interactions
    'ratio_epargne_revenu':       'Ratio épargne / revenu',
    'marge_relative':             'Marge mensuelle (% du revenu)',
    'solde_vs_mensualite':        'Buffer de liquidité (solde / mensualités)',
    'intensite_simul_ponderee':   'Intensité simulation pondérée revenu',
    'fragilite_relative':         'Fragilité relative au revenu',
    'stabilite_x_capacite':       'Stabilité × capacité d\'endettement résiduelle',
}
