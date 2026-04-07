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

CAT_FEATURES = ['type_revenu', 'segment']

LSTM_CONFIG = {
    'input_size':    1,
    'hidden_size':   64,
    'num_layers':    2,
    'dropout':       0.2,
    'bidirectional': False,
    'embedding_dim': 32,
    'sequence_length': 91,
    'batch_size':    256,
    'learning_rate': 0.001,
    'epochs':        50,
    'patience':      10,
}

LSTM_EMBEDDING_COLS = [f'lstm_emb_{i}' for i in range(32)]
FEATURE_COLS = STATIC_FEATURE_COLS + BALANCE_STAT_COLS + LSTM_EMBEDDING_COLS  # 52 features

MODEL_PARAMS = {
    'iterations':           1000,
    'learning_rate':        0.05,
    'depth':                6,
    'loss_function':        'Logloss',
    'eval_metric':          'AUC',
    'random_seed':          42,
    'verbose':              100,
    'early_stopping_rounds': 50,
    'task_type':            'CPU',
    'bootstrap_type':       'Bernoulli',
    'subsample':            0.8,
}

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
    'lstm_embedding':              'Profil temporel du compte (LSTM)',
}
