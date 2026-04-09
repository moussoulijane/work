"""
Chargement de la base Excel + cache Streamlit.
Les 3 fichiers (Excel principal + demographics + financials) sont chargés
UNE SEULE FOIS au démarrage et gardés en mémoire.
"""
import pandas as pd
import streamlit as st
from pathlib import Path

from config import COMMON_FILES


@st.cache_data(show_spinner=False)
def load_clients_database(excel_path):
    """
    Charge le fichier Excel principal.
    Cache Streamlit : chargé une seule fois par session.
    """
    if not Path(excel_path).exists():
        return None
    
    df = pd.read_excel(excel_path)
    
    # Vérifier les colonnes critiques
    required = ['id_client']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes dans l'Excel : {missing}")
        return None
    
    return df


@st.cache_data(show_spinner=False)
def load_enrichment_files():
    """Charge demographics + financials (fichiers fixes)."""
    dfs = {}
    for name, (path, sep) in COMMON_FILES.items():
        try:
            dfs[name] = pd.read_csv(path, sep=sep)
        except FileNotFoundError:
            dfs[name] = None
    return dfs


def find_client(df_main, dfs_enrichment, id_client):
    """
    Recherche un client par id et l'enrichit.
    
    Returns:
        pd.Series du client enrichi, ou None si introuvable
    """
    # Convertir l'id (l'utilisateur peut taper un str)
    try:
        id_num = int(id_client)
    except ValueError:
        return None
    
    # Recherche
    matches = df_main[df_main['id_client'] == id_num]
    if len(matches) == 0:
        return None
    
    client = matches.iloc[0].copy()
    
    # Enrichir avec demographics et financials
    for name, df_enrich in dfs_enrichment.items():
        if df_enrich is not None and 'id_client' in df_enrich.columns:
            enrich_matches = df_enrich[df_enrich['id_client'] == id_num]
            if len(enrich_matches) > 0:
                enrich_row = enrich_matches.iloc[0]
                for col in enrich_row.index:
                    if col != 'id_client' and col not in client.index:
                        client[col] = enrich_row[col]
    
    return client

