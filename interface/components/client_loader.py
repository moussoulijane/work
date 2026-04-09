"""
Chargement du fichier clients (Excel ou CSV) avec cache Streamlit.
Recherche un client par id_client.
"""
import os
import pandas as pd
import streamlit as st


@st.cache_data(ttl=300, show_spinner="Chargement du fichier clients...")
def load_clients_file(path: str) -> pd.DataFrame:
    """
    Charge le fichier clients depuis Excel ou CSV.
    Mis en cache 5 minutes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier clients introuvable : {path}\n"
            "Vérifiez le chemin dans UI_CONFIG['default_excel_path']."
        )

    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    elif ext == '.csv':
        # Tente ; puis ,
        try:
            df = pd.read_csv(path, sep=';', low_memory=False)
            if df.shape[1] <= 1:
                df = pd.read_csv(path, sep=',', low_memory=False)
        except Exception:
            df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Format non supporté : {ext}. Accepté : .xlsx, .xls, .csv")

    return df


def find_client(df: pd.DataFrame, id_client) -> pd.Series | None:
    """
    Recherche un client par id_client dans le DataFrame.
    Retourne la Series correspondante ou None si introuvable.
    """
    if 'id_client' not in df.columns:
        raise KeyError("La colonne 'id_client' est absente du fichier clients.")

    # Convertir id_client dans le type de la colonne pour éviter les mismatch
    col_dtype = df['id_client'].dtype
    try:
        if pd.api.types.is_integer_dtype(col_dtype):
            id_client = int(id_client)
        else:
            id_client = str(id_client)
    except (ValueError, TypeError):
        pass

    result = df[df['id_client'] == id_client]
    if result.empty:
        return None
    return result.iloc[0]
