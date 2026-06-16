"""Pré-processamento alinhado ao notebook MTH_IDS_IoTJ.ipynb."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def zscore_array(X: np.ndarray) -> np.ndarray:
    """Z-score coluna a coluna (artigo / notebook)."""
    Xf = np.asarray(X, dtype=np.float64)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    out = (Xf - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def zscore_normalize(df: pd.DataFrame, *, label_col: str = "Label") -> pd.DataFrame:
    """
    Z-score nas features numéricas (notebook: ``dtypes != 'object'`` com Label em texto).
    Em pandas moderno, restringe a ``is_numeric_dtype`` para evitar colunas ``string``.
    """
    out = df.copy()
    feature_cols = [
        c
        for c in out.columns
        if c != label_col and pd.api.types.is_numeric_dtype(out[c])
    ]
    out[feature_cols] = out[feature_cols].apply(lambda x: (x - x.mean()) / (x.std()))
    return out.fillna(0)


def encode_labels(df: pd.DataFrame, *, label_col: str = "Label") -> tuple[pd.DataFrame, LabelEncoder]:
    out = df.copy()
    encoder = LabelEncoder()
    out[label_col] = encoder.fit_transform(out[label_col].astype(str)).astype("int64")
    return out, encoder


def _fix_hex_columns(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Converte colunas com valores hex (ex: '0x000b') para float64.
    Funciona em qualquer dataset — se não houver hex, não faz nada.
    """
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1000)
            has_hex = sample.apply(lambda x: isinstance(x, str) and str(x).startswith("0x")).any()
            if has_hex:
                df[col] = pd.to_numeric(
                    df[col].apply(
                        lambda x: int(str(x), 16) if isinstance(x, str) and str(x).startswith("0x") else x
                    ),
                    errors="coerce",
                ).fillna(0).astype("float64")
                if verbose:
                    print(f"Coluna '{col}' convertida de hex para float64")
    return df


def _cast_numeric_columns(df: pd.DataFrame, *, label_col: str = "Label", verbose: bool = True) -> pd.DataFrame:
    """Converte colunas object para numérico, preservando a coluna de rótulo."""
    df = df.copy()
    for col in df.columns:
        if col == label_col:
            continue
        if df[col].dtype == object:
            # Trata strings vazias ou espaços como NaN antes de converter
            cleaned = df[col].replace(r"^\s*$", np.nan, regex=True)
            coerced = pd.to_numeric(cleaned, errors="coerce")
            if coerced.notna().any():
                df[col] = coerced
                if verbose:
                    print(f"Coluna '{col}' convertida para numérico (coerce) e valores não numéricos definidos como NaN")
    return df


def load_and_preprocess(
    raw_csv: Path,
    *,
    label_col: str = "Label",
    verbose: bool = True,
    zscore: bool = True,
) -> pd.DataFrame:
    """Fase 1: load CSV; Z-score opcional (CAN artigo: Z-score após k-means, fase 2)."""
    warnings.filterwarnings("ignore")
    if verbose:
        print(f"Carregando dataset: {raw_csv}")
    start = time.time()
    df = pd.read_csv(raw_csv, index_col=False)
    if verbose:
        print(df[label_col].value_counts())
        print(f"Dataset carregado em {time.time() - start:.2f}s")
        print(f"Shape original: {df.shape}")

    df = _fix_hex_columns(df, verbose=verbose)
    df = _cast_numeric_columns(df, label_col=label_col, verbose=verbose)

    if zscore:
        df = zscore_normalize(df, label_col=label_col)
    else:
        df = df.fillna(0)
    if verbose:
        print(f"Pré-processamento concluído. Shape final: {df.shape}")
        if not zscore:
            print("Z-score adiado (artigo CAN: normalização após amostragem k-means).")
    return df
