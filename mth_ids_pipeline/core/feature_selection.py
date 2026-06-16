"""Seleção de atributos: Information Gain + FCBF (notebook e artigo)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    import pandas as pd


def information_gain_feature_subset(
    X_train: np.ndarray,
    feature_names: list[str],
    y_train: np.ndarray,
    *,
    cumulative: float = 0.9,
    random_state: int | None = None,
) -> list[str]:
    """Seleção IG do notebook: MI sem seed, scores arredondados, acumulado até 90%."""
    del random_state
    importances = mutual_info_classif(X_train, y_train)
    f_list = sorted(
        zip(map(lambda x: round(x, 4), importances), feature_names),
        reverse=True,
    )
    total = sum(score for score, _ in f_list) or 1.0
    f_list2 = sorted(
        zip(map(lambda x: round(x, 4), importances / total), feature_names),
        reverse=True,
    )
    acc = 0.0
    selected: list[str] = []
    for score, name in f_list2:
        acc += score
        selected.append(name)
        if acc >= cumulative:
            break
    return selected


def numeric_feature_names(df: pd.DataFrame, label_col: str = "Label") -> list[str]:
    return [
        c
        for c in df.columns
        if c != label_col and __import__("pandas").api.types.is_numeric_dtype(df[c])
    ]


def fit_fcbf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k: int = 20,
    alpha: float | None = None,
    mode: str = "k",
):
    """Ajusta FCBF no treino — modo ``k`` (FCBFK) ou ``alpha`` (FCBF com limiar th)."""
    from mth_ids_pipeline.utils.FCBF_module import FCBF, FCBFK

    if mode == "alpha":
        th = float(alpha if alpha is not None else 0.01)
        fcbf = FCBF(th=th)
    else:
        fcbf = FCBFK(k=k)
    fcbf.fit(X_train, y_train)
    return fcbf


def transform_fcbf(fcbf, X: np.ndarray) -> np.ndarray:
    """Aplica máscara FCBF ajustada no treino."""
    return fcbf.transform(X)


def apply_fcbf(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 20,
    alpha: float | None = None,
    mode: str = "k",
):
    """Ajusta e transforma no mesmo conjunto (legado; preferir fit_fcbf + transform_fcbf)."""
    fcbf = fit_fcbf(X, y, k=k, alpha=alpha, mode=mode)
    return transform_fcbf(fcbf, X), fcbf


class AnomalyFeaturePipeline:
    """
    Pré-processamento anomaly: Z-score → IG (90%) → FCBF.

    Ajuste exclusivamente no treino; transform no teste com os mesmos parâmetros.
    """

    def __init__(
        self,
        *,
        fcbf_k: int = 20,
        ig_cumulative: float = 0.9,
        random_state: int = 0,
    ) -> None:
        self.fcbf_k = fcbf_k
        self.ig_cumulative = ig_cumulative
        self.random_state = random_state
        self.scaler: StandardScaler | None = None
        self.ig_features: list[str] = []
        self.feature_names: list[str] = []
        self.fcbf = None

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
    ) -> np.ndarray:
        """Ajusta no treino e devolve matriz pós-FCBF do treino."""
        self.feature_names = list(feature_names)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        _log_dims("Z-score (fit treino)", X_scaled)

        self.ig_features = information_gain_feature_subset(
            X_scaled,
            self.feature_names,
            y_train,
            cumulative=self.ig_cumulative,
            random_state=self.random_state,
        )
        ig_idx = [self.feature_names.index(n) for n in self.ig_features]
        X_ig = X_scaled[:, ig_idx]
        _log_dims(f"IG {self.ig_cumulative:.0%} ({len(self.ig_features)} feats)", X_ig)

        self.fcbf = fit_fcbf(X_ig, y_train, k=self.fcbf_k)
        X_fcbf = transform_fcbf(self.fcbf, X_ig)
        _log_dims(f"FCBF k={self.fcbf_k}", X_fcbf)
        return X_fcbf

    def transform(self, X: np.ndarray, *, split: str = "teste") -> np.ndarray:
        if self.scaler is None or self.fcbf is None:
            raise RuntimeError("Pipeline não ajustado: chame fit() antes de transform().")
        X_scaled = self.scaler.transform(X)
        _log_dims(f"Z-score (transform {split})", X_scaled)

        ig_idx = [self.feature_names.index(n) for n in self.ig_features]
        X_ig = X_scaled[:, ig_idx]
        _log_dims(f"IG ({len(self.ig_features)} feats, {split})", X_ig)

        X_fcbf = transform_fcbf(self.fcbf, X_ig)
        _log_dims(f"FCBF ({split})", X_fcbf)
        return X_fcbf

    def fcbf_selected_indices(self) -> list[int]:
        """Índices das colunas IG selecionadas pelo FCBF."""
        if self.fcbf is None:
            return []
        return list(self.fcbf.idx_sel)


def _log_dims(step: str, X: np.ndarray) -> None:
    print(f"  [{step}] shape={X.shape}")
