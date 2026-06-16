"""Reducao de dimensionalidade: PCA no ramo anomaly.

As funcoes mantem os nomes ``kpca`` por compatibilidade com os artefatos e
fases existentes, mas a implementacao usa PCA linear para escalar melhor em
datasets grandes como UNSW-NB15.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _prepare_kpca_input(X: np.ndarray, *, dtype: type = np.float32) -> np.ndarray:
    return np.ascontiguousarray(X, dtype=dtype)


def fit_kpca(
    X_train: np.ndarray,
    *,
    n_components: int = 10,
    kernel: str = "rbf",
    dtype: type = np.float32,
) -> PCA:
    """Ajusta PCA somente no treino.

    ``kernel`` e o nome da funcao sao preservados para nao quebrar chamadas
    existentes que ainda falam em KPCA.
    """
    del kernel
    X32 = _prepare_kpca_input(X_train, dtype=dtype)
    max_components = max(1, min(X32.shape))
    n_comp = min(int(n_components), max_components)
    solver = "randomized" if n_comp < min(X32.shape) else "auto"
    print(f"  [PCA fit treino] shape={X32.shape}, n_components={n_comp}")
    pca = PCA(n_components=n_comp, svd_solver=solver, random_state=0)
    try:
        pca.fit(X32)
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        est_gb = (X32.shape[0] * X32.shape[1] * X32.dtype.itemsize) / (1024**3)
        raise MemoryError(
            f"PCA esgotou memoria (~{est_gb:.1f} GiB de entrada para {X32.shape[0]} amostras). "
            "Opcoes: fechar outros apps, usar maquina com mais RAM, ou reduzir amostras na fase 2."
        ) from e
    return pca


def transform_kpca(
    X: np.ndarray,
    kpca: PCA,
    *,
    dtype: type = np.float32,
    split: str = "teste",
) -> np.ndarray:
    """Projeta amostras com PCA ja ajustado no treino."""
    X32 = _prepare_kpca_input(X, dtype=dtype)
    out = kpca.transform(X32)
    print(f"  [PCA transform {split}] shape={out.shape}")
    return out


def apply_kpca(
    X: np.ndarray,
    *,
    n_components: int = 10,
    kernel: str = "rbf",
    y: np.ndarray | None = None,
    dtype: type = np.float32,
) -> tuple[np.ndarray, PCA]:
    """Ajusta e transforma no mesmo conjunto (legado). ``y`` e ignorado."""
    del y
    pca = fit_kpca(X, n_components=n_components, kernel=kernel, dtype=dtype)
    return transform_kpca(X, pca, dtype=dtype, split="fit+transform"), pca
