"""Clustering: amostragem k-means (fase 2) e CL-k-means (anomaly)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min

CL_KMEANS_METRICS = ("euclidean", "manhattan", "cosine", "mahalanobis")


def _sorted_labels_for_encoding(labels) -> list:
    label_values = list(labels)
    try:
        return sorted(label_values, key=lambda x: int(str(x)))
    except ValueError:
        return sorted(label_values, key=lambda x: str(x))


def _encode_labels_for_sampling(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    out = df.copy()
    original_labels = _sorted_labels_for_encoding(out[label_col].unique())
    label_map = {str(orig): i for i, orig in enumerate(original_labels)}
    out[label_col] = out[label_col].astype(str).map(label_map).astype("int64")
    return out


@dataclass
class CLKmeansModel:
    """Modelo leve para variantes de distância sem suporte nativo no MiniBatchKMeans."""

    cluster_centers_: np.ndarray
    metric: str


@dataclass
class CLKmeansResult:
    y_pred: np.ndarray
    accuracy: float
    cluster_confidence: np.ndarray
    benign_clusters: set[int]
    model: Any


def sample_kmeans(
    df: pd.DataFrame,
    *,
    n_clusters: int = 1000,
    random_state: int = 0,
    frac: float = 0.008,
    label_col: str = "Label",
    minority_labels: tuple[int, ...] = (),
) -> pd.DataFrame:
    # Numeric labels keep numeric order; text labels use alphabetical order.
    df = _encode_labels_for_sampling(df, label_col)

    df_minor = df[df[label_col].isin(minority_labels)]
    df_major = df.drop(df_minor.index)

    X = df_major.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    df_major = df_major.copy()
    df_major["klabel"] = kmeans.labels_

    def typical_sampling(group: pd.DataFrame) -> pd.DataFrame:
        return group.sample(frac=frac, random_state=random_state)

    result = df_major.groupby("klabel", group_keys=False).apply(typical_sampling)
    result = result.drop(columns=["klabel"], errors="ignore")
    return pd.concat([result, df_minor], ignore_index=True)


def sample_kmeans_staged(
    df: pd.DataFrame,
    stages: tuple[tuple[tuple[int, ...], float], ...],
    *,
    n_clusters: int = 1000,
    random_state: int = 0,
    label_col: str = "Label",
) -> pd.DataFrame:
    """Aplica amostragem k-means em grupos de labels, um estagio por vez."""
    if not stages:
        return df.copy()

    out = _encode_labels_for_sampling(df, label_col)

    for target_labels, frac in stages:
        if not target_labels:
            raise ValueError("sampling-stage precisa conter ao menos um label")
        if not 0 < frac <= 1:
            raise ValueError(f"frac do sampling-stage deve estar em (0, 1], recebido: {frac}")

        target_mask = out[label_col].isin(target_labels)
        df_target = out[target_mask].copy()
        df_rest = out[~target_mask].copy()
        if df_target.empty:
            continue

        X = df_target.drop(columns=[label_col]).to_numpy(dtype=np.float32)
        stage_clusters = min(n_clusters, len(df_target))
        kmeans = MiniBatchKMeans(n_clusters=stage_clusters, random_state=random_state)
        kmeans.fit(X)
        df_target["klabel"] = kmeans.labels_

        sampled_target = df_target.groupby("klabel", group_keys=False).apply(
            lambda group: group.sample(frac=frac, random_state=random_state)
        )
        sampled_target = sampled_target.drop(columns=["klabel"], errors="ignore")
        out = pd.concat([sampled_target, df_rest], ignore_index=True)

    return out

def _validate_metric(metric: str) -> str:
    m = str(metric).lower()
    if m not in CL_KMEANS_METRICS:
        raise ValueError(f"metric deve ser um de {CL_KMEANS_METRICS}, recebido: {metric!r}")
    return m


def _regularized_covariance(X: np.ndarray, *, shrink: float = 1e-3) -> np.ndarray:
    Xf = np.asarray(X, dtype=np.float64)
    cov = np.cov(Xf, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = np.atleast_2d(cov)
    cov += shrink * np.eye(cov.shape[0])
    return cov


def _mahalanobis_distances(X: np.ndarray, centers: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    """Matriz (n_samples, n_centers) de distâncias de Mahalanobis."""
    Xf = np.asarray(X, dtype=np.float64)
    centers_f = np.asarray(centers, dtype=np.float64)
    diff = Xf[:, np.newaxis, :] - centers_f[np.newaxis, :, :]
    # (x-c)^T Σ^{-1} (x-c) para cada par
    left = np.einsum("nck,kl->ncl", diff, cov_inv)
    d2 = np.einsum("ncl,ncl->nc", left, diff)
    return np.sqrt(np.maximum(d2, 0.0))


def _assign_clusters_mahalanobis(X: np.ndarray, centers: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    dists = _mahalanobis_distances(X, centers, cov_inv)
    return np.argmin(dists, axis=1)


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    Xf = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(Xf, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return Xf / norms


def _assign_clusters(X: np.ndarray, centers: np.ndarray, metric: str) -> np.ndarray:
    labels, _ = pairwise_distances_argmin_min(X, centers, metric=metric)
    return labels


def _minibatch_kmeans_manhattan(
    X: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int = 100,
    random_state: int = 0,
    max_iter: int = 100,
    max_no_improvement: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """MiniBatch k-means com distância de Manhattan (L1)."""
    rng = np.random.RandomState(random_state)
    Xf = np.asarray(X, dtype=np.float64)
    n_samples = Xf.shape[0]
    batch_size = min(batch_size, n_samples)

    bootstrap = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init=1,
    )
    bootstrap.fit(Xf)
    centers = bootstrap.cluster_centers_.copy()
    center_counts = np.zeros(n_clusters, dtype=np.float64)

    prev_inertia = np.inf
    n_iter_no_improve = 0
    for _ in range(max_iter):
        batch_idx = rng.randint(0, n_samples, size=batch_size)
        X_batch = Xf[batch_idx]
        labels_batch = _assign_clusters(X_batch, centers, "manhattan")

        for c in range(n_clusters):
            mask = labels_batch == c
            if not np.any(mask):
                continue
            center_counts[c] += float(mask.sum())
            eta = 1.0 / center_counts[c]
            batch_center = np.median(X_batch[mask], axis=0)
            centers[c] = (1.0 - eta) * centers[c] + eta * batch_center

        all_labels = _assign_clusters(Xf, centers, "manhattan")
        inertia = float(
            np.sum(
                np.abs(Xf - centers[all_labels]).sum(axis=1),
            )
        )
        if inertia >= prev_inertia - 1e-4:
            n_iter_no_improve += 1
            if n_iter_no_improve >= max_no_improvement:
                break
        else:
            n_iter_no_improve = 0
        prev_inertia = inertia

    train_labels = _assign_clusters(Xf, centers, "manhattan")
    return train_labels, centers


def _fit_predict_clusters(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int = 100,
    random_state: int | None = 0,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, Any]:
    metric = _validate_metric(metric)
    kw: dict[str, Any] = {"n_clusters": n_clusters, "batch_size": batch_size}
    if random_state is not None:
        kw["random_state"] = random_state

    if metric == "euclidean":
        km = MiniBatchKMeans(**kw)
        train_labels = km.fit_predict(X_train)
        test_labels = km.predict(X_test)
        return train_labels, test_labels, km

    if metric == "cosine":
        X_train_n = _l2_normalize_rows(X_train)
        X_test_n = _l2_normalize_rows(X_test)
        km = MiniBatchKMeans(**kw)
        train_labels = km.fit_predict(X_train_n)
        test_labels = km.predict(X_test_n)
        return train_labels, test_labels, km

    if metric == "manhattan":
        train_labels, centers = _minibatch_kmeans_manhattan(
            X_train,
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state or 0,
        )
        X_test_f = np.asarray(X_test, dtype=np.float64)
        test_labels = _assign_clusters(X_test_f, centers, "manhattan")
        return train_labels, test_labels, CLKmeansModel(cluster_centers_=centers, metric=metric)

    # mahalanobis: inicialização euclidiana + refinamento com Σ regularizada
    X_train_f = np.asarray(X_train, dtype=np.float64)
    cov = _regularized_covariance(X_train_f)
    cov_inv = np.linalg.inv(cov)
    bootstrap = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state or 0,
        n_init=1,
    )
    bootstrap.fit(X_train_f)
    centers = bootstrap.cluster_centers_.copy()
    center_counts = np.zeros(n_clusters, dtype=np.float64)
    rng = np.random.RandomState(random_state or 0)
    n_samples = X_train_f.shape[0]
    batch_size = min(batch_size, n_samples)
    for _ in range(100):
        batch_idx = rng.randint(0, n_samples, size=batch_size)
        X_batch = X_train_f[batch_idx]
        labels_batch = _assign_clusters_mahalanobis(X_batch, centers, cov_inv)
        for c in range(n_clusters):
            mask = labels_batch == c
            if not np.any(mask):
                continue
            center_counts[c] += float(mask.sum())
            eta = 1.0 / center_counts[c]
            batch_center = np.mean(X_batch[mask], axis=0)
            centers[c] = (1.0 - eta) * centers[c] + eta * batch_center

    train_labels = _assign_clusters_mahalanobis(X_train_f, centers, cov_inv)
    X_test_f = np.asarray(X_test, dtype=np.float64)
    test_labels = _assign_clusters_mahalanobis(X_test_f, centers, cov_inv)
    model = CLKmeansModel(cluster_centers_=centers, metric="mahalanobis")
    setattr(model, "cov_inv_", cov_inv)
    return train_labels, test_labels, model


def cl_kmeans_fit_predict(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int = 100,
    random_state: int | None = 0,
    metric: str = "euclidean",
) -> CLKmeansResult:
    """
    CL-k-means com probabilidade de cluster p_i (fração da classe majoritária no cluster).

    metric: 'euclidean', 'manhattan', 'cosine' ou 'mahalanobis'.
    """
    train_labels, test_labels, model = _fit_predict_clusters(
        X_train,
        X_test,
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        metric=metric,
    )

    attack_counts = np.zeros(n_clusters, dtype=np.float64)
    benign_counts = np.zeros(n_clusters, dtype=np.float64)
    for i, cluster in enumerate(train_labels):
        if y_train[i] == 1:
            attack_counts[cluster] += 1
        else:
            benign_counts[cluster] += 1

    benign_clusters: set[int] = set()
    majority_ratio = np.zeros(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        total = attack_counts[c] + benign_counts[c]
        if total <= 0:
            majority_ratio[c] = 0.0
            benign_clusters.add(c)
            continue
        if attack_counts[c] <= benign_counts[c]:
            benign_clusters.add(c)
            majority_ratio[c] = benign_counts[c] / total
        else:
            majority_ratio[c] = attack_counts[c] / total

    mapped = np.zeros(len(y_test), dtype=np.int64)
    conf = np.zeros(len(y_test), dtype=np.float64)
    for i in range(len(y_test)):
        c = int(test_labels[i])
        mapped[i] = 0 if c in benign_clusters else 1
        conf[i] = majority_ratio[c]

    acc = float(metrics.accuracy_score(y_test, mapped))
    return CLKmeansResult(
        y_pred=mapped,
        accuracy=acc,
        cluster_confidence=conf,
        benign_clusters=benign_clusters,
        model=model,
    )


def cl_kmeans(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int = 100,
    random_state: int | None = 0,
    metric: str = "euclidean",
) -> tuple[np.ndarray, float]:
    """API legada: retorna (predições, acurácia)."""
    res = cl_kmeans_fit_predict(
        X_train,
        X_test,
        y_train,
        y_test,
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        metric=metric,
    )
    return res.y_pred, res.accuracy


@dataclass
class CLKmeansInferenceState:
    """Estado serializável do CL-k-means para inferência (fase 13)."""

    model: Any
    benign_clusters: frozenset[int]
    majority_ratio: np.ndarray
    n_clusters: int
    metric: str
    batch_size: int = 100
    random_state: int | None = 0


def _predict_cluster_labels(X: np.ndarray, model: Any, metric: str) -> np.ndarray:
    """Atribui rótulos de cluster sem re-treinar."""
    metric = _validate_metric(metric)
    if metric == "euclidean":
        return np.asarray(model.predict(X), dtype=np.int64)
    if metric == "cosine":
        X_n = _l2_normalize_rows(X)
        return np.asarray(model.predict(X_n), dtype=np.int64)
    if metric == "manhattan":
        centers = np.asarray(model.cluster_centers_, dtype=np.float64)
        return _assign_clusters(np.asarray(X, dtype=np.float64), centers, "manhattan")
    cov_inv = getattr(model, "cov_inv_", None)
    if cov_inv is None:
        raise ValueError("Modelo mahalanobis sem cov_inv_")
    centers = np.asarray(model.cluster_centers_, dtype=np.float64)
    return _assign_clusters_mahalanobis(np.asarray(X, dtype=np.float64), centers, cov_inv)


def _cluster_label_mapping(
    train_labels: np.ndarray,
    y_train: np.ndarray,
    n_clusters: int,
) -> tuple[frozenset[int], np.ndarray]:
    attack_counts = np.zeros(n_clusters, dtype=np.float64)
    benign_counts = np.zeros(n_clusters, dtype=np.float64)
    for i, cluster in enumerate(train_labels):
        if y_train[i] == 1:
            attack_counts[cluster] += 1
        else:
            benign_counts[cluster] += 1

    benign_clusters: set[int] = set()
    majority_ratio = np.zeros(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        total = attack_counts[c] + benign_counts[c]
        if total <= 0:
            majority_ratio[c] = 0.0
            benign_clusters.add(c)
            continue
        if attack_counts[c] <= benign_counts[c]:
            benign_clusters.add(c)
            majority_ratio[c] = benign_counts[c] / total
        else:
            majority_ratio[c] = attack_counts[c] / total
    return frozenset(benign_clusters), majority_ratio


def cl_kmeans_build_inference_state(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int = 100,
    random_state: int | None = 0,
    metric: str = "euclidean",
) -> CLKmeansInferenceState:
    """Ajusta CL-k-means no treino e devolve estado para inferência."""
    probe = X_train[:1]
    train_labels, _, model = _fit_predict_clusters(
        X_train,
        probe,
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        metric=metric,
    )
    benign_clusters, majority_ratio = _cluster_label_mapping(train_labels, y_train, n_clusters)
    return CLKmeansInferenceState(
        model=model,
        benign_clusters=benign_clusters,
        majority_ratio=majority_ratio,
        n_clusters=n_clusters,
        metric=metric,
        batch_size=batch_size,
        random_state=random_state,
    )


def cl_kmeans_predict_inference(
    state: CLKmeansInferenceState,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prediz rótulo binário e confiança p_i a partir de estado salvo."""
    test_labels = _predict_cluster_labels(X, state.model, state.metric)
    mapped = np.zeros(len(X), dtype=np.int64)
    conf = np.zeros(len(X), dtype=np.float64)
    for i in range(len(X)):
        c = int(test_labels[i])
        mapped[i] = 0 if c in state.benign_clusters else 1
        conf[i] = state.majority_ratio[c]
    return mapped, conf
