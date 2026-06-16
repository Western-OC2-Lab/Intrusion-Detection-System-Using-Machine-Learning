"""Validação cruzada e HPO na validação (artigo Sec. IV-F)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


WORST_HPO_SCORE = 0.0


def sanitize_hpo_score(score: Any, *, worst: float = WORST_HPO_SCORE) -> float:
    """Score finito para maximização; NaN/inf → penalização."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return float(worst)
    if not np.isfinite(value):
        return float(worst)
    return float(np.clip(value, 0.0, 1.0))


def score_to_bo_loss(score: Any, *, worst: float = WORST_HPO_SCORE) -> float:
    """Loss finita para skopt (minimizar)."""
    return 1.0 - sanitize_hpo_score(score, worst=worst)


def stratified_kfold_scores(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 10,
    random_state: int = 0,
    scoring: str = "accuracy",
) -> dict[str, Any]:
    """10-fold CV no conjunto de treino (paper: 70% split, CV no train)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score=0.0
    )
    return {
        "n_splits": n_splits,
        "scoring": scoring,
        "scores": scores.tolist(),
        "mean": sanitize_hpo_score(float(np.mean(scores))),
        "std": float(np.std(scores)),
    }


def hpo_objective_on_validation(
    build_estimator: Callable[[dict[str, Any]], Any],
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_splits: int = 10,
    random_state: int = 0,
) -> float:
    """Retorna acurácia média de CV (maximizar). Usado como loss negativa no Hyperopt."""
    clf = build_estimator(params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, error_score=0.0
    )
    return sanitize_hpo_score(float(np.mean(scores)))


def holdout_accuracy(
    build_estimator: Callable[[dict[str, Any]], Any],
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    clf = build_estimator(params)
    clf.fit(X_train, y_train)
    return float(accuracy_score(y_test, clf.predict(X_test)))
