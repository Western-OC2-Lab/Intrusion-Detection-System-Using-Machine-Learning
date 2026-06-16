"""Otimização de hiperparâmetros: BO-TPE (Hyperopt) e BO-GP (skopt)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score

from mth_ids_pipeline.io.reproducibility import numpy_random_state


@dataclass
class CLKmeansHpoResult:
    best_n_clusters: int
    best_metric: str
    best_accuracy: float
    trials: list[dict[str, Any]]
    objective_metric: str = "accuracy"


@dataclass
class PStarHpoResult:
    best_p_star: float
    best_score: float
    trials: list[dict[str, Any]]
    objective_metric: str = "f1"


@dataclass
class IgAlphaHpoResult:
    best_alpha: float
    best_score: float
    trials: list[dict[str, Any]]
    objective_metric: str = "cv_accuracy"


@dataclass
class FcbfAlphaHpoResult:
    best_alpha: float
    best_score: float
    trials: list[dict[str, Any]]
    objective_metric: str = "cv_accuracy"


@dataclass
class KpcaHpoResult:
    best_n_components: int
    best_kernel: str
    best_score: float
    trials: list[dict[str, Any]]
    objective_metric: str = "cv_accuracy"


def _criterion_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ["gini", "entropy"][int(value)]


def optimize_xgb_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_evals: int = 20,
    use_validation_cv: bool = False,
    cv_folds: int = 10,
    random_state: int = 0,
) -> dict[str, Any]:
    """BO-TPE para XGBoost — test set (notebook) ou CV no treino (artigo)."""
    from hyperopt import STATUS_OK, fmin, hp, tpe
    import xgboost as xgb

    from mth_ids_pipeline.core.validation import hpo_objective_on_validation

    def objective(params):
        p = {
            "n_estimators": int(params["n_estimators"]),
            "max_depth": int(params["max_depth"]),
            "learning_rate": abs(float(params["learning_rate"])),
        }
        if use_validation_cv:

            def build_estimator(pp: dict[str, Any]):
                return xgb.XGBClassifier(**pp, random_state=random_state)

            score = hpo_objective_on_validation(
                build_estimator,
                p,
                X_train,
                y_train,
                n_splits=cv_folds,
                random_state=random_state,
            )
        else:
            clf = xgb.XGBClassifier(**p, random_state=random_state)
            clf.fit(X_train, y_train)
            score = accuracy_score(y_test, clf.predict(X_test))
        return {"loss": -score, "status": STATUS_OK}

    space = {
        "n_estimators": hp.quniform("n_estimators", 10, 100, 5),
        "max_depth": hp.quniform("max_depth", 4, 100, 1),
        "learning_rate": hp.normal("learning_rate", 0.01, 0.9),
    }
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        rstate=numpy_random_state(random_state),
    )
    return {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "learning_rate": abs(float(best["learning_rate"])),
    }


def optimize_sklearn_tree_hyperparams(
    model_factory: Callable[..., Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_evals: int = 20,
    include_n_estimators: bool = True,
    use_validation_cv: bool = False,
    cv_folds: int = 10,
    random_state: int = 0,
) -> dict[str, Any]:
    """BO-TPE para RF/ET/DT — test set (notebook) ou CV no treino (artigo)."""
    from hyperopt import STATUS_OK, fmin, hp, tpe

    from mth_ids_pipeline.core.validation import hpo_objective_on_validation

    def objective(params):
        p2: dict[str, Any] = {
            "max_depth": int(params["max_depth"]),
            "max_features": int(params["max_features"]),
            "min_samples_split": int(params["min_samples_split"]),
            "min_samples_leaf": int(params["min_samples_leaf"]),
            "criterion": _criterion_value(params["criterion"]),
        }
        if include_n_estimators:
            p2["n_estimators"] = int(params["n_estimators"])
        if use_validation_cv:

            def build_estimator(pp: dict[str, Any]):
                kw = dict(pp)
                if include_n_estimators:
                    return model_factory(**kw, random_state=random_state)
                return model_factory(**kw, random_state=random_state)

            score = hpo_objective_on_validation(
                build_estimator,
                p2,
                X_train,
                y_train,
                n_splits=cv_folds,
                random_state=random_state,
            )
            return {"loss": -score, "status": STATUS_OK}
        clf = model_factory(**p2, random_state=random_state)
        clf.fit(X_train, y_train)
        return {"loss": -clf.score(X_test, y_test), "status": STATUS_OK}

    space: dict = {
        "max_depth": hp.quniform("max_depth", 5, 50, 1),
        "max_features": hp.quniform("max_features", 1, 20, 1),
        "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
    }
    if include_n_estimators:
        space["n_estimators"] = hp.quniform("n_estimators", 10, 200, 1)

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        rstate=numpy_random_state(random_state),
    )
    out: dict[str, Any] = {
        "max_depth": int(best["max_depth"]),
        "max_features": int(best["max_features"]),
        "min_samples_split": int(best["min_samples_split"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "criterion": _criterion_value(best["criterion"]),
    }
    if include_n_estimators:
        out["n_estimators"] = int(best["n_estimators"])
    return out


_CL_KMEANS_METRIC_TIEBREAK = {"mahalanobis": 3, "euclidean": 2, "manhattan": 1, "cosine": 0}


def _pick_best_cl_kmeans_trial(trials: list[dict[str, Any]], *, k_anchor: int = 16) -> dict[str, Any]:
    """Em empate de acurácia, prefere euclidiano e k próximo ao notebook (default 16)."""
    return max(
        trials,
        key=lambda t: (
            float(t.get("score", t["accuracy"])),
            _CL_KMEANS_METRIC_TIEBREAK.get(str(t["metric"]), -1),
            -abs(int(t["n_clusters"]) - k_anchor),
        ),
    )


def optimize_cl_kmeans_clusters(
    objective_fn: Callable[[int, str], float],
    *,
    n_calls: int = 20,
    low: int = 2,
    high: int = 50,
    random_state: int = 0,
    metrics: tuple[str, ...] = ("euclidean", "manhattan", "cosine"),
    objective_metric: str = "accuracy",
) -> CLKmeansHpoResult:
    """BO-GP (skopt) para n_clusters e métrica de distância (artigo MTH-IDS)."""
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer

    space = [
        Integer(low, high, name="n_clusters"),
        Categorical(list(metrics), name="metric"),
    ]

    def wrapped(params: list[Any]) -> float:
        n, metric = int(params[0]), str(params[1])
        return 1.0 - objective_fn(n, metric)

    result = gp_minimize(wrapped, space, n_calls=n_calls, random_state=random_state)

    trials: list[dict[str, Any]] = []
    for i, (params, loss) in enumerate(zip(result.x_iters, result.func_vals)):
        n_clusters = int(params[0])
        metric = str(params[1])
        accuracy = 1.0 - float(loss)
        trials.append(
            {
                "trial": i,
                "n_clusters": n_clusters,
                "metric": metric,
                "score": accuracy,
                "accuracy": accuracy,
                "loss": float(loss),
            }
        )

    best_trial = _pick_best_cl_kmeans_trial(trials)
    best_n = int(best_trial["n_clusters"])
    best_metric = str(best_trial["metric"])
    best_acc = float(best_trial["score"])
    return CLKmeansHpoResult(
        best_n_clusters=best_n,
        best_metric=best_metric,
        best_accuracy=best_acc,
        trials=trials,
        objective_metric=objective_metric,
    )


def optimize_p_star_threshold(
    objective_fn: Callable[[float], float],
    *,
    n_calls: int = 20,
    low: float = 0.5,
    high: float = 0.99,
    random_state: int = 0,
    objective_metric: str = "f1",
) -> PStarHpoResult:
    """BO-GP para limiar p* do tier 4 (artigo Sec. IV-D2)."""
    from skopt import gp_minimize
    from skopt.space import Real

    space = [Real(low, high, name="p_star")]

    def wrapped(params: list[Any]) -> float:
        p = float(params[0])
        return 1.0 - objective_fn(p)

    result = gp_minimize(wrapped, space, n_calls=n_calls, random_state=random_state)
    trials: list[dict[str, Any]] = []
    for i, (params, loss) in enumerate(zip(result.x_iters, result.func_vals)):
        p_star = float(params[0])
        score = 1.0 - float(loss)
        trials.append({"trial": i, "p_star": p_star, "score": score, "loss": float(loss)})

    best_p = float(result.x[0])
    best_score = 1.0 - float(result.fun)
    return PStarHpoResult(
        best_p_star=best_p,
        best_score=best_score,
        trials=trials,
        objective_metric=objective_metric,
    )


def _ig_fcbf_cv_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    cumulative: float,
    fcbf_k: int = 20,
    fcbf_alpha: float | None = None,
    fcbf_mode: str = "k",
    cv_folds: int,
    random_state: int,
) -> float:
    """Proxy rápido: RF + CV no treino após IG+FCBF."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from mth_ids_pipeline.core.feature_selection import fit_fcbf, information_gain_feature_subset, transform_fcbf

    feats = information_gain_feature_subset(
        X_train, feature_names, y_train, cumulative=cumulative
    )
    if not feats:
        return 0.0
    idx = [feature_names.index(n) for n in feats]
    X_ig = X_train[:, idx]
    try:
        fcbf = fit_fcbf(
            X_ig,
            y_train,
            k=fcbf_k,
            alpha=fcbf_alpha,
            mode=fcbf_mode,
        )
        X_fcbf = transform_fcbf(fcbf, X_ig)
    except Exception:
        return 0.0
    if X_fcbf.shape[1] == 0:
        return 0.0
    clf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        clf, X_fcbf, y_train, cv=cv, scoring="accuracy", n_jobs=-1, error_score=0.0
    )
    from mth_ids_pipeline.core.validation import sanitize_hpo_score

    return sanitize_hpo_score(float(np.mean(scores)))


def optimize_ig_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    fcbf_k: int = 20,
    n_calls: int = 15,
    cv_folds: int = 10,
    random_state: int = 0,
    low: float = 0.7,
    high: float = 0.99,
) -> IgAlphaHpoResult:
    """BO-GP (skopt) para α acumulado do IG (artigo)."""
    from skopt import gp_minimize
    from skopt.space import Real

    from mth_ids_pipeline.core.validation import sanitize_hpo_score, score_to_bo_loss

    trials: list[dict[str, Any]] = []

    def objective(params: list[Any]) -> float:
        alpha = float(params[0])
        score = _ig_fcbf_cv_score(
            X_train,
            y_train,
            feature_names,
            cumulative=alpha,
            fcbf_k=fcbf_k,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        score = sanitize_hpo_score(score)
        loss = score_to_bo_loss(score)
        trials.append({"alpha": alpha, "score": score, "loss": loss})
        return loss

    result = gp_minimize(
        objective,
        [Real(low, high, name="alpha")],
        n_calls=n_calls,
        random_state=random_state,
    )
    best_alpha = float(result.x[0])
    best_score = sanitize_hpo_score(1.0 - float(result.fun))
    return IgAlphaHpoResult(best_alpha=best_alpha, best_score=best_score, trials=trials)


def optimize_fcbf_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    ig_cumulative: float = 0.9,
    n_calls: int = 15,
    cv_folds: int = 10,
    random_state: int = 0,
    low: float = 0.001,
    high: float = 0.5,
) -> FcbfAlphaHpoResult:
    """BO-GP (skopt) para limiar α do FCBF (classe FCBF, th=α)."""
    from skopt import gp_minimize
    from skopt.space import Real

    from mth_ids_pipeline.core.validation import sanitize_hpo_score, score_to_bo_loss

    trials: list[dict[str, Any]] = []

    def objective(params: list[Any]) -> float:
        alpha = float(params[0])
        score = _ig_fcbf_cv_score(
            X_train,
            y_train,
            feature_names,
            cumulative=ig_cumulative,
            fcbf_alpha=alpha,
            fcbf_mode="alpha",
            cv_folds=cv_folds,
            random_state=random_state,
        )
        score = sanitize_hpo_score(score)
        loss = score_to_bo_loss(score)
        trials.append({"alpha": alpha, "score": score, "loss": loss})
        return loss

    result = gp_minimize(
        objective,
        [Real(low, high, name="fcbf_alpha")],
        n_calls=n_calls,
        random_state=random_state,
    )
    best_alpha = float(result.x[0])
    best_score = sanitize_hpo_score(1.0 - float(result.fun))
    return FcbfAlphaHpoResult(best_alpha=best_alpha, best_score=best_score, trials=trials)


def optimize_kpca_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_calls: int = 15,
    cv_folds: int = 5,
    random_state: int = 0,
    n_low: int = 5,
    n_high: int = 20,
    kernels: tuple[str, ...] = ("rbf", "poly", "sigmoid"),
) -> KpcaHpoResult:
    """BO-GP para n_components e kernel do KernelPCA (artigo)."""
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer
    from sklearn.decomposition import KernelPCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    trials: list[dict[str, Any]] = []
    space = [Integer(n_low, n_high, name="n_components"), Categorical(list(kernels), name="kernel")]

    def objective(params: list[Any]) -> float:
        n_comp = int(params[0])
        kernel = str(params[1])
        n_comp = min(n_comp, X_train.shape[0] - 1, X_train.shape[1])
        if n_comp < 1:
            trials.append({"n_components": n_comp, "kernel": kernel, "score": 0.0})
            return 1.0
        try:
            kpca = KernelPCA(n_components=n_comp, kernel=kernel, copy_X=False)
            X_k = kpca.fit_transform(np.ascontiguousarray(X_train, dtype=np.float32))
        except Exception:
            trials.append({"n_components": n_comp, "kernel": kernel, "score": 0.0})
            return 1.0
        clf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            clf, X_k, y_train, cv=cv, scoring="accuracy", n_jobs=-1, error_score=0.0
        )
        from mth_ids_pipeline.core.validation import sanitize_hpo_score, score_to_bo_loss

        score = sanitize_hpo_score(float(np.mean(scores)))
        trials.append({"n_components": n_comp, "kernel": kernel, "score": score})
        return score_to_bo_loss(score)

    result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state)
    best_n = int(result.x[0])
    best_kernel = str(result.x[1])
    from mth_ids_pipeline.core.validation import sanitize_hpo_score

    best_score = sanitize_hpo_score(1.0 - float(result.fun))
    return KpcaHpoResult(
        best_n_components=best_n,
        best_kernel=best_kernel,
        best_score=best_score,
        trials=trials,
    )


def optimize_cl_kmeans_tpe(
    objective_fn: Callable[[int], float],
    *,
    max_evals: int = 20,
    low: int = 2,
    high: int = 50,
    random_state: int = 0,
) -> tuple[int, float]:
    """BO-TPE alternativo para n_clusters (notebook também testa)."""
    from hyperopt import STATUS_OK, fmin, hp, tpe

    best_acc = -1.0
    best_n = low

    def objective(params):
        nonlocal best_acc, best_n
        n = int(params["n_clusters"])
        acc = objective_fn(n)
        if acc > best_acc:
            best_acc = acc
            best_n = n
        return {"loss": -acc, "status": STATUS_OK}

    space = {"n_clusters": hp.quniform("n_clusters", low, high, 1)}
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        rstate=numpy_random_state(random_state),
    )
    return best_n, best_acc
