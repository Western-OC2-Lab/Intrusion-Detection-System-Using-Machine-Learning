"""
Tier 4 — biased classifiers B1 (FN) e B2 (FP) após CL-k-means (artigo Sec. IV-D2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mth_ids_pipeline.core.evaluation import binary_dr_far_f1

BiasedMode = Literal["both", "b1-only", "b2-only", "none", "auto"]


def load_best_n_clusters(report_dir: Path) -> int | None:
    """Lê best_n_clusters de phase10_anomaly_cluster_hpo.json."""
    path = report_dir / "phase10_anomaly_cluster_hpo.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    n = data.get("best_n_clusters")
    return int(n) if n is not None else None


def load_best_metric(report_dir: Path) -> str | None:
    """Lê best_metric de phase10_anomaly_cluster_hpo.json."""
    path = report_dir / "phase10_anomaly_cluster_hpo.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metric = data.get("best_metric")
    if metric is None:
        best_cfg = data.get("best_config") or {}
        metric = best_cfg.get("metric")
    return str(metric) if metric is not None else None


def pick_best_supervised_model(metrics_path: Path) -> str:
    """Escolhe o modelo com maior f1_weighted em 06_supervised_metrics.json."""
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not rows:
        return "RandomForest (HPO)"
    base_rows = [r for r in rows if "stacking" not in str(r.get("model", "")).lower()]
    pool = base_rows or rows
    best = max(pool, key=lambda r: float(r.get("f1_weighted", 0)))
    return str(best["model"])


def resolve_supervised_family_key(model_name: str) -> str:
    """Mapeia nome da fase 6 (ou stacking meta) para dt/rf/et/xgb."""
    name = model_name.lower()
    if "extratrees" in name or "extra trees" in name:
        return "et"
    if "randomforest" in name or "random forest" in name or name.startswith("rf"):
        return "rf"
    if "decisiontree" in name or "decision tree" in name or name.startswith("dt"):
        return "dt"
    if "xgboost" in name or "xgb" in name:
        return "xgb"
    return "rf"


def _notebook_default_estimator(family: str, *, random_state: int) -> Any:
    """Fallback quando ``models/supervised/*.joblib`` não existem (--no-hpo notebook)."""
    if family == "et":
        return ExtraTreesClassifier(
            n_estimators=53,
            min_samples_leaf=1,
            max_depth=31,
            min_samples_split=5,
            max_features=20,
            criterion="entropy",
            random_state=random_state,
        )
    if family == "rf":
        return RandomForestClassifier(
            n_estimators=71,
            min_samples_leaf=1,
            max_depth=46,
            min_samples_split=9,
            max_features=20,
            criterion="entropy",
            random_state=random_state,
        )
    if family == "xgb":
        return xgb.XGBClassifier(
            learning_rate=0.7340229699980686,
            n_estimators=70,
            max_depth=14,
            random_state=random_state,
        )
    if family == "dt":
        return DecisionTreeClassifier(
            min_samples_leaf=2,
            max_depth=47,
            min_samples_split=3,
            max_features=19,
            criterion="gini",
            random_state=random_state,
        )
    return RandomForestClassifier(n_estimators=100, random_state=random_state)


def estimator_factory_for_supervised(
    model_name: str,
    *,
    random_state: int = 0,
    intermediate_dir: Path | None = None,
) -> Callable[[], Any]:
    """
    Factory do melhor learner do tier 1–2 (mesma família do artigo), para treinar B1/B2
    no espaço de features do anomaly (KPCA).

    Com ``intermediate_dir``, clona hiperparâmetros dos ``.joblib`` da fase 6
    (fallback para ``pipeline_mth_ids_merged`` no LOAO fine).
    """
    family = resolve_supervised_family_key(model_name)
    template: Any | None = None
    models_root: Path | None = None

    if intermediate_dir is not None:
        try:
            from mth_ids_pipeline.io.model_io import (
                load_supervised_classifier_template,
                resolve_supervised_models_dir,
            )
        except ImportError:
            from mth_ids_pipeline.io.model_io import (
                load_supervised_classifier_template,
                resolve_supervised_models_dir,
            )
        try:
            template = load_supervised_classifier_template(intermediate_dir, family)
            models_root = resolve_supervised_models_dir(intermediate_dir)
        except FileNotFoundError:
            template = None

    if template is not None:

        def _from_joblib() -> Any:
            est = clone(template)
            params: dict[str, Any] = {"random_state": random_state}
            # Tier 4 é binário (benigno vs ataque); template da fase 6 pode ser multi-class.
            if isinstance(est, xgb.XGBClassifier):
                obj = est.get_params().get("objective", "")
                if obj and obj != "binary:logistic":
                    params["objective"] = "binary:logistic"
            if hasattr(est, "set_params"):
                est.set_params(**params)
            return est

        if models_root is not None and models_root != intermediate_dir:
            print(
                f"Biased factory: hiperparâmetros de {family}.joblib "
                f"← {models_root / 'models' / 'supervised'}"
            )
        return _from_joblib

    return lambda: _notebook_default_estimator(family, random_state=random_state)


def collect_train_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Índices de FN (ataque previsto como normal) e FP (normal previsto como ataque)."""
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    return fn_idx, fp_idx


def _balanced_sample(
    pool_idx: np.ndarray,
    n: int,
    *,
    random_state: int,
) -> np.ndarray:
    if len(pool_idx) == 0 or n <= 0:
        return np.array([], dtype=int)
    n = min(n, len(pool_idx))
    rng = np.random.default_rng(random_state)
    return rng.choice(pool_idx, size=n, replace=len(pool_idx) < n)


def train_biased_pair(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    *,
    estimator_factory: Callable[[], Any],
    random_state: int = 0,
    mode: BiasedMode = "both",
) -> tuple[Any | None, Any | None, dict[str, int]]:
    """Treina B1 (FN + benignos) e/ou B2 (FP + ataques)."""
    if mode == "none":
        return None, None, {"fn": 0, "fp": 0, "b1_train": 0, "b2_train": 0}

    fn_idx, fp_idx = collect_train_errors(y_train, y_pred_train)
    normal_idx = np.where(y_train == 0)[0]
    attack_idx = np.where(y_train == 1)[0]

    b1, b2 = None, None
    stats = {"fn": len(fn_idx), "fp": len(fp_idx), "b1_train": 0, "b2_train": 0}

    if mode in ("both", "b1-only") and len(fn_idx) > 0:
        norm_sample = _balanced_sample(normal_idx, len(fn_idx), random_state=random_state)
        idx1 = np.concatenate([fn_idx, norm_sample])
        b1 = clone(estimator_factory())
        b1.fit(X_train[idx1], y_train[idx1])
        stats["b1_train"] = len(idx1)

    if mode in ("both", "b2-only") and len(fp_idx) > 0:
        atk_sample = _balanced_sample(attack_idx, len(fp_idx), random_state=random_state + 1)
        idx2 = np.concatenate([fp_idx, atk_sample])
        b2 = clone(estimator_factory())
        b2.fit(X_train[idx2], y_train[idx2])
        stats["b2_train"] = len(idx2)

    return b1, b2, stats


def apply_biased_refinement(
    y_pred: np.ndarray,
    cluster_confidence: np.ndarray,
    X: np.ndarray,
    *,
    b1: Any | None,
    b2: Any | None,
    p_star: float = 0.933,
    mode: BiasedMode = "both",
) -> np.ndarray:
    """Instâncias com p_i < p* passam por B1 (normal) ou B2 (ataque), conforme o modo."""
    if mode == "none":
        return np.ravel(y_pred).copy()

    out = np.ravel(y_pred).copy()
    conf = np.ravel(cluster_confidence)
    uncertain = conf < p_star

    if mode in ("both", "b1-only") and b1 is not None:
        mask = uncertain & (out == 0)
        if mask.any():
            out[mask] = b1.predict(X[mask])

    if mode in ("both", "b2-only") and b2 is not None:
        mask = uncertain & (out == 1)
        if mask.any():
            out[mask] = b2.predict(X[mask])

    return out


def _eval_mode_on_val(
    X_sub: np.ndarray,
    X_val: np.ndarray,
    y_sub: np.ndarray,
    y_val: np.ndarray,
    cl_val: Any,
    *,
    n_clusters: int,
    random_state: int,
    metric: str,
    p_star: float,
    estimator_factory: Callable[[], Any],
    mode: BiasedMode,
) -> tuple[float, dict[str, Any]]:
    from mth_ids_pipeline.core.clustering import cl_kmeans_fit_predict

    if mode == "none":
        return binary_dr_far_f1(y_val, cl_val.y_pred)["f1"], {}

    cl_sub = cl_kmeans_fit_predict(
        X_sub, X_sub, y_sub, y_sub,
        n_clusters=n_clusters,
        random_state=random_state,
        metric=metric,
    )
    b1, b2, stats = train_biased_pair(
        X_sub, y_sub, cl_sub.y_pred,
        estimator_factory=estimator_factory,
        random_state=random_state,
        mode=mode,
    )
    y_ref = apply_biased_refinement(
        cl_val.y_pred,
        cl_val.cluster_confidence,
        X_val,
        b1=b1,
        b2=b2,
        p_star=p_star,
        mode=mode,
    )
    return binary_dr_far_f1(y_val, y_ref)["f1"], stats


def pick_best_biased_mode(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    metric: str,
    p_star: float,
    estimator_factory: Callable[[], Any],
    requested: BiasedMode = "auto",
    val_fraction: float = 0.2,
    force_apply: bool = False,
) -> tuple[BiasedMode, dict[str, Any]]:
    """
    Escolhe none | b1-only | b2-only | both pelo maior F1 no hold-out do treino.
    Se requested != auto, só avalia esse modo (e none para comparar).
    force_apply=True (protocolo paper): aplica requested sem gate de melhoria.
    """
    if force_apply and requested not in ("auto", "none"):
        return requested, {"forced": True, "selected_mode": requested}
    from mth_ids_pipeline.core.clustering import cl_kmeans_fit_predict

    y_train = np.ravel(y_train)
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_train,
    )
    cl_val = cl_kmeans_fit_predict(
        X_sub, X_val, y_sub, y_val,
        n_clusters=n_clusters,
        random_state=random_state,
        metric=metric,
    )
    f1_base = binary_dr_far_f1(y_val, cl_val.y_pred)["f1"]

    scores: dict[str, float] = {"none": f1_base}
    details: dict[str, Any] = {"f1_baseline": f1_base, "min_improvement": 0.01}
    min_gain = 0.01

    if requested != "auto":
        candidates: list[BiasedMode] = [requested]
    else:
        # auto: só tenta B1 (reduz FN); B2/both costumam colapsar recall no teste desta amostra
        candidates = ["b1-only"]

    best_mode: BiasedMode = "none"
    best_f1 = f1_base

    for mode in candidates:
        f1_m, stats = _eval_mode_on_val(
            X_sub, X_val, y_sub, y_val, cl_val,
            n_clusters=n_clusters,
            random_state=random_state,
            metric=metric,
            p_star=p_star,
            estimator_factory=estimator_factory,
            mode=mode,
        )
        scores[mode] = f1_m
        details[f"val_f1_{mode}"] = f1_m
        if stats:
            details[f"train_stats_{mode}"] = stats
        if f1_m > f1_base + min_gain and f1_m >= best_f1:
            best_f1 = f1_m
            best_mode = mode

    if requested == "auto":
        for probe in ("b2-only", "both"):
            f1_p, _ = _eval_mode_on_val(
                X_sub, X_val, y_sub, y_val, cl_val,
                n_clusters=n_clusters,
                random_state=random_state,
                metric=metric,
                p_star=p_star,
                estimator_factory=estimator_factory,
                mode=probe,  # type: ignore[arg-type]
            )
            scores[probe] = f1_p

    details["scores"] = scores
    details["selected_mode"] = best_mode
    details["selected_f1"] = best_f1
    return best_mode, details
