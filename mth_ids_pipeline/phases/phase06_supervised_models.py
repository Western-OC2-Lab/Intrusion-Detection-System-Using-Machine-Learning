"""
Fase 6: modelos supervisionados (XGBoost, RF, DT, ET), HPO opcional (hyperopt),
stacking com meta XGBoost.

Entradas: Parquet da fase 5.

Padrão: protocolo do notebook (SMOTE {2,4}→1000, HPO no hold-out, stacking meta XGBoost).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.tree import DecisionTreeClassifier

try:
    from mth_ids_pipeline.cli import init_paths, phase_parser
    from mth_ids_pipeline.config import (
        DEFAULT_CV_FOLDS,
        DEFAULT_HPO_ON_VALIDATION,
        DEFAULT_META_LEARNER,
        P05_TEST,
        P05_TRAIN_SMOTE,
    )
except ImportError:
    from mth_ids_pipeline.cli import init_paths, phase_parser
    from mth_ids_pipeline.config import (
        DEFAULT_CV_FOLDS,
        DEFAULT_HPO_ON_VALIDATION,
        DEFAULT_META_LEARNER,
        P05_TEST,
        P05_TRAIN_SMOTE,
    )

try:
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.io.reporting import write_report


def _evaluate(name: str, clf, X_test, y_test, *, binary: bool = False) -> dict:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    print(f"\n=== {name} ===\nAccuracy: {acc}\nPrecision: {p}\nRecall: {r}\nF1: {f}")
    print(classification_report(y_test, y_pred))
    row = {"model": name, "accuracy": acc, "precision": float(p), "recall": float(r), "f1_weighted": float(f)}
    if binary or len(np.unique(y_test)) <= 2:
        try:
            from mth_ids_pipeline.core.evaluation import binary_dr_far_f1
        except ImportError:
            from mth_ids_pipeline.core.evaluation import binary_dr_far_f1
        extra = binary_dr_far_f1(y_test, y_pred)
        row.update(
            {
                "detection_rate": extra["detection_rate"],
                "false_alarm_rate": extra["false_alarm_rate"],
                "f1_binary": extra["f1"],
            }
        )
    return row


def _criterion_value(value) -> str:
    if isinstance(value, str):
        return value
    return ["gini", "entropy"][int(value)]


def _resolve_cv_folds(cv_folds: int, hpo_on_validation: bool) -> int:
    if hpo_on_validation and cv_folds <= 0:
        return 10
    return max(0, cv_folds)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _phase_banner(step: int, total: int, title: str) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"[fase 6 — {step}/{total}] {title}", flush=True)
    print(f"{'=' * 60}", flush=True)


def _hyperopt_objective(
    build_estimator: Callable[[dict[str, Any]], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hpo_on_validation: bool,
    cv_folds: int,
    random_state: int = 0,
):
    from hyperopt import STATUS_OK

    try:
        from mth_ids_pipeline.core.validation import holdout_accuracy, hpo_objective_on_validation
    except ImportError:
        from mth_ids_pipeline.core.validation import holdout_accuracy, hpo_objective_on_validation

    def objective(params: dict[str, Any]) -> dict[str, Any]:
        if hpo_on_validation:
            score = hpo_objective_on_validation(
                build_estimator,
                params,
                X_train,
                y_train,
                n_splits=cv_folds,
                random_state=random_state,
            )
        else:
            score = holdout_accuracy(
                build_estimator, params, X_train, y_train, X_test, y_test
            )
        return {"loss": -score, "status": STATUS_OK}

    return objective


def _fmin_best(
    objective,
    space: dict,
    *,
    max_evals: int,
    label: str,
    random_state: int = 0,
) -> dict:
    from hyperopt import Trials, fmin, tpe

    try:
        from mth_ids_pipeline.io.reproducibility import numpy_random_state
    except ImportError:
        from mth_ids_pipeline.io.reproducibility import numpy_random_state

    trials = Trials()
    hpo_start = time.time()
    best_acc = [-float("inf")]

    def _objective_with_progress(params: dict[str, Any]) -> dict[str, Any]:
        trial_idx = len(trials.trials) + 1
        trial_start = time.time()
        result = objective(params)
        acc = float(-result["loss"])
        if acc > best_acc[0]:
            best_acc[0] = acc
        elapsed_total = time.time() - hpo_start
        print(
            f"  [{label}] trial {trial_idx}/{max_evals} "
            f"acc={acc:.4f} (melhor={best_acc[0]:.4f}) "
            f"+{_format_duration(time.time() - trial_start)} "
            f"| total {_format_duration(elapsed_total)}",
            flush=True,
        )
        return result

    print(f"  [{label}] HPO iniciando ({max_evals} trials)...", flush=True)
    best = fmin(
        fn=_objective_with_progress,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=False,
        rstate=numpy_random_state(random_state),
    )
    print(
        f"  [{label}] HPO concluído em {_format_duration(time.time() - hpo_start)} — best: {best}",
        flush=True,
    )
    return best


def _run_cv_reports(
    models: list[tuple[str, Any, np.ndarray, np.ndarray]],
    *,
    n_splits: int,
    random_state: int = 0,
) -> dict[str, dict]:
    try:
        from mth_ids_pipeline.core.validation import stratified_kfold_scores
    except ImportError:
        from mth_ids_pipeline.core.validation import stratified_kfold_scores

    reports: dict[str, dict] = {}
    for name, estimator, X, y in models:
        rep = stratified_kfold_scores(
            estimator, X, y, n_splits=n_splits, random_state=random_state
        )
        reports[name] = rep
        print(f"\n{n_splits}-fold CV ({name}): mean={rep['mean']:.4f} ± {rep['std']:.4f}")
    return reports


def _pick_best_base_name(metrics: list[dict], base_names: set[str]) -> str:
    candidates = [m for m in metrics if m["model"] in base_names]
    if not candidates:
        raise ValueError("Nenhum modelo base encontrado para stacking meta-learner")
    best = max(candidates, key=lambda r: float(r.get("f1_weighted", 0)))
    return str(best["model"])


def main() -> None:
    from mth_ids_pipeline.io.subprocess_env import suppress_worker_warnings

    suppress_worker_warnings()
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 6 — treino supervisionado + stacking")
    parser.add_argument("--no-hpo", action="store_true", help="Hiperparâmetros fixos (sem BO-TPE)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--binary", action="store_true", help="BENIGN vs ataque (Tabela VII)")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, help="Opcional: relatório k-fold CV no treino")
    parser.add_argument(
        "--hpo-on-validation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HPO_ON_VALIDATION,
        help="Notebook: HPO no hold-out (padrão). Use --hpo-on-validation para CV no treino (artigo)",
    )
    parser.add_argument("--meta-learner", choices=("best-base", "xgb"), default=DEFAULT_META_LEARNER)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    paths = init_paths(args)
    output_dir = paths.intermediate
    rs = int(args.random_state)
    tr = pd.read_parquet(output_dir / P05_TRAIN_SMOTE)
    te = pd.read_parquet(output_dir / P05_TEST)
    label_col = "Label"
    X_train = tr.drop(columns=[label_col]).values
    y_train = tr[label_col].values
    X_test = te.drop(columns=[label_col]).values
    y_test = te[label_col].values

    if args.binary:
        y_train = (y_train > 0).astype(np.int64)
        y_test = (y_test > 0).astype(np.int64)

    cv_folds = _resolve_cv_folds(args.cv_folds, args.hpo_on_validation)
    n_features = int(X_train.shape[1])
    max_features_hpo = min(20, n_features)
    print(
        f"Fase 6 — treino: {X_train.shape[0]:,} × {n_features} features | "
        f"teste: {X_test.shape[0]:,} | max_features HPO ≤ {max_features_hpo}",
        flush=True,
    )
    if args.hpo_on_validation:
        print(f"HPO objetivo: acurácia média em {cv_folds}-fold CV (treino)", flush=True)
    elif not args.no_hpo:
        print("HPO objetivo: acurácia no conjunto de teste (hold-out)", flush=True)
    if args.no_hpo:
        print("HPO desligado (--no-hpo): hiperparâmetros fixos", flush=True)

    metrics: list[dict] = []
    phase6_start = time.time()
    total_steps = 5 + (1 if cv_folds > 0 else 0)  # 4 bases + stacking + CV opcional
    cv_reports: dict[str, dict] | None = None  # preenchido se cv_folds > 0

    if not args.no_plots:
        import matplotlib.pyplot as plt
        import seaborn as sns

        def heatmap(y_true, y_pred, title: str) -> None:
            cm = confusion_matrix(y_true, y_pred)
            _, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
            ax.set_xlabel("y_pred")
            ax.set_ylabel("y_true")
            ax.set_title(title)
            plt.show()
    else:

        def heatmap(y_true, y_pred, title: str) -> None:  # noqa: ARG001
            pass

    # ----- XGBoost -----
    _phase_banner(1, total_steps, "XGBoost" + (" — HPO" if not args.no_hpo else " — fit"))
    if args.no_hpo:
        xg = xgb.XGBClassifier(learning_rate=0.7340229699980686, n_estimators=70, max_depth=14, random_state=rs)
    else:
        from hyperopt import hp

        def build_xg(params: dict[str, Any]) -> xgb.XGBClassifier:
            return xgb.XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=abs(float(params["learning_rate"])),
                random_state=rs,
            )

        xg_space = {
            "n_estimators": hp.quniform("n_estimators", 10, 100, 5),
            "max_depth": hp.quniform("max_depth", 4, 100, 1),
            "learning_rate": hp.normal("learning_rate", 0.01, 0.9),
        }
        best = _fmin_best(
            _hyperopt_objective(
                build_xg,
                X_train,
                y_train,
                X_test,
                y_test,
                hpo_on_validation=args.hpo_on_validation,
                cv_folds=cv_folds,
                random_state=rs,
            ),
            xg_space,
            max_evals=20,
            label="XGBoost",
            random_state=rs,
        )
        xg = xgb.XGBClassifier(
            n_estimators=int(best["n_estimators"]),
            max_depth=int(best["max_depth"]),
            learning_rate=abs(float(best["learning_rate"])),
            random_state=rs,
        )

    print("  [XGBoost] treinando modelo final no treino completo...", flush=True)
    xg.fit(X_train, y_train)
    metrics.append(_evaluate("XGBoost (base)", xg, X_test, y_test))
    heatmap(y_test, xg.predict(X_test), "XGBoost")
    xg_train_p = xg.predict(X_train).reshape(-1, 1)
    xg_test_p = xg.predict(X_test).reshape(-1, 1)

    # ----- Random Forest -----
    _phase_banner(2, total_steps, "Random Forest" + (" — HPO" if not args.no_hpo else " — fit"))
    if args.no_hpo:
        rf_hpo = RandomForestClassifier(
            n_estimators=71,
            min_samples_leaf=1,
            max_depth=46,
            min_samples_split=9,
            max_features=max_features_hpo,
            criterion="entropy",
            random_state=rs,
        )
    else:
        from hyperopt import hp

        def build_rf(params: dict[str, Any]) -> RandomForestClassifier:
            return RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                max_features=int(params["max_features"]),
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                criterion=_criterion_value(params["criterion"]),
                random_state=rs,
            )

        rf_space = {
            "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "max_features": hp.quniform("max_features", 1, max_features_hpo, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        }
        best = _fmin_best(
            _hyperopt_objective(
                build_rf,
                X_train,
                y_train,
                X_test,
                y_test,
                hpo_on_validation=args.hpo_on_validation,
                cv_folds=cv_folds,
                random_state=rs,
            ),
            rf_space,
            max_evals=20,
            label="RandomForest",
            random_state=rs,
        )
        crit = _criterion_value(best["criterion"])
        rf_hpo = RandomForestClassifier(
            n_estimators=int(best["n_estimators"]),
            max_depth=int(best["max_depth"]),
            max_features=int(best["max_features"]),
            min_samples_split=int(best["min_samples_split"]),
            min_samples_leaf=int(best["min_samples_leaf"]),
            criterion=crit,
            random_state=rs,
        )

    print("  [RandomForest] treinando modelo final no treino completo...", flush=True)
    rf_hpo.fit(X_train, y_train)
    metrics.append(_evaluate("RandomForest (HPO)", rf_hpo, X_test, y_test))
    heatmap(y_test, rf_hpo.predict(X_test), "RF")
    rf_train_p = rf_hpo.predict(X_train).reshape(-1, 1)
    rf_test_p = rf_hpo.predict(X_test).reshape(-1, 1)

    # ----- Decision Tree -----
    _phase_banner(3, total_steps, "Decision Tree" + (" — HPO (50 trials)" if not args.no_hpo else " — fit"))
    if args.no_hpo:
        dt_hpo = DecisionTreeClassifier(
            min_samples_leaf=2,
            max_depth=47,
            min_samples_split=3,
            max_features=max_features_hpo,
            criterion="gini",
            random_state=rs,
        )
    else:
        from hyperopt import hp

        def build_dt(params: dict[str, Any]) -> DecisionTreeClassifier:
            return DecisionTreeClassifier(
                max_depth=int(params["max_depth"]),
                max_features=int(params["max_features"]),
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                criterion=_criterion_value(params["criterion"]),
                random_state=rs,
            )

        dt_space = {
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "max_features": hp.quniform("max_features", 1, max_features_hpo, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        }
        best = _fmin_best(
            _hyperopt_objective(
                build_dt,
                X_train,
                y_train,
                X_test,
                y_test,
                hpo_on_validation=args.hpo_on_validation,
                cv_folds=cv_folds,
                random_state=rs,
            ),
            dt_space,
            max_evals=50,
            label="DecisionTree",
            random_state=rs,
        )
        crit = _criterion_value(best["criterion"])
        dt_hpo = DecisionTreeClassifier(
            max_depth=int(best["max_depth"]),
            max_features=int(best["max_features"]),
            min_samples_split=int(best["min_samples_split"]),
            min_samples_leaf=int(best["min_samples_leaf"]),
            criterion=crit,
            random_state=rs,
        )

    print("  [DecisionTree] treinando modelo final no treino completo...", flush=True)
    dt_hpo.fit(X_train, y_train)
    metrics.append(_evaluate("DecisionTree (HPO)", dt_hpo, X_test, y_test))
    heatmap(y_test, dt_hpo.predict(X_test), "DT")
    dt_train_p = dt_hpo.predict(X_train).reshape(-1, 1)
    dt_test_p = dt_hpo.predict(X_test).reshape(-1, 1)

    # ----- Extra Trees -----
    _phase_banner(4, total_steps, "Extra Trees" + (" — HPO" if not args.no_hpo else " — fit"))
    if args.no_hpo:
        et_hpo = ExtraTreesClassifier(
            n_estimators=53,
            min_samples_leaf=1,
            max_depth=31,
            min_samples_split=5,
            max_features=max_features_hpo,
            criterion="entropy",
            random_state=rs,
        )
    else:
        from hyperopt import hp

        def build_et(params: dict[str, Any]) -> ExtraTreesClassifier:
            return ExtraTreesClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                max_features=int(params["max_features"]),
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                criterion=_criterion_value(params["criterion"]),
                random_state=rs,
            )

        et_space = {
            "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "max_features": hp.quniform("max_features", 1, max_features_hpo, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        }
        best = _fmin_best(
            _hyperopt_objective(
                build_et,
                X_train,
                y_train,
                X_test,
                y_test,
                hpo_on_validation=args.hpo_on_validation,
                cv_folds=cv_folds,
                random_state=rs,
            ),
            et_space,
            max_evals=20,
            label="ExtraTrees",
            random_state=rs,
        )
        crit = _criterion_value(best["criterion"])
        et_hpo = ExtraTreesClassifier(
            n_estimators=int(best["n_estimators"]),
            max_depth=int(best["max_depth"]),
            max_features=int(best["max_features"]),
            min_samples_split=int(best["min_samples_split"]),
            min_samples_leaf=int(best["min_samples_leaf"]),
            criterion=crit,
            random_state=rs,
        )

    print("  [ExtraTrees] treinando modelo final no treino completo...", flush=True)
    et_hpo.fit(X_train, y_train)
    metrics.append(_evaluate("ExtraTrees (HPO)", et_hpo, X_test, y_test))
    heatmap(y_test, et_hpo.predict(X_test), "ET")
    et_train_p = et_hpo.predict(X_train).reshape(-1, 1)
    et_test_p = et_hpo.predict(X_test).reshape(-1, 1)

    # ----- Stacking (meta features = predições dos 4 modelos) -----
    _phase_banner(5, total_steps, f"Stacking (meta-learner={args.meta_learner})")
    x_train_meta = np.concatenate((dt_train_p, et_train_p, rf_train_p, xg_train_p), axis=1)
    x_test_meta = np.concatenate((dt_test_p, et_test_p, rf_test_p, xg_test_p), axis=1)

    base_entries: list[tuple[str, Any]] = [
        ("XGBoost (base)", xg),
        ("RandomForest (HPO)", rf_hpo),
        ("DecisionTree (HPO)", dt_hpo),
        ("ExtraTrees (HPO)", et_hpo),
    ]
    base_names = {name for name, _ in base_entries}
    meta_label: str
    stk_for_cv: Any | None = None

    if args.meta_learner == "best-base":
        best_base_name = _pick_best_base_name(metrics, base_names)
        best_est = next(est for name, est in base_entries if name == best_base_name)
        print(f"\nStacking meta-learner (best-base): reutilizando '{best_base_name}'")
        meta = clone(best_est)
        meta.fit(x_train_meta, y_train)
        meta_label = f"Stacking meta ({best_base_name})"
        metrics.append(_evaluate(meta_label, meta, x_test_meta, y_test, binary=args.binary))
        heatmap(y_test, meta.predict(x_test_meta), "Stacking meta (best base)")
    else:
        stk = xgb.XGBClassifier(random_state=rs)
        stk.fit(x_train_meta, y_train)
        stk_for_cv = stk
        metrics.append(_evaluate("Stacking (XGB meta)", stk, x_test_meta, y_test))
        heatmap(y_test, stk.predict(x_test_meta), "Stacking")

        if args.no_hpo:
            meta = xgb.XGBClassifier(
                learning_rate=0.19229249758051492, n_estimators=30, max_depth=36, random_state=rs
            )
        else:
            from hyperopt import hp

            def build_meta(params: dict[str, Any]) -> xgb.XGBClassifier:
                return xgb.XGBClassifier(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]),
                    learning_rate=abs(float(params["learning_rate"])),
                    random_state=rs,
                )

            meta_space = {
                "n_estimators": hp.quniform("n_estimators", 10, 100, 5),
                "max_depth": hp.quniform("max_depth", 4, 100, 1),
                "learning_rate": hp.normal("learning_rate", 0.01, 0.9),
            }
            best = _fmin_best(
                _hyperopt_objective(
                    build_meta,
                    x_train_meta,
                    y_train,
                    x_test_meta,
                    y_test,
                    hpo_on_validation=args.hpo_on_validation,
                    cv_folds=cv_folds,
                    random_state=rs,
                ),
                meta_space,
                max_evals=20,
                label="Stacking meta",
                random_state=rs,
            )
            meta = xgb.XGBClassifier(
                n_estimators=int(best["n_estimators"]),
                max_depth=int(best["max_depth"]),
                learning_rate=abs(float(best["learning_rate"])),
                random_state=rs,
            )

        meta.fit(x_train_meta, y_train)
        meta_label = "Stacking meta (HPO XGB)"
        metrics.append(_evaluate(meta_label, meta, x_test_meta, y_test, binary=args.binary))
        heatmap(y_test, meta.predict(x_test_meta), "Stacking HPO")

    if cv_folds > 0:
        _phase_banner(6, total_steps, f"Relatório {cv_folds}-fold CV (modelos finais)")
        cv_models: list[tuple[str, Any, np.ndarray, np.ndarray]] = [
            ("XGBoost (base)", xg, X_train, y_train),
            ("RandomForest (HPO)", rf_hpo, X_train, y_train),
            ("DecisionTree (HPO)", dt_hpo, X_train, y_train),
            ("ExtraTrees (HPO)", et_hpo, X_train, y_train),
        ]
        if stk_for_cv is not None:
            cv_models.append(("Stacking (XGB meta)", stk_for_cv, x_train_meta, y_train))
        cv_models.append((meta_label, meta, x_train_meta, y_train))
        cv_reports = _run_cv_reports(cv_models, n_splits=cv_folds, random_state=rs)

    out_metrics = output_dir / "06_supervised_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMétricas salvas em: {out_metrics}")

    try:
        from mth_ids_pipeline.io.model_io import save_supervised_classifier_artifacts
    except ImportError:
        from mth_ids_pipeline.io.model_io import save_supervised_classifier_artifacts

    save_supervised_classifier_artifacts(
        output_dir,
        dt=dt_hpo,
        rf=rf_hpo,
        et=et_hpo,
        xgb_model=xg,
        stacking_meta=meta,
        meta_label=meta_label,
        meta_learner=args.meta_learner,
        binary=bool(args.binary),
    )

    report = {
        "train_input": str(output_dir / P05_TRAIN_SMOTE),
        "test_input": str(output_dir / P05_TEST),
        "metrics_output": str(out_metrics),
        "train_shape": {"rows": int(tr.shape[0]), "cols": int(tr.shape[1])},
        "test_shape": {"rows": int(te.shape[0]), "cols": int(te.shape[1])},
        "train_label_counts": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().items()},
        "test_label_counts": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
        "no_hpo": bool(args.no_hpo),
        "no_plots": bool(args.no_plots),
        "binary": bool(args.binary),
        "cv_folds": cv_folds,
        "cv_folds_requested": args.cv_folds,
        "hpo_on_validation": bool(args.hpo_on_validation),
        "hpo_objective": "cv_train" if args.hpo_on_validation else "holdout_test",
        "meta_learner": args.meta_learner,
        "stacking_meta_model": meta_label,
        "cv_reports": cv_reports,
        "random_state": rs,
    }
    report_path = write_report(paths.reports, "phase06_supervised_models", report)
    print(f"Relatorio salvo em: {report_path}")
    print(f"Fase 6 concluída em {_format_duration(time.time() - phase6_start)}", flush=True)


if __name__ == "__main__":
    main()
