"""
Fase 11 (anomaly tier 4): CL-k-means + biased classifiers B1/B2 + threshold p* (artigo).

- k padrão: lido de phase10_anomaly_cluster_hpo.json
- B1/B2: mesma **família** (RF/XGB/DT/ET) do melhor modelo da Tabela VII
  (``06_supervised_metrics.json`` no merged, copiado para fine no bootstrap — ver
  ``docs/EXECUCAO.md``)
- Gate: aplica biased só se melhorar F1 no hold-out interno do treino
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

try:
    from mth_ids_pipeline.io.anomaly_io import (
        is_global_table_x_protocol,
        label_value_counts_dict,
        load_anomaly_full_train_smote,
        load_anomaly_splits,
    )
    from mth_ids_pipeline.core.biased_classifiers import (
        apply_biased_refinement,
        estimator_factory_for_supervised,
        pick_best_biased_mode,
        load_best_metric,
        load_best_n_clusters,
        pick_best_supervised_model,
        train_biased_pair,
    )
    from mth_ids_pipeline.core.clustering import cl_kmeans_fit_predict
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.config import DEFAULT_BIASED_MODE
    from mth_ids_pipeline.core.evaluation import binary_dr_far_f1
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.io.anomaly_io import (
        is_global_table_x_protocol,
        label_value_counts_dict,
        load_anomaly_full_train_smote,
        load_anomaly_splits,
    )
    from mth_ids_pipeline.core.biased_classifiers import (
        apply_biased_refinement,
        estimator_factory_for_supervised,
        pick_best_biased_mode,
        load_best_metric,
        load_best_n_clusters,
        pick_best_supervised_model,
        train_biased_pair,
    )
    from mth_ids_pipeline.core.clustering import cl_kmeans_fit_predict
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.config import DEFAULT_BIASED_MODE
    from mth_ids_pipeline.core.evaluation import binary_dr_far_f1
    from mth_ids_pipeline.io.reporting import write_report


def _resolve_n_clusters(report_dir: Path, n_clusters: int | None) -> tuple[int, str]:
    if n_clusters is not None:
        return int(n_clusters), "cli"
    auto = load_best_n_clusters(report_dir)
    if auto is not None:
        print(f"k automático da fase 10: {auto}")
        return auto, "phase10"
    print("Aviso: phase10 não encontrado; usando n_clusters=8")
    return 8, "default"


def _resolve_metric(report_dir: Path, metric: str | None) -> tuple[str, str]:
    if metric is not None:
        return metric, "cli"
    auto = load_best_metric(report_dir)
    if auto is not None:
        print(f"metric automática da fase 10: {auto}")
        return auto, "phase10"
    return "euclidean", "default"


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 11 — CL-k-means + biased classifiers (tier 4)")
    add_work_dir(parser)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--p-star", type=float, default=0.933)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--smote-target",
        type=int,
        default=None,
        help="Notebook: default = nº de BENIGN no treino",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="CAN / artigo: não aplicar SMOTE no treino anomaly",
    )
    parser.add_argument(
        "--metric",
        choices=("euclidean", "manhattan", "cosine", "mahalanobis"),
        default=None,
        help="Distância CL-k-means; default = best_metric da fase 10",
    )
    parser.add_argument(
        "--biased-mode",
        choices=("auto", "both", "b1-only", "b2-only", "none"),
        default=DEFAULT_BIASED_MODE,
        help="Artigo: both (B1+B2); auto = gate por F1 no treino",
    )
    parser.add_argument(
        "--force-biased",
        action="store_true",
        help="Protocolo paper: aplica tier 4 sem gate de melhoria",
    )
    parser.add_argument(
        "--optimize-p-star",
        action="store_true",
        help="BO-GP do limiar p* (artigo Sec. IV-D2)",
    )
    parser.add_argument("--p-star-n-calls", type=int, default=20)
    args = parser.parse_args()
    requested = args.biased_mode
    p_star = float(args.p_star)
    p_star_trials: list[dict] = []

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    report_dir = paths.reports
    n_clusters, n_clusters_source = _resolve_n_clusters(report_dir, args.n_clusters)
    metric, metric_source = _resolve_metric(report_dir, args.metric)
    metrics_path = paths.intermediate / "06_supervised_metrics.json"
    best_model_name = pick_best_supervised_model(metrics_path)
    factory = estimator_factory_for_supervised(
        best_model_name,
        random_state=args.random_state,
        intermediate_dir=paths.intermediate,
    )
    print(f"Biased learners: família de '{best_model_name}'")

    X_train, X_test, y_train, y_test, did_smote = load_anomaly_splits(
        work,
        smote_target=args.smote_target,
        random_state=args.random_state,
        no_smote=args.no_smote,
    )
    train_label_counts = label_value_counts_dict(pd.Series(y_train))
    test_label_counts = label_value_counts_dict(pd.Series(y_test))
    print(
        f"Partição LOAO (fase 11): treino={X_train.shape} labels={train_label_counts} | "
        f"teste={X_test.shape} labels={test_label_counts}"
    )

    cl_res = cl_kmeans_fit_predict(
        X_train,
        X_test,
        y_train,
        y_test,
        n_clusters=n_clusters,
        random_state=args.random_state,
        metric=metric,
    )
    print(f"CL-k-means (n={n_clusters}, metric={metric}): accuracy={cl_res.accuracy:.4f}")
    print(classification_report(y_test, cl_res.y_pred))
    metrics_cl = binary_dr_far_f1(y_test, cl_res.y_pred)
    print(
        f"DR={metrics_cl['detection_rate']:.4f} FAR={metrics_cl['false_alarm_rate']:.4f} "
        f"F1={metrics_cl['f1']:.4f}"
    )

    y_final = cl_res.y_pred
    metrics_final = metrics_cl
    biased_stats: dict = {}
    selection_info: dict = {}
    b1 = None
    b2 = None

    effective_mode, selection_info = pick_best_biased_mode(
        X_train,
        y_train,
        n_clusters=n_clusters,
        random_state=args.random_state,
        metric=metric,
        p_star=p_star,
        estimator_factory=factory,
        requested=requested,  # type: ignore[arg-type]
        force_apply=args.force_biased,
    )
    print(f"\nModo biased selecionado: {effective_mode}")
    if selection_info.get("scores"):
        print("F1 validação interna:", selection_info["scores"])

    if effective_mode != "none":
        train_cl = cl_kmeans_fit_predict(
            X_train,
            X_train,
            y_train,
            y_train,
            n_clusters=n_clusters,
            random_state=args.random_state,
            metric=metric,
        )
        b1, b2, biased_stats = train_biased_pair(
            X_train,
            y_train,
            train_cl.y_pred,
            estimator_factory=factory,
            random_state=args.random_state,
            mode=effective_mode,
        )
        if args.optimize_p_star:
            try:
                from mth_ids_pipeline.core.hyperparameter_optimization import optimize_p_star_threshold
            except ImportError:
                from mth_ids_pipeline.core.hyperparameter_optimization import optimize_p_star_threshold

            def _p_objective(p: float) -> float:
                yp = apply_biased_refinement(
                    cl_res.y_pred,
                    cl_res.cluster_confidence,
                    X_test,
                    b1=b1,
                    b2=b2,
                    p_star=p,
                    mode=effective_mode,
                )
                return float(binary_dr_far_f1(y_test, yp)["f1"])

            hpo_p = optimize_p_star_threshold(
                _p_objective,
                n_calls=args.p_star_n_calls,
                random_state=args.random_state,
            )
            p_star = float(hpo_p.best_p_star)
            p_star_trials = hpo_p.trials
            print(f"BO-GP p*: melhor={p_star:.4f}, F1={hpo_p.best_score:.4f}")

        y_final = apply_biased_refinement(
            cl_res.y_pred,
            cl_res.cluster_confidence,
            X_test,
            b1=b1,
            b2=b2,
            p_star=p_star,
            mode=effective_mode,
        )
        metrics_final = binary_dr_far_f1(y_test, y_final)
        print(f"\nMTH-IDS anomaly (biased mode={effective_mode}, p*={p_star}):")
        print(classification_report(y_test, y_final))
        print(
            f"Acc={metrics_final['accuracy']:.4f} DR={metrics_final['detection_rate']:.4f} "
            f"FAR={metrics_final['false_alarm_rate']:.4f} F1={metrics_final['f1']:.4f}"
        )
        print("CM:\n", confusion_matrix(y_test, y_final))
    else:
        print("\nResultado final = CL-k-means (sem biased).")

    report = {
        "work_dir": str(work),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
        "n_clusters": n_clusters,
        "n_clusters_source": n_clusters_source,
        "p_star": p_star,
        "p_star_optimized": bool(args.optimize_p_star),
        "p_star_hpo_trials": p_star_trials,
        "force_biased": bool(args.force_biased),
        "metric": metric,
        "metric_source": metric_source,
        "random_state": args.random_state,
        "smote_target": args.smote_target,
        "did_smote": did_smote,
        "biased_mode_requested": requested,
        "biased_mode_applied": effective_mode,
        "supervised_model_for_biased": best_model_name,
        "mode_selection": selection_info,
        "cl_kmeans": metrics_cl,
        "mth_ids_anomaly": metrics_final,
        "biased_training": biased_stats,
    }
    report_path = write_report(report_dir, "phase11_anomaly_biased", report)
    print(f"Relatório salvo em: {report_path}")

    try:
        from mth_ids_pipeline.core.clustering import cl_kmeans_build_inference_state
        from mth_ids_pipeline.core.inference import resolve_anomaly_feature_names
        from mth_ids_pipeline.io.model_io import save_anomaly_inference_artifacts
    except ImportError:
        from mth_ids_pipeline.core.clustering import cl_kmeans_build_inference_state
        from mth_ids_pipeline.core.inference import resolve_anomaly_feature_names
        from mth_ids_pipeline.io.model_io import save_anomaly_inference_artifacts

    slice_meta: dict = {}
    slice_path = work / "a06_test_slice.json"
    if slice_path.is_file():
        slice_meta = json.loads(slice_path.read_text(encoding="utf-8"))
    if is_global_table_x_protocol(slice_meta):
        X_full, y_full, _ = load_anomaly_full_train_smote(
            work,
            smote_target=args.smote_target,
            random_state=args.random_state,
        )
        y_fit = np.ravel(y_full).astype(np.int64)
        X_fit = X_full
        print(f"Persistência global: CL-k-means no treino completo {X_fit.shape}")
    else:
        X_fit = X_train
        y_fit = np.ravel(y_train).astype(np.int64)
    cl_state = cl_kmeans_build_inference_state(
        X_fit,
        y_fit,
        n_clusters=n_clusters,
        random_state=args.random_state,
        metric=metric,
    )
    save_b1 = b1 if effective_mode in ("both", "b1-only") else None
    save_b2 = b2 if effective_mode in ("both", "b2-only") else None
    if is_global_table_x_protocol(slice_meta) and effective_mode != "none":
        from mth_ids_pipeline.core.clustering import cl_kmeans_predict_inference

        y_cl_fit, _ = cl_kmeans_predict_inference(cl_state, X_fit)
        save_b1, save_b2, _ = train_biased_pair(
            X_fit,
            y_fit,
            y_cl_fit,
            estimator_factory=factory,
            random_state=args.random_state,
            mode=effective_mode,  # type: ignore[arg-type]
        )
        print("Persistência global: B1/B2 re-treinados no treino completo")

    save_anomaly_inference_artifacts(
        work,
        cl_state=cl_state,
        b1=save_b1,
        b2=save_b2,
        n_clusters=n_clusters,
        metric=metric,
        p_star=p_star,
        biased_mode=effective_mode,
        feature_names=resolve_anomaly_feature_names(work),
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
