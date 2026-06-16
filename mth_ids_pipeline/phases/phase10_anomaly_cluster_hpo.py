"""
Fase 10 (anomaly): BO-GP para otimizar n_clusters e métrica de distância do CL-k-means.

Protocolo paper: objetivo F1 no teste LOAO (Tabela IX); notebook: accuracy.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

try:
    from mth_ids_pipeline.io.anomaly_io import label_value_counts_dict, load_anomaly_splits
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.core.clustering import CL_KMEANS_METRICS, cl_kmeans
    from mth_ids_pipeline.core.evaluation import binary_dr_far_f1
    from mth_ids_pipeline.core.hyperparameter_optimization import optimize_cl_kmeans_clusters
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.io.anomaly_io import label_value_counts_dict, load_anomaly_splits
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.core.clustering import CL_KMEANS_METRICS, cl_kmeans
    from mth_ids_pipeline.core.evaluation import binary_dr_far_f1
    from mth_ids_pipeline.core.hyperparameter_optimization import optimize_cl_kmeans_clusters
    from mth_ids_pipeline.io.reporting import write_report


_CL_KMEANS_METRIC_EVIDENCE = {
    "article": (
        "Artigo Yang et al. (2022) Sec. IV-D: BO-GP otimiza n_clusters do CL-k-means; "
        "métrica de objetivo do HPO não é nomeada explicitamente. "
        "Tabela IX reporta F1/DR/FAR — preset paper usa F1 como alinhamento às métricas de avaliação."
    ),
    "notebook": (
        "MTH_IDS_IoTJ.ipynb: gp_minimize com objective retornando 1 - accuracy_score(y_test, y_pred) "
        "no split LOAO."
    ),
}


def _hpo_score(y_test, y_pred, metric: str) -> float:
    if metric == "f1":
        return float(binary_dr_far_f1(y_test, y_pred)["f1"])
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y_test, y_pred))


def _hpo_metric_provenance(metric: str, source: str) -> dict[str, str]:
    return {
        "hpo_metric": metric,
        "source": source,
        "article_evidence": _CL_KMEANS_METRIC_EVIDENCE["article"],
        "notebook_evidence": _CL_KMEANS_METRIC_EVIDENCE["notebook"],
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 10 — BO-GP CL-k-means")
    add_work_dir(parser)
    parser.add_argument("--n-calls", type=int, default=20)
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
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument(
        "--hpo-metric",
        choices=("accuracy", "f1"),
        default="f1",
        help="Objetivo BO-GP (paper: f1; notebook: accuracy)",
    )
    parser.add_argument(
        "--hpo-metric-source",
        choices=("article", "notebook"),
        default="article",
        help="Fonte da métrica HPO (article | notebook)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Métricas CL-k-means (ex.: euclidean,manhattan,cosine,mahalanobis)",
    )
    args = parser.parse_args()
    metric_space = CL_KMEANS_METRICS
    if args.metrics:
        metric_space = tuple(m.strip() for m in args.metrics.split(",") if m.strip())

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    X_train, X_test, y_train, y_test, did_smote = load_anomaly_splits(
        work,
        smote_target=args.smote_target,
        random_state=args.random_state,
        no_smote=args.no_smote,
    )

    y_base, baseline_score = cl_kmeans(
        X_train, X_test, y_train, y_test, n_clusters=8, random_state=args.random_state
    )
    baseline_score = _hpo_score(y_test, y_base, args.hpo_metric)
    print(
        f"Baseline CL-k-means (n=8, metric=euclidean): "
        f"{args.hpo_metric}={baseline_score:.4f}"
    )

    trials: list[dict] = []
    if args.skip_hpo:
        best_n, best_score, best_metric = 8, baseline_score, "euclidean"
    else:

        def objective(n: int, metric: str) -> float:
            y_pred, _ = cl_kmeans(
                X_train,
                X_test,
                y_train,
                y_test,
                n_clusters=n,
                random_state=args.random_state,
                metric=metric,
            )
            score = _hpo_score(y_test, y_pred, args.hpo_metric)
            print(f"  trial n={n}, metric={metric}: {args.hpo_metric}={score:.4f}")
            return score

        hpo = optimize_cl_kmeans_clusters(
            objective,
            n_calls=args.n_calls,
            random_state=args.random_state,
            metrics=metric_space,
            objective_metric=args.hpo_metric,
        )
        best_n = hpo.best_n_clusters
        best_score = hpo.best_accuracy
        best_metric = hpo.best_metric
        trials = hpo.trials

    print(f"Melhor n_clusters={best_n}, metric={best_metric}, {args.hpo_metric}={best_score:.4f}")
    y_final, final_score = cl_kmeans(
        X_train,
        X_test,
        y_train,
        y_test,
        n_clusters=best_n,
        random_state=args.random_state,
        metric=best_metric,
    )
    final_score = _hpo_score(y_test, y_final, args.hpo_metric)
    print(f"Avaliação final n={best_n}, metric={best_metric}: {args.hpo_metric}={final_score:.4f}")

    train_label_counts = label_value_counts_dict(pd.Series(y_train))
    test_label_counts = label_value_counts_dict(pd.Series(y_test))
    print(
        f"Partição LOAO (fase 10): treino={X_train.shape} labels={train_label_counts} | "
        f"teste={X_test.shape} labels={test_label_counts}"
    )

    metric_provenance = _hpo_metric_provenance(args.hpo_metric, args.hpo_metric_source)
    report = {
        "work_dir": str(work),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
        "baseline_clusters": 8,
        "baseline_metric": "euclidean",
        "baseline_score": float(baseline_score),
        "best_n_clusters": int(best_n),
        "best_metric": best_metric,
        "best_score": float(best_score),
        "best_accuracy": float(best_score),
        "best_f1": float(best_score) if args.hpo_metric == "f1" else None,
        "best_config": {
            "n_clusters": int(best_n),
            "metric": best_metric,
            "score": float(best_score),
            "objective_metric": args.hpo_metric,
        },
        "hpo_trials": trials,
        "final_score": float(final_score),
        "n_calls": args.n_calls,
        "search_space": {
            "n_clusters": [2, 50],
            "metric": list(CL_KMEANS_METRICS),
        },
        "optimizer": "BO-GP (skopt gp_minimize)",
        "objective": f"{args.hpo_metric} on LOAO test split",
        "hpo_metric": args.hpo_metric,
        "hpo_metric_source": args.hpo_metric_source,
        "hpo_metric_provenance": metric_provenance,
        "random_state": args.random_state,
        "smote_target": args.smote_target,
        "did_smote": did_smote,
        "reproducible": True,
    }
    report_path = write_report(paths.reports, "phase10_anomaly_cluster_hpo", report)
    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
