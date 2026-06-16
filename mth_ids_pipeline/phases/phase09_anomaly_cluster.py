"""
Fase 9 (anomaly): SMOTE no treino (partição df1 | df2), CL-k-means no conjunto de teste.

Lê a04_after_kpca.parquet e a06_test_slice.json da fase 8.
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
        label_value_counts_dict,
        load_anomaly_splits,
    )
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.core.clustering import cl_kmeans
    from mth_ids_pipeline.config import A04_AFTER_KPCA, A05_TRAIN_SMOTE, A06_TEST_SLICE_INFO
except ImportError:
    from mth_ids_pipeline.io.anomaly_io import (
        label_value_counts_dict,
        load_anomaly_splits,
    )
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.core.clustering import cl_kmeans
    from mth_ids_pipeline.config import A04_AFTER_KPCA, A05_TRAIN_SMOTE, A06_TEST_SLICE_INFO

try:
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.io.reporting import write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 9 — SMOTE + CL-k-means (anomaly)")
    add_work_dir(parser)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--smote-target",
        type=int,
        default=None,
        help="Notebook: default = nº de BENIGN no treino (18225 no demo PortScan)",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="CAN / artigo: não aplicar SMOTE no treino anomaly",
    )
    args = parser.parse_args()

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    df = pd.read_parquet(work / A04_AFTER_KPCA)
    meta = json.loads((work / A06_TEST_SLICE_INFO).read_text(encoding="utf-8"))
    n_df1 = int(meta.get("n_train_rows", meta["n_df1_rows"]))
    label_col = "Label"
    counts_before = pd.Series(np.ravel(df[label_col].values[:n_df1])).value_counts()

    X_train, X_test, y_train, y_test, did_smote = load_anomaly_splits(
        work,
        smote_target=args.smote_target,
        random_state=args.random_state,
        no_smote=args.no_smote,
    )

    cols = [c for c in df.columns if c != label_col]
    train_out = work / A05_TRAIN_SMOTE
    pd.DataFrame(X_train, columns=cols).assign(**{label_col: y_train}).to_parquet(train_out, index=False)
    print(f"Salvo treino pós-SMOTE: {train_out}")

    pred, acc = cl_kmeans(
        X_train, X_test, y_train, y_test,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )
    print(classification_report(y_test, pred))
    print("Accuracy:", acc)
    print("CM:\n", confusion_matrix(y_test, pred))

    counts_after = pd.Series(y_train).value_counts()
    resolved = int(counts_after.get(1, 0)) if did_smote else args.smote_target
    report = {
        "input": str(work / A04_AFTER_KPCA),
        "train_output": str(train_out),
        "test_slice_meta": str(work / A06_TEST_SLICE_INFO),
        "n_clusters": args.n_clusters,
        "random_state": args.random_state,
        "smote_target": resolved if did_smote else args.smote_target,
        "smote_target_resolved": resolved,
        "did_smote": did_smote,
        "train_counts_before": {str(k): int(v) for k, v in counts_before.items()},
        "train_counts_after": {str(k): int(v) for k, v in counts_after.items()},
        "test_label_counts": label_value_counts_dict(pd.Series(y_test)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "accuracy": float(acc),
    }
    report_path = write_report(paths.reports, "phase09_anomaly_cluster", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
