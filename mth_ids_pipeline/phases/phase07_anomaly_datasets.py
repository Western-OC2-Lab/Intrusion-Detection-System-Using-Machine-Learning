"""
Fase 7 (anomaly branch):
  - ``--mode loao`` (default): leave-one-attack-out — Tabela IX
  - ``--mode global``: treino binário no 80% (Tabela X); hold-out 20% reservado à fase 13

Saídas: a01_without_portscan.parquet, a02_portscan_only.parquet (nomes legados do notebook).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import json
import pandas as pd

try:
    from mth_ids_pipeline.io.anomaly_io import (
        _json_safe,
        build_anomaly_binary_split,
        build_global_binary_train_split,
        label_value_counts_dict,
        log_loao_partition,
        loao_original_label_report,
        require_path,
    )
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir, supervised_path
    from mth_ids_pipeline.config import (
        A00_LOAO_ROUND,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        DEFAULT_TEST_SIZE,
        P02_SAMPLED_KMEANS,
        default_benign_label,
        default_loao_attack_label,
    )
except ImportError:
    from mth_ids_pipeline.io.anomaly_io import (
        _json_safe,
        build_anomaly_binary_split,
        build_global_binary_train_split,
        label_value_counts_dict,
        log_loao_partition,
        loao_original_label_report,
        require_path,
    )
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir, supervised_path
    from mth_ids_pipeline.config import (
        A00_LOAO_ROUND,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        DEFAULT_TEST_SIZE,
        P02_SAMPLED_KMEANS,
        default_benign_label,
        default_loao_attack_label,
    )

try:
    from mth_ids_pipeline.io.reporting import dataset_report, write_report
except ImportError:
    from mth_ids_pipeline.io.reporting import dataset_report, write_report

# PortScan no notebook (LabelEncoder sobre amostra CICIDS2017 fine)
# CAN usa DoS (label 1) — ver default_loao_attack_label()


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 7 — datasets anomaly (LOAO ou global Tabela X)")
    add_work_dir(parser)
    parser.add_argument(
        "--mode",
        choices=("loao", "global"),
        default="loao",
        help="loao=Tabela IX; global=Tabela X (treino 80%%, hold-out 20%% na fase 13)",
    )
    parser.add_argument(
        "--attack-label",
        type=int,
        default=None,
        help="Zero-day LOAO (default: PortScan=5 CICIDS, DoS=1 CAN)",
    )
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    test_size = float(args.test_size) if args.test_size is not None else DEFAULT_TEST_SIZE

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    attack_label = (
        args.attack_label
        if args.attack_label is not None
        else default_loao_attack_label(intermediate_dir=paths.intermediate)
    )
    path_in = supervised_path(paths, P02_SAMPLED_KMEANS)
    require_path(
        path_in,
        hint=(
            "Execute as fases 1–2 antes (gera 02_sampled_kmeans.parquet).\n"
            f"  python -m mth_ids_pipeline.experiment_runner --intermediate-dir {paths.intermediate} "
            "--from-phase 1 --to-phase 2\n"
            "Depois rode a fase 7 com o mesmo --intermediate-dir e --work-dir."
        ),
    )
    df = pd.read_parquet(path_in)
    label_col = "Label"
    benign_label = default_benign_label(intermediate_dir=paths.intermediate)

    p1 = work / A01_WITHOUT_PORTSCAN
    p2 = work / A02_PORTSCAN_ONLY
    round_meta_path = work / A00_LOAO_ROUND

    if args.mode == "global":
        df1, _holdout, global_meta = build_global_binary_train_split(
            df,
            test_size=test_size,
            random_state=int(args.random_state),
            benign_label=benign_label,
            label_col=label_col,
        )
        df2 = df1.iloc[0:0].copy()
        round_meta_path.write_text(
            json.dumps(_json_safe(global_meta), indent=2),
            encoding="utf-8",
        )
        print(
            f"Modo global (Tabela X): treino={df1.shape}, "
            f"hold-out reservado={global_meta['n_holdout_rows']} linhas (fase 13)"
        )
        phase7_meta = {
            **global_meta,
            "zero_day_samples": 0,
            "benign_sampled": 0,
            "benign_label": benign_label,
            "benign_pairing_rule": "none_holdout_reserved",
            "test_binary_label_counts": {},
        }
        log_loao_partition(
            stage="fase 7 (global binário)",
            train_df=df1,
            test_df=df2,
            meta=phase7_meta,
            label_col=label_col,
        )
        report = {
            "input": str(path_in),
            "mode": "global",
            "test_size": test_size,
            "random_state": int(args.random_state),
            "without_portscan_output": str(p1),
            "portscan_output": str(p2),
            "without_portscan": dataset_report(df1, label_col),
            "holdout_rows": global_meta["n_holdout_rows"],
            **global_meta,
        }
    else:
        orig_report = loao_original_label_report(
            df,
            attack_label,
            benign_label=benign_label,
            label_col=label_col,
        )
        df1, df2 = build_anomaly_binary_split(
            df,
            attack_label,
            benign_label=benign_label,
            label_col=label_col,
        )
        round_meta_path.write_text(
            json.dumps(_json_safe(orig_report), indent=2),
            encoding="utf-8",
        )
        phase7_meta = {
            **orig_report,
            "benign_label": benign_label,
            "zero_day_samples": int(len(df2)),
            "benign_sampled": 0,
            "benign_pairing_rule": "fase7_zero_day_only",
            "train_binary_label_counts": label_value_counts_dict(df1[label_col]),
            "test_binary_label_counts": label_value_counts_dict(df2[label_col]),
        }
        log_loao_partition(
            stage="fase 7 (split binário)",
            train_df=df1,
            test_df=df2,
            meta=phase7_meta,
            label_col=label_col,
        )
        report = {
            "input": str(path_in),
            "mode": "loao",
            "attack_label": attack_label,
            "without_portscan_output": str(p1),
            "portscan_output": str(p2),
            "without_portscan": dataset_report(df1, label_col),
            "portscan": dataset_report(df2, label_col),
            **orig_report,
        }

    df1.to_parquet(p1, index=False)
    df2.to_parquet(p2, index=False)
    print(f"Salvo: {p1} {df1.shape}, {p2} {df2.shape}")
    report_path = write_report(paths.reports, "phase07_anomaly_datasets", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
