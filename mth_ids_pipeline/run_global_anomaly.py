"""
Detector anomaly global (Tabela X) — fases 7–11.

Treina um único modelo binário (benigno vs ataque) no mesmo 80%% do split
supervisionado (default 80/20). O hold-out 20%% fica reservado para ``run_eval`` (fase 13).

Use a **mesma** ``--intermediate-dir`` do supervisionado (ex.: merged).

Exemplo (CICIDS2017 / Tabela X):
  python -m mth_ids_pipeline.utils.merge_cicids --profile merged
  python -m mth_ids_pipeline.run_supervised --protocol paper --from 4 --to 6 --test-size 0.3
  python -m mth_ids_pipeline.run_global_anomaly --protocol paper --test-size 0.3
Exemplo (CAN / Tabela X):
  python -m mth_ids_pipeline.run_supervised --protocol can_paper
  python -m mth_ids_pipeline.run_global_anomaly --protocol can_paper
  python -m mth_ids_pipeline.run_eval \\
    --intermediate-dir data/pipeline_can_otids_merged \\
    --work-dir data/pipeline_can_otids_merged/anomaly/global
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mth_ids_pipeline.config import (
    ANOMALY_GLOBAL_WORK_SUBDIR,
    DEFAULT_TEST_SIZE,
    INTERMEDIATE_DIR_MERGED,
)
from mth_ids_pipeline.io.results_io import make_run_log_path
from mth_ids_pipeline.io.run_log import RunLog
from mth_ids_pipeline.io.subprocess_env import configure_stdio_utf8
from mth_ids_pipeline.label_profiles import get_label_profile
from mth_ids_pipeline.protocol import (
    PROTOCOL_CHOICES,
    get_protocol_settings,
    is_can_protocol,
    is_unsw_protocol,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_phase(log: RunLog, module: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *extra]
    log.run_subprocess(cmd, cwd=_repo_root())


def main() -> None:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="MTH-IDS anomaly global (Tabela X)")
    parser.add_argument("--protocol", choices=list(PROTOCOL_CHOICES), default="paper")
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Default: pipeline do perfil CAN (intrusion ou OTIDS) ou pipeline_mth_ids_merged",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Default: <intermediate-dir>/anomaly/global",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--from-phase", type=int, default=7, choices=(7, 8, 9, 10, 11))
    parser.add_argument("--to-phase", type=int, default=11, choices=(7, 8, 9, 10, 11))
    parser.add_argument("--no-hpo", action="store_true")
    args = parser.parse_args()

    ps = get_protocol_settings(args.protocol)
    if args.intermediate_dir is not None:
        intermediate = Path(args.intermediate_dir)
    elif is_can_protocol(args.protocol) or is_unsw_protocol(args.protocol):
        intermediate = get_label_profile(ps.supervised_profile).intermediate_dir
    else:
        intermediate = INTERMEDIATE_DIR_MERGED
    work = Path(args.work_dir) if args.work_dir else intermediate / ANOMALY_GLOBAL_WORK_SUBDIR
    work.mkdir(parents=True, exist_ok=True)
    report_dir = intermediate / "phase_reports"

    common = [
        "--intermediate-dir", str(intermediate),
        "--report-dir", str(report_dir),
        "--work-dir", str(work),
        "--random-state", str(args.random_state),
    ]

    phase10_extra = [
        *common,
        "--n-calls", str(ps.cl_hpo_n_calls),
        "--hpo-metric", ps.cl_hpo_metric,
        "--metrics", ",".join(ps.cl_kmeans_metrics),
    ]
    if args.no_hpo:
        phase10_extra.append("--skip-hpo")
    if ps.skip_anomaly_smote:
        no_smote_flag = ["--no-smote"]
    else:
        no_smote_flag = []

    phases: list[tuple[int, str, list[str]]] = [
        (
            7,
            "mth_ids_pipeline.phases.phase07_anomaly_datasets",
            [*common, "--mode", "global", "--test-size", str(args.test_size)],
        ),
        (
            8,
            "mth_ids_pipeline.phases.phase08_anomaly_features",
            [
                *common,
                "--feature-fit-scope", "train",
                "--fcbf-k", str(ps.fcbf_k),
                "--kpca-components", str(ps.kpca_components),
                "--kpca-kernel", ps.kpca_kernel,
                "--ig-cumulative", str(ps.ig_cumulative),
                "--zscore-scope", ps.anomaly_zscore_scope,
            ]
            + (["--optimize-ig"] if ps.optimize_ig else [])
            + (["--optimize-kpca"] if ps.optimize_kpca else []),
        ),
        (
            9,
            "mth_ids_pipeline.phases.phase09_anomaly_cluster",
            [*common, "--n-clusters", "8", *no_smote_flag],
        ),
        (10, "mth_ids_pipeline.phases.phase10_anomaly_cluster_hpo", [*phase10_extra, *no_smote_flag]),
        (
            11,
            "mth_ids_pipeline.phases.phase11_anomaly_biased",
            [
                *common,
                "--p-star", str(ps.cl_p_star),
                "--biased-mode", ps.biased_mode,
                *no_smote_flag,
            ]
            + (["--force-biased"] if ps.force_biased else [])
            + (
                ["--optimize-p-star", "--p-star-n-calls", str(ps.cl_hpo_n_calls)]
                if ps.optimize_p_star
                else []
            ),
        ),
    ]

    # Bootstrap 06_supervised_metrics if missing (biased tier 4)
    metrics_path = intermediate / "06_supervised_metrics.json"
    if not metrics_path.is_file():
        print(
            f"Aviso: {metrics_path} ausente — execute "
            f"run_supervised --protocol {args.protocol} antes (Tabela VII/VI)."
        )

    print(f"Anomaly global (Tabela X): intermediate={intermediate} work={work}")
    log_path = make_run_log_path(f"global_anomaly_{args.protocol}")
    with RunLog(log_path) as log:
        log.emit(f"intermediate-dir: {intermediate}")
        log.emit(f"work-dir: {work}")
        log.emit(f"protocol: {args.protocol} | fases: {args.from_phase}-{args.to_phase}")
        for phase_num, module, extra in phases:
            if phase_num < args.from_phase or phase_num > args.to_phase:
                continue
            log.emit(f"-> fase {phase_num} ({module}) ...")
            _run_phase(log, module, extra)

    print(f"\nConcluído. Modelos em: {work / 'models' / 'anomaly'}")
    print(f"Log da execução: {log_path}")


if __name__ == "__main__":
    main()
