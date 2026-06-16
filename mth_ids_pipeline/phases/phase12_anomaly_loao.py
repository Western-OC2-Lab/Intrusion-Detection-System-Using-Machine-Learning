"""
Fase 12: leave-one-attack-out no ramo anomaly (Tabela IX do artigo).

Para cada rótulo de ataque no dataset amostrado, executa fases 7→8→9→10→11 em subdiretório
e agrega DR/FAR/F1.
"""

from __future__ import annotations

import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

try:
    from mth_ids_pipeline.io.anomaly_io import discover_attack_labels
    from mth_ids_pipeline.io.loao_reporting import (
        build_loao_summary,
        load_attack_row,
        write_loao_summary,
    )
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import (
        CICIDS2017_FINE_LABEL_NAMES,
        CICIDS2017_MERGED_LABEL_NAMES,
        P02_SAMPLED_KMEANS,
        UNSW_NB15_LABEL_NAMES,
        default_benign_label,
        resolve_can_label_names,
    )
    from mth_ids_pipeline.io.reporting import write_report
    from mth_ids_pipeline.io.results_io import mirror_log
    from mth_ids_pipeline.io.run_log import RunLog
    from mth_ids_pipeline.io.subprocess_env import configure_stdio_utf8
except ImportError:
    from mth_ids_pipeline.io.anomaly_io import discover_attack_labels
    from mth_ids_pipeline.io.loao_reporting import (
        build_loao_summary,
        load_attack_row,
        write_loao_summary,
    )
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import (
        CICIDS2017_FINE_LABEL_NAMES,
        CICIDS2017_MERGED_LABEL_NAMES,
        P02_SAMPLED_KMEANS,
        UNSW_NB15_LABEL_NAMES,
        default_benign_label,
        resolve_can_label_names,
    )
    from mth_ids_pipeline.io.reporting import write_report
    from mth_ids_pipeline.io.results_io import mirror_log
    from mth_ids_pipeline.io.run_log import RunLog
    from mth_ids_pipeline.io.subprocess_env import configure_stdio_utf8


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _resolve_label_names(attacks: list[int], intermediate_dir: Path | None = None) -> dict[int, str]:
    """Nomes legíveis por ID (CAN, CICIDS merged ou fine)."""
    if intermediate_dir and "pipeline_can_otids" in intermediate_dir.as_posix():
        table = resolve_can_label_names(
            attack_labels=attacks,
            pipeline_path=intermediate_dir,
        )
    elif intermediate_dir and "pipeline_unsw_nb15" in intermediate_dir.as_posix():
        table = UNSW_NB15_LABEL_NAMES
    elif max(attacks, default=0) > 6:
        table = CICIDS2017_FINE_LABEL_NAMES
    else:
        table = CICIDS2017_MERGED_LABEL_NAMES
    return {label: table.get(label, f"Label={label}") for label in attacks}


def _run_phase(module: str, extra: list[str], repo_root: Path, log: RunLog) -> float:
    cmd = [sys.executable, "-m", module, *extra]
    return log.run_subprocess(cmd, cwd=repo_root)


def _loao_subphases(args) -> list[tuple[str, str, str]]:
    phases: list[tuple[str, str, str]] = [
        ("7", "mth_ids_pipeline.phases.phase07_anomaly_datasets", "Partição binária LOAO"),
        (
            "8",
            "mth_ids_pipeline.phases.phase08_anomaly_features",
            "Features (Z-score, IG, FCBF, KPCA)",
        ),
    ]
    if not args.skip_phase9:
        phases.append(
            ("9", "mth_ids_pipeline.phases.phase09_anomaly_cluster", "SMOTE + CL-k-means inicial")
        )
    if not args.skip_phase10:
        phases.append(
            ("10", "mth_ids_pipeline.phases.phase10_anomaly_cluster_hpo", "BO-GP n_clusters")
        )
    phases.append(("11", "mth_ids_pipeline.phases.phase11_anomaly_biased", "Biased B1/B2"))
    return phases


def main() -> None:
    configure_stdio_utf8()
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 12 — LOAO anomaly (Tabela IX)")
    parser.add_argument("--output-root", type=Path, default=None, help="Raiz LOAO (default: anomaly/loao)")
    parser.add_argument(
        "--attack-label",
        type=int,
        default=None,
        help="Um único zero-day por LabelEncoder (atalho para --attack-labels N)",
    )
    parser.add_argument(
        "--attack-labels",
        type=str,
        default=None,
        help="Subset de zero-days, ex.: 1 ou 1,5,10 (default: todos em 02_sampled_kmeans)",
    )
    parser.add_argument("--benign-target", type=int, default=None)
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
    parser.add_argument("--hpo-n-calls", type=int, default=20)
    parser.add_argument("--hpo-metric", choices=("accuracy", "f1"), default="f1")
    parser.add_argument("--biased-mode", default="both")
    parser.add_argument("--force-biased", action="store_true")
    parser.add_argument("--optimize-p-star", action="store_true")
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--skip-phase9", action="store_true")
    parser.add_argument("--skip-phase10", action="store_true")
    parser.add_argument("--feature-fit-scope", choices=("combined", "train"), default="combined")
    parser.add_argument("--zscore-scope", choices=("per_split", "combined"), default="combined")
    parser.add_argument("--fcbf-k", type=int, default=20)
    parser.add_argument("--kpca-components", type=int, default=10)
    parser.add_argument("--kpca-kernel", type=str, default="rbf")
    parser.add_argument("--ig-cumulative", type=float, default=0.9)
    parser.add_argument("--optimize-ig", action="store_true")
    parser.add_argument("--optimize-kpca", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--ig-hpo-calls", type=int, default=15)
    parser.add_argument("--kpca-hpo-calls", type=int, default=15)
    args = parser.parse_args()

    if args.attack_label is not None and args.attack_labels:
        parser.error("Use apenas --attack-label ou --attack-labels, não ambos.")
    if args.attack_label is not None:
        args.attack_labels = str(args.attack_label)

    paths = init_paths(args)
    output_root = args.output_root or (paths.anomaly / "loao")
    repo_root = Path(__file__).resolve().parents[2]
    path_in = supervised_path(paths, P02_SAMPLED_KMEANS)
    df = pd.read_parquet(path_in)
    benign_label = default_benign_label(intermediate_dir=paths.intermediate)

    if args.attack_labels:
        requested = [int(x.strip()) for x in args.attack_labels.split(",")]
    else:
        requested = discover_attack_labels(df, benign_label=benign_label)

    attacks = requested

    all_label_names = _resolve_label_names(
        discover_attack_labels(df, benign_label=benign_label),
        paths.intermediate,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    label_names = _resolve_label_names(attacks, paths.intermediate)
    subphases = _loao_subphases(args)
    n_attacks = len(attacks)
    n_sub = len(subphases)
    n_total = len(all_label_names)
    loao_start = time.time()
    succeeded_this_run: list[int] = []
    failed_this_run: list[int] = []

    print(f"\n{'#' * 70}", flush=True)
    print(
        f"LOAO — {n_attacks} ataque(s) nesta execução × {n_sub} subfases "
        f"(7->8->9->10->11) -> {output_root}",
        flush=True,
    )
    print(f"Log por ataque: results/logs/loao/attack_<N>.log (cópia local em attack_<N>/loao_run.log)", flush=True)
    for attack in attacks:
        name = label_names.get(attack, f"Label={attack}")
        print(f"  - [{attack:2d}] {name}", flush=True)
    print(f"{'#' * 70}\n", flush=True)

    for attack_idx, attack in enumerate(attacks, start=1):
        subdir = output_root / f"attack_{attack}"
        subdir.mkdir(parents=True, exist_ok=True)
        attack_name = label_names.get(attack, f"Label={attack}")
        attack_log_path = subdir / "loao_run.log"
        attack_start = time.time()
        elapsed_loao = time.time() - loao_start

        with RunLog(attack_log_path) as log:
            log.emit(f"intermediate-dir: {paths.intermediate}")
            log.emit(f"amostra: {path_in}")
            log.emit(f"output-root: {output_root}")
            log.emit(
                f"[LOAO {attack_idx}/{n_attacks}] zero-day: {attack_name} (Label={attack}) "
                f"| decorrido global {_format_duration(elapsed_loao)}"
            )

            attack_report_dir = subdir / "reports"
            common = ["--intermediate-dir", str(paths.intermediate), "--report-dir", str(attack_report_dir)]
            p7 = [*common, "--work-dir", str(subdir), "--attack-label", str(attack)]
            p8 = [
                *common,
                "--work-dir",
                str(subdir),
                "--random-state",
                str(args.random_state),
                "--feature-fit-scope",
                args.feature_fit_scope,
                "--zscore-scope",
                args.zscore_scope,
                "--fcbf-k",
                str(args.fcbf_k),
                "--kpca-components",
                str(args.kpca_components),
                "--kpca-kernel",
                args.kpca_kernel,
                "--ig-cumulative",
                str(args.ig_cumulative),
                "--cv-folds",
                str(args.cv_folds),
                "--ig-hpo-calls",
                str(args.ig_hpo_calls),
                "--kpca-hpo-calls",
                str(args.kpca_hpo_calls),
            ]
            if args.optimize_ig:
                p8.append("--optimize-ig")
            if args.optimize_kpca:
                p8.append("--optimize-kpca")
            if args.benign_target is not None:
                p8 += ["--benign-target", str(args.benign_target)]

            p9: list[str] = [
                *common,
                "--work-dir",
                str(subdir),
                "--random-state",
                str(args.random_state),
            ]
            if args.no_smote:
                p9.append("--no-smote")
            elif args.smote_target is not None:
                p9 += ["--smote-target", str(args.smote_target)]

            p10: list[str] = [
                *common,
                "--work-dir",
                str(subdir),
                "--random-state",
                str(args.random_state),
                "--n-calls",
                str(args.hpo_n_calls),
                "--hpo-metric",
                args.hpo_metric,
            ]
            if args.no_smote:
                p10.append("--no-smote")
            elif args.smote_target is not None:
                p10 += ["--smote-target", str(args.smote_target)]
            if args.metrics:
                p10 += ["--metrics", args.metrics]

            p11: list[str] = [
                *common,
                "--work-dir",
                str(subdir),
                "--random-state",
                str(args.random_state),
                "--biased-mode",
                args.biased_mode,
            ]
            if args.no_smote:
                p11.append("--no-smote")
            elif args.smote_target is not None:
                p11 += ["--smote-target", str(args.smote_target)]
            if args.force_biased:
                p11.append("--force-biased")
            if args.optimize_p_star:
                p11 += ["--optimize-p-star", "--p-star-n-calls", str(args.hpo_n_calls)]

            phase_args: dict[str, list[str]] = {
                "7": p7,
                "8": p8,
                "9": p9,
                "10": p10,
                "11": p11,
            }

            attack_ok = True
            try:
                for sub_idx, (phase_num, module, desc) in enumerate(subphases, start=1):
                    log.emit(f"  -> [fase {phase_num} - {sub_idx}/{n_sub}] {desc} ...")
                    phase_elapsed = _run_phase(module, phase_args[phase_num], repo_root, log)
                    log.emit(
                        f"  OK fase {phase_num} concluida em {_format_duration(phase_elapsed)}"
                    )
            except subprocess.CalledProcessError as exc:
                attack_ok = False
                failed_this_run.append(attack)
                log.emit(
                    f"  ERRO fase {phase_num} (exit {exc.returncode}) - "
                    f"ataque {attack_name} interrompido; proximo ataque."
                )
                print(
                    f"[LOAO {attack_idx}/{n_attacks}] {attack_name} - "
                    f"falhou na fase {phase_num} (ver {attack_log_path})",
                    flush=True,
                )

            report_file = attack_report_dir / "phase11_anomaly_biased.json"
            if attack_ok and report_file.is_file():
                row = load_attack_row(subdir, attack, attack_name)
                if row is not None:
                    succeeded_this_run.append(attack)
                    dr = row.get("detection_rate")
                    far = row.get("false_alarm_rate")
                    f1 = row.get("f1")
                    metrics = []
                    if dr is not None:
                        metrics.append(f"DR={dr:.4f}")
                    if far is not None:
                        metrics.append(f"FAR={far:.4f}")
                    if f1 is not None:
                        metrics.append(f"F1={f1:.4f}")
                    metrics_str = ", ".join(metrics) if metrics else "metricas indisponiveis"
                    remaining = n_attacks - attack_idx
                    log.emit(
                        f"Resultado - {metrics_str} "
                        f"| rodada {_format_duration(time.time() - attack_start)} "
                        f"| restam {remaining} nesta execucao"
                    )
                    print(
                        f"[LOAO {attack_idx}/{n_attacks}] {attack_name} - {metrics_str} "
                        f"| log: {attack_log_path}",
                        flush=True,
                    )
            elif attack_ok:
                failed_this_run.append(attack)
                log.emit("aviso: phase11_anomaly_biased.json nao encontrado")
                print(
                    f"[LOAO {attack_idx}/{n_attacks}] {attack_name} - "
                    f"falhou (ver {attack_log_path})",
                    flush=True,
                )

        results_log = mirror_log(attack_log_path, "loao", f"attack_{attack}.log")
        if results_log:
            print(f"  log centralizado: {results_log}", flush=True)

    summary = build_loao_summary(
        output_root,
        all_label_names,
        attacks_in_dataset=n_total,
        attacks_planned=requested if args.attack_labels else attacks,
        attacks_succeeded_this_run=succeeded_this_run,
        attacks_failed_this_run=sorted(set(failed_this_run)),
    )
    out_path = write_loao_summary(output_root, summary)
    total_elapsed = time.time() - loao_start
    n_done = int(summary["n_attacks_completed"])
    f1_mean = float(summary["mean_f1"])
    dr_mean = float(summary["mean_detection_rate"])
    far_mean = float(summary["mean_false_alarm_rate"])
    print(
        f"\nLOAO: {len(succeeded_this_run)} ok nesta execucao, "
        f"{n_done}/{n_total} no resumo agregado "
        f"({_format_duration(total_elapsed)}).",
        flush=True,
    )
    if n_done:
        print(f"Media F1={f1_mean:.4f} | DR={dr_mean:.4f} | FAR={far_mean:.4f}", flush=True)
    else:
        print("Nenhum ataque concluido ainda; loao_summary.json atualizado com pendencias.", flush=True)
    if summary.get("pending_attack_labels"):
        print(f"Pendentes: {summary['pending_attack_labels']}", flush=True)
    print(f"Resumo: {out_path}", flush=True)
    write_report(paths.reports, "phase12_anomaly_loao", summary)


if __name__ == "__main__":
    main()
