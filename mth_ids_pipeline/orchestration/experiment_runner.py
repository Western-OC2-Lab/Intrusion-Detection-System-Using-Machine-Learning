"""
Orquestrador MTH-IDS.

  run_supervised  → fases 1–6 (merged no paper)
  run_anomaly     → fases 7–12 (fine no paper)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from mth_ids_pipeline.io.subprocess_env import configure_stdio_utf8, utf8_subprocess_env
from mth_ids_pipeline.config import (
    INTERMEDIATE_DIR_FINE,
    INTERMEDIATE_DIR_MERGED,
    P02_SAMPLED_KMEANS,
    P04_TEST_FSS,
    P04_TRAIN_FSS,
    P05_TEST,
    P05_TRAIN_SMOTE,
    REPORTS_DIR,
    default_loao_attack_label,
    ensure_pipeline_dirs,
    get_pipeline_paths,
)
from mth_ids_pipeline.io.results_io import make_run_log_path
from mth_ids_pipeline.io.reproducibility import DEFAULT_RANDOM_STATE, log_run_config, set_global_seeds
from mth_ids_pipeline.io.run_log import RunLog
from mth_ids_pipeline.label_profiles import LabelProfile, LabelProfileKind, get_label_profile
from mth_ids_pipeline.protocol import (
    MthIdsProtocol,
    PROTOCOL_CHOICES,
    get_protocol_settings,
    is_can_protocol,
)

SUPERVISED = frozenset(range(1, 7))
ANOMALY = frozenset(range(7, 13))
EVAL = frozenset({13})

PHASES = {
    1: "mth_ids_pipeline.phases.phase01_load_preprocess",
    2: "mth_ids_pipeline.phases.phase02_sample_kmeans",
    4: "mth_ids_pipeline.phases.phase04_feature_engineering",
    5: "mth_ids_pipeline.phases.phase05_smote",
    6: "mth_ids_pipeline.phases.phase06_supervised_models",
    7: "mth_ids_pipeline.phases.phase07_anomaly_datasets",
    8: "mth_ids_pipeline.phases.phase08_anomaly_features",
    9: "mth_ids_pipeline.phases.phase09_anomaly_cluster",
    10: "mth_ids_pipeline.phases.phase10_anomaly_cluster_hpo",
    11: "mth_ids_pipeline.phases.phase11_anomaly_biased",
    12: "mth_ids_pipeline.phases.phase12_anomaly_loao",
    13: "mth_ids_pipeline.phases.phase13_full_system_eval",
}

SUPERVISED_PHASE_LABELS: dict[int, str] = {
    1: "Carregamento e pré-processamento",
    2: "Amostragem k-means",
    4: "Engenharia de features (IG, FCBF)",
    5: "SMOTE",
    6: "Modelos supervisionados + stacking",
}

PHASE_LABELS: dict[int, str] = {
    **SUPERVISED_PHASE_LABELS,
    7: "Datasets anomaly (LOAO/global)",
    8: "Features anomaly (IG, FCBF, KPCA)",
    9: "CL-k-means",
    10: "HPO CL-k-means",
    11: "Biased classifiers B1/B2",
    12: "LOAO (orquestração)",
    13: "Avaliação sistema completo",
}


@dataclass
class ExperimentConfig:
    protocol: str = MthIdsProtocol.PAPER.value
    branch: str = "supervised"
    random_state: int = DEFAULT_RANDOM_STATE
    from_phase: int = 1
    to_phase: int = 6
    only_phase: int | None = None
    skip_phase6: bool = False
    run_hpo: bool = True
    run_loao: bool = False
    skip_bootstrap: bool = False
    skip_smote: bool = False
    skip_anomaly_smote: bool = False
    intermediate_dir: Path | None = None
    raw_csv: Path | None = None
    report_dir: Path | None = None
    profile: LabelProfile | None = None
    test_size: float = 0.2
    cv_folds: int = 0
    hpo_on_validation: bool = False
    smote_strategy: dict[int, int] = field(default_factory=dict)
    kmeans_frac: float = 0.008
    kmeans_sampling_stages: tuple[str, ...] = ()
    skip_kmeans_sampling: bool = False
    anomaly_benign_target: int | None = None
    anomaly_smote_target: int | None = None
    biased_mode: str = "both"
    force_biased: bool = False
    optimize_p_star: bool = False
    cl_hpo_metric: str = "accuracy"
    cl_hpo_metric_source: str = "notebook"
    cl_hpo_n_calls: int = 20
    cl_kmeans_metrics: tuple[str, ...] = ("euclidean", "manhattan", "cosine")
    meta_learner: str = "xgb"
    cl_p_star: float = 0.933
    optimize_ig: bool = False
    optimize_kpca: bool = False
    ig_cumulative: float = 0.9
    fcbf_k: int = 20
    fcbf_scope: str = "train"
    supervised_scale_mode: str = "split"
    feature_fit_scope: str = "combined"
    anomaly_zscore_scope: str = "combined"
    kpca_components: int = 10
    kpca_kernel: str = "rbf"
    hpo_n_calls: int = 15
    loao_attack_labels: str | None = None
    fixed_supervised_features: tuple[str, ...] | None = None
    phase1_zscore: bool = True
    post_sample_zscore: bool = False

    @classmethod
    def from_protocol(
        cls,
        protocol: str,
        *,
        branch: str = "supervised",
        label_profile: str | None = None,
        **kw,
    ) -> ExperimentConfig:
        ps = get_protocol_settings(protocol)
        profile_name = label_profile or (
            ps.anomaly_profile if branch == "anomaly" else ps.supervised_profile
        )
        profile = get_label_profile(profile_name)
        cfg = cls(
            protocol=ps.name,
            branch=branch,
            test_size=ps.test_size,
            cv_folds=ps.cv_folds,
            hpo_on_validation=ps.hpo_on_validation,
            smote_strategy=dict(ps.smote_targets),
            skip_smote=ps.skip_smote,
            skip_anomaly_smote=ps.skip_anomaly_smote,
            kmeans_frac=ps.kmeans_frac,
            skip_kmeans_sampling=ps.skip_kmeans_sampling,
            anomaly_benign_target=ps.anomaly_benign_target,
            anomaly_smote_target=ps.anomaly_smote_target,
            biased_mode=ps.biased_mode,
            force_biased=ps.force_biased,
            optimize_p_star=ps.optimize_p_star,
            cl_hpo_metric=ps.cl_hpo_metric,
            cl_hpo_metric_source=ps.cl_hpo_metric_source,
            cl_hpo_n_calls=ps.cl_hpo_n_calls,
            cl_kmeans_metrics=ps.cl_kmeans_metrics,
            meta_learner=ps.meta_learner,
            cl_p_star=ps.cl_p_star,
            optimize_ig=ps.optimize_ig,
            optimize_kpca=ps.optimize_kpca,
            ig_cumulative=ps.ig_cumulative,
            fcbf_k=ps.fcbf_k,
            fcbf_scope=ps.fcbf_scope,
            supervised_scale_mode=ps.supervised_scale_mode,
            feature_fit_scope=ps.feature_fit_scope,
            anomaly_zscore_scope=ps.anomaly_zscore_scope,
            kpca_components=ps.kpca_components,
            kpca_kernel=ps.kpca_kernel,
            hpo_n_calls=ps.hpo_n_calls,
            fixed_supervised_features=ps.fixed_supervised_features,
            phase1_zscore=ps.phase1_zscore,
            post_sample_zscore=ps.post_sample_zscore,
            intermediate_dir=profile.intermediate_dir,
            raw_csv=profile.raw_csv,
            profile=profile,
        )
        for k, v in kw.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        return cfg


def _reports(cfg: ExperimentConfig) -> Path:
    if cfg.report_dir:
        return cfg.report_dir
    if cfg.intermediate_dir:
        return cfg.intermediate_dir / "phase_reports"
    return REPORTS_DIR


def _resolve_phase7_attack_label(cfg: ExperimentConfig) -> int:
    """Zero-day da fase 7: CLI > default por dataset (DoS no CAN, PortScan no CICIDS)."""
    if cfg.loao_attack_labels:
        first = cfg.loao_attack_labels.split(",")[0].strip()
        if first:
            return int(first)
    return default_loao_attack_label(
        intermediate_dir=cfg.intermediate_dir,
        protocol=cfg.protocol,
    )


def _phase_args(phase: int, cfg: ExperimentConfig) -> list[str]:
    p = cfg.profile
    assert p is not None
    extra = [
        "--report-dir", str(_reports(cfg)),
        "--intermediate-dir", str(cfg.intermediate_dir),
    ]
    if phase == 1 and cfg.raw_csv:
        extra += ["--input", str(cfg.raw_csv)]
        if not cfg.phase1_zscore:
            extra.append("--no-zscore")
    if phase == 2:
        extra += [
            "--n-clusters", "1000",
            "--frac", str(cfg.kmeans_frac),
            "--random-state", str(cfg.random_state),
        ]
        if cfg.kmeans_sampling_stages:
            for stage in cfg.kmeans_sampling_stages:
                extra += ["--sampling-stage", stage]
        elif cfg.skip_kmeans_sampling:
            extra.append("--skip-sampling")
        elif getattr(p, "kmeans_sample_all_classes", False):
            extra.append("--sample-all-classes")
        elif p.auto_minority:
            extra.append("--auto-minority")
        elif p.minority_labels:
            extra += ["--minority-labels", p.minority_labels_csv() or ""]
        if cfg.post_sample_zscore:
            extra.append("--zscore-after-sample")
        else:
            extra.append("--no-zscore-after-sample")
    if phase == 4:
        extra += [
            "--fcbf-k", str(cfg.fcbf_k),
            "--random-state", str(cfg.random_state),
            "--test-size", str(cfg.test_size),
            "--fcbf-scope", cfg.fcbf_scope,
            "--scale-mode", cfg.supervised_scale_mode,
            "--ig-cumulative", str(cfg.ig_cumulative),
            "--cv-folds", str(cfg.cv_folds or 10),
            "--ig-hpo-calls", str(cfg.hpo_n_calls),
        ]
        if cfg.optimize_ig:
            extra.append("--optimize-ig")
        if cfg.fixed_supervised_features:
            extra += ["--ig-features", ",".join(cfg.fixed_supervised_features)]
    if phase == 5:
        extra += ["--random-state", str(cfg.random_state)]
        if cfg.smote_strategy:
            extra += ["--smote-strategy", json.dumps({str(k): v for k, v in cfg.smote_strategy.items()})]
    if phase == 6:
        if not cfg.run_hpo:
            extra.append("--no-hpo")
        extra += [
            "--no-plots",
            "--cv-folds",
            str(cfg.cv_folds),
            "--meta-learner",
            cfg.meta_learner,
            "--random-state",
            str(cfg.random_state),
        ]
        if cfg.hpo_on_validation:
            extra.append("--hpo-on-validation")
    if phase == 7:
        extra += ["--attack-label", str(_resolve_phase7_attack_label(cfg))]
    if phase == 8:
        extra += [
            "--random-state", str(cfg.random_state),
            "--fcbf-k", str(cfg.fcbf_k),
            "--kpca-components", str(cfg.kpca_components),
            "--kpca-kernel", cfg.kpca_kernel,
            "--ig-cumulative", str(cfg.ig_cumulative),
            "--feature-fit-scope", cfg.feature_fit_scope,
            "--zscore-scope", cfg.anomaly_zscore_scope,
            "--cv-folds", str(cfg.cv_folds or 10),
            "--ig-hpo-calls", str(cfg.hpo_n_calls),
            "--kpca-hpo-calls", str(cfg.hpo_n_calls),
        ]
        if cfg.optimize_ig:
            extra.append("--optimize-ig")
        if cfg.optimize_kpca:
            extra.append("--optimize-kpca")
        if cfg.anomaly_benign_target is not None:
            extra += ["--benign-target", str(cfg.anomaly_benign_target)]
    if phase == 9:
        extra += [
            "--n-clusters", "8",
            "--random-state", str(cfg.random_state),
        ]
        if cfg.skip_anomaly_smote:
            extra.append("--no-smote")
        elif cfg.anomaly_smote_target is not None:
            extra += ["--smote-target", str(cfg.anomaly_smote_target)]
    if phase == 10:
        extra += [
            "--random-state", str(cfg.random_state),
            "--n-calls", str(cfg.cl_hpo_n_calls),
            "--hpo-metric", cfg.cl_hpo_metric,
            "--hpo-metric-source", cfg.cl_hpo_metric_source,
            "--metrics", ",".join(cfg.cl_kmeans_metrics),
        ]
        if cfg.skip_anomaly_smote:
            extra.append("--no-smote")
        elif cfg.anomaly_smote_target is not None:
            extra += ["--smote-target", str(cfg.anomaly_smote_target)]
        if not cfg.run_hpo:
            extra.append("--skip-hpo")
    if phase == 11:
        extra += [
            "--p-star", str(cfg.cl_p_star),
            "--random-state", str(cfg.random_state),
            "--biased-mode", cfg.biased_mode,
        ]
        if cfg.skip_anomaly_smote:
            extra.append("--no-smote")
        elif cfg.anomaly_smote_target is not None:
            extra += ["--smote-target", str(cfg.anomaly_smote_target)]
        if cfg.force_biased:
            extra.append("--force-biased")
        if cfg.optimize_p_star:
            extra += ["--optimize-p-star", "--p-star-n-calls", str(cfg.cl_hpo_n_calls)]
    if phase == 12:
        extra += [
            "--random-state", str(cfg.random_state),
            "--hpo-n-calls", str(cfg.cl_hpo_n_calls),
            "--hpo-metric", cfg.cl_hpo_metric,
            "--biased-mode", cfg.biased_mode,
            "--feature-fit-scope", cfg.feature_fit_scope,
            "--zscore-scope", cfg.anomaly_zscore_scope,
            "--fcbf-k", str(cfg.fcbf_k),
            "--kpca-components", str(cfg.kpca_components),
            "--kpca-kernel", cfg.kpca_kernel,
            "--ig-cumulative", str(cfg.ig_cumulative),
            "--cv-folds", str(cfg.cv_folds or 10),
            "--ig-hpo-calls", str(cfg.hpo_n_calls),
            "--kpca-hpo-calls", str(cfg.hpo_n_calls),
            "--metrics", ",".join(cfg.cl_kmeans_metrics),
        ]
        if cfg.skip_anomaly_smote:
            extra.append("--no-smote")
        elif cfg.anomaly_smote_target is not None:
            extra += ["--smote-target", str(cfg.anomaly_smote_target)]
        if cfg.optimize_ig:
            extra.append("--optimize-ig")
        if cfg.optimize_kpca:
            extra.append("--optimize-kpca")
        if cfg.force_biased:
            extra.append("--force-biased")
        if cfg.optimize_p_star:
            extra.append("--optimize-p-star")
        if cfg.anomaly_benign_target is not None:
            extra += ["--benign-target", str(cfg.anomaly_benign_target)]
        if cfg.loao_attack_labels:
            extra += ["--attack-labels", cfg.loao_attack_labels]
    if phase == 13:
        sup_dir = _resolve_table_vii_dir(cfg)
        extra += [
            "--random-state", str(cfg.random_state),
            "--test-size", str(cfg.test_size),
            "--no-plots",
            "--intermediate-dir", str(sup_dir),
        ]
        if cfg.intermediate_dir and cfg.intermediate_dir != sup_dir:
            extra += ["--work-dir", str(cfg.intermediate_dir)]
    return extra


def _resolve_table_vii_dir(cfg: ExperimentConfig) -> Path:
    if cfg.profile and cfg.profile.paired_supervised_dir:
        return cfg.profile.paired_supervised_dir
    if cfg.intermediate_dir and cfg.profile and cfg.profile.kind == LabelProfileKind.MERGED:
        return cfg.intermediate_dir
    return INTERMEDIATE_DIR_MERGED


def _resolve_table_vii_profile_name(cfg: ExperimentConfig) -> str:
    if cfg.profile and cfg.profile.table_vii_profile:
        return cfg.profile.table_vii_profile
    return "merged"


def _link_fcbf_to_smote_paths(intermediate_dir: Path) -> None:
    """CAN / sem SMOTE: fase 6 lê ``05_*`` espelhando a saída da fase 4."""
    src_train = intermediate_dir / P04_TRAIN_FSS
    src_test = intermediate_dir / P04_TEST_FSS
    if not src_train.is_file() or not src_test.is_file():
        raise FileNotFoundError(
            f"Fase 4 incompleta em {intermediate_dir}: "
            f"esperado {P04_TRAIN_FSS} e {P04_TEST_FSS}"
        )
    shutil.copy2(src_train, intermediate_dir / P05_TRAIN_SMOTE)
    shutil.copy2(src_test, intermediate_dir / P05_TEST)
    print(
        f"[supervised] Sem SMOTE: {P04_TRAIN_FSS} → {P05_TRAIN_SMOTE}, "
        f"{P04_TEST_FSS} → {P05_TEST}"
    )


def _sampled_kmeans_path(intermediate_dir: Path) -> Path:
    return intermediate_dir / P02_SAMPLED_KMEANS.replace(".csv", ".parquet")


def _supervised_metrics_path(intermediate_dir: Path) -> Path:
    return intermediate_dir / "06_supervised_metrics.json"


def _bootstrap_supervised(
    cfg: ExperimentConfig,
    start: int,
    end: int,
    *,
    intermediate_dir: Path | None = None,
    profile: LabelProfile | None = None,
) -> None:
    """Executa fases supervisionadas ``start``–``end`` (pula fase 3)."""
    updates: dict = {
        "branch": "supervised",
        "from_phase": start,
        "to_phase": end,
        "only_phase": None,
        "run_loao": False,
    }
    if intermediate_dir is not None:
        updates["intermediate_dir"] = intermediate_dir
    if profile is not None:
        updates["profile"] = profile
        updates["raw_csv"] = profile.raw_csv
    bootstrap = replace(cfg, **updates)
    _run_phase_range(bootstrap, start, end)


def _ensure_merged_table_vii(cfg: ExperimentConfig) -> Path:
    """Gera ``06_supervised_metrics.json`` no merged pareado (Tabela VII / VI CAN)."""
    merged_dir = _resolve_table_vii_dir(cfg)
    merged_metrics = _supervised_metrics_path(merged_dir)
    if merged_metrics.is_file():
        return merged_metrics

    profile_name = _resolve_table_vii_profile_name(cfg)
    print(
        f"\n[anomaly] Tabela VII ausente: bootstrap fases 1–6 em {merged_dir} "
        f"(perfil {profile_name} — melhor learner para biased tier 4).\n"
    )
    merged_profile = get_label_profile(profile_name)
    _bootstrap_supervised(
        cfg,
        1,
        6,
        intermediate_dir=merged_dir,
        profile=merged_profile,
    )
    if not merged_metrics.is_file():
        raise FileNotFoundError(
            f"Não foi possível gerar {merged_metrics}.\n"
            f"Execute manualmente:\n"
            f"  python -m mth_ids_pipeline.run_supervised --protocol {cfg.protocol} "
            f"--intermediate-dir {merged_dir}"
        )
    return merged_metrics


def _copy_supervised_metrics_for_biased(src: Path, dest: Path) -> None:
    """Copia métricas da Tabela VII para a pasta fine (fase 11 lê ``dest``)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"[anomaly] 06_supervised_metrics.json ← {src} (Tabela VII → tier 4 biased)")


def ensure_anomaly_prerequisites(cfg: ExperimentConfig) -> None:
    """
    Tabela IX — pré-requisitos antes das fases 7–12.

    Perfil **fine** (paper):
      - ``02_sampled_kmeans.parquet``: fases 1–2 no fine (``frac=0.008``,
        ``FINE_DEFAULT_MINORITY_LABELS`` — escala ~notebook)
      - ``06_supervised_metrics.json``: Tabela VII no **merged** (copiado para fine)

    Perfil **merged** (notebook / demo): bootstrap 1–6 ou 4–6 na mesma pasta.
    """
    assert cfg.intermediate_dir is not None
    assert cfg.profile is not None

    fine_dir = cfg.intermediate_dir
    sampled = _sampled_kmeans_path(fine_dir)
    metrics = _supervised_metrics_path(fine_dir)
    if sampled.is_file() and metrics.is_file():
        return

    if cfg.profile.kind == LabelProfileKind.FINE:
        if not sampled.is_file():
            print(
                f"\n[anomaly] Bootstrap fases 1–2 (fine) em {fine_dir} "
                f"— amostra k-means para LOAO.\n"
            )
            _bootstrap_supervised(cfg, 1, 2)

        if not metrics.is_file():
            merged_metrics = _ensure_merged_table_vii(cfg)
            _copy_supervised_metrics_for_biased(merged_metrics, metrics)
        return

    if not sampled.is_file():
        start, end = 1, 6
    else:
        start, end = 4, 6

    print(
        f"\n[anomaly] Pré-requisitos em {fine_dir}: "
        f"bootstrap automático das fases {start}–{end} (perfil merged).\n"
    )
    _bootstrap_supervised(cfg, start, end)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _phases_to_run(cfg: ExperimentConfig, start: int, end: int) -> list[int]:
    phases: list[int] = []
    # LOAO: fase 12 reexecuta 7–11 por ataque; 7–11 globais usam zero-day errado no CAN.
    skip_pre_loao = cfg.run_loao and end >= 12
    for phase in range(start, end + 1):
        if phase == 3:
            continue
        if phase == 6 and cfg.skip_phase6:
            continue
        if phase == 5 and cfg.skip_smote:
            continue
        if phase == 12 and not cfg.run_loao:
            continue
        if skip_pre_loao and 7 <= phase <= 11:
            continue
        if phase in PHASES:
            phases.append(phase)
    return phases


def _run_log_stem(cfg: ExperimentConfig, start: int, end: int) -> str:
    profile = cfg.profile.kind.value if cfg.profile else "custom"
    return f"{cfg.branch}_{profile}_{cfg.protocol}_phases{start}-{end}"


def _run_phase_range(cfg: ExperimentConfig, start: int, end: int) -> None:
    root = Path(__file__).resolve().parents[2]
    phases = _phases_to_run(cfg, start, end)
    if not phases:
        return

    log_path = make_run_log_path(_run_log_stem(cfg, start, end))
    profile_kind = cfg.profile.kind.value if cfg.profile else "?"

    def _run_one(phase: int, phase_idx: int, n_phases: int, log: RunLog) -> None:
        if phase == 12:
            log.emit(
                ">>> Fase 12 — LOAO (leave-one-attack-out; pode levar muitas horas)"
            )
        cmd = [sys.executable, "-m", PHASES[phase], *_phase_args(phase, cfg)]
        desc = PHASE_LABELS.get(phase, f"fase {phase}")
        log.emit(f"-> [fase {phase} - {phase_idx}/{n_phases}] {desc} ...")
        elapsed = log.run_subprocess(cmd, cwd=root)
        if phase == 4 and cfg.skip_smote and cfg.intermediate_dir:
            _link_fcbf_to_smote_paths(cfg.intermediate_dir)
            log.emit(
                f"Sem SMOTE: parquets FCBF copiados para caminhos da fase 5 "
                f"em {cfg.intermediate_dir}"
            )
        log.emit(f"OK fase {phase} concluida em {_format_duration(elapsed)}")

    with RunLog(log_path) as log:
        log.emit(f"intermediate-dir: {cfg.intermediate_dir}")
        log.emit(f"protocolo: {cfg.protocol} | ramo: {cfg.branch} | perfil: {profile_kind}")
        log.emit(f"fases: {start}-{end} ({', '.join(str(p) for p in phases)})")
        if cfg.run_loao and end >= 12 and start <= 11:
            log.emit(
                "LOAO: fases 7–11 globais omitidas; a fase 12 executa 7–11 por ataque "
                "em anomaly/loao/attack_<N>/"
            )
        n_phases = len(phases)
        for phase_idx, phase in enumerate(phases, start=1):
            _run_one(phase, phase_idx, n_phases, log)
    print(f"Log da execução: {log_path}", flush=True)


def run_experiment(cfg: ExperimentConfig) -> None:
    configure_stdio_utf8()
    set_global_seeds(cfg.random_state)
    if cfg.intermediate_dir:
        ensure_pipeline_dirs(get_pipeline_paths(cfg.intermediate_dir, _reports(cfg)))

    ps = get_protocol_settings(cfg.protocol)
    log_run_config(
        _reports(cfg),
        run_name="experiment_runner",
        config={**asdict(cfg), "protocol_description": ps.description},
    )
    print(f"Protocolo: {cfg.protocol} | ramo: {cfg.branch} | perfil: {cfg.profile.kind.value}")
    print(f"Artefatos: {cfg.intermediate_dir}")

    if cfg.branch == "anomaly" and not cfg.skip_bootstrap:
        ensure_anomaly_prerequisites(cfg)

    start = cfg.only_phase or cfg.from_phase
    end = cfg.only_phase or cfg.to_phase
    _run_phase_range(cfg, start, end)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--protocol", choices=list(PROTOCOL_CHOICES), default="paper")
    p.add_argument(
        "--label-profile",
        choices=[
            "merged",
            "fine",
            "can_merged",
            "can_fine",
            "can_intrusion_merged",
            "can_intrusion_fine",
            "can_otids_merged",
            "can_otids_fine",
        ],
        default=None,
    )
    p.add_argument("--from", dest="from_phase", type=int, default=1)
    p.add_argument("--to", type=int, default=6)
    p.add_argument("--only", type=int, default=None)
    p.add_argument("--intermediate-dir", type=Path, default=None)
    p.add_argument("--raw-csv", type=Path, default=None)
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument(
        "--kmeans-stage",
        action="append",
        default=[],
        metavar="LABELS:FRAC",
        help=(
            "Fase 2: amostragem k-means em estagios. "
            "Ex.: --kmeans-stage 3,7:0.008 --kmeans-stage 5,6,4,8:0.10"
        ),
    )
    p.add_argument("--no-hpo", action="store_true")
    p.add_argument("--skip-phase6", action="store_true")
    p.add_argument(
        "--skip-smote",
        action="store_true",
        help="Pula fase 5; copia parquets FCBF (fase 4) para caminhos 05_*",
    )
    p.add_argument("--loao", action="store_true")
    p.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Anomaly: não preparar pré-requisitos (02_ fine / 06_ merged→fine)",
    )
    p.add_argument(
        "--attack-label",
        type=int,
        default=None,
        help="LOAO: um zero-day (LabelEncoder); default DoS (CAN) ou PortScan (CICIDS)",
    )
    p.add_argument(
        "--attack-labels",
        type=str,
        default=None,
        help="LOAO fase 12: subset, ex. 1,5,10",
    )
    return p


def config_from_args(args: argparse.Namespace, *, branch: str) -> ExperimentConfig:
    cfg = ExperimentConfig.from_protocol(
        args.protocol,
        branch=branch,
        label_profile=args.label_profile,
        from_phase=args.from_phase,
        to_phase=args.to,
        only_phase=args.only,
        random_state=args.random_state,
        kmeans_sampling_stages=tuple(args.kmeans_stage or ()),
        run_loao=args.loao or args.to >= 12,
    )
    if args.intermediate_dir:
        cfg.intermediate_dir = args.intermediate_dir
    if args.raw_csv:
        cfg.raw_csv = args.raw_csv
    if args.no_hpo:
        cfg.run_hpo = False
    if args.skip_phase6:
        cfg.skip_phase6 = True
    if args.skip_smote:
        cfg.skip_smote = True
    if args.skip_bootstrap:
        cfg.skip_bootstrap = True
    if args.attack_label is not None and args.attack_labels:
        raise SystemExit("Use apenas --attack-label ou --attack-labels, não ambos.")
    if args.attack_label is not None:
        cfg.loao_attack_labels = str(args.attack_label)
    elif args.attack_labels:
        cfg.loao_attack_labels = args.attack_labels
    elif branch == "eval" and args.intermediate_dir is None:
        cfg.intermediate_dir = INTERMEDIATE_DIR_FINE
    return cfg


def main() -> None:
    args = build_arg_parser("MTH-IDS").parse_args()
    if args.loao:
        args.to = max(args.to, 12)
    if args.from_phase >= 13 or args.to >= 13 or args.only == 13:
        branch = "eval"
    elif args.from_phase >= 7 or args.loao or args.to >= 7:
        branch = "anomaly"
    else:
        branch = "supervised"
    run_experiment(config_from_args(args, branch=branch))


if __name__ == "__main__":
    main()
