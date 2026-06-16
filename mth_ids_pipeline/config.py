"""Caminhos e nomes de arquivos intermediários do pipeline MTH-IDS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"
RESULTS_CONFIG_DIR = RESULTS_DIR / "config"
INTERMEDIATE_DIR = DATA_DIR / "pipeline_mth_ids"
ANOMALY_DIR = INTERMEDIATE_DIR / "anomaly"
REPORTS_DIR = INTERMEDIATE_DIR / "phase_reports"

DEFAULT_MINORITY_LABELS: tuple[int, ...] = (6, 1, 4)

# Fine: ver ``label_profiles.compute_fine_minority_labels_notebook_aligned()`` (Bot/Infiltration/WebAttack)
CICIDS2017_FINE_LABEL_NAMES: dict[int, str] = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack Brute Force",
    13: "Web Attack Sql Injection",
    14: "Web Attack XSS",
}

# CICIDS2017 merged — LabelEncoder alfabético:
# 0=BENIGN 1=Bot 2=BruteForce 3=DoS 4=Infiltration 5=PortScan 6=WebAttack
CICIDS2017_MERGED_LABEL_NAMES: dict[int, str] = {
    0: "BENIGN",
    1: "Bot",
    2: "BruteForce",
    3: "DoS",
    4: "Infiltration",
    5: "PortScan",
    6: "WebAttack",
}

# SMOTE supervisionado — notebook MTH_IDS_IoTJ.ipynb
NOTEBOOK_SMOTE_TARGETS: dict[int, int] = {
    2: 1_000,  # BruteForce
    4: 1_000,  # Infiltration
}

# SMOTE supervisionado — artigo original (Yang et al. 2022); protocolo paper usa NOTEBOOK_SMOTE_TARGETS
PAPER_SMOTE_TARGETS: dict[int, int] = {
    1: 100_000,  # Bot
    2: 100_000,  # BruteForce
    4: 100_000,  # Infiltration
    6: 100_000,  # WebAttack
}

DEFAULT_TEST_SIZE = 0.2  # notebook IoTJ: split 80/20 na engenharia de features
PAPER_TEST_SIZE = 0.3  # artigo Sec. IV-F: 70% treino / 30% teste
DEFAULT_KMEANS_FRAC = 0.008  # notebook IoTJ / CICIDS2017 (fase 2)
NOTEBOOK_KMEANS_FRAC = DEFAULT_KMEANS_FRAC
PAPER_KMEANS_FRAC = DEFAULT_KMEANS_FRAC  # CICIDS2017 paper; CAN usa CAN_PAPER_KMEANS_FRAC
DEFAULT_CV_FOLDS = 0  # notebook: HPO por acurácia no hold-out
PAPER_CV_FOLDS = 10
DEFAULT_META_LEARNER = "xgb"  # notebook: stacking meta XGBoost + HPO
PAPER_META_LEARNER = "best-base"  # artigo tier 2: clone do melhor base (maior F1 no hold-out)
DEFAULT_SMOTE_TARGETS = NOTEBOOK_SMOTE_TARGETS
DEFAULT_HPO_ON_VALIDATION = False  # notebook: accuracy_score(y_test, y_pred)
PAPER_HPO_ON_VALIDATION = True  # artigo Sec. IV-F: HPO por acurácia em validação
PAPER_IG_CUMULATIVE = 0.9
PAPER_FCBF_K = 20
PAPER_FCBF_ALPHA = 0.01  # FCBF (th) — classe FCBF do módulo; preset paper usa k=20
PAPER_KPCA_COMPONENTS = 10
PAPER_KPCA_KERNEL = "rbf"
PAPER_FEATURE_FIT_SCOPE = "combined"  # anomaly tier 3: IG/FCBF/KPCA no conjunto combinado
PAPER_FCBF_SCOPE = "train"  # supervisionado: FCBF só no treino
PAPER_SUPERVISED_SCALE = "split"  # StandardScaler após split (artigo)
NOTEBOOK_SUPERVISED_SCALE = "phase1"  # Z-score só da fase 1 (IoTJ)
NOTEBOOK_ANOMALY_ZSCORE = "per_split"  # Z-score em df1/df2 separados (IoTJ)
PAPER_ANOMALY_ZSCORE = "combined"  # Z-score no conjunto concatenado
PAPER_HPO_N_CALLS = 15
DEFAULT_BIASED_MODE = "both"  # tier 4: B1 + B2
# Notebook IoTJ: SMOTE(sampling_strategy={1: N}) com N = nº de BENIGN no treino df1 (18225 no demo PortScan).
DEFAULT_ANOMALY_SMOTE_TARGET: int | None = None
NOTEBOOK_ANOMALY_SMOTE_DEMO_BENIGN = 18225
DEFAULT_CL_P_STAR = 0.933


@dataclass(frozen=True)
class PipelinePaths:
    intermediate: Path
    anomaly: Path
    reports: Path


def parse_minority_labels(value: str) -> tuple[int, ...]:
    labels = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not labels:
        raise ValueError("minority-labels não pode ser vazio")
    return labels


def get_pipeline_paths(
    intermediate_dir: Path | None = None,
    report_dir: Path | None = None,
) -> PipelinePaths:
    inter = Path(intermediate_dir) if intermediate_dir is not None else INTERMEDIATE_DIR
    reports = Path(report_dir) if report_dir is not None else inter / "phase_reports"
    return PipelinePaths(intermediate=inter, anomaly=inter / "anomaly", reports=reports)


def ensure_pipeline_dirs(paths: PipelinePaths) -> None:
    paths.intermediate.mkdir(parents=True, exist_ok=True)
    paths.anomaly.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)

# Entrada bruta CICIDS2017 (ver label_profiles.py: merged vs fine)
DEFAULT_RAW_CSV = DATA_DIR / "CICIDS2017.csv"
DEFAULT_RAW_CSV_FINE = DATA_DIR / "CICIDS2017_fine.csv"
INTERMEDIATE_DIR_MERGED = DATA_DIR / "pipeline_mth_ids_merged"
INTERMEDIATE_DIR_FINE = DATA_DIR / "pipeline_mth_ids_fine"

# CAN — Car-Hacking (intrusion) e repack OTIDS em pastas distintas
CAN_OTIDS_DATA_DIR = DATA_DIR / "CAN_OTIDS_DATA"
CAN_INTRUSION_DATA_DIR = DATA_DIR  # Car-Hacking: CAN_normal_run_data.txt, CAN_*_dataset.csv

# Car-Hacking original (artigo Tabela IV — 5 classes)
DEFAULT_RAW_CSV_CAN_INTRUSION = DATA_DIR / "CAN_intrusion_Dataset.csv"
CAN_INTRUSION_DATASET_META = DATA_DIR / "can_intrusion_meta.json"
INTERMEDIATE_DIR_CAN_INTRUSION_MERGED = DATA_DIR / "pipeline_can_intrusion_merged"
INTERMEDIATE_DIR_CAN_INTRUSION_FINE = DATA_DIR / "pipeline_can_intrusion_fine"
RESULTS_DIR_CAN_INTRUSION = RESULTS_DIR / "can_intrusion"

# Repack OTIDS (4 classes)
DEFAULT_RAW_CSV_CAN_OTIDS = DATA_DIR / "CAN_OTIDS_Dataset.csv"
CAN_OTIDS_DATASET_META = DATA_DIR / "can_otids_meta.json"
INTERMEDIATE_DIR_CAN_OTIDS_MERGED = DATA_DIR / "pipeline_can_otids_merged"
INTERMEDIATE_DIR_CAN_OTIDS_FINE = DATA_DIR / "pipeline_can_otids_fine"
RESULTS_DIR_CAN_OTIDS = RESULTS_DIR / "can_otids"

# Aliases legados (preset ``can`` / ``can_paper`` → intrusion)
DEFAULT_RAW_CSV_CAN = DEFAULT_RAW_CSV_CAN_INTRUSION
INTERMEDIATE_DIR_CAN_MERGED = INTERMEDIATE_DIR_CAN_INTRUSION_MERGED
INTERMEDIATE_DIR_CAN_FINE = INTERMEDIATE_DIR_CAN_INTRUSION_FINE
RESULTS_DIR_CAN = RESULTS_DIR_CAN_INTRUSION
CAN_DATASET_META = CAN_INTRUSION_DATASET_META
CAN_KMEANS_FRAC = 0.008  # notebook / CICIDS2017 (can_notebook)
CAN_TEST_SIZE = DEFAULT_TEST_SIZE  # alias legado (80/20)
# CAN paper: 1% (0,01); notebook/legado 0,8%; artigo Tabela VI cita 10%
CAN_PAPER_KMEANS_FRAC = 0.01
CAN_PAPER_TEST_SIZE = 0.3

# UNSW-NB15 — rede externa (ver docs/PROTOCOLO_UNSW_NB15.md)
DEFAULT_RAW_CSV_UNSW_NB15 = DATA_DIR / "UNSW-NB15_merged.csv"
UNSW_NB15_DATASET_META = DATA_DIR / "unsw_nb15_meta.json"
INTERMEDIATE_DIR_UNSW_NB15_MERGED = DATA_DIR / "pipeline_unsw_nb15_merged"
INTERMEDIATE_DIR_UNSW_NB15_FINE = DATA_DIR / "pipeline_unsw_nb15_fine"
RESULTS_DIR_UNSW_NB15 = RESULTS_DIR / "unsw_nb15"
UNSW_NB15_KMEANS_FRAC = 0.008  # Benign: k-means 15%; ataques preservados intactos

# LabelEncoder alfabético — UNSW-NB15 merged (Benign = 3 neste workspace)
UNSW_NB15_LABEL_NAMES: dict[int, str] = {
    0: "Analysis",
    1: "Backdoors",
    2: "Backdoor",
    3: "Benign",
    4: "DoS",
    5: "Exploits",
    6: "Fuzzers",
    7: "Generic",
    8: "Reconnaissance",
    9: "Shellcode",
    10: "Worms",
}

# Fase 2: classes preservadas intactas; só Benign passa pelo k-means
UNSW_NB15_PRESERVED_ATTACK_LABELS: tuple[int, ...] = (4, 5, 6, 7, 8)

# Fase 5: SMOTE supervisionado (docs/PROTOCOLO_UNSW_NB15.md)
UNSW_NB15_SMOTE_TARGETS: dict[int, int] = {
    2: 1000,
    4: 1000,
    6: 1200,
    8: 1000,
    10: 500,
}
# Tabela VI CAN — 4 features citadas no artigo (referência; preset can_paper usa BO-GP α IG)
CAN_PAPER_IG_FEATURES: tuple[str, ...] = ("CAN_ID", "DATA_1", "DATA_3", "DATA_5")

# LabelEncoder alfabético — Car-Hacking original (5 classes, Tabela IV)
CAN_INTRUSION_LABEL_NAMES: dict[int, str] = {
    0: "BENIGN",
    1: "DoS",
    2: "Fuzzy",
    3: "Gear",
    4: "RPM",
}

# Repack OTIDS (4 classes; Impersonation já unificado no .txt)
CAN_OTIDS_LABEL_NAMES: dict[int, str] = {
    0: "BENIGN",
    1: "DoS",
    2: "Fuzzy",
    3: "Impersonation",
}

CAN_LABEL_NAMES = CAN_OTIDS_LABEL_NAMES  # alias legado (OTIDS)
CAN_DATASET_META = DATA_DIR / "can_dataset_meta.json"

# Zero-day padrão quando fase 7 roda isolada (demo / debug)
DEFAULT_LOAO_ATTACK_LABEL_CICIDS = 5  # PortScan (fine CICIDS2017)
DEFAULT_LOAO_ATTACK_LABEL_CAN = 1  # DoS (CAN_LABEL_NAMES)
DEFAULT_LOAO_ATTACK_LABEL_UNSW = 0  # primeira classe de ataque no UNSW-NB15 encodado

# Log de sessão do ramo supervisionado (experiment_runner / run_supervised)
SUPERVISED_RUN_LOG = "supervised_run.log"  # legado; logs novos vão em results/logs/


def ensure_results_dirs() -> None:
    """Garante pastas de saída: tabelas, logs de execução e configs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Fase supervisionada (known attacks)
P01_PREPROCESSED = "01_preprocessed.csv"
P02_SAMPLED_KMEANS = "02_sampled_kmeans.csv"
P03_TRAIN = "03_train.csv"
P03_TEST = "03_test.csv"
P04_TRAIN_FSS = "04_train_after_fcbf.parquet"
P04_TEST_FSS = "04_test_after_fcbf.parquet"
P04_SELECTED_FEATURES = "04_selected_features.txt"
P05_TRAIN_SMOTE = "05_train_after_smote.parquet"
P05_TEST = "05_test_unchanged.parquet"

# Ramo anomaly-based
A00_LOAO_ROUND = "a00_loao_round.json"
A01_WITHOUT_PORTSCAN = "a01_without_portscan.parquet"
A02_PORTSCAN_ONLY = "a02_portscan_only.parquet"
A03_COMBINED_NORMALIZED = "a03_combined_normalized.parquet"
A04_AFTER_KPCA = "a04_after_kpca.parquet"
A05_TRAIN_SMOTE = "a05_train_after_smote.parquet"
A06_TEST_SLICE_INFO = "a06_test_slice.json"

# Artefatos de inferência (fases 4/6/11 salvam; fase 13 carrega)
MODELS_DIR = "models"
MODELS_SUPERVISED_DIR = "models/supervised"
MODELS_ANOMALY_DIR = "models/anomaly"
MODEL_MANIFEST = "manifest.json"
SUP_SCALER = "scaler.joblib"
SUP_FCBF = "fcbf.joblib"
SUP_FEATURE_NAMES = "feature_names.txt"
SUP_IG_FEATURES = "ig_features.txt"
SUP_DT = "dt.joblib"
SUP_RF = "rf.joblib"
SUP_ET = "et.joblib"
SUP_XGB = "xgb.joblib"
SUP_STACKING_META = "stacking_meta.joblib"
ANOM_CL_STATE = "cl_kmeans_state.joblib"
ANOM_B1 = "b1.joblib"
ANOM_B2 = "b2.joblib"
DEFAULT_ANOMALY_ATTACK_PRED_LABEL = 99
GLOBAL_TABLE_X_PROTOCOL = "global_table_x"
ANOMALY_GLOBAL_WORK_SUBDIR = "anomaly/global"
PAPER_TABLE_X_REFERENCE = {
    "cicids2017": {
        "accuracy_pct": 99.88,
        "detection_rate_pct": 99.77,
        "false_alarm_rate_pct": 0.10,
        "f1": 0.9988,
    },
    "can": {
        "accuracy_pct": 99.99,
        "detection_rate_pct": 100.0,
        "false_alarm_rate_pct": 0.00005,
        "f1": 0.9999,
    },
}

# Artigo — Tabela VI (CAN supervisionado, linha MTH-IDS)
PAPER_REFERENCE_SUPERVISED_CAN = {
    "MTH-IDS (Multi-Class Model)": {
        "accuracy_pct": 99.999,
        "detection_rate_pct": 99.999,
        "false_alarm_rate_pct": 0.0006,
        "f1": 0.99999,
    },
}

# Artigo — Tabela VIII (CAN LOAO, média MTH-IDS)
PAPER_REFERENCE_LOAO_CAN = {
    "mean_f1": 0.96307,
    "mean_dr_pct": 93.740,
    "mean_far_pct": 0.128,
}


def resolve_can_label_names(
    *,
    attack_labels: list[int] | None = None,
    meta_path: Path | None = None,
    pipeline_path: Path | str | None = None,
) -> dict[int, str]:
    """Car-Hacking (Gear + RPM separados) vs repack OTIDS (Impersonation)."""
    if meta_path is None and pipeline_path is not None:
        meta_path = can_dataset_meta_for_pipeline(pipeline_path)
    path = meta_path or CAN_INTRUSION_DATASET_META
    if path.is_file():
        import json  # noqa: PLC0415 — evita import circular no load do config

        profile = json.loads(path.read_text(encoding="utf-8")).get("label_profile")
        if profile == "intrusion":
            return CAN_INTRUSION_LABEL_NAMES
        if profile == "otids":
            return CAN_OTIDS_LABEL_NAMES
    if pipeline_path is not None and is_can_otids_pipeline_path(pipeline_path):
        return CAN_OTIDS_LABEL_NAMES
    if attack_labels and max(attack_labels) >= 4:
        return CAN_INTRUSION_LABEL_NAMES
    return CAN_INTRUSION_LABEL_NAMES


def is_can_intrusion_pipeline_path(path: Path | str) -> bool:
    return "pipeline_can_intrusion" in Path(path).as_posix()


def is_can_otids_pipeline_path(path: Path | str) -> bool:
    return "pipeline_can_otids" in Path(path).as_posix()


def is_can_pipeline_path(path: Path | str) -> bool:
    """True para ``pipeline_can_intrusion_*`` ou ``pipeline_can_otids_*``."""
    p = Path(path).as_posix()
    return "pipeline_can_intrusion" in p or "pipeline_can_otids" in p


def is_can_raw_input(path: Path | str) -> bool:
    """True se o CSV de entrada é um dataset CAN automotivo."""
    s = Path(path).as_posix().lower()
    markers = (
        "can_intrusion",
        "can_otids",
        "can_intrusion_dataset",
        "can_otids_dataset",
        "pipeline_can",
    )
    return any(m in s for m in markers)


def is_can_feature_columns(columns) -> bool:
    """True se as colunas indicam features de barramento CAN."""
    upper = {str(c).strip().upper() for c in columns}
    return "CAN_ID" in upper and ("DATA_0" in upper or "DATA_1" in upper)


def is_can_automotive_context(
    *,
    intermediate_dir: Path | str | None = None,
    input_path: Path | str | None = None,
    columns: list | None = None,
) -> bool:
    """Detecta contexto CAN (pasta pipeline, CSV ou colunas)."""
    if intermediate_dir is not None and is_can_pipeline_path(intermediate_dir):
        return True
    if input_path is not None and is_can_raw_input(input_path):
        return True
    if columns is not None and is_can_feature_columns(columns):
        return True
    return False


def is_unsw_pipeline_path(path: Path | str) -> bool:
    return "pipeline_unsw_nb15" in Path(path).as_posix()


def can_dataset_meta_for_pipeline(path: Path | str) -> Path:
    if is_can_otids_pipeline_path(path):
        return CAN_OTIDS_DATASET_META
    return CAN_INTRUSION_DATASET_META


def results_dir_for_can_pipeline(path: Path | str) -> Path:
    if is_can_otids_pipeline_path(path):
        return RESULTS_DIR_CAN_OTIDS
    return RESULTS_DIR_CAN_INTRUSION


def default_raw_csv_for_can_source(source: str) -> Path:
    if source in ("original", "intrusion"):
        return DEFAULT_RAW_CSV_CAN_INTRUSION
    return DEFAULT_RAW_CSV_CAN_OTIDS


def default_meta_for_can_source(source: str) -> Path:
    if source in ("original", "intrusion"):
        return CAN_INTRUSION_DATASET_META
    return CAN_OTIDS_DATASET_META


def default_loao_attack_label(
    *,
    intermediate_dir: Path | str | None = None,
    protocol: str | None = None,
) -> int:
    """Zero-day padrão da fase 7: PortScan (CICIDS) ou DoS (CAN)."""
    if intermediate_dir is not None and is_can_pipeline_path(intermediate_dir):
        return DEFAULT_LOAO_ATTACK_LABEL_CAN
    if protocol is not None:
        from mth_ids_pipeline.protocol import is_can_protocol

        if is_can_protocol(protocol):
            return DEFAULT_LOAO_ATTACK_LABEL_CAN
        from mth_ids_pipeline.protocol import is_unsw_protocol

        if is_unsw_protocol(protocol):
            return DEFAULT_LOAO_ATTACK_LABEL_UNSW
    return DEFAULT_LOAO_ATTACK_LABEL_CICIDS


def default_benign_label(
    *,
    intermediate_dir: Path | str | None = None,
    protocol: str | None = None,
) -> int:
    """Rótulo encodado da classe benigna para o pipeline atual."""
    if intermediate_dir is not None:
        path = Path(intermediate_dir).as_posix().lower()
        if "pipeline_unsw_nb15" in path:
            return 3
    if protocol is not None:
        from mth_ids_pipeline.protocol import is_unsw_protocol

        if is_unsw_protocol(protocol):
            return 3
    return 0


def ensure_intermediate_dirs(intermediate_dir: Path | None = None) -> PipelinePaths:
    paths = get_pipeline_paths(intermediate_dir)
    ensure_pipeline_dirs(paths)
    return paths
