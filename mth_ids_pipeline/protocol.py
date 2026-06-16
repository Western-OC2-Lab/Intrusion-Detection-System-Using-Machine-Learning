"""
Presets paper (artigo Yang et al. 2022) vs notebook (IoTJ) vs CAN-OTIDS.

CICIDS2017 — ``paper``:
  - Supervisionado: merged, k-means frac=0.008, 70/30, 10-fold CV, SMOTE notebook,
    sem Z-score fase 1; StandardScaler pós-split (fase 4); FCBF no treino, BO-GP α IG
  - Anomaly LOAO: fine, IG/FCBF/KPCA no conjunto combinado, BO-GP α/KPCA, tier 3–4

CICIDS2017 — ``notebook``: IoTJ publicado (α fixo, FCBF full, hold-out, meta XGBoost).

CAN-OTIDS (pastas ``pipeline_can_otids_*``):
  - ``can_paper`` (alias ``can``): Tabela VI — k-means 1%, split 70/30, sem SMOTE;
    IG α=0,9 dinâmico; Z-score pós k-means; FCBF full; 10-fold CV; meta best-base
  - ``can_notebook``: fluxo IoTJ adaptado (k-means 0,8%, 80/20, IG dinâmico α=0,9, FCBF full, meta XGBoost)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mth_ids_pipeline.core.clustering import CL_KMEANS_METRICS
from mth_ids_pipeline.config import (
    CAN_KMEANS_FRAC,
    CAN_PAPER_KMEANS_FRAC,
    CAN_PAPER_TEST_SIZE,
    DEFAULT_ANOMALY_SMOTE_TARGET,
    DEFAULT_BIASED_MODE,
    DEFAULT_CL_P_STAR,
    DEFAULT_META_LEARNER,
    PAPER_META_LEARNER,
    DEFAULT_TEST_SIZE,
    NOTEBOOK_KMEANS_FRAC,
    NOTEBOOK_SMOTE_TARGETS,
    UNSW_NB15_KMEANS_FRAC,
    UNSW_NB15_SMOTE_TARGETS,
    PAPER_CV_FOLDS,
    PAPER_FCBF_K,
    PAPER_FCBF_SCOPE,
    PAPER_FEATURE_FIT_SCOPE,
    PAPER_HPO_N_CALLS,
    PAPER_HPO_ON_VALIDATION,
    PAPER_IG_CUMULATIVE,
    PAPER_KMEANS_FRAC,
    PAPER_KPCA_COMPONENTS,
    PAPER_KPCA_KERNEL,
    PAPER_SUPERVISED_SCALE,
    PAPER_ANOMALY_ZSCORE,
    PAPER_TEST_SIZE,
    NOTEBOOK_SUPERVISED_SCALE,
    NOTEBOOK_ANOMALY_ZSCORE,
)

# Aliases — Car-Hacking (intrusion) vs repack OTIDS
CAN_INTRUSION_PAPER_ALIASES = frozenset(
    {"can", "can_paper", "can-paper", "can_artigo", "can_intrusion", "can-intrusion"}
)
CAN_OTIDS_PAPER_ALIASES = frozenset({"can_otids", "can-otids"})
CAN_PAPER_ALIASES = CAN_INTRUSION_PAPER_ALIASES  # alias legado
CAN_NOTEBOOK_ALIASES = frozenset({"can_notebook", "can-notebook", "can_nb", "can-nb"})
CAN_OTIDS_NOTEBOOK_ALIASES = frozenset(
    {"can_otids_notebook", "can-otids-notebook", "can_otids_nb"}
)
CAN_PROTOCOL_ALIASES = (
    CAN_INTRUSION_PAPER_ALIASES
    | CAN_OTIDS_PAPER_ALIASES
    | CAN_NOTEBOOK_ALIASES
    | CAN_OTIDS_NOTEBOOK_ALIASES
)

UNSW_NB15_ALIASES = frozenset({"unsw", "unsw_nb15", "unsw-nb15", "unsw_nb_15"})

PROTOCOL_CHOICES: tuple[str, ...] = (
    "paper",
    "notebook",
    "can",
    "can_paper",
    "can_intrusion",
    "can_otids",
    "can_notebook",
    "can_otids_notebook",
    "unsw",
    "unsw_nb15",
)


class MthIdsProtocol(str, Enum):
    PAPER = "paper"
    NOTEBOOK = "notebook"
    CAN_PAPER = "can_paper"
    CAN_INTRUSION = "can_intrusion"
    CAN_OTIDS = "can_otids"
    CAN_NOTEBOOK = "can_notebook"
    CAN_OTIDS_NOTEBOOK = "can_otids_notebook"
    CAN = "can"  # alias legado → can_paper (intrusion)
    UNSW_NB15 = "unsw_nb15"


@dataclass(frozen=True)
class ProtocolSettings:
    name: str
    description: str
    supervised_profile: str
    anomaly_profile: str
    test_size: float
    cv_folds: int
    hpo_on_validation: bool
    smote_targets: dict[int, int]
    skip_smote: bool
    skip_anomaly_smote: bool
    kmeans_frac: float
    skip_kmeans_sampling: bool
    anomaly_benign_target: int | None
    anomaly_smote_target: int | None
    biased_mode: str
    force_biased: bool
    optimize_p_star: bool
    cl_hpo_metric: str
    cl_hpo_metric_source: str
    cl_hpo_n_calls: int
    optimize_ig: bool
    optimize_kpca: bool
    ig_cumulative: float
    fcbf_k: int
    fcbf_scope: str
    supervised_scale_mode: str
    feature_fit_scope: str
    anomaly_zscore_scope: str
    kpca_components: int
    kpca_kernel: str
    hpo_n_calls: int
    meta_learner: str = DEFAULT_META_LEARNER
    cl_p_star: float = DEFAULT_CL_P_STAR
    cl_kmeans_metrics: tuple[str, ...] = CL_KMEANS_METRICS
    fixed_supervised_features: tuple[str, ...] | None = None
    phase1_zscore: bool = True
    post_sample_zscore: bool = False


PAPER = ProtocolSettings(
    name="paper",
    description=(
        "Artigo MTH-IDS: sup k-means 0.8% + 70/30 + 10-fold CV + SMOTE notebook; "
        "normalização pós-split (fase 4); HPO em validação; stacking meta best-base; "
        "anomaly LOAO fine + BO-GP tier 3–4."
    ),
    supervised_profile="merged",
    anomaly_profile="fine",
    test_size=PAPER_TEST_SIZE,
    cv_folds=PAPER_CV_FOLDS,
    hpo_on_validation=PAPER_HPO_ON_VALIDATION,
    smote_targets=dict(NOTEBOOK_SMOTE_TARGETS),
    skip_smote=False,
    skip_anomaly_smote=False,
    kmeans_frac=PAPER_KMEANS_FRAC,
    skip_kmeans_sampling=False,
    anomaly_benign_target=None,
    anomaly_smote_target=DEFAULT_ANOMALY_SMOTE_TARGET,
    biased_mode="both",
    force_biased=True,
    optimize_p_star=True,
    cl_hpo_metric="f1",
    cl_hpo_metric_source="article",
    cl_hpo_n_calls=PAPER_HPO_N_CALLS,
    optimize_ig=True,
    optimize_kpca=True,
    ig_cumulative=PAPER_IG_CUMULATIVE,
    fcbf_k=PAPER_FCBF_K,
    fcbf_scope=PAPER_FCBF_SCOPE,
    supervised_scale_mode=PAPER_SUPERVISED_SCALE,
    feature_fit_scope=PAPER_FEATURE_FIT_SCOPE,
    anomaly_zscore_scope=PAPER_ANOMALY_ZSCORE,
    kpca_components=PAPER_KPCA_COMPONENTS,
    kpca_kernel=PAPER_KPCA_KERNEL,
    hpo_n_calls=PAPER_HPO_N_CALLS,
    meta_learner=PAPER_META_LEARNER,
    phase1_zscore=False,
)

NOTEBOOK = ProtocolSettings(
    name="notebook",
    description="Notebook IoTJ: merged, hold-out 20%, PortScan demo, parâmetros fixos.",
    supervised_profile="merged",
    anomaly_profile="merged",
    test_size=0.2,
    cv_folds=0,
    hpo_on_validation=False,
    smote_targets=dict(NOTEBOOK_SMOTE_TARGETS),
    skip_smote=False,
    skip_anomaly_smote=False,
    kmeans_frac=NOTEBOOK_KMEANS_FRAC,
    skip_kmeans_sampling=False,
    anomaly_benign_target=1255,
    anomaly_smote_target=DEFAULT_ANOMALY_SMOTE_TARGET,
    biased_mode=DEFAULT_BIASED_MODE,
    force_biased=False,
    optimize_p_star=False,
    cl_hpo_metric="accuracy",
    cl_hpo_metric_source="notebook",
    cl_hpo_n_calls=20,
    optimize_ig=False,
    optimize_kpca=False,
    ig_cumulative=0.9,
    fcbf_k=20,
    fcbf_scope="full",
    supervised_scale_mode=NOTEBOOK_SUPERVISED_SCALE,
    feature_fit_scope="combined",
    anomaly_zscore_scope=NOTEBOOK_ANOMALY_ZSCORE,
    kpca_components=10,
    kpca_kernel="rbf",
    hpo_n_calls=20,
    meta_learner=DEFAULT_META_LEARNER,
    cl_kmeans_metrics=("euclidean", "manhattan", "cosine"),
)

CAN_PAPER = ProtocolSettings(
    name="can_paper",
    description=(
        "Car-Hacking (Tabela VI artigo): k-means 1%, split 70/30, sem SMOTE; "
        "IG α=0,9 dinâmico; Z-score pós k-means; FCBF full; "
        "10-fold CV; meta best-base; BO-GP KPCA/p* (anomaly)."
    ),
    supervised_profile="can_intrusion_merged",
    anomaly_profile="can_intrusion_fine",
    test_size=CAN_PAPER_TEST_SIZE,
    cv_folds=PAPER_CV_FOLDS,
    hpo_on_validation=PAPER_HPO_ON_VALIDATION,
    smote_targets={},
    skip_smote=True,
    skip_anomaly_smote=True,
    kmeans_frac=CAN_PAPER_KMEANS_FRAC,
    skip_kmeans_sampling=False,
    anomaly_benign_target=None,
    anomaly_smote_target=None,
    biased_mode="both",
    force_biased=True,
    optimize_p_star=True,
    cl_hpo_metric="f1",
    cl_hpo_metric_source="article",
    cl_hpo_n_calls=PAPER_HPO_N_CALLS,
    optimize_ig=False,
    optimize_kpca=True,
    ig_cumulative=PAPER_IG_CUMULATIVE,
    fcbf_k=PAPER_FCBF_K,
    fcbf_scope="full",
    supervised_scale_mode=NOTEBOOK_SUPERVISED_SCALE,
    feature_fit_scope=PAPER_FEATURE_FIT_SCOPE,
    anomaly_zscore_scope=PAPER_ANOMALY_ZSCORE,
    kpca_components=PAPER_KPCA_COMPONENTS,
    kpca_kernel=PAPER_KPCA_KERNEL,
    hpo_n_calls=PAPER_HPO_N_CALLS,
    meta_learner=PAPER_META_LEARNER,
    fixed_supervised_features=None,
    phase1_zscore=False,
    post_sample_zscore=True,
)

CAN_OTIDS = ProtocolSettings(
    name="can_otids",
    description=(
        "CAN-OTIDS repack (Tabela VI): mesmo preset que can_paper (artigo); "
        "pastas pipeline_can_otids_* e CSV CAN_OTIDS_Dataset.csv."
    ),
    supervised_profile="can_otids_merged",
    anomaly_profile="can_otids_fine",
    test_size=CAN_PAPER_TEST_SIZE,
    cv_folds=PAPER_CV_FOLDS,
    hpo_on_validation=PAPER_HPO_ON_VALIDATION,
    smote_targets={},
    skip_smote=True,
    skip_anomaly_smote=True,
    kmeans_frac=CAN_PAPER_KMEANS_FRAC,
    skip_kmeans_sampling=False,
    anomaly_benign_target=None,
    anomaly_smote_target=None,
    biased_mode="both",
    force_biased=True,
    optimize_p_star=True,
    cl_hpo_metric="f1",
    cl_hpo_metric_source="article",
    cl_hpo_n_calls=PAPER_HPO_N_CALLS,
    optimize_ig=False,
    optimize_kpca=True,
    ig_cumulative=PAPER_IG_CUMULATIVE,
    fcbf_k=PAPER_FCBF_K,
    fcbf_scope="full",
    supervised_scale_mode=NOTEBOOK_SUPERVISED_SCALE,
    feature_fit_scope=PAPER_FEATURE_FIT_SCOPE,
    anomaly_zscore_scope=PAPER_ANOMALY_ZSCORE,
    kpca_components=PAPER_KPCA_COMPONENTS,
    kpca_kernel=PAPER_KPCA_KERNEL,
    hpo_n_calls=PAPER_HPO_N_CALLS,
    meta_learner=PAPER_META_LEARNER,
    fixed_supervised_features=None,
    phase1_zscore=False,
    post_sample_zscore=True,
)

CAN_NOTEBOOK = ProtocolSettings(
    name="can_notebook",
    description=(
        "Car-Hacking notebook IoTJ: k-means 0,8%, 80/20, sem SMOTE; α IG=0,9; "
        "FCBF full; HPO hold-out; meta XGBoost; KPCA fixo (anomaly)."
    ),
    supervised_profile="can_intrusion_merged",
    anomaly_profile="can_intrusion_fine",
    test_size=DEFAULT_TEST_SIZE,
    cv_folds=0,
    hpo_on_validation=False,
    smote_targets={},
    skip_smote=True,
    skip_anomaly_smote=True,
    kmeans_frac=CAN_KMEANS_FRAC,
    skip_kmeans_sampling=False,
    anomaly_benign_target=None,
    anomaly_smote_target=None,
    biased_mode=DEFAULT_BIASED_MODE,
    force_biased=False,
    optimize_p_star=False,
    cl_hpo_metric="accuracy",
    cl_hpo_metric_source="notebook",
    cl_hpo_n_calls=20,
    optimize_ig=False,
    optimize_kpca=False,
    ig_cumulative=0.9,
    fcbf_k=20,
    fcbf_scope="full",
    supervised_scale_mode=NOTEBOOK_SUPERVISED_SCALE,
    feature_fit_scope="combined",
    anomaly_zscore_scope=NOTEBOOK_ANOMALY_ZSCORE,
    kpca_components=10,
    kpca_kernel="rbf",
    hpo_n_calls=20,
    meta_learner=DEFAULT_META_LEARNER,
    cl_kmeans_metrics=("euclidean", "manhattan", "cosine"),
    fixed_supervised_features=None,
)

CAN_OTIDS_NOTEBOOK = ProtocolSettings(
    name="can_otids_notebook",
    description="CAN-OTIDS repack + preset can_notebook; pastas pipeline_can_otids_*.",
    supervised_profile="can_otids_merged",
    anomaly_profile="can_otids_fine",
    test_size=DEFAULT_TEST_SIZE,
    cv_folds=0,
    hpo_on_validation=False,
    smote_targets={},
    skip_smote=True,
    skip_anomaly_smote=True,
    kmeans_frac=CAN_KMEANS_FRAC,
    skip_kmeans_sampling=False,
    anomaly_benign_target=None,
    anomaly_smote_target=None,
    biased_mode=DEFAULT_BIASED_MODE,
    force_biased=False,
    optimize_p_star=False,
    cl_hpo_metric="accuracy",
    cl_hpo_metric_source="notebook",
    cl_hpo_n_calls=20,
    optimize_ig=False,
    optimize_kpca=False,
    ig_cumulative=0.9,
    fcbf_k=20,
    fcbf_scope="full",
    supervised_scale_mode=NOTEBOOK_SUPERVISED_SCALE,
    feature_fit_scope="combined",
    anomaly_zscore_scope=NOTEBOOK_ANOMALY_ZSCORE,
    kpca_components=10,
    kpca_kernel="rbf",
    hpo_n_calls=20,
    meta_learner=DEFAULT_META_LEARNER,
    cl_kmeans_metrics=("euclidean", "manhattan", "cosine"),
    fixed_supervised_features=None,
)

# Alias legado: ``--protocol can`` → Car-Hacking (artigo)
CAN = CAN_PAPER

UNSW = ProtocolSettings(
    name="unsw_nb15",
    description=(
        "UNSW-NB15: Benign k-means 10%; ataques preservados; SMOTE Analysis/Backdoors/"
        "Shellcode→5000, Worms→2000; BO-GP α IG; 10-fold CV; meta best-base."
    ),
    supervised_profile="unsw_nb15_merged",
    anomaly_profile="unsw_nb15_fine",
    test_size=DEFAULT_TEST_SIZE,
    cv_folds=PAPER_CV_FOLDS,
    hpo_on_validation=PAPER_HPO_ON_VALIDATION,
    smote_targets=dict(UNSW_NB15_SMOTE_TARGETS),
    skip_smote=False,
    skip_anomaly_smote=False,
    kmeans_frac=UNSW_NB15_KMEANS_FRAC,
    skip_kmeans_sampling=False,
    anomaly_benign_target=None,
    anomaly_smote_target=DEFAULT_ANOMALY_SMOTE_TARGET,
    biased_mode="both",
    force_biased=True,
    optimize_p_star=True,
    cl_hpo_metric="f1",
    cl_hpo_metric_source="article",
    cl_hpo_n_calls=PAPER_HPO_N_CALLS,
    optimize_ig=True,
    optimize_kpca=True,
    ig_cumulative=PAPER_IG_CUMULATIVE,
    fcbf_k=PAPER_FCBF_K,
    fcbf_scope=PAPER_FCBF_SCOPE,
    supervised_scale_mode=PAPER_SUPERVISED_SCALE,
    feature_fit_scope=PAPER_FEATURE_FIT_SCOPE,
    anomaly_zscore_scope=PAPER_ANOMALY_ZSCORE,
    kpca_components=PAPER_KPCA_COMPONENTS,
    kpca_kernel=PAPER_KPCA_KERNEL,
    hpo_n_calls=PAPER_HPO_N_CALLS,
    meta_learner=PAPER_META_LEARNER,
    phase1_zscore=False,
)


def _normalize_protocol_key(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def is_can_protocol(name: str) -> bool:
    """True para presets CAN (intrusion ou OTIDS)."""
    return _normalize_protocol_key(name) in {
        a.replace("-", "_") for a in CAN_PROTOCOL_ALIASES
    }


def is_can_otids_protocol(name: str) -> bool:
    key = _normalize_protocol_key(name)
    return key in {a.replace("-", "_") for a in CAN_OTIDS_PAPER_ALIASES | CAN_OTIDS_NOTEBOOK_ALIASES}


def is_unsw_protocol(name: str) -> bool:
    return _normalize_protocol_key(name) in {a.replace("-", "_") for a in UNSW_NB15_ALIASES}


def get_protocol_settings(name: str) -> ProtocolSettings:
    key = _normalize_protocol_key(name)
    if key in ("paper", "article", "artigo"):
        return PAPER
    if key in ("notebook", "nb", "ioj"):
        return NOTEBOOK
    if key in {a.replace("-", "_") for a in CAN_OTIDS_PAPER_ALIASES}:
        return CAN_OTIDS
    if key in {a.replace("-", "_") for a in CAN_OTIDS_NOTEBOOK_ALIASES}:
        return CAN_OTIDS_NOTEBOOK
    if key in {a.replace("-", "_") for a in CAN_INTRUSION_PAPER_ALIASES}:
        return CAN_PAPER
    if key in {a.replace("-", "_") for a in CAN_NOTEBOOK_ALIASES}:
        return CAN_NOTEBOOK
    if key in {a.replace("-", "_") for a in UNSW_NB15_ALIASES}:
        return UNSW
    raise ValueError(
        f"Protocolo desconhecido: {name!r}. "
        f"Use {', '.join(PROTOCOL_CHOICES)}."
    )
