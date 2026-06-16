"""
Perfis de rótulos CICIDS2017 para o pipeline MTH-IDS.

- merged: famílias agregadas (Tabela VII) — ~7 classes
- fine: rótulos originais — amostra k-means estilo notebook

Fase 2 fine: preserva inteiros os fine cuja família merged está no ``df_minor`` do notebook
(Bot, Infiltration, WebAttack), não “todos os rótulos que o merge não agrega” (PortScan
é amostrado). Ver ``compute_fine_minority_labels_notebook_aligned()`` e
``docs/EXECUCAO.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from mth_ids_pipeline.config import (
    CICIDS2017_FINE_LABEL_NAMES,
    DATA_DIR,
    DEFAULT_MINORITY_LABELS,
    DEFAULT_RAW_CSV_CAN_INTRUSION,
    DEFAULT_RAW_CSV_CAN_OTIDS,
    DEFAULT_RAW_CSV_UNSW_NB15,
    INTERMEDIATE_DIR_CAN_INTRUSION_FINE,
    INTERMEDIATE_DIR_CAN_INTRUSION_MERGED,
    INTERMEDIATE_DIR_CAN_OTIDS_FINE,
    INTERMEDIATE_DIR_CAN_OTIDS_MERGED,
    INTERMEDIATE_DIR_UNSW_NB15_FINE,
    INTERMEDIATE_DIR_UNSW_NB15_MERGED,
    UNSW_NB15_PRESERVED_ATTACK_LABELS,
)

# Separador corrompido quando UTF-8 (U+FFFD) é lido com encoding="latin1"
_WEB_ATTACK_SEP_MOJIBAKE = "\xef\xbf\xbd"

# Mapeamento notebook / artigo (subtipos → família)
CICIDS_LABEL_MERGE: dict[str, str] = {
    "DoS Hulk": "DoS",
    "DDoS": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack \ufffd Brute Force": "WebAttack",
    "Web Attack \ufffd XSS": "WebAttack",
    "Web Attack \ufffd Sql Injection": "WebAttack",
    f"Web Attack {_WEB_ATTACK_SEP_MOJIBAKE} Brute Force": "WebAttack",
    f"Web Attack {_WEB_ATTACK_SEP_MOJIBAKE} XSS": "WebAttack",
    f"Web Attack {_WEB_ATTACK_SEP_MOJIBAKE} Sql Injection": "WebAttack",
    "Web Attack – Brute Force": "WebAttack",
    "Web Attack – XSS": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
}


def normalize_cicids_label_text(labels) -> "pd.Series":
    """
    Normaliza strings de Label do CICIDS2017 antes do agrupamento merged.

    O CSV oficial usa um separador não-ASCII entre 'Web Attack' e o subtipo.
    Com ``encoding='latin1'`` (notebook / merge_cicids), U+FFFD vira três
    caracteres ``\\xef\\xbf\\xbd``, que não casam com ``\\ufffd`` no dict.
    """
    import pandas as pd

    s = labels.astype(str).str.strip()
    s = s.str.replace(_WEB_ATTACK_SEP_MOJIBAKE, "\ufffd", regex=False)
    return s


class LabelProfileKind(str, Enum):
    MERGED = "merged"
    FINE = "fine"


@dataclass(frozen=True)
class LabelProfile:
    """Caminhos e defaults por perfil de rótulo."""

    kind: LabelProfileKind
    raw_csv: Path
    intermediate_dir: Path
    minority_labels: tuple[int, ...] | None
    description: str
    paired_supervised_dir: Path | None = None
    table_vii_profile: str | None = None
    # CAN: k-means 0,8% em BENIGN + ataques (nenhuma classe preservada intacta).
    kmeans_sample_all_classes: bool = False

    @property
    def auto_minority(self) -> bool:
        return self.minority_labels is None and not self.kmeans_sample_all_classes

    def minority_labels_csv(self) -> str | None:
        if self.minority_labels is None:
            return None
        return ",".join(str(x) for x in self.minority_labels)


def apply_cicids_label_merge(df, *, inplace: bool = False):
    """Agrupa subtipos de ataque em famílias (perfil merged)."""
    out = df if inplace else df.copy()
    labels = normalize_cicids_label_text(out["Label"])
    labels = labels.replace(CICIDS_LABEL_MERGE)
    # Qualquer variante restante de Web Attack → WebAttack (notebook)
    web = labels.str.startswith("Web Attack", na=False)
    labels = labels.where(~web, "WebAttack")
    out["Label"] = labels
    return out


# Famílias preservadas intactas no notebook merged (df_minor: Label 6, 1, 4 → WebAttack, Bot, Infiltration)
NOTEBOOK_MERGED_PRESERVED_FAMILIES = frozenset({"Bot", "Infiltration", "WebAttack"})

# Fine-only: ataques ultra-raros que somem no k-means 0,8% mas entram no LOAO (Tabela IX).
# Heartbleed = 11 linhas no CICIDS2017; família merged DoS → não entra no df_minor do notebook.
FINE_ULTRA_RARE_PRESERVED_LABELS: tuple[int, ...] = (8,)


def merged_family_for_fine_label(label: str) -> str:
    """Mapeia rótulo fine → família merged (``CICIDS_LABEL_MERGE`` + Web Attack → WebAttack)."""
    s = label.strip().replace(_WEB_ATTACK_SEP_MOJIBAKE, "\ufffd")
    if s in CICIDS_LABEL_MERGE:
        return CICIDS_LABEL_MERGE[s]
    if s.startswith("Web Attack"):
        return "WebAttack"
    return s


def compute_fine_minority_labels_notebook_aligned(
    fine_label_names: dict[int, str] | None = None,
) -> tuple[int, ...]:
    """
    IDs fine (LabelEncoder alfabético) preservados inteiros na fase 2 — espelha o notebook.

    Critério: família merged ∈ ``NOTEBOOK_MERGED_PRESERVED_FAMILIES`` (Bot, Infiltration,
    WebAttack), igual ao ``df_minor`` merged:

        df_minor = df[(df['Label']==6)|(df['Label']==1)|(df['Label']==4)]

    DoS (exceto Heartbleed), PortScan, BruteForce e BENIGN passam pelo k-means 0,8% no merged
    **e** no fine. Heartbleed (label 8) é preservado à parte — 11 amostras no dataset completo.

    **Não** confundir com “rótulos que o merge não agrega”: PortScan não é agregado, mas
    **não** é preservado; subtipos Web Attack **são** agregados em WebAttack, mas **são**
    preservados porque a família WebAttack está no ``df_minor``.
    """
    names = fine_label_names or CICIDS2017_FINE_LABEL_NAMES
    preserved = {
        int(idx)
        for idx, name in names.items()
        if int(idx) != 0 and merged_family_for_fine_label(name) in NOTEBOOK_MERGED_PRESERVED_FAMILIES
    }
    preserved.update(FINE_ULTRA_RARE_PRESERVED_LABELS)
    return tuple(sorted(preserved))


FINE_DEFAULT_MINORITY_LABELS = compute_fine_minority_labels_notebook_aligned()


def get_label_profile(name: str) -> LabelProfile:
    key = name.strip().lower()
    if key in (LabelProfileKind.MERGED.value, "merge", "7"):
        return MERGED_PROFILE
    if key in (LabelProfileKind.FINE.value, "fine-grained", "finegrained", "14"):
        return FINE_PROFILE
    if key in (
        "can_intrusion_merged",
        "can-intrusion-merged",
        "can_merged",
        "can-merged",
        "can_sup",
        "can",
    ):
        return CAN_INTRUSION_MERGED_PROFILE
    if key in (
        "can_intrusion_fine",
        "can-intrusion-fine",
        "can_fine",
        "can-fine",
        "can_loao",
    ):
        return CAN_INTRUSION_FINE_PROFILE
    if key in ("can_otids_merged", "can-otids-merged"):
        return CAN_OTIDS_MERGED_PROFILE
    if key in ("can_otids_fine", "can-otids-fine", "can_otids_loao"):
        return CAN_OTIDS_FINE_PROFILE
    if key in ("unsw_nb15_merged", "unsw-nb15-merged", "unsw_merged", "unsw"):
        return UNSW_NB15_MERGED_PROFILE
    if key in ("unsw_nb15_fine", "unsw-nb15-fine", "unsw_fine", "unsw_loao"):
        return UNSW_NB15_FINE_PROFILE
    raise ValueError(
        f"Perfil desconhecido: {name!r}. "
        "Use merged/fine, can_intrusion_merged/fine, can_otids_merged/fine "
        "ou unsw_nb15_merged/fine."
    )


MERGED_PROFILE = LabelProfile(
    kind=LabelProfileKind.MERGED,
    raw_csv=DATA_DIR / "CICIDS2017.csv",
    intermediate_dir=DATA_DIR / "pipeline_mth_ids_merged",
    minority_labels=DEFAULT_MINORITY_LABELS,
    description="Famílias agregadas (Tabela VII): BENIGN + 6 ataques; LOAO com 6 rodadas.",
)

FINE_PROFILE = LabelProfile(
    kind=LabelProfileKind.FINE,
    raw_csv=DATA_DIR / "CICIDS2017_fine.csv",
    intermediate_dir=DATA_DIR / "pipeline_mth_ids_fine",
    minority_labels=FINE_DEFAULT_MINORITY_LABELS,
    description=(
        "Rótulos originais; fase 2 preserva fine equivalentes ao df_minor merged "
        "(Bot, Infiltration, WebAttack) + Heartbleed (ultra-raro) + k-means 0,8% em "
        "DoS/PortScan/BruteForce/BENIGN (~escala notebook). LOAO nas labels presentes em "
        "02_sampled_kmeans."
    ),
)

CAN_INTRUSION_MERGED_PROFILE = LabelProfile(
    kind=LabelProfileKind.MERGED,
    raw_csv=DEFAULT_RAW_CSV_CAN_INTRUSION,
    intermediate_dir=INTERMEDIATE_DIR_CAN_INTRUSION_MERGED,
    minority_labels=None,
    kmeans_sample_all_classes=True,
    description=(
        "Car-Hacking (Tabela VI artigo): 5 classes; k-means 0,8% em todas; "
        "split 70/30; IG α=0,9 + 4 features; Z-score pós k-means; sem SMOTE."
    ),
)

CAN_INTRUSION_FINE_PROFILE = LabelProfile(
    kind=LabelProfileKind.FINE,
    raw_csv=DEFAULT_RAW_CSV_CAN_INTRUSION,
    intermediate_dir=INTERMEDIATE_DIR_CAN_INTRUSION_FINE,
    minority_labels=None,
    kmeans_sample_all_classes=True,
    paired_supervised_dir=INTERMEDIATE_DIR_CAN_INTRUSION_MERGED,
    table_vii_profile="can_intrusion_merged",
    description="Car-Hacking LOAO (Tabela VIII): 4 zero-days (DoS, Fuzzy, Gear, RPM).",
)

CAN_OTIDS_MERGED_PROFILE = LabelProfile(
    kind=LabelProfileKind.MERGED,
    raw_csv=DEFAULT_RAW_CSV_CAN_OTIDS,
    intermediate_dir=INTERMEDIATE_DIR_CAN_OTIDS_MERGED,
    minority_labels=None,
    kmeans_sample_all_classes=True,
    description=(
        "CAN-OTIDS repack (Tabela VI): 4 classes; mesmo preset can_paper "
        "(k-means 0,8%, split 70/30, 4 features IG); sem SMOTE."
    ),
)

CAN_OTIDS_FINE_PROFILE = LabelProfile(
    kind=LabelProfileKind.FINE,
    raw_csv=DEFAULT_RAW_CSV_CAN_OTIDS,
    intermediate_dir=INTERMEDIATE_DIR_CAN_OTIDS_FINE,
    minority_labels=None,
    kmeans_sample_all_classes=True,
    paired_supervised_dir=INTERMEDIATE_DIR_CAN_OTIDS_MERGED,
    table_vii_profile="can_otids_merged",
    description="CAN-OTIDS LOAO (Tabela VIII): 3 zero-days (DoS, Fuzzy, Impersonation).",
)

# Aliases legados
CAN_MERGED_PROFILE = CAN_INTRUSION_MERGED_PROFILE
CAN_FINE_PROFILE = CAN_INTRUSION_FINE_PROFILE

UNSW_NB15_MERGED_PROFILE = LabelProfile(
    kind=LabelProfileKind.MERGED,
    raw_csv=DEFAULT_RAW_CSV_UNSW_NB15,
    intermediate_dir=INTERMEDIATE_DIR_UNSW_NB15_MERGED,
    minority_labels=UNSW_NB15_PRESERVED_ATTACK_LABELS,
    description=(
        "UNSW-NB15 merged: Benign k-means 10%; Generic/Exploits/Fuzzers/DoS/Reconnaissance "
        "preservados; SMOTE Analysis/Backdoors/Shellcode→5000, Worms→2000."
    ),
)

UNSW_NB15_FINE_PROFILE = LabelProfile(
    kind=LabelProfileKind.FINE,
    raw_csv=DEFAULT_RAW_CSV_UNSW_NB15,
    intermediate_dir=INTERMEDIATE_DIR_UNSW_NB15_FINE,
    minority_labels=UNSW_NB15_PRESERVED_ATTACK_LABELS,
    paired_supervised_dir=INTERMEDIATE_DIR_UNSW_NB15_MERGED,
    table_vii_profile="unsw_nb15_merged",
    description="UNSW-NB15 LOAO: 9 zero-days (uma rodada por attack_cat).",
)

ALL_PROFILES: dict[str, LabelProfile] = {
    LabelProfileKind.MERGED.value: MERGED_PROFILE,
    LabelProfileKind.FINE.value: FINE_PROFILE,
    "can_intrusion_merged": CAN_INTRUSION_MERGED_PROFILE,
    "can_intrusion_fine": CAN_INTRUSION_FINE_PROFILE,
    "can_otids_merged": CAN_OTIDS_MERGED_PROFILE,
    "can_otids_fine": CAN_OTIDS_FINE_PROFILE,
    "can_merged": CAN_INTRUSION_MERGED_PROFILE,
    "can_fine": CAN_INTRUSION_FINE_PROFILE,
    "unsw_nb15_merged": UNSW_NB15_MERGED_PROFILE,
    "unsw_nb15_fine": UNSW_NB15_FINE_PROFILE,
}


def minority_labels_all_attacks(
    df,
    *,
    label_col: str = "Label",
    benign_label: int = 0,
) -> tuple[int, ...]:
    """
    Rótulos inteiros de ataque preservados no k-means sampling (fase 2).

    Assume BENIGN → 0 após LabelEncoder (ordem alfabética no CICIDS2017).
    """
    from mth_ids_pipeline.core.preprocessing import encode_labels

    encoded, _ = encode_labels(df, label_col=label_col)
    return tuple(
        sorted(int(x) for x in encoded[label_col].unique() if int(x) != benign_label)
    )
