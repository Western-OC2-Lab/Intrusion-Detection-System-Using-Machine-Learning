"""Agregação de resultados LOAO (Tabela IX) a partir de ``attack_*``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys
from pathlib import Path

from mth_ids_pipeline.config import (
    A06_TEST_SLICE_INFO,
    PAPER_REFERENCE_LOAO_CAN,
    RESULTS_LOGS_DIR,
    is_can_pipeline_path,
)
from mth_ids_pipeline.core.biased_classifiers import load_best_n_clusters

PAPER_REFERENCE_CICIDS2017 = {
    "mean_f1": 0.80013,
    "mean_dr_pct": 75.943,
    "mean_far_pct": 13.882,
}


def resolve_attack_log(subdir: Path, attack: int) -> Path | None:
    """Prefer ``results/logs/loao/attack_<N>.log``; fallback para cópia local em ``data/``."""
    mirrored = RESULTS_LOGS_DIR / "loao" / f"attack_{attack}.log"
    local = subdir / "loao_run.log"
    if mirrored.is_file():
        return mirrored
    if local.is_file():
        return local
    return None


def load_attack_row(subdir: Path, attack: int, attack_name: str) -> dict[str, Any] | None:
    """Métricas de ``attack_<N>/reports/phase11_anomaly_biased.json``."""
    attack_report_dir = subdir / "reports"
    report_file = attack_report_dir / "phase11_anomaly_biased.json"
    if not report_file.is_file():
        return None
    rep = json.loads(report_file.read_text(encoding="utf-8"))
    m = rep.get("mth_ids_anomaly") or rep.get("cl_kmeans") or {}
    slice_meta: dict[str, Any] = {}
    slice_path = subdir / A06_TEST_SLICE_INFO
    if slice_path.is_file():
        slice_meta = json.loads(slice_path.read_text(encoding="utf-8"))
    best_k = load_best_n_clusters(attack_report_dir)
    return {
        "attack_label": attack,
        "attack_name": attack_name,
        "n_clusters": rep.get("n_clusters") or best_k,
        "zero_day_samples": slice_meta.get("zero_day_samples"),
        "benign_sampled": slice_meta.get("benign_sampled"),
        "train_rows": slice_meta.get("n_train_rows"),
        "test_rows": slice_meta.get("n_test_rows"),
        "train_binary_label_counts": slice_meta.get("train_binary_label_counts"),
        "test_binary_label_counts": slice_meta.get("test_binary_label_counts"),
        "train_original_label_counts": slice_meta.get("train_original_label_counts"),
        "zero_day_fully_excluded_from_train": slice_meta.get(
            "zero_day_fully_excluded_from_train"
        ),
        "accuracy": m.get("accuracy"),
        "detection_rate": m.get("detection_rate"),
        "false_alarm_rate": m.get("false_alarm_rate"),
        "f1": m.get("f1"),
        "output_dir": str(subdir),
        "log_file": str(log) if (log := resolve_attack_log(subdir, attack)) else None,
    }


def collect_summary_rows(
    output_root: Path,
    label_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Todos os ``attack_*`` com fase 11 concluída."""
    rows: list[dict[str, Any]] = []
    if not output_root.is_dir():
        return rows
    for subdir in sorted(output_root.glob("attack_*")):
        try:
            attack = int(subdir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        name = label_names.get(attack, f"Label={attack}")
        row = load_attack_row(subdir, attack, name)
        if row is not None:
            rows.append(row)
    return rows


def build_loao_summary(
    output_root: Path,
    label_names: dict[int, str],
    *,
    attacks_in_dataset: int,
    attacks_planned: list[int] | None = None,
    attacks_succeeded_this_run: list[int] | None = None,
    attacks_failed_this_run: list[int] | None = None,
) -> dict[str, Any]:
    """Monta dict do resumo LOAO (com ou sem ataques concluídos)."""
    per_attack = collect_summary_rows(output_root, label_names)
    completed_labels = {int(r["attack_label"]) for r in per_attack}
    all_attack_labels = sorted(label_names.keys())
    pending = [label for label in all_attack_labels if label not in completed_labels]

    if per_attack:
        dr_mean = sum(float(r["detection_rate"] or 0) for r in per_attack) / len(per_attack)
        far_mean = sum(float(r["false_alarm_rate"] or 0) for r in per_attack) / len(per_attack)
        f1_mean = sum(float(r["f1"] or 0) for r in per_attack) / len(per_attack)
    else:
        dr_mean = far_mean = f1_mean = 0.0

    return {
        "n_attacks": len(per_attack),
        "n_attacks_completed": len(per_attack),
        "n_attacks_in_dataset": attacks_in_dataset,
        "n_attacks_pending": len(pending),
        "pending_attack_labels": pending,
        "attacks_planned_this_run": attacks_planned or [],
        "attacks_succeeded_this_run": attacks_succeeded_this_run or [],
        "attacks_failed_this_run": attacks_failed_this_run or [],
        "loao_phases": "7,8,9,10,11",
        "benign_pairing": "paper_table_ix_1_to_1 (unless --benign-target)",
        "mean_detection_rate": dr_mean,
        "mean_false_alarm_rate": far_mean,
        "mean_f1": f1_mean,
        "per_attack": per_attack,
        "log_pattern": str(RESULTS_LOGS_DIR / "loao" / "attack_<N>.log"),
        **(
            {"paper_reference_can": dict(PAPER_REFERENCE_LOAO_CAN)}
            if is_can_pipeline_path(output_root)
            else {"paper_reference_cicids2017": dict(PAPER_REFERENCE_CICIDS2017)}
        ),
    }


def write_loao_summary(output_root: Path, summary: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "loao_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
