"""Métricas de avaliação e comparação com resultados do artigo."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_classifier(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    result = {
        "model": name,
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1_weighted": float(f),
    }
    if verbose:
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.6f}  Precision: {p:.6f}  Recall: {r:.6f}  F1: {f:.6f}")
        print(classification_report(y_true, y_pred))
    return result


def binary_dr_far_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Detection rate, false alarm rate e F1 binários (artigo, eq. 8-11)."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    dr = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "accuracy": acc,
        "detection_rate": dr,
        "false_alarm_rate": far,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# Referências do artigo (Tabela VII — CICIDS2017 multi-class, stacking tier)
PAPER_REFERENCE_SUPERVISED = {
    "MTH-IDS (Multi-Class Model)": {
        "accuracy_pct": 99.879,
        "detection_rate_pct": 99.818,
        "false_alarm_rate_pct": 0.101,
        "f1": 0.99879,
    },
    "MTH-IDS (Without FS & HPO)": {
        "accuracy_pct": 99.861,
        "f1": 0.99860,
    },
}

# Notebook (sampled dataset, hold-out test ~5360)
NOTEBOOK_REFERENCE_SUPERVISED = {
    "XGBoost HPO": {"accuracy": 0.9957, "f1_weighted": 0.9957},
    "RandomForest HPO": {"accuracy": 0.9951, "f1_weighted": 0.9951},
    "DecisionTree HPO": {"accuracy": 0.9937, "f1_weighted": 0.9938},
    "ExtraTrees HPO": {"accuracy": 0.9955, "f1_weighted": 0.9955},
    "Stacking meta HPO": {"accuracy": 0.9957, "f1_weighted": 0.9957},
}

NOTEBOOK_REFERENCE_ANOMALY = {
    "CL-k-means n=8": {"accuracy": 0.598},
    "CL-k-means BO-GP n=16": {"accuracy": 0.9195},
    "CL-k-means final n=16": {"accuracy": 0.945, "f1_macro": 0.94},
}


def compare_metrics(
    reproduced: dict[str, float],
    reference: dict[str, float],
    *,
    metric_keys: tuple[str, ...] = ("accuracy", "f1_weighted"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in metric_keys:
        if key not in reproduced or key not in reference:
            continue
        rep = float(reproduced[key])
        ref = float(reference[key])
        abs_diff = rep - ref
        pct_diff = (abs_diff / ref * 100.0) if ref else 0.0
        rows.append(
            {
                "metric": key,
                "reference": ref,
                "reproduced": rep,
                "absolute_diff": abs_diff,
                "percent_diff": pct_diff,
            }
        )
    return rows
