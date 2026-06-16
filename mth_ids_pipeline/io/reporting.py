"""Helpers for writing per-phase JSON reports."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _shape(df: pd.DataFrame) -> dict[str, int]:
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}


def write_report(report_dir: Path, phase: str, payload: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"{phase}.json"
    payload = dict(payload)
    payload.setdefault("phase", phase)
    payload.setdefault("timestamp", int(time.time()))
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return out


def dataset_report(df: pd.DataFrame, label_col: str) -> dict[str, Any]:
    return {
        "shape": _shape(df),
        "label_col": label_col,
        "label_value_counts": _value_counts(df[label_col]),
    }
