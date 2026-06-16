"""Caminhos de saída em ``results/`` (tabelas, logs, configs de execução)."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from mth_ids_pipeline.config import RESULTS_CONFIG_DIR, RESULTS_LOGS_DIR, ensure_results_dirs


def _safe_stem(stem: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", stem.strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "run"


def make_run_log_path(stem: str, *, timestamp: bool = True) -> Path:
    """``results/logs/<stem>[_YYYYMMDD_HHMMSS].log``"""
    ensure_results_dirs()
    base = _safe_stem(stem)
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return RESULTS_LOGS_DIR / f"{base}_{ts}.log"
    return RESULTS_LOGS_DIR / f"{base}.log"


def mirror_log(source: Path, subdir: str, dest_name: str) -> Path | None:
    """Copia um ``.log`` existente para ``results/logs/<subdir>/``."""
    if not source.is_file():
        return None
    ensure_results_dirs()
    dest_dir = RESULTS_LOGS_DIR / _safe_stem(subdir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dest_name
    shutil.copy2(source, dest)
    return dest


def write_results_config(payload: str, filename: str) -> Path:
    """Grava JSON de configuração em ``results/config/``."""
    ensure_results_dirs()
    out = RESULTS_CONFIG_DIR / filename
    out.write_text(payload, encoding="utf-8")
    return out
