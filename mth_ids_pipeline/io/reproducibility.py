"""Controle de reprodutibilidade: seeds globais, versões e registro de execução."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from importlib import metadata
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RANDOM_STATE = 0

PACKAGE_VERSIONS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "imbalanced-learn",
    "hyperopt",
    "scikit-optimize",
)


def set_global_seeds(seed: int = DEFAULT_RANDOM_STATE) -> None:
    """Define seeds globais para Python, NumPy e variável de ambiente PYTHONHASHSEED."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def numpy_random_state(seed: int = DEFAULT_RANDOM_STATE) -> np.random.Generator:
    """RNG NumPy para Hyperopt ``fmin(..., rstate=...)`` (requer ``Generator`` desde hyperopt 0.2.6+)."""
    return np.random.default_rng(seed)


def collect_environment_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    for pkg in PACKAGE_VERSIONS:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def _json_safe(value: Any) -> Any:
    """Converte valores comuns do pipeline para tipos serializáveis em JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def log_run_config(
    report_dir: Path,
    *,
    run_name: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persiste parâmetros e versões de bibliotecas de uma execução."""
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "timestamp": int(time.time()),
        "environment": collect_environment_versions(),
        "config": _json_safe(config),
        "non_deterministic_components": [
            "KernelPCA (sklearn, sem random_state)",
            "MiniBatchKMeans / paralelismo (n_jobs=-1) pode variar ordem numérica",
        ],
    }
    if extra:
        payload.update(_json_safe(extra))
    out = report_dir / f"{run_name}_config.json"
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    out.write_text(text, encoding="utf-8")
    try:
        from mth_ids_pipeline.io.results_io import write_results_config

        write_results_config(text, out.name)
    except ImportError:
        pass
    return out
