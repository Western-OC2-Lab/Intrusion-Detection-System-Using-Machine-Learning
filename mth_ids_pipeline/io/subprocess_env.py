"""Ambiente de subprocesso compatível com Windows (stdout UTF-8)."""

from __future__ import annotations

import os
import sys
import warnings


def suppress_worker_warnings() -> None:
    """Silencia avisos conhecidos de workers sklearn/joblib (pkg_resources / multiprocessing)."""
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.queues")


def _python_warnings_env() -> str:
    return ",".join(
        (
            "ignore:pkg_resources is deprecated as an API:UserWarning",
            "ignore::UserWarning:multiprocessing.queues",
        )
    )


def configure_stdio_utf8() -> None:
    """Evita ``UnicodeEncodeError`` no console Windows (cp1252)."""
    suppress_worker_warnings()
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def utf8_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    extra = _python_warnings_env()
    existing = env.get("PYTHONWARNINGS", "").strip()
    if not existing:
        env["PYTHONWARNINGS"] = extra
    elif extra not in existing:
        env["PYTHONWARNINGS"] = f"{existing},{extra}"
    return env
