"""Avaliação end-to-end MTH-IDS (fase 13 / Tabela X).

Exemplo (CICIDS2017 — detector anomaly global):
  python -m mth_ids_pipeline.run_global_anomaly
  python -m mth_ids_pipeline.run_eval \\
    --intermediate-dir data/pipeline_mth_ids_merged \\
    --work-dir data/pipeline_mth_ids_merged/anomaly/global
  python -m mth_ids_pipeline.report_paper_tables --table x \\
    --intermediate-dir data/pipeline_mth_ids_merged
"""

from __future__ import annotations

import sys
from pathlib import Path

from mth_ids_pipeline.io.results_io import make_run_log_path
from mth_ids_pipeline.io.run_log import RunLog
from mth_ids_pipeline.io.subprocess_env import configure_stdio_utf8


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    configure_stdio_utf8()
    log_path = make_run_log_path("eval_phase13")
    cmd = [sys.executable, "-m", "mth_ids_pipeline.phases.phase13_full_system_eval", *sys.argv[1:]]
    with RunLog(log_path) as log:
        log.emit(" ".join(cmd))
        log.run_subprocess(cmd, cwd=_repo_root())
    print(f"Log da execução: {log_path}")


if __name__ == "__main__":
    main()
