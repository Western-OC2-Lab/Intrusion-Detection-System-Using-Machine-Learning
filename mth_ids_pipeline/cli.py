"""CLI mínimo compartilhado pelas fases do pipeline MTH-IDS."""

from __future__ import annotations

import argparse
from pathlib import Path

from mth_ids_pipeline.config import PipelinePaths, ensure_intermediate_dirs, get_pipeline_paths
from mth_ids_pipeline.io.subprocess_env import configure_stdio_utf8


def phase_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Raiz dos artefatos (default: data/pipeline_mth_ids)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Relatórios JSON (default: <intermediate>/phase_reports)",
    )
    return parser


def init_paths(args) -> PipelinePaths:
    configure_stdio_utf8()
    paths = get_pipeline_paths(args.intermediate_dir, args.report_dir)
    ensure_intermediate_dirs(paths.intermediate)
    return paths


def as_parquet(filename: str) -> str:
    return filename.replace(".csv", ".parquet")


def supervised_path(paths: PipelinePaths, filename: str) -> Path:
    return paths.intermediate / as_parquet(filename)


def add_work_dir(parser: argparse.ArgumentParser) -> None:
    """Diretório de trabalho do ramo anomaly (entrada e saída da fase)."""
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Pasta anomaly da fase ou subpasta LOAO (default: <intermediate>/anomaly)",
    )


def resolve_work_dir(args, paths: PipelinePaths) -> Path:
    work = getattr(args, "work_dir", None) or paths.anomaly
    work.mkdir(parents=True, exist_ok=True)
    return work
