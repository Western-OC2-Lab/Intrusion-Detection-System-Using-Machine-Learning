"""
Fase 1: carrega CSV bruto, Z-score opcional nas colunas numéricas, preenche NaN com 0.

CAN (artigo Yang et al., 2022): **sem** Z-score nesta fase — normalização após k-means (fase 2).
CICIDS / UNSW: Z-score no carregamento (padrão).

Saída:
data/pipeline_mth_ids/01_preprocessed.parquet
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

try:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import DEFAULT_RAW_CSV, P01_PREPROCESSED, is_can_automotive_context
except ImportError:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import DEFAULT_RAW_CSV, P01_PREPROCESSED, is_can_automotive_context

try:
    from mth_ids_pipeline.io.reporting import dataset_report, write_report
except ImportError:
    from mth_ids_pipeline.io.reporting import dataset_report, write_report


try:
    from mth_ids_pipeline.core.preprocessing import load_and_preprocess as _load_and_preprocess
except ImportError:
    from mth_ids_pipeline.core.preprocessing import load_and_preprocess as _load_and_preprocess


def load_and_preprocess(raw_csv: Path, *, zscore: bool = True) -> pd.DataFrame:
    return _load_and_preprocess(raw_csv, zscore=zscore)


def main() -> None:
    total_start = time.time()

    parser = phase_parser("Fase 1 — load + preprocess")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_CSV, help="CSV bruto")
    parser.add_argument(
        "--no-zscore",
        action="store_true",
        help="Não aplicar Z-score no carregamento (padrão automático para datasets CAN)",
    )
    parser.add_argument(
        "--zscore",
        action="store_true",
        help="Forçar Z-score no carregamento (sobrescreve detecção CAN)",
    )
    args = parser.parse_args()

    paths = init_paths(args)
    out = supervised_path(paths, P01_PREPROCESSED)
    is_can = is_can_automotive_context(
        intermediate_dir=args.intermediate_dir,
        input_path=args.input,
    )
    if args.zscore:
        apply_zscore = True
    elif args.no_zscore:
        apply_zscore = False
    else:
        apply_zscore = not is_can
    if is_can and not apply_zscore:
        print(
            "CAN detectado: Z-score adiado para a fase 2 (pós k-means), "
            "conforme protocolo Yang et al. (2022)."
        )
    df = load_and_preprocess(args.input, zscore=apply_zscore)

    # Salva parquet
    print("Salvando arquivo parquet...")

    save_start = time.time()

    df.to_parquet(out, index=False)

    print(f"Arquivo salvo em: {out}")
    print(f"Tempo de salvamento: {time.time() - save_start:.2f}s")
    print(f"Tempo total do pipeline: {time.time() - total_start:.2f}s")

    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    numeric = df.select_dtypes(include="number").columns
    feature_cols = [c for c in numeric if c != label_col]
    report = dataset_report(df, label_col)
    report.update(
        {
            "input": str(args.input),
            "output": str(out),
            "feature_count": len(feature_cols),
            "zscore": apply_zscore,
            "can_context": is_can,
            "duration_s": round(time.time() - total_start, 4),
        }
    )
    report_path = write_report(paths.reports, "phase01_load_preprocess", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()