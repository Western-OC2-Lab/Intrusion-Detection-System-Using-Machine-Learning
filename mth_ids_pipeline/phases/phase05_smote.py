"""
Fase 5: SMOTE no conjunto de treino (notebook: BruteForce e Infiltration → 1000).

CAN (artigo): SMOTE **desativado** por padrão — o dataset já possui >490k amostras/classe
minoritária; oversampling distorce a representação do barramento.

Saída: 05_train_after_smote.parquet, 05_test_unchanged.parquet
"""

from __future__ import annotations

import inspect
import json
import shutil
import warnings
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE

try:
    from mth_ids_pipeline.cli import init_paths, phase_parser
    from mth_ids_pipeline.config import (
        CICIDS2017_MERGED_LABEL_NAMES,
        DEFAULT_SMOTE_TARGETS,
        P04_TEST_FSS,
        P04_TRAIN_FSS,
        P05_TEST,
        P05_TRAIN_SMOTE,
        is_can_automotive_context,
    )
except ImportError:
    from mth_ids_pipeline.cli import init_paths, phase_parser
    from mth_ids_pipeline.config import (
        CICIDS2017_MERGED_LABEL_NAMES,
        DEFAULT_SMOTE_TARGETS,
        P04_TEST_FSS,
        P04_TRAIN_FSS,
        P05_TEST,
        P05_TRAIN_SMOTE,
        is_can_automotive_context,
    )

try:
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.io.reporting import write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 5 — SMOTE")
    parser.add_argument(
        "--smote-strategy",
        type=str,
        default=None,
        help='JSON, ex.: {"2":1000,"4":1000} (default: config do protocolo)',
    )
    parser.add_argument(
        "--force-smote",
        action="store_true",
        help="Forçar SMOTE mesmo em datasets CAN (padrão: ignorado no CAN)",
    )
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    paths = init_paths(args)
    output_dir = paths.intermediate
    tr_path = output_dir / P04_TRAIN_FSS
    te_path = output_dir / P04_TEST_FSS
    label_col = "Label"

    train_df = pd.read_parquet(tr_path)
    test_df = pd.read_parquet(te_path)

    is_can = is_can_automotive_context(
        intermediate_dir=args.intermediate_dir,
        columns=train_df.columns,
    )
    train_out = output_dir / P05_TRAIN_SMOTE
    test_out = output_dir / P05_TEST

    if is_can and not args.force_smote:
        shutil.copy2(tr_path, train_out)
        shutil.copy2(te_path, test_out)
        print(
            "CAN detectado: SMOTE ignorado (artigo Yang et al., 2022 — "
            "dataset já balanceado com >490k amostras/classe minoritária)."
        )
        print(f"Salvo: {train_out} shape={train_df.shape} (cópia da fase 4)")
        print(f"Salvo: {test_out} shape={test_df.shape} (cópia da fase 4)")
        report = {
            "train_input": str(tr_path),
            "test_input": str(te_path),
            "train_output": str(train_out),
            "test_output": str(test_out),
            "smote_skipped": True,
            "can_context": True,
            "reason": "CAN artigo: SMOTE desnecessário e prejudicial",
            "random_state": int(args.random_state),
        }
        report_path = write_report(paths.reports, "phase05_smote", report)
        print(f"Relatorio salvo em: {report_path}")
        return

    y_train = train_df[label_col].values
    X_train = train_df.drop(columns=[label_col]).values

    orig_counts = pd.Series(y_train).value_counts()
    print("Contagem de rótulos no conjunto de treino original:")
    print(orig_counts)

    strategy = (
        {int(k): int(v) for k, v in json.loads(args.smote_strategy).items()}
        if args.smote_strategy
        else dict(DEFAULT_SMOTE_TARGETS)
    )
    print(
        "Estratégia SMOTE (rótulo → alvo):",
        {
            f"{k} ({CICIDS2017_MERGED_LABEL_NAMES.get(k, '?')})": v
            for k, v in strategy.items()
        },
    )
    kw: dict = {"sampling_strategy": strategy}
    if "n_jobs" in inspect.signature(SMOTE.__init__).parameters:
        kw["n_jobs"] = -1
    if "random_state" in inspect.signature(SMOTE.__init__).parameters:
        kw["random_state"] = int(args.random_state)
    smote = SMOTE(**kw)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    smote_counts = pd.Series(y_res).value_counts()
    print("Contagem de rótulos no conjunto de treino após SMOTE:")
    print(smote_counts)

    fcols = [c for c in train_df.columns if c != label_col]
    out_train = pd.DataFrame(X_res, columns=fcols)
    out_train[label_col] = y_res
    out_train.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)
    print(f"Salvo: {train_out} shape={out_train.shape}")
    print(f"Salvo: {test_out} shape={test_df.shape}")

    report = {
        "train_input": str(tr_path),
        "test_input": str(te_path),
        "train_output": str(train_out),
        "test_output": str(test_out),
        "smote_skipped": False,
        "can_context": is_can,
        "smote_sampling_strategy": {str(k): v for k, v in strategy.items()},
        "train_counts_before": {str(k): int(v) for k, v in orig_counts.items()},
        "train_counts_after": {str(k): int(v) for k, v in smote_counts.items()},
        "random_state": int(args.random_state),
    }
    report_path = write_report(paths.reports, "phase05_smote", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
