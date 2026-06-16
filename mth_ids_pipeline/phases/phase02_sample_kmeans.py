"""
Fase 2: LabelEncoder + MiniBatchKMeans (k=1000) + amostragem 0.8% + minoritárias intactas.

CAN (artigo): Z-score **após** amostragem k-means (padrão ``--zscore-after-sample``).
"""

from __future__ import annotations

import argparse
import time
import warnings

import pandas as pd

try:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import (
        DEFAULT_KMEANS_FRAC,
        DEFAULT_MINORITY_LABELS,
        P01_PREPROCESSED,
        P02_SAMPLED_KMEANS,
        parse_minority_labels,
    )
    from mth_ids_pipeline.core.clustering import sample_kmeans as _sample_kmeans
    from mth_ids_pipeline.core.clustering import sample_kmeans_staged as _sample_kmeans_staged
    from mth_ids_pipeline.label_profiles import minority_labels_all_attacks
    from mth_ids_pipeline.io.reporting import dataset_report, write_report
except ImportError:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import (
        DEFAULT_KMEANS_FRAC,
        DEFAULT_MINORITY_LABELS,
        P01_PREPROCESSED,
        P02_SAMPLED_KMEANS,
        parse_minority_labels,
    )
    from mth_ids_pipeline.core.clustering import sample_kmeans as _sample_kmeans
    from mth_ids_pipeline.core.clustering import sample_kmeans_staged as _sample_kmeans_staged
    from label_profiles import minority_labels_all_attacks
    from mth_ids_pipeline.io.reporting import dataset_report, write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    total_start = time.time()

    parser = phase_parser("Fase 2 — sampling k-means")
    parser.add_argument("--n-clusters", type=int, default=1000)
    parser.add_argument("--frac", type=float, default=DEFAULT_KMEANS_FRAC)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--minority-labels",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MINORITY_LABELS),
    )
    parser.add_argument("--auto-minority", action="store_true")
    parser.add_argument(
        "--sample-all-classes",
        action="store_true",
        help="k-means em todas as classes (CAN: sem preservar ataques intactos; frac do protocolo)",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Pula amostragem k-means e usa o dataset completo da fase 1",
    )
    parser.add_argument(
        "--zscore-after-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Z-score após amostragem k-means (padrão True; artigo CAN Tabela VI). "
            "Use --no-zscore-after-sample para desativar (ex.: CICIDS artigo)."
        ),
    )
    parser.add_argument(
        "--sampling-stage",
        action="append",
        default=[],
        metavar="LABELS:FRAC",
        help=(
            "Amostragem k-means em estagios. Ex.: --sampling-stage 3,7:0.008 "
            "--sampling-stage 5,6,4,8:0.10"
        ),
    )
    args = parser.parse_args()

    if not args.zscore_after_sample:
        warnings.warn(
            "Z-score pós-amostragem desativado. O protocolo CAN exige normalização "
            "imediatamente após o k-means (Yang et al., 2022, Tabela VI).",
            stacklevel=1,
        )

    paths = init_paths(args)
    df = pd.read_parquet(supervised_path(paths, P01_PREPROCESSED))

    sampling_stages = _parse_sampling_stages(args.sampling_stage)

    if args.skip_sampling:
        sampled = df.copy()
        minority_labels: tuple[int, ...] = ()
    elif sampling_stages:
        minority_labels = ()
        sampled = _sample_kmeans_staged(
            df,
            sampling_stages,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
        )
    elif args.sample_all_classes:
        minority_labels = ()
        sampled = _sample_kmeans(
            df,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            frac=args.frac,
            minority_labels=minority_labels,
        )
    else:
        if args.auto_minority:
            minority_labels = minority_labels_all_attacks(df)
        else:
            minority_labels = parse_minority_labels(args.minority_labels)
        sampled = _sample_kmeans(
            df,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            frac=args.frac,
            minority_labels=minority_labels,
        )

    if args.zscore_after_sample:
        try:
            from mth_ids_pipeline.core.preprocessing import zscore_normalize
        except ImportError:
            from mth_ids_pipeline.core.preprocessing import zscore_normalize
        label_col_pre = "Label" if "Label" in sampled.columns else sampled.columns[-1]
        sampled = zscore_normalize(sampled, label_col=label_col_pre)
        print("Z-score aplicado após amostragem k-means (protocolo CAN artigo).")

    out = supervised_path(paths, P02_SAMPLED_KMEANS)
    sampled.to_parquet(out, index=False)
    label_col = "Label" if "Label" in sampled.columns else sampled.columns[-1]
    print(sampled[label_col].value_counts())
    print(f"Salvo: {out} shape={sampled.shape}")

    report = dataset_report(sampled, label_col)
    report.update(
        {
            "input": str(supervised_path(paths, P01_PREPROCESSED)),
            "output": str(out),
            "n_clusters": args.n_clusters,
            "frac": args.frac,
            "minority_labels": list(minority_labels),
            "auto_minority": args.auto_minority,
            "sample_all_classes": args.sample_all_classes,
            "skip_sampling": args.skip_sampling,
            "zscore_after_sample": args.zscore_after_sample,
            "sampling_stages": [
                {"labels": list(labels), "frac": frac}
                for labels, frac in sampling_stages
            ],
            "duration_s": round(time.time() - total_start, 4),
        }
    )
    write_report(paths.reports, "phase02_sample_kmeans", report)


def _parse_sampling_stages(values: list[str]) -> tuple[tuple[tuple[int, ...], float], ...]:
    stages: list[tuple[tuple[int, ...], float]] = []
    for value in values:
        if ":" not in value:
            raise SystemExit(
                f"sampling-stage invalido: {value!r}. Use LABELS:FRAC, ex. 3,7:0.008"
            )
        labels_part, frac_part = value.split(":", 1)
        labels = tuple(int(x.strip()) for x in labels_part.split(",") if x.strip())
        if not labels:
            raise SystemExit(f"sampling-stage sem labels: {value!r}")
        try:
            frac = float(frac_part)
        except ValueError as exc:
            raise SystemExit(f"frac invalido em sampling-stage: {value!r}") from exc
        if not 0 < frac <= 1:
            raise SystemExit(f"frac fora de (0, 1] em sampling-stage: {value!r}")
        stages.append((labels, frac))
    return tuple(stages)


if __name__ == "__main__":
    main()
