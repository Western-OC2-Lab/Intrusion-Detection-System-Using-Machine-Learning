"""
Fase 13: avaliação end-to-end do MTH-IDS (tiers 1–4 em cascata).

Pré-requisitos (artefatos salvos automaticamente pelas fases anteriores):
  - Fase 4/6: ``<intermediate>/models/supervised/``
  - Fase 8:   ``fitted_*.joblib`` em ``--anomaly-work-dir``
  - Fase 11:  ``<anomaly-work-dir>/models/anomaly/``

Não re-treina modelos; apenas carrega artefatos e avalia o hold-out (05_test).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

try:
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.core.inference import run_full_system_inference
    from mth_ids_pipeline.config import default_benign_label
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.core.inference import run_full_system_inference
    from mth_ids_pipeline.config import default_benign_label
    from mth_ids_pipeline.io.reporting import write_report


def _load_test_size_from_report(report_dir: Path, default: float) -> float:
    report_path = report_dir / "phase04_feature_engineering.json"
    if not report_path.is_file():
        return default
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return float(data.get("test_size", default))


def _plot_confusion_matrix(
    cm: np.ndarray,
    *,
    title: str,
    labels: list[str] | None,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap_kw: dict = {
        "data": cm,
        "annot": True,
        "fmt": "d",
        "linewidths": 0.5,
        "cmap": "Blues",
        "ax": ax,
    }
    if labels is not None:
        heatmap_kw["xticklabels"] = labels
        heatmap_kw["yticklabels"] = labels
    sns.heatmap(**heatmap_kw)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figura salva: {output_path}")


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 13 — avaliação end-to-end MTH-IDS")
    add_work_dir(parser)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--benign-label", type=int, default=None)
    parser.add_argument(
        "--anomaly-attack-label",
        type=int,
        default=99,
        help="Classe atribuída quando tier anomaly detecta ataque (sem subtipo)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Pasta para confusion matrices (default: <intermediate>/figures)",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    intermediate = paths.intermediate
    benign_label = (
        args.benign_label
        if args.benign_label is not None
        else default_benign_label(intermediate_dir=intermediate)
    )
    test_size = (
        float(args.test_size)
        if args.test_size is not None
        else _load_test_size_from_report(paths.reports, 0.2)
    )

    print(f"Avaliação end-to-end: supervised={intermediate} | anomaly={work}")
    print(f"Hold-out: test_size={test_size}, random_state={args.random_state}")

    result = run_full_system_inference(
        intermediate_dir=intermediate,
        anomaly_work_dir=work,
        test_size=test_size,
        random_state=args.random_state,
        benign_label=benign_label,
        anomaly_attack_pred_label=args.anomaly_attack_label,
    )

    binary = result["binary"]
    print(
        f"\n=== MTH-IDS end-to-end (hold-out) ===\n"
        f"Accuracy: {result['accuracy']:.6f}\n"
        f"F1 (weighted): {result['f1_weighted']:.6f}\n"
        f"DR: {binary['detection_rate']:.6f}  FAR: {binary['false_alarm_rate']:.6f}  "
        f"F1 (binário): {binary['f1']:.6f}"
    )
    print(f"Roteamento: {result['route_stats']}")
    print(f"CM multi-classe:\n{np.array(result['confusion_matrix_multiclass'])}")
    print(f"CM binária:\n{np.array(result['confusion_matrix_binary'])}")

    figures_dir = args.figures_dir or (intermediate / "figures")
    report = {
        "intermediate_dir": str(intermediate),
        "anomaly_work_dir": str(work),
        "test_size": test_size,
        "random_state": args.random_state,
        "benign_label": benign_label,
        "anomaly_attack_pred_label": args.anomaly_attack_label,
        "accuracy": result["accuracy"],
        "f1_weighted": result["f1_weighted"],
        "precision_weighted": result["precision_weighted"],
        "recall_weighted": result["recall_weighted"],
        "binary_metrics": binary,
        "confusion_matrix_multiclass": result["confusion_matrix_multiclass"],
        "confusion_matrix_binary": result["confusion_matrix_binary"],
        "route_stats": result["route_stats"],
        "figures_dir": str(figures_dir),
    }
    report_path = write_report(paths.reports, "phase13_full_system_eval", report)
    print(f"Relatório salvo em: {report_path}")

    if not args.no_plots:
        cm_multi = np.array(result["confusion_matrix_multiclass"])
        cm_bin = np.array(result["confusion_matrix_binary"])
        _plot_confusion_matrix(
            cm_multi,
            title="MTH-IDS — Confusion Matrix (multi-classe)",
            labels=None,
            output_path=figures_dir / "fig_multiclass_cm.png",
        )
        _plot_confusion_matrix(
            cm_bin,
            title="MTH-IDS — Confusion Matrix (binária)",
            labels=["BENIGN", "ATAQUE"],
            output_path=figures_dir / "fig_binary_cm.png",
        )


if __name__ == "__main__":
    main()
