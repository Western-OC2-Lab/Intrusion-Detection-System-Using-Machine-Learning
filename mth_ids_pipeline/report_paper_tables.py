"""
Gera tabelas de comparação com o artigo MTH-IDS (Tabelas VII, IX e X).

Uso (comparativo completo — grava em results/ fora de data/):
  python -m mth_ids_pipeline.report_paper_tables --table all \\
    --merged-dir data/pipeline_mth_ids_merged \\
    --loao-root data/pipeline_mth_ids_fine/anomaly/loao

Saída padrão: results/paper_comparison.json + results/tables_report.txt
Tabelas individuais: --table vii | ix | x | notebook
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

from mth_ids_pipeline.config import (
    resolve_can_label_names,
    CICIDS2017_FINE_LABEL_NAMES,
    INTERMEDIATE_DIR_CAN_INTRUSION_FINE,
    INTERMEDIATE_DIR_CAN_INTRUSION_MERGED,
    INTERMEDIATE_DIR_CAN_OTIDS_FINE,
    INTERMEDIATE_DIR_CAN_OTIDS_MERGED,
    INTERMEDIATE_DIR_FINE,
    INTERMEDIATE_DIR_MERGED,
    INTERMEDIATE_DIR_UNSW_NB15_FINE,
    INTERMEDIATE_DIR_UNSW_NB15_MERGED,
    PAPER_REFERENCE_LOAO_CAN,
    PAPER_REFERENCE_SUPERVISED_CAN,
    PAPER_TABLE_X_REFERENCE,
    RESULTS_DIR,
    RESULTS_DIR_UNSW_NB15,
    UNSW_NB15_LABEL_NAMES,
    ensure_results_dirs,
    is_can_otids_pipeline_path,
    is_can_pipeline_path,
    is_unsw_pipeline_path,
    results_dir_for_can_pipeline,
)
from mth_ids_pipeline.core.evaluation import (
    NOTEBOOK_REFERENCE_SUPERVISED,
    PAPER_REFERENCE_SUPERVISED,
    compare_metrics,
)
from mth_ids_pipeline.io.loao_reporting import PAPER_REFERENCE_CICIDS2017
from mth_ids_pipeline.io.results_io import make_run_log_path

# Modelo do artigo na Tabela VII (tier multi-class / stacking)
PAPER_TABLE_VII_MODEL = "MTH-IDS (Multi-Class Model)"
# Notebook protocol (--meta-learner xgb + HPO)
NOTEBOOK_STACKING_MODEL = "Stacking meta (HPO XGB)"
# Prefixo usado no paper protocol (--meta-learner best-base)
PAPER_STACKING_PREFIX = "Stacking meta ("


def _dataset_key(merged_dir: Path, loao_root: Path) -> str:
    for path in (merged_dir, loao_root):
        if is_unsw_pipeline_path(path):
            return "unsw_nb15"
    for path in (merged_dir, loao_root):
        if is_can_otids_pipeline_path(path):
            return "can_otids"
    if is_can_pipeline_path(merged_dir) or is_can_pipeline_path(loao_root):
        return "can"
    return "cicids2017"


def _dataset_display_name(dataset: str) -> str:
    return {
        "can": "CAN-intrusion",
        "can_otids": "CAN-OTIDS",
        "unsw_nb15": "UNSW-NB15",
    }.get(dataset, "CICIDS2017")


def _has_paper_reference(dataset: str) -> bool:
    return dataset != "unsw_nb15"


def _supervised_table_label(dataset: str) -> str:
    if dataset == "unsw_nb15":
        return "Supervisionado"
    return "VI" if dataset in ("can", "can_otids") else "VII"


def _loao_table_label(dataset: str) -> str:
    if dataset == "unsw_nb15":
        return "LOAO"
    return "VIII" if dataset in ("can", "can_otids") else "IX"


def _supervised_paper_reference(dataset: str) -> dict:
    key = PAPER_TABLE_VII_MODEL
    if dataset in ("can", "can_otids"):
        return PAPER_REFERENCE_SUPERVISED_CAN[key]
    return PAPER_REFERENCE_SUPERVISED[key]


def _loao_paper_reference(summary: dict, dataset: str) -> dict:
    if dataset in ("can", "can_otids"):
        return summary.get("paper_reference_can", PAPER_REFERENCE_LOAO_CAN)
    return summary.get("paper_reference_cicids2017", PAPER_REFERENCE_CICIDS2017)


def _loao_attack_label_names(loao_root: Path) -> dict[int, str]:
    if is_unsw_pipeline_path(loao_root):
        return {k: v for k, v in UNSW_NB15_LABEL_NAMES.items() if v != "Benign"}
    if is_can_pipeline_path(loao_root):
        table = resolve_can_label_names(pipeline_path=loao_root)
        return {k: v for k, v in table.items() if k > 0}
    return {k: v for k, v in CICIDS2017_FINE_LABEL_NAMES.items() if k > 0}


def _default_loao_root(merged_dir: Path) -> Path:
    if is_unsw_pipeline_path(merged_dir):
        base = INTERMEDIATE_DIR_UNSW_NB15_FINE
    elif is_can_otids_pipeline_path(merged_dir):
        base = INTERMEDIATE_DIR_CAN_OTIDS_FINE
    elif is_can_pipeline_path(merged_dir):
        base = INTERMEDIATE_DIR_CAN_INTRUSION_FINE
    else:
        base = INTERMEDIATE_DIR_FINE
    return base / "anomaly" / "loao"


def _default_merged_dir(dataset: str) -> Path:
    if dataset == "unsw_nb15":
        return INTERMEDIATE_DIR_UNSW_NB15_MERGED
    if dataset == "can_otids":
        return INTERMEDIATE_DIR_CAN_OTIDS_MERGED
    if dataset == "can":
        return INTERMEDIATE_DIR_CAN_INTRUSION_MERGED
    return INTERMEDIATE_DIR_MERGED


def _find_stacking_row(metrics: list[dict]) -> dict | None:
    """Stacking no JSON: paper → ``Stacking meta (<melhor base>)``; notebook → ``Stacking meta (HPO XGB)``."""
    for row in metrics:
        if row.get("model") == NOTEBOOK_STACKING_MODEL:
            return row
    for row in metrics:
        name = str(row.get("model", ""))
        if name.startswith(PAPER_STACKING_PREFIX) and name.endswith(")"):
            return row
    for legacy in ("Stacking (XGB meta)",):
        for row in metrics:
            if row.get("model") == legacy:
                return row
    return None


def _load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def report_notebook_comparison(intermediate_dir: Path) -> None:
    """Comparação vs referências do notebook IoTJ."""
    metrics_path = intermediate_dir / "06_supervised_metrics.json"
    if not metrics_path.is_file():
        print(f"Notebook: métricas não encontradas em {metrics_path}")
        return
    metrics = _load_json(metrics_path)
    model_map = {
        "XGBoost (base)": "XGBoost HPO",
        "RandomForest (HPO)": "RandomForest HPO",
        "DecisionTree (HPO)": "DecisionTree HPO",
        "ExtraTrees (HPO)": "ExtraTrees HPO",
        NOTEBOOK_STACKING_MODEL: "Stacking meta HPO",
    }
    rows: list[dict] = []
    for entry in metrics:
        ref_key = model_map.get(entry["model"])
        if not ref_key and str(entry.get("model", "")).startswith(PAPER_STACKING_PREFIX):
            ref_key = "Stacking meta HPO"
        if not ref_key or ref_key not in NOTEBOOK_REFERENCE_SUPERVISED:
            continue
        ref = NOTEBOOK_REFERENCE_SUPERVISED[ref_key]
        for cmp in compare_metrics(entry, ref, metric_keys=("accuracy", "f1_weighted")):
            cmp["label"] = f"{entry['model']} / {cmp['metric']}"
            rows.append(cmp)
    if not rows:
        return
    print("\n" + "=" * 72)
    print("Comparação vs Notebook (hold-out)")
    print("=" * 72)
    print(f"{'Modelo/Métrica':<28} {'Ref':>10} {'Reprod':>10} {'Diff abs':>10} {'Diff %':>8}")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['label']:<28} "
            f"{row['reference']:>10.6f} "
            f"{row['reproduced']:>10.6f} "
            f"{row['absolute_diff']:>10.6f} "
            f"{row['percent_diff']:>7.2f}%"
        )


def _paper_ref_pct(ref: dict, key_pct: str) -> float:
    return float(ref.get(key_pct, 0)) / 100.0


def report_table_vii(intermediate_dir: Path, *, dataset: str = "cicids2017") -> dict | None:
    metrics_path = intermediate_dir / "06_supervised_metrics.json"
    cv_path = intermediate_dir / "phase_reports" / "phase06_supervised_models.json"
    table_no = _supervised_table_label(dataset)
    if not metrics_path.is_file():
        print(f"Tabela {table_no}: métricas não encontradas em {metrics_path}")
        return None

    metrics = _load_json(metrics_path)
    cv_report = _load_json(cv_path) if cv_path else None
    paper = _supervised_paper_reference(dataset)
    ref_acc = _paper_ref_pct(paper, "accuracy_pct")
    ref_f1 = float(paper.get("f1", 0))

    dataset_name = _dataset_display_name(dataset)
    print("\n" + "=" * 72)
    print(f"TABELA {table_no} — Supervisionado ({dataset_name}, multi-class, hold-out 80/20)")
    print("=" * 72)
    print(f"{'Modelo':<24} {'Acc':>10} {'F1(w)':>10}")
    print("-" * 72)
    stacking_row = _find_stacking_row(metrics)
    for row in metrics:
        name = row.get("model", "?")
        acc = float(row.get("accuracy", 0))
        f1 = float(row.get("f1_weighted", 0))
        print(f"{name:<24} {acc:>10.6f} {f1:>10.6f}")

    repro_model = stacking_row.get("model") if stacking_row else NOTEBOOK_STACKING_MODEL
    out: dict = {
        "dataset": dataset,
        "table": table_no,
        "paper_model": PAPER_TABLE_VII_MODEL,
        "repro_model": repro_model,
    }
    if _has_paper_reference(dataset):
        print("\nComparação vs artigo (tier multi-class / stacking):")
        print(f"{'Métrica':<12} {'Reprod':>12} {'Artigo':>12} {'Diff':>12}")
        print("-" * 72)
        if stacking_row:
            acc = float(stacking_row.get("accuracy", 0))
            f1 = float(stacking_row.get("f1_weighted", 0))
            print(f"{'Acc':<12} {acc:>12.6f} {ref_acc:>12.6f} {acc - ref_acc:>+12.6f}")
            print(f"{'F1(w)':<12} {f1:>12.6f} {ref_f1:>12.6f} {f1 - ref_f1:>+12.6f}")
            out["accuracy"] = {"reproduced": acc, "reference": ref_acc, "delta": acc - ref_acc}
            out["f1_weighted"] = {"reproduced": f1, "reference": ref_f1, "delta": f1 - ref_f1}
        else:
            print(
                f"  (nenhum stacking encontrado em 06_supervised_metrics.json; "
                f"esperado '{NOTEBOOK_STACKING_MODEL}' ou '{PAPER_STACKING_PREFIX}…)')"
            )
    elif stacking_row:
        acc = float(stacking_row.get("accuracy", 0))
        f1 = float(stacking_row.get("f1_weighted", 0))
        print("\nMétricas do stacking (sem referência do artigo para UNSW-NB15):")
        print(f"  Acc:   {acc:.6f}")
        print(f"  F1(w): {f1:.6f}")
        out["accuracy"] = {"reproduced": acc}
        out["f1_weighted"] = {"reproduced": f1}
    else:
        print(
            f"\n  (nenhum stacking encontrado em 06_supervised_metrics.json; "
            f"esperado '{NOTEBOOK_STACKING_MODEL}' ou '{PAPER_STACKING_PREFIX}…)')"
        )

    if cv_report and cv_report.get("cv_reports"):
        print("\n10-fold CV no treino (reprodução):")
        for name, rep in cv_report["cv_reports"].items():
            print(f"  {name}: {rep['mean']:.4f} ± {rep['std']:.4f}")
    return out


def report_table_ix(loao_root: Path, *, dataset: str = "cicids2017") -> dict | None:
    table_no = _loao_table_label(dataset)
    summary_path = loao_root / "loao_summary.json"
    summary = _load_json(summary_path)
    if not summary:
        alt = loao_root.parent / "phase_reports" / "phase12_anomaly_loao.json"
        summary = _load_json(alt) if alt.is_file() else None
    if not summary and loao_root.is_dir():
        try:
            from mth_ids_pipeline.io.loao_reporting import build_loao_summary, write_loao_summary

            label_names = _loao_attack_label_names(loao_root)
            summary = build_loao_summary(
                loao_root,
                label_names,
                attacks_in_dataset=len(label_names),
            )
            if summary.get("per_attack"):
                write_loao_summary(loao_root, summary)
        except ImportError:
            pass
    if not summary:
        print(f"Tabela {table_no}: resumo LOAO nao encontrado em {loao_root}")
        return None

    ref = _loao_paper_reference(summary, dataset)
    mean_f1 = float(summary.get("mean_f1", 0))
    mean_dr = float(summary.get("mean_detection_rate", 0))
    mean_far = float(summary.get("mean_false_alarm_rate", 0))
    ref_f1 = float(ref.get("mean_f1", 0.80013 if dataset == "cicids2017" else 0.96307))
    ref_dr = float(ref.get("mean_dr_pct", 75.943 if dataset == "cicids2017" else 93.740)) / 100.0
    ref_far = float(ref.get("mean_far_pct", 13.882 if dataset == "cicids2017" else 0.128)) / 100.0
    n_done = len(summary.get("per_attack", []))
    n_planned = int(
        summary.get(
            "attacks_in_dataset",
            {"can": 3, "can_otids": 3, "unsw_nb15": 9}.get(dataset, 14),
        )
    )

    dataset_name = _dataset_display_name(dataset)
    print("\n" + "=" * 72)
    print(f"TABELA {table_no} — Anomaly LOAO ({dataset_name}, média sobre ataques concluídos)")
    print("=" * 72)
    print(f"Ataques concluídos: {n_done}/{n_planned}")
    if _has_paper_reference(dataset):
        print(f"{'Métrica':<12} {'Reprod':>12} {'Artigo':>12} {'Diff':>12}")
        print("-" * 72)
        print(f"{'F1':<12} {mean_f1:>12.5f} {ref_f1:>12.5f} {mean_f1 - ref_f1:>+12.5f}")
        print(f"{'DR':<12} {mean_dr:>12.4f} {ref_dr:>12.4f} {mean_dr - ref_dr:>+12.4f}")
        print(f"{'FAR':<12} {mean_far:>12.4f} {ref_far:>12.4f} {mean_far - ref_far:>+12.4f}")
    else:
        print(f"{'Métrica':<12} {'Reprod':>12}")
        print("-" * 72)
        print(f"{'F1':<12} {mean_f1:>12.5f}")
        print(f"{'DR':<12} {mean_dr:>12.4f}")
        print(f"{'FAR':<12} {mean_far:>12.4f}")
    print(f"\n{'Ataque':<6} {'Nome':<22} {'F1':>8} {'DR':>8} {'FAR':>8} {'N test':>8}")
    print("-" * 72)
    for row in summary.get("per_attack", []):
        print(
            f"{row.get('attack_label', '?'):<6} "
            f"{str(row.get('attack_name', ''))[:22]:<22} "
            f"{(row.get('f1') or 0):>8.4f} "
            f"{(row.get('detection_rate') or 0):>8.4f} "
            f"{(row.get('false_alarm_rate') or 0):>8.4f} "
            f"{(row.get('test_rows') or 0):>8}"
        )
    if n_done < n_planned:
        note = "execute LOAO completo"
        if _has_paper_reference(dataset):
            note += " para comparar com o artigo"
        print(f"\nNota: média parcial — {note}.")
    out = {
        "dataset": dataset,
        "table": table_no,
        "attacks_done": n_done,
        "attacks_planned": n_planned,
        "per_attack": summary.get("per_attack", []),
    }
    if _has_paper_reference(dataset):
        out["mean_f1"] = {"reproduced": mean_f1, "reference": ref_f1, "delta": mean_f1 - ref_f1}
        out["mean_dr"] = {"reproduced": mean_dr, "reference": ref_dr, "delta": mean_dr - ref_dr}
        out["mean_far"] = {"reproduced": mean_far, "reference": ref_far, "delta": mean_far - ref_far}
    else:
        out["mean_f1"] = {"reproduced": mean_f1}
        out["mean_dr"] = {"reproduced": mean_dr}
        out["mean_far"] = {"reproduced": mean_far}
    return out


def report_table_x(intermediate_dir: Path, *, dataset: str = "cicids2017") -> dict | None:
    """Tabela X — sistema completo no hold-out (fase 13)."""
    report_path = intermediate_dir / "phase_reports" / "phase13_full_system_eval.json"
    data = _load_json(report_path)
    protocol_flag = {
        "can": "can",
        "can_otids": "can_otids",
        "unsw_nb15": "unsw",
    }.get(dataset, "paper")
    if not data:
        print(
            f"Tabela X: relatório não encontrado em {report_path}\n"
            "Execute:\n"
            f"  python -m mth_ids_pipeline.run_global_anomaly --protocol {protocol_flag}\n"
            "  python -m mth_ids_pipeline.run_eval "
            f"--intermediate-dir {intermediate_dir} "
            f"--work-dir {intermediate_dir / 'anomaly' / 'global'}"
        )
        return None

    binary = data.get("binary_metrics") or {}
    ref = PAPER_TABLE_X_REFERENCE.get(dataset)
    acc_pct = float(data.get("accuracy", 0)) * 100.0
    dr_pct = float(binary.get("detection_rate", 0)) * 100.0
    far_pct = float(binary.get("false_alarm_rate", 0)) * 100.0
    f1 = float(binary.get("f1", 0))
    split_note = data.get("test_split") or "hold-out 80/20 (artigo: 70/30)"

    dataset_name = _dataset_display_name(dataset)
    print("\n" + "=" * 72)
    print(f"TABELA X — Sistema completo no hold-out ({dataset_name})")
    print("=" * 72)
    print(f"Protocolo: {split_note}")
    if ref is not None:
        print(f"{'Métrica':<12} {'Reprod':>12} {'Artigo':>12} {'Diff':>12}")
        print("-" * 72)
        print(f"{'Acc (%)':<12} {acc_pct:>12.2f} {ref['accuracy_pct']:>12.2f} {acc_pct - ref['accuracy_pct']:>+12.2f}")
        print(f"{'DR (%)':<12} {dr_pct:>12.2f} {ref['detection_rate_pct']:>12.2f} {dr_pct - ref['detection_rate_pct']:>+12.2f}")
        print(f"{'FAR (%)':<12} {far_pct:>12.4f} {ref['false_alarm_rate_pct']:>12.4f} {far_pct - ref['false_alarm_rate_pct']:>+12.4f}")
        print(f"{'F1':<12} {f1:>12.4f} {ref['f1']:>12.4f} {f1 - ref['f1']:>+12.4f}")
    else:
        print(f"{'Métrica':<12} {'Reprod':>12}")
        print("-" * 72)
        print(f"{'Acc (%)':<12} {acc_pct:>12.2f}")
        print(f"{'DR (%)':<12} {dr_pct:>12.2f}")
        print(f"{'FAR (%)':<12} {far_pct:>12.4f}")
        print(f"{'F1':<12} {f1:>12.4f}")
    if data.get("route_stats"):
        print(f"\nRoteamento: {data['route_stats']}")
    figures = intermediate_dir / "figures"
    print(f"CM: {figures / 'fig_multiclass_cm.png'} | {figures / 'fig_binary_cm.png'}")
    out = {
        "dataset": dataset,
        "protocol": split_note,
    }
    if ref is not None:
        out["accuracy_pct"] = {"reproduced": acc_pct, "reference": ref["accuracy_pct"], "delta": acc_pct - ref["accuracy_pct"]}
        out["detection_rate_pct"] = {"reproduced": dr_pct, "reference": ref["detection_rate_pct"], "delta": dr_pct - ref["detection_rate_pct"]}
        out["false_alarm_rate_pct"] = {"reproduced": far_pct, "reference": ref["false_alarm_rate_pct"], "delta": far_pct - ref["false_alarm_rate_pct"]}
        out["f1"] = {"reproduced": f1, "reference": ref["f1"], "delta": f1 - ref["f1"]}
    else:
        out["accuracy_pct"] = {"reproduced": acc_pct}
        out["detection_rate_pct"] = {"reproduced": dr_pct}
        out["false_alarm_rate_pct"] = {"reproduced": far_pct}
        out["f1"] = {"reproduced": f1}
    return out


def _run_reports(
    *,
    merged_dir: Path,
    loao_root: Path,
    table: str,
) -> tuple[dict, str]:
    """Executa relatórios; retorna (comparison dict, texto formatado)."""
    buf = io.StringIO()
    dataset = _dataset_key(merged_dir, loao_root)

    comparison: dict = {
        "merged_dir": str(merged_dir),
        "loao_root": str(loao_root),
        "dataset": dataset,
    }

    with contextlib.redirect_stdout(buf):
        if table == "all":
            print("=" * 72)
            sup_label = _supervised_table_label(dataset)
            loao_label = _loao_table_label(dataset)
            title = _dataset_display_name(dataset)
            header = f"COMPARATIVO MTH-IDS ({title})"
            if _has_paper_reference(dataset):
                header = f"COMPARATIVO MTH-IDS vs artigo ({title})"
            print(header)
            print("=" * 72)
            print(f"Supervisionado / Tabela {sup_label} / X: {merged_dir}")
            print(f"LOAO / Tabela {loao_label}:          {loao_root}")

        if table in ("notebook", "all") and dataset == "cicids2017":
            report_notebook_comparison(merged_dir)
        if table in ("vii", "all"):
            comparison["table_vii"] = report_table_vii(merged_dir, dataset=dataset)
        if table in ("ix", "all"):
            comparison["table_ix"] = report_table_ix(loao_root, dataset=dataset)
        if table in ("x", "all"):
            comparison["table_x"] = report_table_x(merged_dir, dataset=dataset)

    report_text = buf.getvalue()
    print(report_text, end="" if report_text.endswith("\n") else "\n")
    return comparison, report_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Relatórios Tabela VII / IX / X vs artigo")
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=INTERMEDIATE_DIR_MERGED,
        help="Pipeline merged (supervisionado + anomaly global / Tabela X)",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Alias de --merged-dir (retrocompatível)",
    )
    parser.add_argument(
        "--loao-root",
        type=Path,
        default=None,
        help="Raiz LOAO (default: pipeline_can_otids_fine ou pipeline_mth_ids_fine conforme --merged-dir)",
    )
    parser.add_argument(
        "--table",
        choices=("vii", "ix", "x", "notebook", "all"),
        default="all",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Salvar tabelas em JSON + TXT fora de data/ (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Salvar comparativo estruturado (JSON); sobrescreve caminho em --results-dir",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Não gravar arquivos em disco (só imprimir no terminal)",
    )
    args = parser.parse_args()

    merged_dir = args.intermediate_dir or args.merged_dir
    loao_root = args.loao_root or _default_loao_root(merged_dir)
    dataset = _dataset_key(merged_dir, loao_root)
    if args.results_dir == RESULTS_DIR and dataset in ("can", "can_otids"):
        results_dir = results_dir_for_can_pipeline(merged_dir)
    elif args.results_dir == RESULTS_DIR and dataset == "unsw_nb15":
        results_dir = RESULTS_DIR_UNSW_NB15
    else:
        results_dir = args.results_dir

    comparison, report_text = _run_reports(
        merged_dir=merged_dir,
        loao_root=loao_root,
        table=args.table,
    )

    if args.no_save:
        return

    ensure_results_dirs()
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.save_json or (results_dir / "paper_comparison.json")
    txt_path = results_dir / "tables_report.txt"
    log_path = make_run_log_path(f"report_tables_{args.table}")
    json_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    txt_path.write_text(report_text, encoding="utf-8")
    log_path.write_text(report_text, encoding="utf-8")
    print(f"\nComparativo JSON: {json_path}")
    print(f"Relatório texto:  {txt_path}")
    print(f"Log do relatório: {log_path}")


if __name__ == "__main__":
    main()
