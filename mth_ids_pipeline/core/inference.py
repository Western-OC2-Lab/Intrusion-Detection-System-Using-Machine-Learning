"""Inferência em cascata MTH-IDS (tiers 1–4) — fase 13."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from mth_ids_pipeline.config import (
    A01_WITHOUT_PORTSCAN,
    DEFAULT_ANOMALY_ATTACK_PRED_LABEL,
    P02_SAMPLED_KMEANS,
    P05_TEST,
)
from mth_ids_pipeline.core.biased_classifiers import apply_biased_refinement
from mth_ids_pipeline.core.clustering import cl_kmeans_predict_inference
from mth_ids_pipeline.core.dimensionality_reduction import transform_kpca
from mth_ids_pipeline.core.evaluation import binary_dr_far_f1
from mth_ids_pipeline.core.feature_selection import transform_fcbf
from mth_ids_pipeline.io.anomaly_io import load_anomaly_fitted_artifacts
from mth_ids_pipeline.io.model_io import load_anomaly_bundle, load_supervised_bundle


def _stacking_meta_features(
    dt_p: np.ndarray,
    et_p: np.ndarray,
    rf_p: np.ndarray,
    xg_p: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        (
            np.ravel(dt_p).reshape(-1, 1),
            np.ravel(et_p).reshape(-1, 1),
            np.ravel(rf_p).reshape(-1, 1),
            np.ravel(xg_p).reshape(-1, 1),
        ),
        axis=1,
    )


def predict_supervised_stacking(supervised: dict[str, Any], X_fss: np.ndarray) -> np.ndarray:
    """Tier 2: predições multi-classe via stacking."""
    dt_p = supervised["dt"].predict(X_fss)
    et_p = supervised["et"].predict(X_fss)
    rf_p = supervised["rf"].predict(X_fss)
    xg_p = supervised["xgb"].predict(X_fss)
    x_meta = _stacking_meta_features(dt_p, et_p, rf_p, xg_p)
    return np.ravel(supervised["stacking_meta"].predict(x_meta)).astype(np.int64)


def transform_raw_to_kpca(
    X_raw: np.ndarray,
    *,
    anomaly_work_dir: Path,
    feature_names: list[str],
) -> np.ndarray:
    """Z-score → IG → FCBF → KPCA usando artefatos da fase 8."""
    artifacts = load_anomaly_fitted_artifacts(anomaly_work_dir)
    scaler = artifacts["scaler"]
    ig_features: list[str] = artifacts["ig_features"]
    fcbf = artifacts["fcbf"]
    kpca = artifacts["kpca"]

    X_scaled = scaler.transform(X_raw)
    ig_idx = [feature_names.index(n) for n in ig_features]
    X_ig = X_scaled[:, ig_idx]
    X_fcbf = transform_fcbf(fcbf, X_ig)
    return transform_kpca(X_fcbf, kpca, split="inferência")


def predict_anomaly_tier(
    X_kpca: np.ndarray,
    anomaly_bundle: dict[str, Any],
) -> np.ndarray:
    """Tier 3–4: CL-k-means + biased B1/B2."""
    manifest = anomaly_bundle["manifest"]
    y_cl, conf = cl_kmeans_predict_inference(anomaly_bundle["cl_state"], X_kpca)
    return apply_biased_refinement(
        y_cl,
        conf,
        X_kpca,
        b1=anomaly_bundle.get("b1"),
        b2=anomaly_bundle.get("b2"),
        p_star=float(manifest.get("p_star", 0.933)),
        mode=manifest.get("biased_mode", "both"),
    )


def run_full_system_inference(
    *,
    intermediate_dir: Path,
    anomaly_work_dir: Path,
    test_size: float,
    random_state: int,
    benign_label: int = 0,
    anomaly_attack_pred_label: int = DEFAULT_ANOMALY_ATTACK_PRED_LABEL,
) -> dict[str, Any]:
    """
    Cascata completa no hold-out supervisionado (05_test_unchanged).

    - Stacking != BENIGN → classe do stacking
    - Stacking == BENIGN → tier anomaly (binário); ataque → ``anomaly_attack_pred_label``
    """
    supervised = load_supervised_bundle(intermediate_dir)
    anomaly_bundle = load_anomaly_bundle(anomaly_work_dir)

    test_df = pd.read_parquet(intermediate_dir / P05_TEST)
    label_col = "Label"
    X_fss = test_df.drop(columns=[label_col]).values.astype(np.float64)
    y_true = np.ravel(test_df[label_col].values).astype(np.int64)

    y_stack = predict_supervised_stacking(supervised, X_fss)
    y_final = y_stack.copy()

    normal_mask = y_stack == benign_label
    n_routed = int(normal_mask.sum())
    route_stats: dict[str, Any] = {
        "n_test": int(len(y_true)),
        "n_stacking_normal": n_routed,
        "n_stacking_attack": int((~normal_mask).sum()),
    }

    if n_routed > 0:
        sampled_path = intermediate_dir / P02_SAMPLED_KMEANS.replace(".csv", ".parquet")
        df_all = pd.read_parquet(sampled_path)
        feature_names = list(df_all.drop(columns=[label_col]).columns)
        X_all = df_all.drop(columns=[label_col]).values.astype(np.float64)
        y_all = np.ravel(df_all[label_col].values)
        _, X_test_raw, _, y_test_raw = train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            random_state=random_state,
            stratify=y_all,
        )
        if len(X_test_raw) != len(y_true):
            raise ValueError(
                f"Split reproduzido ({len(X_test_raw)} linhas) != teste supervisionado "
                f"({len(y_true)}). Verifique --test-size e --random-state."
            )

        manifest = anomaly_bundle["manifest"]
        anom_feature_names = manifest.get("feature_names") or feature_names
        X_raw_normal = X_test_raw[normal_mask]
        X_kpca = transform_raw_to_kpca(
            X_raw_normal,
            anomaly_work_dir=anomaly_work_dir,
            feature_names=anom_feature_names,
        )
        y_anom = predict_anomaly_tier(X_kpca, anomaly_bundle)
        y_final[normal_mask] = np.where(
            y_anom == 1,
            anomaly_attack_pred_label,
            benign_label,
        )
        route_stats["anomaly_attack_detected"] = int((y_anom == 1).sum())
        route_stats["anomaly_normal_confirmed"] = int((y_anom == 0).sum())

    y_true_bin = (y_true != benign_label).astype(np.int64)
    y_pred_bin = (y_final != benign_label).astype(np.int64)
    binary_metrics = binary_dr_far_f1(y_true_bin, y_pred_bin)

    acc = float(accuracy_score(y_true, y_final))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_final, average="weighted", zero_division=0)

    return {
        "y_true": y_true,
        "y_pred": y_final,
        "y_stacking": y_stack,
        "accuracy": acc,
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f),
        "binary": binary_metrics,
        "confusion_matrix_multiclass": confusion_matrix(y_true, y_final).tolist(),
        "confusion_matrix_binary": confusion_matrix(y_true_bin, y_pred_bin).tolist(),
        "route_stats": route_stats,
        "anomaly_attack_pred_label": int(anomaly_attack_pred_label),
        "supervised_manifest": supervised["manifest"],
        "anomaly_manifest": anomaly_bundle["manifest"],
    }


def resolve_anomaly_feature_names(work_dir: Path) -> list[str]:
    """Nomes de colunas brutas para transform anomaly."""
    for candidate in (work_dir / A01_WITHOUT_PORTSCAN, work_dir / "a01_without_portscan.parquet"):
        if candidate.is_file():
            df = pd.read_parquet(candidate)
            cols = [c for c in df.columns if c != "Label"]
            return list(cols)
    manifest_path = work_dir / "models" / "anomaly" / "manifest.json"
    if manifest_path.is_file():
        import json

        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = meta.get("feature_names")
        if names:
            return list(names)
    raise FileNotFoundError(
        f"Não foi possível resolver feature_names em {work_dir}. "
        "Execute fases 7–8 ou fase 11 com dados disponíveis."
    )
