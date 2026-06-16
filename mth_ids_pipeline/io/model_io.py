"""Persistência de modelos treinados para inferência (fase 13)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from mth_ids_pipeline.config import (
    ANOM_B1,
    ANOM_B2,
    ANOM_CL_STATE,
    MODEL_MANIFEST,
    MODELS_ANOMALY_DIR,
    MODELS_SUPERVISED_DIR,
    SUP_DT,
    SUP_ET,
    SUP_FCBF,
    SUP_FEATURE_NAMES,
    SUP_IG_FEATURES,
    SUP_RF,
    SUP_SCALER,
    SUP_STACKING_META,
    SUP_XGB,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def supervised_models_dir(intermediate_dir: Path) -> Path:
    return intermediate_dir / MODELS_SUPERVISED_DIR


_SUPERVISED_FAMILY_FILES: dict[str, str] = {
    "dt": SUP_DT,
    "rf": SUP_RF,
    "et": SUP_ET,
    "xgb": SUP_XGB,
}


def resolve_supervised_models_dir(intermediate_dir: Path) -> Path:
    """
    Pasta ``models/supervised`` com artefatos da fase 6.

    LOAO (fine) só recebe ``06_supervised_metrics.json`` no bootstrap; os ``.joblib``
    ficam em ``pipeline_mth_ids_merged`` — faz fallback automático.
    """
    from mth_ids_pipeline.config import INTERMEDIATE_DIR_MERGED

    local = supervised_models_dir(intermediate_dir)
    if (local / SUP_ET).is_file():
        return intermediate_dir
    if intermediate_dir != INTERMEDIATE_DIR_MERGED:
        merged_models = supervised_models_dir(INTERMEDIATE_DIR_MERGED)
        if (merged_models / SUP_ET).is_file():
            return INTERMEDIATE_DIR_MERGED
    return intermediate_dir


def load_supervised_classifier_template(intermediate_dir: Path, family: str) -> Any:
    """Carrega um classificador base (dt/rf/et/xgb) treinado na fase 6."""
    key = family.strip().lower()
    if key not in _SUPERVISED_FAMILY_FILES:
        raise ValueError(f"Família supervisionada desconhecida: {family!r}. Use dt, rf, et ou xgb.")
    root = resolve_supervised_models_dir(intermediate_dir)
    path = supervised_models_dir(root) / _SUPERVISED_FAMILY_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(
            f"Modelo supervisionado não encontrado: {path}\n"
            "Execute a fase 6 em pipeline_mth_ids_merged antes da fase 11."
        )
    return joblib.load(path)


def anomaly_models_dir(work_dir: Path) -> Path:
    return work_dir / MODELS_ANOMALY_DIR


def save_supervised_preprocess_artifacts(
    intermediate_dir: Path,
    *,
    scaler: Any | None,
    fcbf: Any,
    ig_features: list[str],
    feature_names: list[str],
    scale_mode: str,
    test_size: float,
    random_state: int,
    fcbf_scope: str,
    ig_cumulative: float,
) -> dict[str, str]:
    """Persiste preprocessors supervisionados (fase 4)."""
    out = supervised_models_dir(intermediate_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {
        "fcbf": str(out / SUP_FCBF),
        "ig_features": str(out / SUP_IG_FEATURES),
        "feature_names": str(out / SUP_FEATURE_NAMES),
        "manifest": str(out / MODEL_MANIFEST),
    }
    if scaler is not None:
        scaler_path = out / SUP_SCALER
        joblib.dump(scaler, scaler_path)
        paths["scaler"] = str(scaler_path)
    joblib.dump(fcbf, paths["fcbf"])
    Path(paths["ig_features"]).write_text("\n".join(ig_features), encoding="utf-8")
    Path(paths["feature_names"]).write_text("\n".join(feature_names), encoding="utf-8")
    manifest = {
        "kind": "supervised_preprocess",
        "scale_mode": scale_mode,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "fcbf_scope": fcbf_scope,
        "ig_cumulative": float(ig_cumulative),
        "artifact_paths": paths,
    }
    Path(paths["manifest"]).write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    return paths


def save_supervised_classifier_artifacts(
    intermediate_dir: Path,
    *,
    dt: Any,
    rf: Any,
    et: Any,
    xgb_model: Any,
    stacking_meta: Any,
    meta_label: str,
    meta_learner: str,
    binary: bool,
) -> dict[str, str]:
    """Persiste modelos supervisionados + stacking (fase 6)."""
    out = supervised_models_dir(intermediate_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "dt": str(out / SUP_DT),
        "rf": str(out / SUP_RF),
        "et": str(out / SUP_ET),
        "xgb": str(out / SUP_XGB),
        "stacking_meta": str(out / SUP_STACKING_META),
        "manifest": str(out / MODEL_MANIFEST),
    }
    joblib.dump(dt, paths["dt"])
    joblib.dump(rf, paths["rf"])
    joblib.dump(et, paths["et"])
    joblib.dump(xgb_model, paths["xgb"])
    joblib.dump(stacking_meta, paths["stacking_meta"])

    manifest_path = out / MODEL_MANIFEST
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "kind": "supervised_full",
            "meta_label": meta_label,
            "meta_learner": meta_learner,
            "binary": bool(binary),
            "base_order": ["dt", "et", "rf", "xgb"],
            "classifier_paths": {
                "dt": paths["dt"],
                "rf": paths["rf"],
                "et": paths["et"],
                "xgb": paths["xgb"],
                "stacking_meta": paths["stacking_meta"],
            },
        }
    )
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    print(f"Modelos supervisionados salvos em: {out}")
    return paths


def save_anomaly_inference_artifacts(
    work_dir: Path,
    *,
    cl_state: Any,
    b1: Any | None,
    b2: Any | None,
    n_clusters: int,
    metric: str,
    p_star: float,
    biased_mode: str,
    feature_names: list[str],
    random_state: int,
) -> dict[str, str]:
    """Persiste CL-k-means + B1/B2 (fase 11)."""
    out = anomaly_models_dir(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {
        "cl_kmeans_state": str(out / ANOM_CL_STATE),
        "manifest": str(out / MODEL_MANIFEST),
    }
    joblib.dump(cl_state, paths["cl_kmeans_state"])
    if b1 is not None:
        paths["b1"] = str(out / ANOM_B1)
        joblib.dump(b1, paths["b1"])
    if b2 is not None:
        paths["b2"] = str(out / ANOM_B2)
        joblib.dump(b2, paths["b2"])

    manifest = {
        "kind": "anomaly_inference",
        "n_clusters": int(n_clusters),
        "metric": metric,
        "p_star": float(p_star),
        "biased_mode": biased_mode,
        "feature_names": feature_names,
        "random_state": int(random_state),
        "artifact_paths": paths,
    }
    Path(paths["manifest"]).write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    print(f"Modelos anomaly salvos em: {out}")
    return paths


def load_supervised_bundle(intermediate_dir: Path) -> dict[str, Any]:
    """Carrega preprocessors + classificadores supervisionados."""
    out = supervised_models_dir(intermediate_dir)
    manifest_path = out / MODEL_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest supervisionado não encontrado: {manifest_path}\n"
            "Execute as fases 4 e 6 antes da fase 13."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clf_paths = manifest.get("classifier_paths", {})
    bundle: dict[str, Any] = {
        "manifest": manifest,
        "fcbf": joblib.load(out / SUP_FCBF),
        "ig_features": (out / SUP_IG_FEATURES).read_text(encoding="utf-8").splitlines(),
        "feature_names": (out / SUP_FEATURE_NAMES).read_text(encoding="utf-8").splitlines(),
        "dt": joblib.load(clf_paths.get("dt", out / SUP_DT)),
        "rf": joblib.load(clf_paths.get("rf", out / SUP_RF)),
        "et": joblib.load(clf_paths.get("et", out / SUP_ET)),
        "xgb": joblib.load(clf_paths.get("xgb", out / SUP_XGB)),
        "stacking_meta": joblib.load(clf_paths.get("stacking_meta", out / SUP_STACKING_META)),
    }
    scaler_path = out / SUP_SCALER
    bundle["scaler"] = joblib.load(scaler_path) if scaler_path.is_file() else None
    return bundle


def load_anomaly_bundle(work_dir: Path) -> dict[str, Any]:
    """Carrega CL-k-means + B1/B2 + manifest anomaly."""
    out = anomaly_models_dir(work_dir)
    manifest_path = out / MODEL_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest anomaly não encontrado: {manifest_path}\n"
            "Execute a fase 11 (--work-dir) antes da fase 13."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest.get("artifact_paths", {})
    bundle: dict[str, Any] = {
        "manifest": manifest,
        "cl_state": joblib.load(paths.get("cl_kmeans_state", out / ANOM_CL_STATE)),
        "b1": None,
        "b2": None,
    }
    b1_path = paths.get("b1", out / ANOM_B1)
    b2_path = paths.get("b2", out / ANOM_B2)
    if Path(b1_path).is_file():
        bundle["b1"] = joblib.load(b1_path)
    if Path(b2_path).is_file():
        bundle["b2"] = joblib.load(b2_path)
    return bundle
