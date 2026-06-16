"""
Fase 8 (anomaly): partição LOAO, depois Z-score → IG → FCBF → KPCA.

--feature-fit-scope combined (artigo tier 3): fit no conjunto treino+teste, split por índice.
--feature-fit-scope train: fit só no treino (rigoroso; não é o padrão do artigo).

Saídas: a03_combined_normalized.parquet, a04_after_kpca.parquet, a06_test_slice.json,
        fitted_*.joblib / fitted_ig_features.txt
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from mth_ids_pipeline.io.anomaly_io import (
        build_global_anomaly_partition,
        build_loao_train_test_split,
        is_global_table_x_protocol,
        log_loao_partition,
        numeric_feature_columns,
        require_path,
        save_anomaly_fitted_artifacts,
        validate_loao_partition,
    )
    from mth_ids_pipeline.core.dimensionality_reduction import fit_kpca, transform_kpca
    from mth_ids_pipeline.core.feature_selection import (
        AnomalyFeaturePipeline,
        fit_fcbf,
        information_gain_feature_subset,
        transform_fcbf,
    )
    from mth_ids_pipeline.core.preprocessing import zscore_array
except ImportError:
    from mth_ids_pipeline.io.anomaly_io import (
        build_global_anomaly_partition,
        build_loao_train_test_split,
        is_global_table_x_protocol,
        log_loao_partition,
        numeric_feature_columns,
        require_path,
        save_anomaly_fitted_artifacts,
        validate_loao_partition,
    )
    from mth_ids_pipeline.core.dimensionality_reduction import fit_kpca, transform_kpca
    from mth_ids_pipeline.core.feature_selection import (
        AnomalyFeaturePipeline,
        fit_fcbf,
        information_gain_feature_subset,
        transform_fcbf,
    )
    from mth_ids_pipeline.core.preprocessing import zscore_array

try:
    from mth_ids_pipeline.utils.bootstrap import ensure_repo_on_path
except ImportError:
    from mth_ids_pipeline.utils.bootstrap import ensure_repo_on_path

try:
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.config import (
        A00_LOAO_ROUND,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        A03_COMBINED_NORMALIZED,
        A04_AFTER_KPCA,
    )
except ImportError:
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.config import (
        A00_LOAO_ROUND,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        A03_COMBINED_NORMALIZED,
        A04_AFTER_KPCA,
    )

try:
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.io.reporting import write_report


def _df_from_scaled(
    X: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    *,
    label_col: str = "Label",
) -> pd.DataFrame:
    out = pd.DataFrame(X, columns=feature_names)
    out[label_col] = y
    return out


def _df_from_kpca(X_kpca: np.ndarray, y: np.ndarray, *, label_col: str = "Label") -> pd.DataFrame:
    cols = [f"kpca_{i}" for i in range(X_kpca.shape[1])]
    out = pd.DataFrame(X_kpca, columns=cols)
    out[label_col] = y
    return out


def _scale_combined_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    *,
    label_col: str,
    zscore_scope: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Normaliza features para IG/FCBF/KPCA combinado.

    per_split: IoTJ — Z-score em treino e teste separados, depois concatena.
    combined: artigo — Z-score no vetor concatenado.
    """
    n_train = len(train_df)
    if zscore_scope == "per_split":
        from mth_ids_pipeline.core.preprocessing import zscore_normalize

        train_z = zscore_normalize(train_df, label_col=label_col)
        test_z = zscore_normalize(test_df, label_col=label_col)
        X_scaled = np.vstack(
            [
                train_z[feature_names].values.astype(np.float64),
                test_z[feature_names].values.astype(np.float64),
            ]
        )
        y = np.concatenate(
            [
                np.ravel(train_z[label_col].values),
                np.ravel(test_z[label_col].values),
            ]
        )
        return X_scaled, y, n_train

    combined = pd.concat([train_df, test_df], ignore_index=True)
    X = combined[feature_names].values.astype(np.float64)
    y = np.ravel(combined[label_col].values)
    return zscore_array(X), y, n_train


def _run_combined_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    *,
    label_col: str,
    fcbf_k: int,
    ig_cumulative: float,
    kpca_components: int,
    kpca_kernel: str,
    zscore_scope: str,
    optimize_ig: bool,
    optimize_kpca: bool,
    ig_hpo_calls: int,
    kpca_hpo_calls: int,
    cv_folds: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """IG/FCBF/KPCA no conjunto combinado; partição por índice."""
    X_scaled, y, n_train = _scale_combined_splits(
        train_df, test_df, feature_names, label_col=label_col, zscore_scope=zscore_scope
    )

    ig_hpo_report = None
    alpha = ig_cumulative
    if optimize_ig:
        try:
            from mth_ids_pipeline.core.hyperparameter_optimization import optimize_ig_alpha
        except ImportError:
            from mth_ids_pipeline.core.hyperparameter_optimization import optimize_ig_alpha
        hpo = optimize_ig_alpha(
            X_scaled[:n_train],
            y[:n_train],
            feature_names,
            fcbf_k=fcbf_k,
            n_calls=ig_hpo_calls,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        alpha = hpo.best_alpha
        ig_hpo_report = {
            "best_alpha": hpo.best_alpha,
            "best_cv_accuracy": hpo.best_score,
            "trials": hpo.trials,
        }
        print(f"BO-GP IG (treino LOAO): alpha={alpha:.4f}, CV acc={hpo.best_score:.4f}")

    ig_features = information_gain_feature_subset(
        X_scaled, feature_names, y, cumulative=alpha
    )
    ig_idx = [feature_names.index(n) for n in ig_features]
    X_ig = X_scaled[:, ig_idx]
    fcbf = fit_fcbf(X_ig, y, k=fcbf_k)
    X_fcbf = transform_fcbf(fcbf, X_ig)

    kpca_hpo_report = None
    n_comp = kpca_components
    kernel = kpca_kernel
    if optimize_kpca:
        print("  [PCA] --optimize-kpca ignorado: KPCA foi substituido por PCA linear.")

    kpca = fit_kpca(X_fcbf, n_components=n_comp, kernel=kernel)
    X_kpca = transform_kpca(X_fcbf, kpca, split="combinado")

    meta = {
        "ig_features": ig_features,
        "ig_cumulative": alpha,
        "fcbf_k": fcbf_k,
        "kpca_components": n_comp,
        "kpca_kernel": kernel,
        "ig_hpo": ig_hpo_report,
        "kpca_hpo": kpca_hpo_report,
        "fcbf": fcbf,
        "kpca": kpca,
        "n_train_rows": n_train,
    }
    y_train = y[:n_train]
    y_test = y[n_train:]
    return (
        X_scaled[:n_train],
        X_scaled[n_train:],
        X_kpca[:n_train],
        X_kpca[n_train:],
        meta,
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_repo_on_path()

    parser = phase_parser("Fase 8 - features anomaly (IG+FCBF+PCA)")
    add_work_dir(parser)
    parser.add_argument("--fcbf-k", type=int, default=20)
    parser.add_argument("--kpca-components", type=int, default=10)
    parser.add_argument("--kpca-kernel", type=str, default="rbf")
    parser.add_argument("--ig-cumulative", type=float, default=0.9)
    parser.add_argument(
        "--feature-fit-scope",
        choices=("combined", "train"),
        default="combined",
        help="combined=IoTJ/artigo tier 3; train=fit só no treino",
    )
    parser.add_argument(
        "--zscore-scope",
        choices=("per_split", "combined"),
        default="combined",
        help="per_split=IoTJ (Z-score df1/df2); combined=Z-score no vetor concatenado",
    )
    parser.add_argument("--optimize-ig", action="store_true")
    parser.add_argument("--optimize-kpca", action="store_true")
    parser.add_argument("--ig-hpo-calls", type=int, default=15)
    parser.add_argument("--kpca-hpo-calls", type=int, default=15)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--benign-target", type=int, default=None, help="Benignos no teste (LOAO usa 1:1)")
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    start = time.time()
    total_steps = 5 if args.feature_fit_scope == "combined" else 6
    step = 0
    label_col = "Label"

    def tick(message: str) -> None:
        nonlocal step
        step += 1
        bar_len = 20
        filled = int(bar_len * step / total_steps)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - start
        print(f"[{step}/{total_steps}] [{bar}] {message} (+{elapsed:.1f}s)")

    p1 = work / A01_WITHOUT_PORTSCAN
    p2 = work / A02_PORTSCAN_ONLY
    phase7_hint = (
        f"Execute a fase 7 antes:\n"
        f"  python -m mth_ids_pipeline.phase07_anomaly_datasets "
        f"--intermediate-dir {paths.intermediate} --work-dir {work} --attack-label <N>"
    )
    require_path(p1, hint=phase7_hint)
    require_path(p2, hint=phase7_hint)
    df1 = pd.read_parquet(p1)
    df2 = pd.read_parquet(p2)
    tick("Dados carregados (fase 7)")

    orig_meta: dict = {}
    round_meta_path = work / A00_LOAO_ROUND
    if round_meta_path.is_file():
        orig_meta = json.loads(round_meta_path.read_text(encoding="utf-8"))
    benign_label = 0

    if is_global_table_x_protocol(orig_meta):
        train_df, test_df, partition_meta = build_global_anomaly_partition(
            df1,
            round_meta=orig_meta,
            label_col=label_col,
        )
        log_loao_partition(
            stage="fase 8 (global — pré-normalização)",
            train_df=train_df,
            test_df=test_df,
            meta=partition_meta,
            label_col=label_col,
        )
        print(
            f"Partição global (Tabela X): treino={train_df.shape}, "
            f"teste interno vazio (hold-out na fase 13)"
        )
    else:
        train_df, test_df, partition_meta = build_loao_train_test_split(
            df1,
            df2,
            benign_label=benign_label,
            label_col=label_col,
            benign_target=args.benign_target,
            random_state=args.random_state,
        )
        if orig_meta:
            partition_meta.update(
                {
                    k: orig_meta[k]
                    for k in (
                        "zero_day_label",
                        "train_original_label_counts",
                        "train_attack_labels_present",
                        "zero_day_fully_excluded_from_train",
                    )
                    if k in orig_meta
                }
            )
        validate_loao_partition(train_df, test_df, partition_meta, label_col=label_col)
        log_loao_partition(
            stage="fase 8 (pré-normalização)",
            train_df=train_df,
            test_df=test_df,
            meta=partition_meta,
            label_col=label_col,
        )
        print(f"Partição LOAO: treino={train_df.shape}, teste={test_df.shape}")
    feature_names = numeric_feature_columns(train_df, label_col=label_col)
    print(f"features={len(feature_names)}")
    tick("Partição treino/teste")

    rs = args.random_state if args.random_state is not None else 0
    ig_hpo_report = None
    kpca_hpo_report = None
    pipeline: AnomalyFeaturePipeline | None = None
    kpca = None
    ig_features: list[str] = []
    ig_cumulative = float(args.ig_cumulative)
    n_comp = args.kpca_components
    kernel = args.kpca_kernel

    if args.feature_fit_scope == "combined":
        (
            train_z,
            test_z,
            X_train_kpca,
            X_test_kpca,
            combined_meta,
        ) = _run_combined_features(
            train_df,
            test_df,
            feature_names,
            label_col=label_col,
            fcbf_k=args.fcbf_k,
            ig_cumulative=ig_cumulative,
            kpca_components=n_comp,
            kpca_kernel=kernel,
            zscore_scope=args.zscore_scope,
            optimize_ig=args.optimize_ig,
            optimize_kpca=args.optimize_kpca,
            ig_hpo_calls=args.ig_hpo_calls,
            kpca_hpo_calls=args.kpca_hpo_calls,
            cv_folds=args.cv_folds,
            random_state=rs,
        )
        ig_features = combined_meta["ig_features"]
        ig_cumulative = combined_meta["ig_cumulative"]
        n_comp = combined_meta["kpca_components"]
        kernel = combined_meta["kpca_kernel"]
        kpca = combined_meta["kpca"]
        ig_hpo_report = combined_meta.get("ig_hpo")
        kpca_hpo_report = combined_meta.get("kpca_hpo")
        fcbf = combined_meta["fcbf"]
        y_train = np.ravel(train_df[label_col].values)
        y_test = np.ravel(test_df[label_col].values)
        tick(f"Z-score ({args.zscore_scope}) + IG + FCBF + PCA (combinado)")
        protocol_name = f"combined_{args.zscore_scope}"
    else:
        X_train = train_df[feature_names].values
        y_train = np.ravel(train_df[label_col].values)
        X_test = test_df[feature_names].values
        y_test = np.ravel(test_df[label_col].values)
        skip_test_transform = len(test_df) == 0
        pipeline = AnomalyFeaturePipeline(fcbf_k=args.fcbf_k, ig_cumulative=ig_cumulative, random_state=rs)
        X_train_fcbf = pipeline.fit(X_train, y_train, feature_names)
        if skip_test_transform:
            X_test_fcbf = np.empty((0, X_train_fcbf.shape[1]), dtype=X_train_fcbf.dtype)
            print("  [Tabela X] teste interno vazio — transform teste omitido (hold-out na fase 13)")
        else:
            X_test_fcbf = pipeline.transform(X_test, split="teste")
        tick("Z-score + IG + FCBF (fit treino, transform teste)")
        if args.optimize_kpca:
            print("  [PCA] --optimize-kpca ignorado: KPCA foi substituido por PCA linear.")
        kpca = fit_kpca(X_train_fcbf, n_components=n_comp, kernel=kernel)
        X_train_kpca = transform_kpca(X_train_fcbf, kpca, split="treino")
        if skip_test_transform:
            X_test_kpca = np.empty((0, n_comp), dtype=np.float32)
        else:
            X_test_kpca = transform_kpca(X_test_fcbf, kpca, split="teste")
        ig_features = pipeline.ig_features
        train_z = pipeline.scaler.transform(X_train)
        test_z = (
            np.empty((0, len(feature_names)), dtype=X_train.dtype)
            if skip_test_transform
            else pipeline.scaler.transform(X_test)
        )
        fcbf = pipeline.fcbf
        protocol_name = "fit_train_transform_test"
        tick("PCA (fit treino, transform teste)")

    df_norm = pd.concat(
        [
            _df_from_scaled(train_z, feature_names, y_train, label_col=label_col),
            _df_from_scaled(test_z, feature_names, y_test, label_col=label_col),
        ],
        ignore_index=True,
    )
    combined_path = work / A03_COMBINED_NORMALIZED
    df_norm.to_parquet(combined_path, index=False)

    df_kpca = pd.concat(
        [
            _df_from_kpca(X_train_kpca, y_train, label_col=label_col),
            _df_from_kpca(X_test_kpca, y_test, label_col=label_col),
        ],
        ignore_index=True,
    )
    kpca_path = work / A04_AFTER_KPCA
    df_kpca.to_parquet(kpca_path, index=False)

    partition_meta["n_train_rows"] = len(train_df)
    partition_meta["n_test_rows"] = len(test_df)

    if pipeline is not None:
        artifact_paths = save_anomaly_fitted_artifacts(
            work,
            scaler=pipeline.scaler,
            ig_features=pipeline.ig_features,
            fcbf=pipeline.fcbf,
            kpca=kpca,
            partition_meta={
                **partition_meta,
                "fcbf_k": args.fcbf_k,
                "fcbf_selected_indices": pipeline.fcbf_selected_indices(),
                "kpca_components": n_comp,
                "kpca_kernel": kernel,
                "dimensionality_reduction": "pca",
                "ig_cumulative": ig_cumulative,
                "feature_count_raw": len(feature_names),
                "feature_count_ig": len(pipeline.ig_features),
                "feature_fit_scope": args.feature_fit_scope,
            },
        )
    else:
        from sklearn.preprocessing import StandardScaler

        dummy_scaler = StandardScaler()
        dummy_scaler.fit(train_df[feature_names].values)
        artifact_paths = save_anomaly_fitted_artifacts(
            work,
            scaler=dummy_scaler,
            ig_features=ig_features,
            fcbf=fcbf,
            kpca=kpca,
            partition_meta={
                **partition_meta,
                "fcbf_k": args.fcbf_k,
                "fcbf_selected_indices": list(fcbf.idx_sel) if fcbf else [],
                "kpca_components": n_comp,
                "kpca_kernel": kernel,
                "dimensionality_reduction": "pca",
                "ig_cumulative": ig_cumulative,
                "feature_count_raw": len(feature_names),
                "feature_count_ig": len(ig_features),
                "feature_fit_scope": args.feature_fit_scope,
            },
        )
    tick("Artefatos e parquets salvos")

    meta = partition_meta
    print(f"Salvo: {combined_path} shape={df_norm.shape}")
    print(f"Salvo: {kpca_path} shape={df_kpca.shape}")
    print(f"Meta/partição: {artifact_paths['partition']}")

    report = {
        "input_without_portscan": str(p1),
        "input_portscan": str(p2),
        "combined_output": str(combined_path),
        "kpca_output": str(kpca_path),
        "test_slice_meta": artifact_paths["partition"],
        "artifact_paths": artifact_paths,
        "protocol": protocol_name,
        "feature_fit_scope": args.feature_fit_scope,
        "zscore_scope": args.zscore_scope,
        "fcbf_k": args.fcbf_k,
        "kpca_components": n_comp,
        "kpca_kernel": kernel,
        "dimensionality_reduction": "pca",
        "ig_cumulative": ig_cumulative,
        "optimize_ig": args.optimize_ig,
        "optimize_kpca": args.optimize_kpca,
        "zero_day_samples": meta["zero_day_samples"],
        "benign_sample_n": meta["benign_sampled"],
        "benign_available": meta["benign_available_in_train"],
        "benign_pairing_rule": meta["benign_pairing_rule"],
        "random_state": args.random_state,
        "train_shape": {"rows": int(train_df.shape[0]), "cols": int(len(feature_names))},
        "test_shape": {"rows": int(test_df.shape[0]), "cols": int(len(feature_names))},
        "kpca_shape": {"rows": int(df_kpca.shape[0]), "cols": int(df_kpca.shape[1] - 1)},
        "ig_feature_count": len(ig_features),
    }
    if ig_hpo_report:
        report["ig_hpo"] = ig_hpo_report
    if kpca_hpo_report:
        report["kpca_hpo"] = kpca_hpo_report
    report["fcbf_feature_count"] = int(X_train_kpca.shape[1])
    report_path = write_report(paths.reports, "phase08_anomaly_features", report)
    print(f"Relatorio salvo em: {report_path}")
    tick("Relatorio JSON")


if __name__ == "__main__":
    main()
