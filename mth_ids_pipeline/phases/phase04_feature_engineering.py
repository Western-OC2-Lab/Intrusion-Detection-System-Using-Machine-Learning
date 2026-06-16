"""
Fase 4: split estratificado → escala → IG (α acumulado) → FCBF (k=20).

CAN (padrão): ``--fcbf-scope full`` + IG dinâmico (``--ig-cumulative``).
CICIDS artigo: ``--fcbf-scope train`` + ``--scale-mode split``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from mth_ids_pipeline.utils.bootstrap import ensure_repo_on_path
except ImportError:
    from mth_ids_pipeline.utils.bootstrap import ensure_repo_on_path

try:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import (
        P02_SAMPLED_KMEANS,
        P04_SELECTED_FEATURES,
        P04_TEST_FSS,
        P04_TRAIN_FSS,
    )
    from mth_ids_pipeline.core.feature_selection import (
        fit_fcbf,
        information_gain_feature_subset,
        transform_fcbf,
    )
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import (
        P02_SAMPLED_KMEANS,
        P04_SELECTED_FEATURES,
        P04_TEST_FSS,
        P04_TRAIN_FSS,
    )
    from mth_ids_pipeline.core.feature_selection import fit_fcbf, information_gain_feature_subset, transform_fcbf
    from mth_ids_pipeline.io.reporting import write_report


def _scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_all: np.ndarray,
    *,
    scale_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """phase1 = IoTJ (sem reescalar); split = StandardScaler fit no treino."""
    if scale_mode == "phase1":
        return X_train, X_test, X_all, None
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X_all)
    return X_train_s, X_test_s, X_all_s, scaler


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_repo_on_path()
    try:
        from mth_ids_pipeline.utils.FCBF_module import FCBFK
    except ImportError as e:
        raise ImportError("Coloque FCBF_module.py na raiz do repositório.") from e

    parser = phase_parser("Fase 4 — IG + FCBF + split")
    parser.add_argument("--fcbf-k", type=int, default=20)
    parser.add_argument(
        "--fcbf-mode",
        choices=("k", "alpha"),
        default="k",
        help="k=FCBFK(k) notebook/artigo; alpha=FCBF(th=α) com limiar configurável",
    )
    parser.add_argument("--fcbf-alpha", type=float, default=0.01, help="Limiar th do FCBF (modo alpha)")
    parser.add_argument("--optimize-fcbf", action="store_true", help="BO-GP para α FCBF (modo alpha)")
    parser.add_argument("--fcbf-hpo-calls", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--fcbf-scope",
        choices=("train", "full"),
        default="full",
        help=(
            "full=FCBF no dataset completo (padrão CAN; evita inconsistência de índices). "
            "train=FCBF só no treino (artigo CICIDS)"
        ),
    )
    parser.add_argument(
        "--scale-mode",
        choices=("phase1", "split"),
        default=None,
        help="phase1=IoTJ (Z-score fase 1); split=StandardScaler no treino (artigo)",
    )
    parser.add_argument("--ig-cumulative", type=float, default=0.9)
    parser.add_argument("--optimize-ig", action="store_true", help="BO-GP para α IG (artigo)")
    parser.add_argument(
        "--ig-features",
        type=str,
        default=None,
        help="Features IG fixas (opcional; padrão: seleção dinâmica por α acumulado)",
    )
    parser.add_argument("--ig-hpo-calls", type=int, default=15)
    parser.add_argument("--cv-folds", type=int, default=10, help="CV para BO-GP α")
    args = parser.parse_args()

    scale_mode = args.scale_mode or ("phase1" if args.fcbf_scope == "full" else "split")
    train_size = 1.0 - float(args.test_size)

    paths = init_paths(args)
    output_dir = paths.intermediate
    df = pd.read_parquet(supervised_path(paths, P02_SAMPLED_KMEANS))
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    feature_names = list(df.drop(columns=[label_col]).columns)
    ig_features_arg = args.ig_features
    X = df.drop(columns=[label_col]).values.astype(np.float64)
    y = np.ravel(df[label_col].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    X_train_s, X_test_s, X_all_s, fitted_scaler = _scale_features(
        X_train, X_test, X, scale_mode=scale_mode
    )

    ig_cumulative = float(args.ig_cumulative)
    fcbf_mode = str(args.fcbf_mode)
    fcbf_alpha = float(args.fcbf_alpha)
    ig_hpo_report: dict | None = None
    fcbf_hpo_report: dict | None = None
    if args.optimize_ig:
        try:
            from mth_ids_pipeline.core.hyperparameter_optimization import optimize_ig_alpha
        except ImportError:
            from mth_ids_pipeline.core.hyperparameter_optimization import optimize_ig_alpha
        hpo = optimize_ig_alpha(
            X_train_s,
            y_train,
            feature_names,
            fcbf_k=args.fcbf_k,
            n_calls=args.ig_hpo_calls,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
        ig_cumulative = hpo.best_alpha
        ig_hpo_report = {
            "best_alpha": hpo.best_alpha,
            "best_cv_accuracy": hpo.best_score,
            "trials": hpo.trials,
        }
        print(f"BO-GP IG: alpha={ig_cumulative:.4f}, CV acc={hpo.best_score:.4f}")

    if args.optimize_fcbf:
        if fcbf_mode != "alpha":
            raise ValueError("--optimize-fcbf requer --fcbf-mode alpha")
        try:
            from mth_ids_pipeline.core.hyperparameter_optimization import optimize_fcbf_alpha
        except ImportError:
            from mth_ids_pipeline.core.hyperparameter_optimization import optimize_fcbf_alpha
        hpo = optimize_fcbf_alpha(
            X_train_s,
            y_train,
            feature_names,
            ig_cumulative=ig_cumulative,
            n_calls=args.fcbf_hpo_calls,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
        fcbf_alpha = hpo.best_alpha
        fcbf_hpo_report = {
            "best_alpha": hpo.best_alpha,
            "best_cv_accuracy": hpo.best_score,
            "trials": hpo.trials,
        }
        print(f"BO-GP FCBF: alpha={fcbf_alpha:.4f}, CV acc={hpo.best_score:.4f}")

    if ig_features_arg:
        ig_features = [f.strip() for f in ig_features_arg.split(",") if f.strip()]
        missing = [f for f in ig_features if f not in feature_names]
        if missing:
            raise ValueError(
                f"Features fixas ausentes no dataset: {missing}. "
                f"Disponíveis: {feature_names}"
            )
        print(f"IG fixo (artigo): {ig_features}")
    else:
        ig_features = information_gain_feature_subset(
            X_train_s, feature_names, y_train, cumulative=ig_cumulative
        )
        print(f"IG dinâmico (α={ig_cumulative:.0%}): {ig_features}")
    (output_dir / P04_SELECTED_FEATURES).write_text("\n".join(ig_features), encoding="utf-8")
    ig_idx = [feature_names.index(n) for n in ig_features]

    if args.fcbf_scope == "full":
        X_ig = X_all_s[:, ig_idx]
        if fcbf_mode == "alpha":
            from mth_ids_pipeline.utils.FCBF_module import FCBF

            fcbf = FCBF(th=fcbf_alpha)
            X_fss = fcbf.fit_transform(X_ig, y)
        else:
            fcbf = FCBFK(k=args.fcbf_k)
            X_fss = fcbf.fit_transform(X_ig, y)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X_fss,
            y,
            train_size=train_size,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
    else:
        X_train_ig = X_train_s[:, ig_idx]
        X_test_ig = X_test_s[:, ig_idx]
        fcbf = fit_fcbf(
            X_train_ig,
            y_train,
            k=args.fcbf_k,
            alpha=fcbf_alpha,
            mode=fcbf_mode,
        )
        X_train2 = transform_fcbf(fcbf, X_train_ig)
        X_test2 = transform_fcbf(fcbf, X_test_ig)
        y_train2, y_test2 = y_train, y_test

    cols = [f"f{i}" for i in range(X_train2.shape[1])]
    train_out = pd.DataFrame(X_train2, columns=cols)
    train_out[label_col] = y_train2
    test_out = pd.DataFrame(X_test2, columns=cols)
    test_out[label_col] = y_test2

    train_path = output_dir / P04_TRAIN_FSS
    test_path = output_dir / P04_TEST_FSS
    train_out.to_parquet(train_path, index=False)
    test_out.to_parquet(test_path, index=False)
    print(f"Salvo: {train_path} {train_out.shape}")
    print(f"Salvo: {test_path} {test_out.shape}")

    report = {
        "ig_feature_count": len(ig_features),
        "ig_cumulative": ig_cumulative,
        "post_fcbf_feature_count": len(cols),
        "test_size": float(args.test_size),
        "fcbf_scope": args.fcbf_scope,
        "scale_mode": scale_mode,
        "optimize_ig": args.optimize_ig,
        "fcbf_mode": fcbf_mode,
        "fcbf_k": int(args.fcbf_k),
        "fcbf_alpha": float(fcbf_alpha),
        "optimize_fcbf": args.optimize_fcbf,
        "fixed_ig_features": ig_features_arg,
        "train_output": str(train_path),
        "test_output": str(test_path),
    }
    if ig_hpo_report:
        report["ig_hpo"] = ig_hpo_report
    if fcbf_hpo_report:
        report["fcbf_hpo"] = fcbf_hpo_report
    write_report(paths.reports, "phase04_feature_engineering", report)

    try:
        from mth_ids_pipeline.io.model_io import save_supervised_preprocess_artifacts
    except ImportError:
        from mth_ids_pipeline.io.model_io import save_supervised_preprocess_artifacts

    artifact_paths = save_supervised_preprocess_artifacts(
        output_dir,
        scaler=fitted_scaler,
        fcbf=fcbf,
        ig_features=ig_features,
        feature_names=feature_names,
        scale_mode=scale_mode,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        fcbf_scope=args.fcbf_scope,
        ig_cumulative=ig_cumulative,
    )
    print(f"Preprocessores supervisionados salvos: {artifact_paths['manifest']}")


if __name__ == "__main__":
    main()
