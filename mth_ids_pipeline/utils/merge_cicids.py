"""
Concatena MachineLearningCSV → CSV CICIDS2017 com perfil de rótulos escolhido.

  python -m mth_ids_pipeline.utils.merge_cicids --profile merged   # data/CICIDS2017.csv
  python -m mth_ids_pipeline.utils.merge_cicids --profile fine     # data/CICIDS2017_fine.csv
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "data" / "MachineLearningCSV"

try:
    from ..label_profiles import apply_cicids_label_merge, get_label_profile
except ImportError:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from mth_ids_pipeline.label_profiles import apply_cicids_label_merge, get_label_profile


def build_cicids_csv(*, profile: str, output: Path | None = None) -> Path:
    profile_obj = get_label_profile(profile)
    out_path = output or profile_obj.raw_csv

    arquivos_csv = sorted(glob.glob(str(INPUT_DIR / "*.csv")))
    if not arquivos_csv:
        raise FileNotFoundError(
            f"Nenhum CSV em {INPUT_DIR}. Coloque os arquivos do CICIDS2017 em data/MachineLearningCSV/"
        )

    print(f"Perfil: {profile_obj.kind.value} — {profile_obj.description}")
    print(f"Lendo {len(arquivos_csv)} arquivos de {INPUT_DIR}")
    df = pd.concat(
        [pd.read_csv(arquivo, encoding="latin1", low_memory=False) for arquivo in arquivos_csv],
        ignore_index=True,
    )
    df.columns = df.columns.str.strip()
    print("Labels brutos:\n", df["Label"].value_counts())

    if profile_obj.kind.value == "merged":
        df = apply_cicids_label_merge(df)
        print("Labels após agrupamento (merged):\n", df["Label"].value_counts())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    n_attack = df["Label"].nunique() - (1 if "BENIGN" in df["Label"].values else 0)
    print(f"Salvo: {out_path} ({len(df):,} linhas, {df['Label'].nunique()} rótulos, ~{n_attack} ataques)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera CSV CICIDS2017 (merged ou fine)")
    parser.add_argument(
        "--profile",
        choices=["merged", "fine"],
        default="merged",
        help="merged: famílias (Tabela VII); fine: rótulos originais (Tabela IX)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Caminho do CSV de saída")
    args = parser.parse_args()
    build_cicids_csv(profile=args.profile, output=args.output)


if __name__ == "__main__":
    main()
