"""
Concatena o CAN-intrusion-dataset (Car-Hacking / OTIDS) → ``data/CAN_OTIDS_Dataset.csv``.

Tratamento conforme artigo MTH-IDS (Yang et al. 2022), Sec. V-B / Fig. 1:
  - descarta timestamp e flag R/T;
  - 10 features: CAN_ID, DLC, DATA_0…DATA_7 (bytes do payload, padding com 0);
  - rótulos Car-Hacking: BENIGN, DoS, Fuzzy, Gear, RPM (Tabela IV — sem unificar);
  - rótulos OTIDS: BENIGN, DoS, Fuzzy, Impersonation (repack);
  - concatena e embaralha (``random_state=42``).

Fontes suportadas (``--source``):
  - ``otids``: ``CAN_OTIDS_*.txt`` em ``data/CAN_OTIDS_DATA/`` (repack OTIDS);
  - ``original``: arquivos Car-Hacking em ``data/`` (``CAN_normal_run_data.txt``,
    ``CAN_*_dataset.csv``);
  - ``auto``: OTIDS se existir; senão original.

Uso:
  python -m mth_ids_pipeline.utils.merge_can
  python -m mth_ids_pipeline.utils.merge_can --source original
  python -m mth_ids_pipeline.utils.merge_can --input-dir data/CAN_OTIDS_DATA
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from mth_ids_pipeline.config import (
    CAN_INTRUSION_DATA_DIR,
    CAN_OTIDS_DATA_DIR,
    default_meta_for_can_source,
    default_raw_csv_for_can_source,
)

_LINE_RE = re.compile(
    r"Timestamp:\s+([\d.]+)\s+ID:\s+(\w+)\s+(\d+)\s+DLC:\s+(\d+)\s+(.*)"
)

_FEATURE_COLUMNS: tuple[str, ...] = (
    "CAN_ID",
    "DLC",
    "DATA_0",
    "DATA_1",
    "DATA_2",
    "DATA_3",
    "DATA_4",
    "DATA_5",
    "DATA_6",
    "DATA_7",
    "Label",
)

_CAN_LABEL_BY_STEM: dict[str, str] = {
    "CAN_OTIDS_Attack_free_dataset": "BENIGN",
}

# Car-Hacking original (HCRL / CAN-intrusion-dataset)
_CAN_ORIGINAL_FILES: tuple[tuple[str, str], ...] = (
    ("CAN_normal_run_data.txt", "BENIGN"),
    ("CAN_DoS_dataset.csv", "DoS"),
    ("CAN_Fuzzy_dataset.csv", "Fuzzy"),
    ("CAN_gear_dataset.csv", "Gear"),
    ("CAN_RPM_dataset.csv", "RPM"),
)

_LABEL_PROFILE_BY_SOURCE: dict[str, str] = {
    "original": "intrusion",
    "otids": "otids",
}


def _attack_label_from_stem(stem: str) -> str:
    if stem in _CAN_LABEL_BY_STEM:
        return _CAN_LABEL_BY_STEM[stem]
    if stem.startswith("CAN_OTIDS_"):
        return stem.split("_")[2]
    raise ValueError(f"Nome de arquivo CAN-OTIDS não reconhecido: {stem}")


def _parse_payload_bytes(payload: str, dlc: int) -> list[int]:
    parts = payload.strip().split()
    out: list[int] = []
    for token in parts[: max(0, dlc)]:
        out.append(int(token, 16))
    while len(out) < 8:
        out.append(0)
    return out[:8]


def _row_from_fields(can_id: int, dlc: int, data: list[int], label: str) -> dict:
    row: dict = {"CAN_ID": can_id, "DLC": dlc, "Label": label}
    for i, val in enumerate(data[:8]):
        row[f"DATA_{i}"] = val
    for i in range(len(data), 8):
        row[f"DATA_{i}"] = 0
    return row


def _parse_can_txt_file(path: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as infile:
        for line in infile:
            match = _LINE_RE.match(line.strip())
            if not match:
                continue
            _ts, can_id, _flag, dlc_s, payload = match.groups()
            dlc = int(dlc_s)
            data = _parse_payload_bytes(payload, dlc)
            rows.append(_row_from_fields(int(can_id, 16), dlc, data, label))
    return rows


def _parse_can_csv_line(line: str, label: str) -> dict | None:
    """CSV Car-Hacking: timestamp, ID hex, DLC, N bytes hex (N=DLC), flag (R/T)."""
    parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
    if len(parts) < 4:
        return None
    try:
        can_id = int(parts[1], 16)
        dlc = int(parts[2])
    except ValueError:
        return None
    
    # Remove as linhas rotuladas como R (pacotes normais) nos conjuntos de dados de ataque
    flag_index = 3 + dlc
    if flag_index < len(parts):
        flag = parts[flag_index].strip().upper()
        if flag == "R":
            return None

    data_tokens = parts[3 : 3 + dlc]
    if len(data_tokens) < dlc:
        return None
    data = [int(token, 16) for token in data_tokens]
    while len(data) < 8:
        data.append(0)
    return _row_from_fields(can_id, dlc, data[:8], label)


def _parse_can_csv_file(path: Path, label: str) -> pd.DataFrame:
    rows: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as infile:
        for line in infile:
            row = _parse_can_csv_line(line, label)
            if row is None:
                skipped += 1
                continue
            rows.append(row)
    if skipped:
        print(f"  ({path.name}: {skipped:,} linhas ignoradas — formato inválido)")
    return pd.DataFrame(rows)


def _has_otids_files(input_dir: Path) -> bool:
    return bool(list(input_dir.glob("CAN_OTIDS_*.txt")))


def _has_original_files(input_dir: Path) -> bool:
    return any((input_dir / name).is_file() for name, _ in _CAN_ORIGINAL_FILES)


def discover_source(input_dir: Path) -> str:
    if _has_otids_files(input_dir):
        return "otids"
    if _has_original_files(input_dir):
        return "original"
    raise FileNotFoundError(
        f"Nenhum arquivo CAN reconhecido em {input_dir}. "
        "Esperado CAN_OTIDS_*.txt (OTIDS) ou CAN_normal_run_data.txt / CAN_*_dataset.csv (original)."
    )


def resolve_input_dir(
    *,
    source: str,
    input_dir: Path | None,
) -> tuple[Path, str]:
    key = source.strip().lower()
    if input_dir is not None:
        resolved = input_dir
        if key == "auto":
            key = discover_source(resolved)
        return resolved, key

    if key == "otids":
        return CAN_OTIDS_DATA_DIR, "otids"
    if key == "original":
        return CAN_INTRUSION_DATA_DIR, "original"
    if key == "auto":
        if _has_otids_files(CAN_OTIDS_DATA_DIR):
            return CAN_OTIDS_DATA_DIR, "otids"
        if _has_original_files(CAN_INTRUSION_DATA_DIR):
            return CAN_INTRUSION_DATA_DIR, "original"
        raise FileNotFoundError(
            f"Nenhuma fonte CAN encontrada em {CAN_OTIDS_DATA_DIR} nem em {CAN_INTRUSION_DATA_DIR}."
        )
    raise ValueError(f"--source inválido: {source!r}. Use auto, otids ou original.")


def _load_otids_source(input_dir: Path) -> list[pd.DataFrame]:
    txt_files = sorted(input_dir.glob("CAN_OTIDS_*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Nenhum CAN_OTIDS_*.txt em {input_dir}.")

    frames: list[pd.DataFrame] = []
    for path in txt_files:
        stem = path.stem
        label = _attack_label_from_stem(stem)
        rows = _parse_can_txt_file(path, label)
        if not rows:
            print(f"Aviso: nenhuma linha parseada em {path.name}")
            continue
        frames.append(pd.DataFrame(rows))
        print(f"{path.name}: {len(rows):,} linhas -> Label={label}")
    return frames


def _load_original_source(input_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for filename, label in _CAN_ORIGINAL_FILES:
        path = input_dir / filename
        if not path.is_file():
            print(f"Aviso: ausente {filename} — ignorado")
            continue
        if path.suffix.lower() == ".txt":
            rows = _parse_can_txt_file(path, label)
            if not rows:
                print(f"Aviso: nenhuma linha parseada em {filename}")
                continue
            frame = pd.DataFrame(rows)
        else:
            frame = _parse_can_csv_file(path, label)
        frames.append(frame)
        print(f"{filename}: {len(frame):,} linhas -> Label={label}")
    return frames


def build_can_dataset(
    input_dir: Path,
    *,
    source: str = "auto",
    random_state: int = 42,
) -> pd.DataFrame:
    resolved_source = source if source != "auto" else discover_source(input_dir)

    if resolved_source == "otids":
        frames = _load_otids_source(input_dir)
    elif resolved_source == "original":
        frames = _load_original_source(input_dir)
    else:
        raise ValueError(f"Fonte CAN desconhecida: {resolved_source}")

    if not frames:
        raise ValueError(f"Nenhum registro CAN válido em {input_dir} (source={resolved_source})")

    df = pd.concat(frames, ignore_index=True)
    df = df[list(_FEATURE_COLUMNS)]
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera CAN_OTIDS_Dataset.csv (merge + tratamento artigo MTH-IDS)"
    )
    parser.add_argument(
        "--source",
        choices=("auto", "otids", "original"),
        default="auto",
        help="Fonte: OTIDS (txt), Car-Hacking original (txt/csv) ou auto (default)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Pasta de entrada (default: CAN_OTIDS_DATA ou data/ conforme --source)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV de saída (default: CAN_intrusion_Dataset.csv ou CAN_OTIDS_Dataset.csv)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_dir, source = resolve_input_dir(source=args.source, input_dir=args.input_dir)
    output = args.output or default_raw_csv_for_can_source(source)
    meta_path = default_meta_for_can_source(source)
    print(f"Fonte: {source} | entrada: {input_dir}")

    df = build_can_dataset(input_dir, source=source, random_state=args.random_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    meta = {
        "source": source,
        "label_profile": _LABEL_PROFILE_BY_SOURCE.get(source, source),
        "labels": sorted(df["Label"].astype(str).unique().tolist()),
        "n_rows": int(len(df)),
        "output_csv": str(output),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Meta: {meta_path} ({meta['label_profile']}, {len(meta['labels'])} classes)")

    print(f"\nShape: {df.shape}")
    print(df["Label"].value_counts())
    print(f"Colunas: {list(df.columns)}")
    print(f"Salvo em: {output}")


if __name__ == "__main__":
    main()
