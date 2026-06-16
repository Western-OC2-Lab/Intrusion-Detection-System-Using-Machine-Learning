# Protocolo CAN-intrusion (Car-Hacking original)

Dataset intra-veicular usado no artigo MTH-IDS (Yang et al., IEEE IoT Journal 2022) — **Tabela IV** e métricas das **Tabelas VI / VIII / X** para CAN.

Fonte: [CAN-intrusion-dataset](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset) (HCRL / Car-Hacking, 2018).

Documentos relacionados: [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) (índice) · [PROTOCOLO_CAN_OTIDS.md](PROTOCOLO_CAN_OTIDS.md) (repack OTIDS) · [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md)

---

## Identificação

| Item | Valor |
|------|--------|
| Nome no artigo | CAN-intrusion-dataset |
| Protocolo | **`--protocol can`** (alias `can_paper`, `can_intrusion`) |
| Merge | `merge_can --source original` |
| CSV | `data/CAN_intrusion_Dataset.csv` |
| Meta | `data/can_intrusion_meta.json` |
| Pipeline merged | `data/pipeline_can_intrusion_merged/` |
| Pipeline fine (LOAO) | `data/pipeline_can_intrusion_fine/` |
| Resultados | `results/can_intrusion/` |

**Não sobrescreve** o repack OTIDS — pastas distintas. Ver [PROTOCOLO_CAN_OTIDS.md](PROTOCOLO_CAN_OTIDS.md).

---

## Arquivos fonte

Coloque na pasta `data/`:

| Arquivo | Rótulo no CSV |
|---------|---------------|
| `CAN_normal_run_data.txt` | **BENIGN** |
| `CAN_DoS_dataset.csv` | **DoS** |
| `CAN_Fuzzy_dataset.csv` | **Fuzzy** |
| `CAN_gear_dataset.csv` | **Gear** |
| `CAN_RPM_dataset.csv` | **RPM** |

**Gear** e **RPM** permanecem **classes separadas** (Tabela IV).

### Classes após `LabelEncoder`

| ID | Rótulo |
|----|--------|
| 0 | BENIGN |
| 1 | DoS |
| 2 | Fuzzy |
| 3 | Gear |
| 4 | RPM |

`config.CAN_INTRUSION_LABEL_NAMES`

---

## Preparação do CSV

```powershell
python -m mth_ids_pipeline.utils.merge_can --source original
```

Saída: `data/CAN_intrusion_Dataset.csv` + `data/can_intrusion_meta.json`.

---

## Pipeline MTH-IDS

```powershell
python -m mth_ids_pipeline.run_supervised --protocol can
python -m mth_ids_pipeline.run_anomaly --protocol can --loao
```

### LOAO (Tabela VIII) — 4 zero-days

| ID | Ataque |
|----|--------|
| 1 | DoS |
| 2 | Fuzzy |
| 3 | Gear |
| 4 | RPM |

```powershell
python -m mth_ids_pipeline.run_anomaly --protocol can --loao --attack-labels 1,2,3,4
```

### Tabela X (sistema completo)

```powershell
python -m mth_ids_pipeline.run_global_anomaly --protocol can
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_can_intrusion_merged `
  --work-dir data/pipeline_can_intrusion_merged/anomaly/global
```

---

## Tabelas — `report_paper_tables`

Flags legados do script: `vii` = **Tabela VI**, `ix` = **Tabela VIII**, `x` = **Tabela X**. Detecção automática pela pasta `pipeline_can_intrusion_*`; saída em `results/can_intrusion/`.

| Artigo | Flag `--table` | Pré-requisito |
|--------|----------------|---------------|
| Tabela VI (supervisionado) | `vii` | `06_supervised_metrics.json` |
| Tabela VIII (LOAO) | `ix` | `anomaly/loao/loao_summary.json` |
| Tabela X (sistema completo) | `x` | `phase13_full_system_eval.json` |
| Todas | `all` | Os três acima |

```powershell
# Tabela VI — supervisionado
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_can_intrusion_merged

# Tabela VIII — LOAO (4 zero-days)
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --merged-dir data/pipeline_can_intrusion_merged `
  --loao-root data/pipeline_can_intrusion_fine/anomaly/loao

# Tabela X — sistema completo
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_can_intrusion_merged

# Todas de uma vez
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_can_intrusion_merged `
  --loao-root data/pipeline_can_intrusion_fine/anomaly/loao
```

**Saída:** `results/can_intrusion/paper_comparison.json` + `results/can_intrusion/tables_report.txt`

```powershell
# Só terminal, sem gravar
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_can_intrusion_merged `
  --loao-root data/pipeline_can_intrusion_fine/anomaly/loao `
  --no-save
```

---

## Erros comuns

### CSV em `CAN_OTIDS_Dataset.csv`

Use `--source original`; o default do merge grava em `CAN_intrusion_Dataset.csv`.

### Artefatos no lugar errado

Use **`--protocol can`** (não `can_otids`). OTIDS usa `--protocol can_otids` e `pipeline_can_otids_*`.
