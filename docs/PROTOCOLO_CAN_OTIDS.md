# Protocolo CAN-OTIDS (repack OTIDS)

Benchmark **reempacotado** (KIA Soul), nomenclatura `CAN_OTIDS_*`. Pipeline e preset iguais ao Car-Hacking, mas **pastas e CSV próprios**.

Documentos relacionados: [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) · [PROTOCOLO_CAN_INTRUSION.md](PROTOCOLO_CAN_INTRUSION.md) · [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md)

---

## Identificação

| Item | Valor |
|------|--------|
| Protocolo | **`--protocol can_otids`** (ou `can_otids_notebook`) |
| Merge | `merge_can --source otids` |
| CSV | `data/CAN_OTIDS_Dataset.csv` |
| Meta | `data/can_otids_meta.json` |
| Pipeline merged | `data/pipeline_can_otids_merged/` |
| Pipeline fine (LOAO) | `data/pipeline_can_otids_fine/` |
| Resultados | `results/can_otids/` |

**Não sobrescreve** o Car-Hacking — pastas distintas.

---

## Arquivos fonte

Em `data/CAN_OTIDS_DATA/`:

| Arquivo | Rótulo |
|---------|--------|
| `CAN_OTIDS_Attack_free_dataset.txt` | BENIGN |
| `CAN_OTIDS_DoS_attack_dataset.txt` | DoS |
| `CAN_OTIDS_Fuzzy_attack_dataset.txt` | Fuzzy |
| `CAN_OTIDS_Impersonation_attack_dataset.txt` | Impersonation |

4 classes (Gear + RPM já unificados no `.txt`).

---

## Preparação e pipeline

```powershell
python -m mth_ids_pipeline.utils.merge_can --source otids
python -m mth_ids_pipeline.run_supervised --protocol can_otids
python -m mth_ids_pipeline.run_anomaly --protocol can_otids --loao
```

### LOAO — 3 zero-days

| ID | Ataque |
|----|--------|
| 1 | DoS |
| 2 | Fuzzy |
| 3 | Impersonation |

```powershell
python -m mth_ids_pipeline.run_anomaly --protocol can_otids --loao --attack-labels 1,2,3
```

### Tabela X (sistema completo)

```powershell
python -m mth_ids_pipeline.run_global_anomaly --protocol can_otids
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_can_otids_merged `
  --work-dir data/pipeline_can_otids_merged/anomaly/global
```

---

## Tabelas — `report_paper_tables`

Flags legados do script: `vii` = **Tabela VI**, `ix` = **Tabela VIII**, `x` = **Tabela X**. Detecção automática pela pasta `pipeline_can_otids_*`; saída em `results/can_otids/`.

| Artigo | Flag `--table` | Pré-requisito |
|--------|----------------|---------------|
| Tabela VI (supervisionado) | `vii` | `06_supervised_metrics.json` |
| Tabela VIII (LOAO) | `ix` | `anomaly/loao/loao_summary.json` |
| Tabela X (sistema completo) | `x` | `phase13_full_system_eval.json` |
| Todas | `all` | Os três acima |

```powershell
# Tabela VI — supervisionado
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_can_otids_merged

# Tabela VIII — LOAO (3 zero-days)
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --merged-dir data/pipeline_can_otids_merged `
  --loao-root data/pipeline_can_otids_fine/anomaly/loao

# Tabela X — sistema completo
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_can_otids_merged

# Todas de uma vez
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_can_otids_merged `
  --loao-root data/pipeline_can_otids_fine/anomaly/loao
```

**Saída:** `results/can_otids/paper_comparison.json` + `results/can_otids/tables_report.txt`

```powershell
# Só terminal, sem gravar
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_can_otids_merged `
  --loao-root data/pipeline_can_otids_fine/anomaly/loao `
  --no-save
```
