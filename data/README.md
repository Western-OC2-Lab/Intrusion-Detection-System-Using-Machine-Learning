# Dados — MTH-IDS

## Entrada bruta

| Caminho | Uso |
|---------|-----|
| `MachineLearningCSV/` | CSVs originais do CICIDS2017 (fonte do `merge_cicids`) |
| `CICIDS2017.csv` | Perfil **merged** (Tabela VII) — `merge_cicids --profile merged` |
| `CICIDS2017_fine.csv` | Perfil **fine** (Tabela IX / LOAO) — `merge_cicids --profile fine` |

Gerar os CSVs (uma vez):
n
```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.utils.merge_cicids --profile fine
```

## Artefatos do pipeline

| Pasta | Perfil | Comando |
|-------|--------|---------|
| `pipeline_mth_ids_merged/` | merged | `run_supervised --protocol paper` |
| `pipeline_mth_ids_fine/` | fine | `run_anomaly --protocol paper --loao` |

O notebook IoTJ usava amostras `CICIDS2017_sample_km*.csv`; o pipeline atual gera `02_sampled_kmeans.parquet` nas pastas acima (não versionar — ver `.gitignore`).
