# Protocolo CICIDS2017 — Tabelas VII, IX e X (MTH-IDS)

Guia do pipeline MTH-IDS no **CICIDS2017** (rede externa / IoV) com **`--protocol paper`** (artigo) ou **`--protocol notebook`** (IoTJ).

Documentos relacionados: [EXECUCAO.md](EXECUCAO.md) · [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) · [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) · [PROTOCOLO_UNSW_NB15.md](PROTOCOLO_UNSW_NB15.md) · [REPRODUCAO_CICIDS2017_VALIDACAO.md](REPRODUCAO_CICIDS2017_VALIDACAO.md)

---

## Princípio

| CICIDS2017 (externo) | CAN (intra-veicular) |
|----------------------|----------------------|
| `data/CICIDS2017.csv` / `CICIDS2017_fine.csv` | `data/CAN_OTIDS_Dataset.csv` (Car-Hacking ou OTIDS — ver [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md)) |
| `data/pipeline_mth_ids_merged/` | `data/pipeline_can_otids_merged/` |
| `data/pipeline_mth_ids_fine/` | `data/pipeline_can_otids_fine/` |
| **`--protocol paper`** / `notebook` | **`--protocol can`** |
| Tabela **VII** (7 famílias) | Tabela **VI** (4 classes) |
| Tabela **IX** (~14 LOAO) | Tabela **VIII** (3 LOAO) |
| Tabela **X** | Tabela **X** (CAN) |

**Regra:** CICIDS2017 é o **preset padrão** do pipeline (`--protocol paper`). Não misture pastas `pipeline_mth_ids_*` com `pipeline_can_otids_*`.

Constantes em `mth_ids_pipeline/config.py`:

- `DEFAULT_RAW_CSV` → `data/CICIDS2017.csv`
- `DEFAULT_RAW_CSV_FINE` → `data/CICIDS2017_fine.csv`
- `INTERMEDIATE_DIR_MERGED` → `data/pipeline_mth_ids_merged`
- `INTERMEDIATE_DIR_FINE` → `data/pipeline_mth_ids_fine`

---

## Dois protocolos CICIDS

| Parâmetro | `paper` (padrão) | `notebook` (IoTJ) |
|-----------|------------------|-------------------|
| Split | **80/20** | **80/20** |
| α IG | **BO-GP** (15 calls, CV 10-fold) | **0,9 fixo** |
| FCBF supervisionado | k=20, **só treino** | k=20, **dataset completo** |
| Normalização sup. | StandardScaler **após split** | Z-score **fase 1** |
| HPO fase 6 | **10-fold CV**, objetivo validação | **Hold-out**, objetivo teste |
| SMOTE fase 5 | BruteForce + Infiltration → 1000 | Igual notebook |
| Stacking meta | **`best-base`** | **XGBoost** + HPO |
| Perfil LOAO | **`fine`** (~14 ataques) | **`merged`** (demo PortScan) |
| Anomaly BO-GP | α IG, KPCA, k, p* | KPCA fixo; HPO só `n_clusters` |

```powershell
# Artigo (Tabelas VII, IX, X)
python -m mth_ids_pipeline.run_supervised --protocol paper

# Notebook IoTJ publicado
python -m mth_ids_pipeline.run_supervised --protocol notebook
```

**Atenção:** `paper` e `notebook` gravam na **mesma pasta** `pipeline_mth_ids_merged` se usar o mesmo perfil. Regenerar ao trocar de preset.

---

## Dataset e preparação dos CSVs

### Origem

CSVs brutos do [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) em `data/MachineLearningCSV/`.

### Geração dos CSVs

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.utils.merge_cicids --profile fine
```

| CSV | Perfil | Classes | Usado em |
|-----|--------|---------|----------|
| `data/CICIDS2017.csv` | **merged** | 7 (BENIGN + 6 famílias) | Tabelas **VII** e **X** |
| `data/CICIDS2017_fine.csv` | **fine** | ~15 subtipos originais | Tabela **IX** (LOAO) |

O **merged** agrega subtipos (ex.: DoS Hulk + DDoS → **DoS**). O **fine** mantém cada subtipo separado para LOAO.

### Amostragem k-means (fase 2)

No CICIDS2017, BENIGN, DoS, PortScan, BruteForce passam pelo k-means **0,8%** (`frac=0.008`). Bot, Infiltration e WebAttack (família merged) ficam **preservados inteiros** — espelha o `df_minor` do notebook.

No perfil **fine** (LOAO), a mesma regra via `label_profiles.py`: preservados os fine equivalentes a Bot/Infiltration/WebAttack; **PortScan é amostrado** (não confundir com “rótulo não agregado pelo merge”).

Resultado típico pós fase 2: **~27k linhas** (merged e fine alinhados ao notebook).

---

## Perfis `merged` e `fine`

| | **merged** | **fine** |
|---|------------|----------|
| Tabelas | VII, X | IX |
| Pasta | `pipeline_mth_ids_merged/` | `pipeline_mth_ids_fine/` |
| Modelos anomaly | 1 global (`anomaly/global/`) | ~14 LOAO (`anomaly/loao/attack_*`) |
| SMOTE anomaly | Sim (paper) | Sim (paper) |

LOAO e global são experimentos **distintos** — ver [ARQUITETURA.md](ARQUITETURA.md).

---

## Tabela VII — supervisionado (ataques conhecidos)

| Item | Valor |
|------|-------|
| Pasta | `data/pipeline_mth_ids_merged` |
| CSV | `data/CICIDS2017.csv` |
| Perfil | `merged` (7 famílias) |
| Split | 80% treino / 20% hold-out |
| Validação HPO | 10-fold CV no treino |
| SMOTE | BruteForce (2) e Infiltration (4) → **1000** cada |
| IG | BO-GP α acumulado |
| Stacking | Meta-learner = clone do melhor base (`best-base`) |

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.run_supervised --protocol paper
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_mth_ids_merged
```

---

## Tabela IX — LOAO anomaly (zero-day)

| Item | Valor |
|------|-------|
| Pasta | `data/pipeline_mth_ids_fine` |
| CSV | `data/CICIDS2017_fine.csv` |
| Perfil | `fine` (~14 ataques) |
| Bootstrap | Fases 1–2 no fine + Tabela VII no merged (`06_…` copiado) |
| LOAO | ~14 rodadas; teste = zero-day + benignos **1:1** |
| Features | Z-score → IG → FCBF → KPCA no combinado |
| BO-GP | α IG, KPCA, n_clusters, p* |
| SMOTE binário | Sim (treino anomaly) |

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile fine
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

**Um ataque** (ex.: Bot, label 1):

```powershell
python -m mth_ids_pipeline.run_all --label-profile fine `
  --protocol paper --from 12 --to 12 --skip-bootstrap --attack-label 1
```

Bootstrap: [EXECUCAO.md — Bootstrap automático](EXECUCAO.md#bootstrap-automático). Retomar fases 9–11: [PIPELINE_PHASES.md](PIPELINE_PHASES.md#retomar-um-ataque-loao-fases-911-manuais).

---

## Tabela X — sistema completo

| Item | Valor |
|------|-------|
| Pasta | `data/pipeline_mth_ids_merged` |
| Pré-requisitos | Tabela VII (fases 1–6) + anomaly global (7–11) |
| Avaliação | Fase 13 — cascata tiers 1→4 no hold-out 20% |
| Split pipeline | **80/20** (artigo reporta **70/30** na Tabela X — comparação aproximada) |

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper
python -m mth_ids_pipeline.run_global_anomaly --protocol paper
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_mth_ids_merged `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_mth_ids_merged
```

Hold-out real (`05_test_unchanged.parquet`) só entra no `run_eval` — fases 7–11 reservam o 20% supervisionado.

---

## Mapa de pastas

```text
data/
├── CICIDS2017.csv                    # merged (7 classes)
├── CICIDS2017_fine.csv               # fine (~15 classes)
├── pipeline_mth_ids_merged/          # Tabela VII + X
│   ├── 06_supervised_metrics.json
│   ├── anomaly/global/
│   └── phase_reports/phase13_…json
└── pipeline_mth_ids_fine/            # Tabela IX
    └── anomaly/loao/
        ├── attack_1/ … attack_14/
        └── loao_summary.json

results/                              # report_paper_tables (default)
├── paper_comparison.json
└── tables_report.txt
```

---

## Tabelas — `report_paper_tables`

O script `report_paper_tables` usa flags legados (`vii`, `ix`, `x`, `all`). Detecta CICIDS2017 pela pasta `pipeline_mth_ids_*` e grava em `results/` (padrão).

| Artigo | Flag `--table` | Conteúdo |
|--------|----------------|----------|
| **Tabela VII** | `vii` | Supervisionado multi-classe + stacking |
| **Tabela IX** | `ix` | LOAO anomaly (média + por ataque) |
| **Tabela X** | `x` | Sistema completo (cascata tiers 1→4) |
| Todas | `all` | VII + IX + X (+ notebook, se CICIDS) |

Comparação numérica vs artigo Yang et al. incluída nas três tabelas.

### Pré-requisitos

| Tabela | Flag `--table` | Pré-requisito no pipeline |
|--------|----------------|---------------------------|
| VII (supervisionado) | `vii` | `06_supervised_metrics.json` (fases 1–6) |
| IX (LOAO) | `ix` | `anomaly/loao/loao_summary.json` (fases 7–12) |
| X (sistema completo) | `x` | `phase13_full_system_eval.json` (`run_global_anomaly` + `run_eval`) |
| Todas | `all` | Os três acima |

### Comandos

```powershell
# Tabela VII — supervisionado
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_mth_ids_merged

# Tabela IX — LOAO
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao

# Tabela X — sistema completo
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_mth_ids_merged

# Todas de uma vez
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

**Saída:** `results/paper_comparison.json` + `results/tables_report.txt`

Só imprimir no terminal (sem gravar):

```powershell
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao `
  --no-save
```

---

## Defaults dos entrypoints

| Script | `--protocol` | Perfil | Pasta |
|--------|--------------|--------|-------|
| `run_supervised` | `paper` | `merged` | `pipeline_mth_ids_merged` |
| `run_anomaly --loao` | `paper` | `fine` | `pipeline_mth_ids_fine` |
| `run_global_anomaly` | `paper` | — | `pipeline_mth_ids_merged/anomaly/global` |
| `run_eval` | — | merged (via `--intermediate-dir`) | idem |

---

## Fluxo completo

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.utils.merge_cicids --profile fine

python -m mth_ids_pipeline.run_supervised --protocol paper          # VII
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao     # IX
python -m mth_ids_pipeline.run_global_anomaly --protocol paper       # X (treino)
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_mth_ids_merged `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global

python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

Validação numérica de uma run completa: [REPRODUCAO_CICIDS2017_VALIDACAO.md](REPRODUCAO_CICIDS2017_VALIDACAO.md).

---

## Erros comuns

| Problema | Solução |
|----------|---------|
| `02_sampled_kmeans.parquet` ausente | `run_anomaly` sem `--skip-bootstrap`, ou fases 1–2 no perfil correto |
| Tabela IX vazia | LOAO incompleto; reconstruir `loao_summary.json` |
| `phase13_…json` ausente | `run_global_anomaly` + `run_eval` antes de `--table x` |
| Comparativo X ≠ artigo | Split **80/20** no pipeline vs **70/30** no artigo |
| Confundir LOAO com Tabela X | LOAO = `fine/loao/`; X = `merged/anomaly/global/` |

Mais: [EXECUCAO.md — Solução de problemas](EXECUCAO.md#solução-de-problemas).

---

## Índice

- [EXECUCAO.md](EXECUCAO.md) — comandos e bootstrap (CICIDS + CAN)
- [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) — tabela comparativa `paper` vs `notebook`
- [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) — CAN (índice) · [INTRUSION](PROTOCOLO_CAN_INTRUSION.md) · [OTIDS](PROTOCOLO_CAN_OTIDS.md) · [UNSW](PROTOCOLO_UNSW_NB15.md)
- [PIPELINE_PHASES.md](PIPELINE_PHASES.md) — referência de cada fase
