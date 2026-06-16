# Protocolo UNSW-NB15 — MTH-IDS (rede externa)

Guia do pipeline MTH-IDS no **UNSW-NB15** com **`--protocol unsw`** (ou `unsw_nb15`). Estrutura alinhada a [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) e [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md).

Documentos relacionados: [EXECUCAO.md](EXECUCAO.md) · [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) · [PIPELINE_PHASES.md](PIPELINE_PHASES.md) · [ARQUITETURA.md](ARQUITETURA.md)

---

## Princípio

| CICIDS2017 | UNSW-NB15 |
|------------|-----------|
| `data/CICIDS2017.csv` | `data/UNSW-NB15_merged.csv` |
| `data/pipeline_mth_ids_merged/` | `data/pipeline_unsw_nb15_merged/` |
| `data/pipeline_mth_ids_fine/` | `data/pipeline_unsw_nb15_fine/` |
| **`--protocol paper`** | **`--protocol unsw`** |
| BENIGN k-means 0,8% | **Benign** k-means **10%** |
| SMOTE BruteForce + Infiltration → 1000 | SMOTE em 5 classes (ver tabela abaixo) |

**Regra:** UNSW-NB15 usa pastas **`pipeline_unsw_nb15_*`** — não misture com `pipeline_mth_ids_*` nem `pipeline_can_*`.

Constantes em `mth_ids_pipeline/config.py`:

- `DEFAULT_RAW_CSV_UNSW_NB15` → `data/UNSW-NB15_merged.csv`
- `INTERMEDIATE_DIR_UNSW_NB15_MERGED` → `data/pipeline_unsw_nb15_merged`
- `INTERMEDIATE_DIR_UNSW_NB15_FINE` → `data/pipeline_unsw_nb15_fine`

---

## Identificação

| Item | Valor |
|------|--------|
| Dataset | [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| Protocolo | **`--protocol unsw`** (alias `unsw_nb15`, `unsw-nb15`) |
| CSV | `data/UNSW-NB15_merged.csv` |
| Meta | `data/unsw_nb15_meta.json` (opcional) |
| Pipeline merged | `data/pipeline_unsw_nb15_merged/` |
| Pipeline fine (LOAO) | `data/pipeline_unsw_nb15_fine/` |
| Resultados | `results/unsw_nb15/` |

---

## Dataset e CSV merged

O arquivo **`UNSW-NB15_merged.csv`** deve estar em `data/` com:

- Coluna de rótulo: **`Label`** (última coluna ou nome explícito na fase 1)
- Classe benigna: **`Benign`** (renomeie `Normal` do dataset original, se necessário)
- **10 classes de ataque** no nível `attack_cat` (sem subtipos adicionais)

### Classes esperadas

| Classe | Papel no protocolo |
|--------|-------------------|
| Benign | Tráfego normal |
| Generic | Ataque genérico |
| Exploits | Exploração de vulnerabilidades |
| Fuzzers | Fuzzing |
| DoS | Denial of Service |
| Reconnaissance | Reconhecimento |
| Analysis | Análise de tráfego malicioso |
| Backdoors | Backdoor (UNSW oficial) |
| Shellcode | Shellcode |
| Worms | Worms |

> **Nota:** a tabela de amostragem cita **Backdoor** e **Backdoors** separadamente. O UNSW-NB15 oficial usa apenas **Backdoors**. Se o CSV tiver só `Backdoors`, aplique SMOTE → 5.000 nessa classe. Se existir um rótulo `Backdoor` distinto, trate-o também com SMOTE → 5.000.

### Classes após `LabelEncoder` (ordem alfabética)

| ID | Rótulo |
|----|--------|
| 0 | Analysis |
| 1 | Backdoors |
| 2 | Benign |
| 3 | DoS |
| 4 | Exploits |
| 5 | Fuzzers |
| 6 | Generic |
| 7 | Reconnaissance |
| 8 | Shellcode |
| 9 | Worms |

`config.UNSW_NB15_LABEL_NAMES` — IDs podem variar se houver rótulo extra (`Backdoor`); confira `01_preprocessed` após a fase 1.

---

## Estratégia de amostragem (fases 2 e 5)

### Fase 2 — k-means (MiniBatchKMeans, k=1000)

| Classe | Estratégia |
|--------|------------|
| **Benign** | KMeans → **10%** (`frac=0.10`) |
| Generic | **Manter** (preservada intacta) |
| Exploits | **Manter** |
| Fuzzers | **Manter** |
| DoS | **Manter** |
| Reconnaissance | **Manter** |
| Analysis | **Manter** (SMOTE na fase 5) |
| Backdoor | **Manter** (SMOTE na fase 5, se existir no CSV) |
| Backdoors | **Manter** (SMOTE na fase 5) |
| Shellcode | **Manter** (SMOTE na fase 5) |
| Worms | **Manter** (SMOTE na fase 5) |

**Mecanismo:** todas as classes de **ataque** entram em `minority_labels` (preservadas). Somente **Benign** passa pelo k-means com `frac=0.10` — espelha o CICIDS2017, onde minoritárias ficam intactas e só a majoritária é amostrada.

### Fase 5 — SMOTE (somente treino, após split 80/20)

| Classe | Alvo |
|--------|------|
| Analysis | **5.000** |
| Backdoor | **5.000** (se existir no CSV) |
| Backdoors | **5.000** |
| Shellcode | **5.000** |
| Worms | **2.000** |

Classes com **Manter** na fase 2 **não** recebem SMOTE. Generic, Exploits, Fuzzers, DoS e Reconnaissance permanecem com a contagem original do treino.

`config.UNSW_NB15_SMOTE_TARGETS` mapeia rótulo numérico → alvo (ver `UNSW_NB15_LABEL_NAMES`).

---

## Preset `unsw` — parâmetros do pipeline

Implementado em `mth_ids_pipeline/protocol.py` (`ProtocolSettings` UNSW).

| Item | UNSW (`unsw`) | CICIDS2017 (`paper`) |
|------|---------------|----------------------|
| Perfil supervisionado | `unsw_nb15_merged` | `merged` |
| Perfil LOAO | `unsw_nb15_fine` | `fine` |
| Split | **80/20** | 80/20 |
| Validação HPO (fase 6) | 10-fold CV | 10-fold CV |
| Amostragem fase 2 | Benign k-means **10%**; ataques preservados | BENIGN/DoS/PortScan/BruteForce k-means 0,8%; Bot/Infiltration/WebAttack preservados |
| SMOTE supervisionado (fase 5) | Analysis, Backdoor(s), Shellcode, Worms | BruteForce + Infiltration → 1000 |
| BO-GP | α IG + KPCA + p* | Igual |
| Meta-learner stacking | `best-base` | Igual |
| SMOTE anomaly (fases 9–11) | Sim (binário no treino) | Sim |

```powershell
python -m mth_ids_pipeline.run_supervised --protocol unsw
python -m mth_ids_pipeline.run_anomaly --protocol unsw --loao
```

---

## Supervisionado (multi-classe)

| Item | Valor |
|------|-------|
| Pasta | `data/pipeline_unsw_nb15_merged` |
| CSV | `data/UNSW-NB15_merged.csv` |
| Perfil | `unsw_nb15_merged` (10 classes + Benign) |
| Amostragem | Benign k-means `frac=0.10`; ataques intactos |
| Split | **80% treino / 20% teste** |
| SMOTE | Analysis/Backdoor(s)/Shellcode → 5000; Worms → 2000 |
| IG | BO-GP α acumulado |
| Stacking | Meta-learner = clone do melhor base (`best-base`) |

```powershell
# 1) Coloque UNSW-NB15_merged.csv em data/

# 2) Pipeline supervisionado
python -m mth_ids_pipeline.run_supervised --protocol unsw
```

---

## LOAO anomaly (zero-day)

| Item | Valor |
|------|-------|
| Pasta | `data/pipeline_unsw_nb15_fine` |
| CSV | `data/UNSW-NB15_merged.csv` (mesmo arquivo; 9 ataques distintos) |
| Perfil | `unsw_nb15_fine` |
| Bootstrap | Fases 1–2 no fine + supervisionado no merged (`06_…` copiado) |
| LOAO | **9 rodadas** (um zero-day por classe de ataque) |
| Teste LOAO | zero-day + benignos **1:1** |
| Amostragem fase 2 | Igual ao merged (Benign 10%; ataques preservados) |
| SMOTE binário | Sim (treino anomaly) |

### LOAO — 9 zero-days

| ID | Ataque |
|----|--------|
| 0 | Analysis |
| 1 | Backdoors |
| 3 | DoS |
| 4 | Exploits |
| 5 | Fuzzers |
| 6 | Generic |
| 7 | Reconnaissance |
| 8 | Shellcode |
| 9 | Worms |

> Benign = 2 (nunca é zero-day). IDs seguem `LabelEncoder` alfabético — confira após fase 1 se o CSV tiver rótulos extras.

```powershell
python -m mth_ids_pipeline.run_anomaly --protocol unsw --loao
python -m mth_ids_pipeline.run_anomaly --protocol unsw --loao --attack-labels 0,1,3,4,5,6,7,8,9
```

---

## Sistema completo (cascata tiers 1→4)

```powershell
python -m mth_ids_pipeline.run_supervised --protocol unsw --from 1 --to 2
python -m mth_ids_pipeline.run_supervised --protocol unsw --from 4 --to 6
python -m mth_ids_pipeline.run_global_anomaly --protocol unsw
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_unsw_nb15_merged `
  --work-dir data/pipeline_unsw_nb15_merged/anomaly/global
```

Hold-out real (`05_test_unchanged.parquet`) só entra no `run_eval` — fases 7–11 reservam o 20% supervisionado.

---

## Mapa de pastas

```text
data/
├── UNSW-NB15_merged.csv              # entrada (10 ataques + Benign)
├── unsw_nb15_meta.json               # opcional
├── pipeline_unsw_nb15_merged/        # supervisionado + anomaly global
│   ├── 02_sampled_kmeans.parquet
│   ├── 06_supervised_metrics.json
│   └── anomaly/global/
└── pipeline_unsw_nb15_fine/          # LOAO
    └── anomaly/loao/
        ├── attack_0/ … attack_9/
        └── loao_summary.json

results/
└── unsw_nb15/
```

---

## Tabelas — `report_paper_tables`

Use os mesmos flags `vii` / `ix` / `x` / `all` dos outros protocolos (nomes legados do script). O relatório detecta automaticamente o dataset pela pasta `pipeline_unsw_nb15_*` e grava em `results/unsw_nb15/`.

| Relatório | Flag `--table` | Equivalente no artigo |
|-----------|----------------|------------------------|
| Supervisionado | `vii` | — (sem tabela no artigo) |
| LOAO | `ix` | — |
| Sistema completo | `x` | — |
| Todas | `all` | — |

> UNSW-NB15 **não** tem referência numérica no artigo Yang et al. — as tabelas mostram só as métricas reproduzidas (sem coluna “Artigo”).

### Pré-requisitos

| Tabela | Flag `--table` | Pré-requisito no pipeline |
|--------|----------------|---------------------------|
| Supervisionado | `vii` | `06_supervised_metrics.json` (fases 1–6) |
| LOAO | `ix` | `anomaly/loao/loao_summary.json` (fases 7–12) |
| Sistema completo | `x` | `phase13_full_system_eval.json` (`run_global_anomaly` + `run_eval`) |
| Todas | `all` | Os três acima |

### Comandos

```powershell
# Supervisionado (multi-classe, stacking)
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_unsw_nb15_merged

# LOAO (média + por ataque)
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --merged-dir data/pipeline_unsw_nb15_merged `
  --loao-root data/pipeline_unsw_nb15_fine/anomaly/loao

# Sistema completo (cascata tiers 1→4)
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_unsw_nb15_merged

# Todas de uma vez
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_unsw_nb15_merged `
  --loao-root data/pipeline_unsw_nb15_fine/anomaly/loao
```

**Saída:** `results/unsw_nb15/paper_comparison.json` + `results/unsw_nb15/tables_report.txt`

Só imprimir no terminal (sem gravar):

```powershell
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_unsw_nb15_merged `
  --loao-root data/pipeline_unsw_nb15_fine/anomaly/loao `
  --no-save
```

---

## Defaults dos entrypoints

| Script | `--protocol` | Perfil | Pasta |
|--------|--------------|--------|-------|
| `run_supervised` | `unsw` | `unsw_nb15_merged` | `pipeline_unsw_nb15_merged` |
| `run_anomaly --loao` | `unsw` | `unsw_nb15_fine` | `pipeline_unsw_nb15_fine` |
| `run_global_anomaly` | `unsw` | — | `pipeline_unsw_nb15_merged/anomaly/global` |
| `run_eval` | — | merged (via `--intermediate-dir`) | idem |

---

## Fluxo completo (copiar e colar)

```powershell
# Pré-requisito: data/UNSW-NB15_merged.csv (Label = Benign + 10 ataques)

python -m mth_ids_pipeline.run_supervised --protocol unsw
python -m mth_ids_pipeline.run_anomaly --protocol unsw --loao
python -m mth_ids_pipeline.run_global_anomaly --protocol unsw
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_unsw_nb15_merged `
  --work-dir data/pipeline_unsw_nb15_merged/anomaly/global

python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_unsw_nb15_merged `
  --loao-root data/pipeline_unsw_nb15_fine/anomaly/loao
```

---

## Erros comuns

| Problema | Solução |
|----------|---------|
| `Normal` em vez de `Benign` | Renomeie no CSV ou na fase 1; o protocolo assume `Benign` |
| Benign não amostrado a 10% | Use `--protocol unsw`; não use `paper` (frac 0,8% no CICIDS) |
| SMOTE não aplicado | Confira `UNSW_NB15_SMOTE_TARGETS` e fase 5 (`phase05_smote.json`) |
| LOAO com IDs errados | Rode fase 1 e verifique `Label` em `01_preprocessed.parquet` |
| Pastas CICIDS sobrescritas | Use sempre `--protocol unsw` e `pipeline_unsw_nb15_*` |

### Regenerar k-means

```powershell
Remove-Item data\pipeline_unsw_nb15_merged\02_sampled_kmeans.parquet
Remove-Item data\pipeline_unsw_nb15_fine\02_sampled_kmeans.parquet
python -m mth_ids_pipeline.run_supervised --protocol unsw --from 1 --to 2
```

---

## Índice

- [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) — CICIDS2017 (referência metodológica)
- [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) — CAN (intra-veicular)
- [EXECUCAO.md](EXECUCAO.md) — comandos e bootstrap
- [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) — comparativo de presets
