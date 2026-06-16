# Execução — tabelas, pastas e comandos

Guia único para rodar Tabelas **VII / IX / X** (CICIDS2017) e **VI / VIII / X** (CAN).

Documentos relacionados: [ARQUITETURA.md](ARQUITETURA.md) · [EXECUCAO.md](EXECUCAO.md) · [PIPELINE_PHASES.md](PIPELINE_PHASES.md) · [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) · [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) · [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md)

---

## Visão geral

Cada tabela é um **experimento distinto** — não há um script que treine as três de uma vez.

| Tabela | O que mede | Perfil | Script(s) |
|--------|------------|--------|-----------|
| **VII** / **VI** | Ataques conhecidos (tiers 1–2) | merged / can_merged | `run_supervised` |
| **IX** / **VIII** | Zero-day LOAO (tiers 3–4) | fine / can_fine | `run_anomaly --loao` |
| **X** | Cascata completa (tiers 1→4) | merged / can_merged | `run_global_anomaly` + `run_eval` |

`report_paper_tables` **só lê** artefatos em `data/` e grava relatórios em `results/`.

### Ordem recomendada

**CICIDS2017:** `merge_cicids` → `run_supervised` (VII) → `run_anomaly --loao` (IX, paralelo) → `run_global_anomaly` + `run_eval` (X) → `report_paper_tables`

**CAN:** `merge_can` → `run_supervised --protocol can` (VI) → `run_anomaly --protocol can --loao` (VIII) → global + eval (X opcional) → `report_paper_tables --results-dir results/can_otids`

A Tabela **X depende da VII/VI** (stacking + hold-out). A **IX/VIII é independente**.

---

## Pré-requisitos

Na raiz do repositório, com `.venv` ativo:

```powershell
cd C:\caminho\para\Intrusion-Detection-System-Using-Machine-Learning
.venv\Scripts\Activate.ps1
```

**CICIDS2017** (uma vez):

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.utils.merge_cicids --profile fine
```

**CAN** — dois estudos em pastas separadas (podem coexistir):

```powershell
# Car-Hacking → pipeline_can_intrusion_* / results/can_intrusion/
python -m mth_ids_pipeline.utils.merge_can --source original
python -m mth_ids_pipeline.run_supervised --protocol can

# OTIDS → pipeline_can_otids_* / results/can_otids/
python -m mth_ids_pipeline.utils.merge_can --source otids
python -m mth_ids_pipeline.run_supervised --protocol can_otids
```

Detalhes: [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) · [PROTOCOLO_CAN_INTRUSION.md](PROTOCOLO_CAN_INTRUSION.md) · [PROTOCOLO_CAN_OTIDS.md](PROTOCOLO_CAN_OTIDS.md) · CICIDS: [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md).

Use `--protocol paper` (CICIDS) ou `--protocol can` (CAN) em todos os orquestradores.

---

## CICIDS2017

### Tabela VII — supervisionado

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_mth_ids_merged
```

Pasta: `data/pipeline_mth_ids_merged/`. Retomar: `--from 4 --to 6`.

### Tabela IX — LOAO

```powershell
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

Pasta: `data/pipeline_mth_ids_fine/anomaly/loao/attack_<N>/`. ~14 rodadas; fase 8 ~1 h/ataque.

Um ataque (ex.: Bot, label 1):

```powershell
python -m mth_ids_pipeline.run_all --label-profile fine `
  --protocol paper --from 12 --to 12 --skip-bootstrap --attack-label 1
```

Retomar fases 9–11: [PIPELINE_PHASES.md — Retomar LOAO](PIPELINE_PHASES.md#retomar-um-ataque-loao-fases-911-manuais).

### Tabela X — sistema completo

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper
python -m mth_ids_pipeline.run_global_anomaly --protocol paper
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_mth_ids_merged `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_mth_ids_merged
```

Fases 7–11 treinam em `anomaly/global/` **sem** o hold-out; o teste real só entra no `run_eval` (fase 13) via `05_test_unchanged.parquet`.

### Comparativo completo

```powershell
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

---

## CAN (intra-veicular)

Dois datasets: [Car-Hacking original](PROTOCOLO_CAN_INTRUSION.md) (`--source original`) e [repack OTIDS](PROTOCOLO_CAN_OTIDS.md) (`--source otids`). Pipeline compartilhado: [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md).

```powershell
python -m mth_ids_pipeline.run_supervised --protocol can          # intrusion
# python -m mth_ids_pipeline.run_supervised --protocol can_otids  # OTIDS
python -m mth_ids_pipeline.run_anomaly --protocol can --loao
python -m mth_ids_pipeline.run_global_anomaly --protocol can
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_can_otids_merged `
  --work-dir data/pipeline_can_otids_merged/anomaly/global
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_can_otids_merged `
  --loao-root data/pipeline_can_otids_fine/anomaly/loao `
  --results-dir results/can_otids
```

Labels LOAO — Car-Hacking: `1`=DoS, `2`=Fuzzy, `3`=Gear, `4`=RPM. OTIDS: `1`=DoS, `2`=Fuzzy, `3`=Impersonation.

---

## Pastas e artefatos

```
results/                              # relatórios (fora de data/)
├── paper_comparison.json
├── tables_report.txt
├── logs/                             # execuções timestampadas
└── can/                              # CAN (--results-dir)

data/
├── pipeline_mth_ids_merged/          # VII + X
│   ├── 06_supervised_metrics.json
│   ├── anomaly/global/
│   └── phase_reports/phase13_full_system_eval.json
├── pipeline_mth_ids_fine/            # IX
│   └── anomaly/loao/attack_1 … attack_14/
├── pipeline_can_otids_merged/              # VI + X
└── pipeline_can_otids_fine/                # VIII (attack_1 … attack_3)
```

| Script | Default perfil | Fases | Tabela |
|--------|----------------|-------|--------|
| `run_supervised` | merged | 1–6 | VII / VI (`--protocol can`) |
| `run_anomaly --loao` | fine | 7–12 | IX / VIII |
| `run_global_anomaly` | merged | 7–11 global | X (pré-requisito) |
| `run_eval` | — | 13 | X |
| `report_paper_tables` | — | — | exporta para `results/` |

---

## Bootstrap automático

`run_anomaly --loao` chama `ensure_anomaly_prerequisites()` antes das fases 7–12:

| Artefato | Gerado onde | Motivo |
|----------|-------------|--------|
| `02_sampled_kmeans.parquet` | Fases 1–2 no **fine** | Entrada da fase 7 (LOAO) |
| `06_supervised_metrics.json` | Fases 4–6 no **merged** → **cópia** no fine | Fase 11: família RF/XGB/DT/ET para B₁/B₂ |

A fase 11 **não** carrega o modelo da Tabela VII — só escolhe a **família** vencedora. Por isso o JSON vem do merged, não de um re-treino fine.

CAN: mesma lógica com `pipeline_can_otids_merged` / `pipeline_can_otids_fine`. Use sempre `--protocol can`.

Desativar: `--skip-bootstrap` (quando `02_` e `06_` já existem no fine).

### Fase 2 no fine (CICIDS)

Preserva inteiros os rótulos fine equivalentes ao `df_minor` merged (Bot, Infiltration, WebAttack). **PortScan não é preservado** — passa pelo k-means 0,8% como no notebook merged. Implementação: `label_profiles.py`.

---

## Pasta `results/`

| Flag | Efeito |
|------|--------|
| (padrão) | Grava `paper_comparison.json` + `tables_report.txt` em `results/` |
| `--results-dir results/can_otids` | Outra pasta de saída |
| `--no-save` | Só imprime no terminal |

Artefatos de treino (parquets, modelos) ficam em `data/`.

---

## Solução de problemas

| Sintoma | Solução |
|---------|---------|
| `02_sampled_kmeans.parquet` ausente (global) | `run_supervised --from 1 --to 2` no **merged** |
| `02_…` ausente (LOAO) | `run_anomaly` sem `--skip-bootstrap` |
| `06_supervised_metrics.json` ausente no fine | `run_supervised` no merged |
| `phase13_…json` ausente | `run_global_anomaly` + `run_eval` |
| Tabela IX vazia | LOAO incompleto; reconstruir `loao_summary.json` — [PIPELINE_PHASES.md](PIPELINE_PHASES.md#retomar-um-ataque-loao-fases-911-manuais) |
| Fase 9: SMOTE `n_jobs` | Atualizar código; `imbalanced-learn` ≥ 0.12 |
| BO-GP: `n_calls >= 10` | `--n-calls 15` ou `--hpo-n-calls 15` |
| CAN bootstrap puxa CICIDS | Faltou `--protocol can` |
| Comparativo X ≠ artigo | Pipeline usa split **80/20**; artigo CICIDS reporta **70/30** |

---

## Fidelidade ao artigo

- Split padrão: **80/20** (notebook); Tabela X do artigo cita 70/30 — comparação aproximada. Ver [NOTA_REPRODUCAO.md](NOTA_REPRODUCAO.md).
- Protocolo `paper`: BO-GP em IG, KPCA, CL-k-means e p* — runs longos.
- LOAO e global medem coisas diferentes; ambos são necessários para reproduzir VII+IX+X.
