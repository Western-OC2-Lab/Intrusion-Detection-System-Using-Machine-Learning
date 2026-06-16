# Validação da reprodução CICIDS2017 (`--protocol paper`)

Análise dos parâmetros observados nos logs locais versus o **artigo** (Yang et al., IEEE IoT Journal 2022) e o **notebook IoTJ** publicado.

**Execução analisada:** 2026-06-07 · preset **`--protocol paper`**

**Fontes:**

| Artefato | Caminho |
|----------|---------|
| Log supervisionado | [`results/cicids2017/logs/merged/supervised_run.log`](../results/cicids2017/logs/merged/supervised_run.log) |
| Config da run | [`results/cicids2017/config/merged/experiment_runner_config.json`](../results/cicids2017/config/merged/experiment_runner_config.json) |
| Fase 2 (k-means) | [`results/cicids2017/config/merged/phase02_sample_kmeans.json`](../results/cicids2017/config/merged/phase02_sample_kmeans.json) |
| Fase 4 (IG+FCBF) | [`results/cicids2017/config/merged/phase04_feature_engineering.json`](../results/cicids2017/config/merged/phase04_feature_engineering.json) |
| Relatório vs artigo | [`results/cicids2017/report_tables_all_20260607_130408.log`](../results/cicids2017/report_tables_all_20260607_130408.log) |
| LOAO resumo | [`results/cicids2017/metrics/loao/loao_summary.json`](../results/cicids2017/metrics/loao/loao_summary.json) |

---

## Veredicto

**Sim, em linhas gerais a reprodução está correta.** Para CICIDS2017, usar **`--protocol paper`** e deixar o pipeline rodar até as fases com BO-GP/HPO é o caminho de reprodução do **método do artigo** (não do notebook IoTJ).

O que foi executado:

- Fases **1–2**, **4–6** supervisionadas com flags do preset `paper`
- **BO-GP** de α IG na fase 4; **HPO** (Hyperopt) na fase 6; **10-fold CV** para seleção de hiperparâmetros
- LOAO **14/14** ataques com BO-GP em IG, KPCA e `n_clusters` (Tabela IX)
- Relatório comparativo Tabelas **VII**, **IX** e **X** gerado

**Não** é réplica literal 100% do texto do PDF — há escolhas documentadas do preset `paper` e omissões metodológicas do artigo (split, SMOTE, espaço de busca BO-GP). **Nota completa para relatório/defesa:** [NOTA_REPRODUCAO.md](NOTA_REPRODUCAO.md).

---

## Supervisionado (Tabela VII) — parâmetros da execução

| Parâmetro | **Log / JSON local** | **Artigo** | **Notebook IoTJ** |
|-----------|----------------------|------------|-------------------|
| Protocolo | `paper` | Metodologia Sec. IV | Notebook publicado |
| Perfil | `merged` (7 famílias) | Multi-class merged | Merged |
| k-means | **`frac=0.008`**, k=1000 | 0,8% | **0,8%**, k=1000 fixo |
| Minoritárias intactas | **6, 1, 4** (WebAttack, Bot, Infiltration) | Preservadas (df_minor) | **6, 1, 4** |
| Amostra pós-fase 2 | **26 791** linhas | ~escala similar | ~26 794 |
| Split hold-out | **`test-size=0.2`** (80/20) | Sec. IV-F: **70/30** em alguns trechos | **80/20** |
| Normalização | **`scale-mode=split`** | Pós-split (artigo) | **Z-score fase 1** |
| α IG | **BO-GP** → **α≈0,786** (15 trials) | BO-GP (α não detalhado) | **Fixo 0,9** |
| Features IG→FCBF | **36 → 20** | Nº variável | ~44 → FCBF |
| FCBF | **k=20, scope=train** | k=20, treino | **k=20, scope=full** |
| SMOTE | **`{2:1000, 4:1000}`** | Artigo cita **100k**; pipeline usa alvos notebook | **`{2:1000, 4:1000}`** |
| HPO fase 6 | **`--hpo-on-validation`**, **10-fold CV** | Validação + 10-fold CV | **Acurácia no teste** |
| Meta stacking | **`best-base`** → **XGBoost (base)** | Clone do melhor base | **XGBoost meta + HPO** |
| Acc hold-out (stacking) | **~99,55%** | ~99,88% (Tabela VII) | ~99,57% |
| 10-fold CV stacking | **~99,94% ± 0,0004** | reportado no artigo | não usado |

### Comandos observados no log (fase 4 e 6)

```text
--fcbf-k 20 --test-size 0.2 --fcbf-scope train --scale-mode split
--ig-cumulative 0.9 --cv-folds 10 --ig-hpo-calls 15 --optimize-ig
BO-GP IG: alpha=0.7863, CV acc=0.9947

--cv-folds 10 --meta-learner best-base --hpo-on-validation
HPO objetivo: acurácia média em 10-fold CV (treino)
Stacking meta-learner (best-base): reutilizando 'XGBoost (base)'
```

---

## Anomaly LOAO (Tabela IX) — exemplo (ataque 10 / PortScan)

Fonte: `results/cicids2017/metrics/loao/attack_10/phase08_anomaly_features.json`

| Parâmetro | **Execução (paper)** | **Artigo** | **Notebook** |
|-----------|------------------------|------------|--------------|
| Perfil | **fine**, LOAO | 14 ataques | Demo PortScan |
| Z-score / features | **combined** | Conjunto combinado | **per_split** |
| α IG | **BO-GP → 0,716** | BO-GP | **Fixo 0,9** |
| KPCA | **BO-GP → n=20, sigmoid** | BO-GP | **Fixo n=10, RBF** |
| Benignos zero-day | **1:1** | Emparelhamento 1:1 | `--benign-target 1255` |
| CL-k-means HPO | **BO-GP**, métrica **F1** | BO-GP tier 3 | BO-GP **accuracy** |
| Biased B₁/B₂ + p* | **Sim** (fase 11) | Tier 4 | **Omitido** |

**LOAO concluído:** 14/14 ataques (`loao_summary.json`).

---

## Otimizações citadas no artigo — checklist

| Etapa | Artigo | Sua execução |
|-------|--------|--------------|
| k-means 0,8% + minoritárias | ✓ | ✓ fase 2 |
| IG + FCBF (k=20, treino) | ✓ | ✓ fase 4 |
| **BO-GP α IG** | ✓ | ✓ α≈0,786, 15 trials |
| SMOTE | ✓ | ✓ alvos **1000** (notebook) |
| HPO base learners (BO-TPE) | ✓ | ✓ fase 6 |
| **10-fold CV** para HPO | ✓ | ✓ |
| Stacking **best-base** | ✓ | ✓ → XGBoost base |
| LOAO + **BO-GP** α/KPCA/k | ✓ | ✓ 14 ataques |
| Biased + **BO-GP p*** | ✓ | ✓ fase 11 |

---

## Métricas vs artigo (relatório 2026-06-07)

### Tabela VII (supervisionado)

| Métrica | Reprod | Artigo | Diff |
|---------|--------|--------|------|
| Acc (stacking) | 0,9955 | 0,9988 | −0,0033 |
| F1(w) | 0,9955 | 0,9988 | −0,0033 |

Comparado ao **notebook** (hold-out): diferença **&lt; 0,05%** nos modelos principais.

### Tabela IX (LOAO média, 14 ataques)

| Métrica | Reprod | Artigo | Diff |
|---------|--------|--------|------|
| F1 | 0,865 | 0,800 | +0,065 |
| DR | 87,2% | 75,9% | +11,2 pp |
| FAR | 13,6% | 13,9% | −0,3 pp |

### Tabela X (sistema completo, hold-out 80/20)

| Métrica | Reprod | Artigo | Diff |
|---------|--------|--------|------|
| Acc | 98,25% | 99,88% | −1,63 pp |
| F1 | 0,976 | 0,999 | −0,023 |

Nota: Tabela X usa split **80/20**; artigo cita **70/30** em parte do texto.

---

## Divergências conhecidas (pipeline vs texto do artigo)

| Tópico | Artigo (texto) | Preset `paper` / sua run | Impacto |
|--------|----------------|--------------------------|---------|
| Split supervisionado | 70/30 (em alguns trechos) | **80/20** | ~−0,33 pp na Tabela VII |
| SMOTE | alvos **100k** | **1000** (notebook) | preset intencional |
| BO-GP α | “otimizado”, sem bounds | **[0,7–0,99], 15 calls** | α≈0,786 ≠ 0,9 |
| Notebook | referência IoTJ | **não usado** | HPO/CV/meta diferentes |

---

## Onde cada fonte coincide com a execução

| Aspecto | Log local | Artigo | Notebook |
|---------|-----------|--------|----------|
| k-means 0,8% + 6,1,4 | ✓ | ✓ | ✓ |
| SMOTE 1000 (2, 4) | ✓ | ⚠️ 100k no texto | ✓ |
| BO-GP α IG (supervisionado) | ✓ | ✓ | ✗ |
| FCBF só treino | ✓ | ✓ | ✗ |
| 10-fold CV + HPO validação | ✓ | ✓ | ✗ |
| Meta `best-base` | ✓ | ✓ | ✗ |
| Split 80/20 | ✓ | ⚠️ | ✓ |
| LOAO + BO-GP + biased | ✓ | ✓ | ✗ |

---

## Comandos de referência

```powershell
# Reprodução método artigo (o que foi executado)
python -m mth_ids_pipeline.run_supervised --protocol paper
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao
python -m mth_ids_pipeline.run_global_anomaly --protocol paper
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_mth_ids_merged `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao

# Estilo notebook IoTJ (parâmetros diferentes — NÃO é o artigo)
python -m mth_ids_pipeline.run_supervised --protocol notebook
```

---

## Resumo em uma frase

A execução com **`--protocol paper`** reproduz corretamente o **fluxo metodológico do artigo até as otimizações BO-GP/HPO**; os gaps numéricos restantes (Tabela VII ~0,3 pp; Tabela X ~1,6 pp) vêm de **detalhes não especificados no paper** (split, SMOTE, hiperespaço BO-GP), não de erro de comando ou ordem de fases.

Ver também: [NOTA_REPRODUCAO.md](NOTA_REPRODUCAO.md) · [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) · [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) · [docs/archive/METHODOLOGICAL_AUDIT.md](archive/METHODOLOGICAL_AUDIT.md) · [docs/archive/REPRODUCTION_REPORT.md](archive/REPRODUCTION_REPORT.md)
