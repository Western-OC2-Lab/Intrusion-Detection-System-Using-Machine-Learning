# Nota sobre a reprodução MTH-IDS

Texto de referência para relatório, artigo ou defesa — **limitações metodológicas** da reprodução, com foco na **dualidade artigo × notebook IoTJ**, **excluindo** bugs de implementação já corrigidos no código.

Documentos relacionados: [REPRODUCAO_CICIDS2017_VALIDACAO.md](REPRODUCAO_CICIDS2017_VALIDACAO.md) · [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) · [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) · [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md)

---

## Escopo declarado

Esta reprodução segue o preset **`--protocol paper`** (CICIDS2017) e **`--protocol can`** (CAN-OTIDS), implementados em `mth_ids_pipeline/protocol.py`. O objetivo é reproduzir o **método descrito no artigo** (Yang et al., IEEE IoT Journal 2022): BO-GP nos tiers 3–4, 10-fold CV no supervisionado, LOAO multi-ataque, biased B₁/B₂, cascata Tabela X.

**Não** se trata de reexecutar linha a linha o notebook [`MTH_IDS_IoTJ.ipynb`](../paper_and_notebooks/MTH_IDS_IoTJ.ipynb). Esse notebook é um artefato **paralelo** ao texto do artigo, com escolhas diferentes em vários pontos.

---

## O problema central: três fontes, nenhuma idêntica à outra

| Fonte | O que representa | Papel na reprodução |
|-------|------------------|---------------------|
| **Artigo (PDF)** | Metodologia de alto nível + tabelas numéricas | **Alvo** das comparações |
| **Notebook IoTJ** | Código publicado pelos autores (2021) | Referência operacional parcial |
| **Preset `paper`** | Híbrido documentado neste repositório | **O que de fato foi executado** |

O preset `paper` **não** é nem o artigo puro nem o notebook puro. Combina, por exemplo:

- **Do artigo:** BO-GP (α IG, KPCA, k, p*), 10-fold CV no HPO supervisionado, meta-learner `best-base`, LOAO fine (~14 ataques), tier 4 biased.
- **Do notebook:** split **80/20**, SMOTE com alvos **1000** (não 100k citados no texto), amostragem k-means 0,8% + minoritárias `{6,1,4}`, escala ~27k linhas.

Por isso, **coincidência numérica com o PDF não implica réplica literal**, e **proximidade ao notebook em hold-out não valida o artigo**.

---

## Limitações estruturais (artigo × notebook × preset `paper`)

Estas limitações **persistem mesmo com o pipeline correto** — não são falhas de comando ou ordem de fases.

### 1. Split treino / teste

| Fonte | Split |
|-------|-------|
| Artigo (Sec. IV-F, Tabela X) | **70/30** em trechos do texto |
| Notebook IoTJ | **80/20** estratificado |
| Preset `paper` | **80/20** |

**Efeito:** Tabela VII ~0,3 pp abaixo do artigo; Tabela X ~1,6 pp (Acc/F1). Comparar com Tabela X do PDF exige explicitar que o hold-out é 20%, não 30%.

### 2. SMOTE (supervisionado)

| Fonte | Alvo |
|-------|------|
| Artigo (texto) | até **100 000** amostras sintéticas |
| Notebook + preset `paper` | **1000** por classe (BruteForce=2, Infiltration=4) |

**Efeito:** escolha **intencional** do preset para alinhar escala ao notebook; diverge do número citado no paper.

### 3. Information Gain (α)

| Fonte | α |
|-------|---|
| Notebook | **0,9 fixo** |
| Artigo | BO-GP (“otimizado”, sem bounds publicados) |
| Preset `paper` | BO-GP, **15 calls**, bounds **[0,7–0,99]** → α≈0,79 (sup.) / ≈0,72 (LOAO) |

**Efeito:** número de features pós-IG/FCBF difere do notebook (~36→20 vs ~44) e do texto (não especificado).

### 4. FCBF e normalização (supervisionado)

| Item | Artigo | Notebook | Preset `paper` |
|------|--------|----------|----------------|
| FCBF | k=20, treino | k=20, **dataset completo** | k=20, **só treino** |
| Z-score | pós-split (espírito) | **fase 1** (global) | **após split** (`scale-mode=split`) |

**Efeito:** hold-out supervisionado fica **mais próximo do notebook** em Acc (~99,55% vs ~99,57%), mas o **procedimento** segue o artigo (sem leakage de FCBF).

### 5. HPO e stacking (Tabela VII)

| Item | Artigo | Notebook | Preset `paper` |
|------|--------|----------|----------------|
| Objetivo HPO | validação / CV | **acurácia no teste** | **média 10-fold CV** |
| Meta-learner | clone do melhor base | **XGBoost + HPO** | **`best-base`** (XGB base na run validada) |

**Efeito:** CV reportada ~99,94% alinha ao artigo; hold-out ~99,55% alinha ao notebook — **métricas diferentes medem coisas diferentes**.

### 6. Anomaly / LOAO (Tabela IX)

| Item | Artigo | Notebook | Preset `paper` |
|------|--------|----------|----------------|
| Escopo | 14 zero-days, 1:1 benignos | demo **PortScan** (`merged`) | **fine**, 14 LOAO, 1:1 |
| Z-score / features | combinado (texto) | **per_split** (df1/df2) | **combined**, fit **só no treino** |
| KPCA | BO-GP | **fixo** n=10, RBF | BO-GP → n≈20, sigmoid |
| Benignos teste | 1:1 | `--benign-target 1255` | 1:1 automático |
| Tier 4 (B₁/B₂, p*) | sim | **omitido** (~5% do código) | sim (fase 11) |
| HPO fase 10 | BO-GP k + métrica | acc no teste LOAO | BO-GP, métrica **F1** |

**Efeito:** LOAO reproduz **estrutura** da Tabela IX; métricas podem **superar** o artigo (F1 médio ~0,87 vs ~0,80) por combinação de split, α, KPCA, biased e rigor estatístico no fit de features — **não** por “erro” de protocolo.

### 7. Sistema completo (Tabela X)

- Cascata tiers 1→4 implementada (fase 13).
- Hold-out = mesmo 20% da Tabela VII (`05_test_unchanged.parquet`).
- Treino anomaly global (fases 7–11) **não usa** esse hold-out; teste final só no `run_eval`.

**Efeito:** gap Acc ~98,25% vs ~99,88% no artigo explica-se sobretudo por **split 80/20** + detalhes de SMOTE/BO-GP acima, não por ausência de cascata.

### 8. CAN-OTIDS (Tabelas VI/VIII)

| Limitação | Detalhe |
|-----------|---------|
| Features | Artigo cita **4** (`CAN_ID`, `DATA_1`, `DATA_3`, `DATA_5`); BO-GP α IG pode selecionar **~7** pós-FCBF |
| Amostragem | k-means 0,8% em **todas** as classes (diferente do CICIDS, que preserva minoritárias) |
| SMOTE | desligado (artigo CAN não usa; preset correto) |
| LOAO | **3** ataques (não 14); nomenclatura interna legada (`a01_without_portscan`, etc.) |
| Tabela X CAN | suportada; referência numérica extrema no artigo (Acc ~99,99%) — validar com cautela |

---

## O que **não** entra nesta nota (correções já incorporadas)

Bugs e desvios **corrigidos no código** — partition LOAO, bootstrap merged→fine para `06_supervised_metrics.json`, fases 7–11 globais com `--loao`, leakage IG/KPCA (fit só treino), SMOTE `n_jobs`, teste vazio no modo global, etc. — **não** são limitações da reprodução; são pré-requisitos para que o preset `paper` funcione como documentado.

Para histórico de correções: [archive/](archive/README.md).

---

## Síntese numérica (CICIDS2017, run 2026-06-07)

| Tabela | Reprodução | Artigo | Interpretação |
|--------|------------|--------|---------------|
| VII Acc | ~99,55% | ~99,88% | Split 80/20 + SMOTE 1000 + α BO-GP |
| IX F1 médio | ~0,865 | ~0,800 | LOAO completo + tier 4; pode superar artigo |
| X Acc | ~98,25% | ~99,88% | Split 80/20 + cascata; maior gap esperado |

Comparado ao **notebook** (hold-out): diferença **&lt; 0,05%** na Tabela VII — confirma alinhamento operacional ao IoTJ, **não** equivalência ao PDF.

---

## Frase sugerida para o relatório

> A reprodução utiliza o preset `--protocol paper` do pipeline modular MTH-IDS, que implementa o fluxo metodológico do artigo (BO-GP, 10-fold CV, LOAO multi-ataque, biased classifiers e cascata Tabela X), incorporando decisões operacionais do notebook publicado (split 80/20, SMOTE com alvo 1000 amostras, amostragem k-means 0,8%). O texto do artigo omite ou contradiz detalhes em split (70/30 vs 80/20), escala SMOTE (100k vs 1000) e hiperespaço BO-GP; por isso, as métricas reportadas são **comparáveis em ordem de grandeza**, mas **não réplicas literais** das Tabelas VII e X do PDF. A Tabela IX reproduz a estrutura LOAO do artigo; diferenças numéricas refletem escolhas documentadas do preset e rigor estatístico no ajuste de features, não falhas de implementação. Para CAN, aplicou-se `--protocol can` com as mesmas ressalvas sobre seleção dinâmica de features e escala de amostragem.

---

## Checklist antes de citar resultados

- [ ] Declarar preset: `paper` / `can`, **não** `notebook`, salvo comparação explícita.
- [ ] Citar split **80/20** ao comparar Tabela X com o artigo.
- [ ] Mencionar SMOTE **1000** vs 100k do texto.
- [ ] Separar métrica de **CV** (artigo) vs **hold-out** (notebook) na Tabela VII.
- [ ] LOAO: 14 ataques (CICIDS) / 3 (CAN); não confundir com demo PortScan do notebook.
- [ ] Anexar logs em `results/logs/` e configs em `results/config/` ou `phase_reports/`.

---

## Referências internas

- Validação CICIDS2017 com logs: [REPRODUCAO_CICIDS2017_VALIDACAO.md](REPRODUCAO_CICIDS2017_VALIDACAO.md)
- Auditoria metodológica histórica: [archive/METHODOLOGICAL_AUDIT.md](archive/METHODOLOGICAL_AUDIT.md)
- Comparativo de presets: [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md)
