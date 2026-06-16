# Auditoria LOAO — Reprodutibilidade MTH-IDS (Tabela IX)

**Data:** 2026-06-05  
**Escopo:** ramo anômalo (fases 7–12), protocolo Leave-One-Attack-Out  
**Referência:** Yang et al., *MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles*, IEEE IoT Journal, 2022; notebook `paper_and_notebooks/MTH_IDS_IoTJ.ipynb`

---

## 1. Veredicto executivo

| Critério LOAO (Tabela IX) | Status |
|---------------------------|--------|
| Zero-day removido completamente do treino | **Conforme** |
| Todas as amostras zero-day no teste | **Conforme** |
| Teste com mesma quantidade de benignos (1:1) | **Conforme** |
| Demais ataques permanecem no treino (colapsados em classe 1) | **Conforme** |
| IG / FCBF / KPCA sem vazamento do teste | **Conforme** (mais rigoroso que notebook) |
| Normalização sem vazamento estatístico | **Conforme** (diverge do notebook em detalhe) |
| HPO (fase 10) usando informação do teste | **Conforme ao artigo/notebook** |

**Conclusão:** o **particionamento LOAO** reproduz fielmente a Tabela IX do artigo. O pré-processamento de features (fase 8) é **estatisticamente mais rigoroso** que o notebook publicado (fit exclusivo no treino). A fase 10 otimiza `n_clusters` pela acurácia no teste LOAO, como no notebook — comportamento esperado para reproduzir métricas tier 3 do artigo.

---

## 2. Mapa da pipeline anômala

```text
02_sampled_kmeans.parquet
        │
        ▼
[Fase 7]  build_anomaly_binary_split
          → a01 (treino sem zero-day, binário) + a02 (só zero-day)
          → a00_loao_round.json (rótulos originais)
        ▼
[Fase 8]  build_loao_train_test_split → Z-score/IG/FCBF/KPCA (fit treino)
          → a03, a04, a06_test_slice.json, fitted_*.joblib
        ▼
[Fase 9]  SMOTE só treino + CL-k-means baseline
        ▼
[Fase 10] BO-GP n_clusters ∈ [2,50] (objetivo = acurácia teste LOAO)
        ▼
[Fase 11] CL-k-means (k da fase 10) + biased B₁/B₂
        ▼
[Fase 12] Repete 7→11 para cada rótulo de ataque
```

---

## 3. Onde treino e teste são construídos

| Fase | Arquivo | Função | Papel |
|------|---------|--------|-------|
| 7 | `phase07_anomaly_datasets.py` | `build_anomaly_binary_split` | Remove zero-day; colapsa demais ataques → 1 |
| 8 | `anomaly_io.py` | `build_loao_train_test_split` | Emparelha benignos 1:1; define índices treino\|teste |
| 8 | `phase08_anomaly_features.py` | `main` | Fit treino / transform teste (Z-score, IG, FCBF, KPCA) |
| 9–11 | `anomaly_io.py` | `load_anomaly_splits` | Recorta `a04_after_kpca.parquet` por `n_train_rows` |

### 3.1 Fase 7 — split binário

```python
df1 = df[df[label_col] != attack_label]   # treino: sem zero-day
df1.loc[df1[label_col] > 0, label_col] = 1  # ataques conhecidos → classe 1
df2 = df[df[label_col] == attack_label]   # teste parcial: só zero-day
df2.loc[:, label_col] = 1
```

### 3.2 Fase 8 — partição final + emparelhamento benigno

```python
sample_n = min(n_zero_day, available_benign)  # Tabela IX: 1:1
test_df = concat(df2_attack, benign_sampled)
train_df = df1  # intacto; benignos do teste permanecem no treino (protocolo artigo)
```

---

## 4. Verificação por invariante LOAO

### 4.1 Zero-day excluído do treino

- **Implementação:** filtro `label != attack_label` em `build_anomaly_binary_split`.
- **Validação:** `loao_original_label_report` → `zero_day_fully_excluded_from_train`; `validate_loao_partition` falha se o rótulo original do zero-day aparecer em `train_original_label_counts`.

### 4.2 Todas as amostras zero-day no teste

- **Implementação:** `df2` integralmente concatenado em `test_df`.
- **Validação:** `test.label[1] == zero_day_samples`.

### 4.3 Emparelhamento benigno 1:1 (Tabela IX)

- **Implementação:** `benign_sample_size_for_zero_day` → `min(n_zero_day, benignos_disponíveis)`.
- **Validação:** se benignos suficientes, `benign_sampled == zero_day_samples`; caso contrário, aviso explícito.

### 4.4 Demais ataques no treino

- Rótulos originais > 0 colapsados em classe binária 1.
- Metadados `train_attack_labels_present` e `train_original_label_counts` em `a00_loao_round.json`.

### 4.5 Sobreposição benigno treino/teste

- Benignos amostrados para o teste **permanecem no treino** (`benign_overlap_train_test`).
- **Impacto:** leve otimismo nas métricas de benignos no teste; **fiel ao notebook/artigo**.

---

## 5. Relatório de problemas, impacto e correções

### P1 — Logs insuficientes de rótulos e tamanhos

| Campo | Valor |
|-------|-------|
| **Arquivo** | `phase07_anomaly_datasets.py`, `phase08_anomaly_features.py`, `phase09_anomaly_cluster.py`, `phase10_anomaly_cluster_hpo.py`, `phase11_anomaly_biased.py`, `anomaly_io.py` |
| **Função** | `log_loao_partition`, `load_anomaly_splits`, `main()` |
| **Problema** | Fases 10–11 não registravam explicitamente contagens treino/teste nos relatórios JSON. |
| **Impacto metodológico** | Dificulta auditoria pós-execução e comparação entre rodadas LOAO. |
| **Correção** | **Implementada (2026-06-05):** logs em fases 7–11; campos `train_rows`, `test_rows`, `train_label_counts`, `test_label_counts` nos relatórios 10–11; `loao_summary.json` enriquecido na fase 12. |

---

### P2 — Validação incompleta dos invariantes LOAO

| Campo | Valor |
|-------|-------|
| **Arquivo** | `anomaly_io.py`, `phase08_anomaly_features.py` |
| **Função** | `validate_loao_partition` |
| **Problema** | Validação verificava apenas composição do teste; não checava exclusão do zero-day no treino nem 1:1 estrito quando benignos suficientes. |
| **Impacto metodológico** | Erros de partição (ex.: zero-day residual no treino) poderiam passar despercebidos. |
| **Correção** | **Implementada (2026-06-05):** checagem de `zero_day_fully_excluded_from_train`, ausência do rótulo zero-day em `train_original_label_counts`, e `benign_sampled == zero_day_samples` quando `benign_available >= zero_day_samples`. Revalidação pós-merge de metadados na fase 8. |

---

### P3 — IG / FCBF / KPCA no conjunto combinado (notebook)

| Campo | Valor |
|-------|-------|
| **Arquivo** | Notebook `MTH_IDS_IoTJ.ipynb` (~células 74–82) |
| **Função** | `mutual_info_classif`, `fcbf.fit_transform`, `KernelPCA.fit` |
| **Problema (notebook)** | Features ajustadas com treino **e** teste concatenados antes do split por índice. |
| **Impacto metodológico** | Vazamento de informação do zero-day e dos benignos de teste na seleção de atributos e KPCA. |
| **Correção no pipeline** | **Já corrigido:** `AnomalyFeaturePipeline` + `fit_kpca`/`transform_kpca` — fit exclusivamente no treino, transform no teste. **Divergência consciente** vs notebook; alinhado à Tabela IX em espírito, mais rigoroso estatisticamente. |

---

### P4 — Normalização Z-score: notebook vs pipeline

| Campo | Valor |
|-------|-------|
| **Arquivo** | Notebook (célula 68); `feature_selection.py` / `phase08` |
| **Função** | `StandardScaler` |
| **Problema** | Notebook normaliza **df1 e df2 separadamente**; pipeline ajusta scaler no treino e transforma o teste. |
| **Impacto metodológico** | Estatísticas de normalização diferem para fluxos zero-day. Pipeline evita vazamento; notebook usa estatísticas só do zero-day no df2. **Não idênticos.** |
| **Correção** | **Mantido pipeline atual** (fit treino → transform teste). Documentado como divergência consciente. Para reprodução numérica exata do notebook PortScan-demo, seria necessário normalização separada — fora do escopo LOAO Tabela IX. |

---

### P5 — HPO (fase 10) otimiza acurácia no teste LOAO

| Campo | Valor |
|-------|-------|
| **Arquivo** | `phase10_anomaly_cluster_hpo.py` |
| **Função** | `objective()` → `cl_kmeans(..., X_test, y_test)` |
| **Problema** | Seleção de `n_clusters` usa o conjunto de teste da rodada LOAO. |
| **Impacto metodológico** | Infla métricas finais vs hold-out independente; **reproduz notebook e tier 3 do artigo**. |
| **Correção** | **Nenhuma** (comportamento intencional). Para avaliação estrita: `--skip-phase10` e k fixo. |

---

### P6 — Emparelhamento benigno fixo 1255 (versão antiga)

| Campo | Valor |
|-------|-------|
| **Arquivo** | `phase08_anomaly_features.py`, `experiment_runner.py` (versão anterior) |
| **Função** | `--benign-target` default |
| **Problema** | Default `1255` em todas as rodadas LOAO contradiz Tabela IX (1:1 por ataque). |
| **Impacto metodológico** | DoS (3042 fluxos) ficava com teste desbalanceado (1255 benignos + 3042 ataques). |
| **Correção** | **Já corrigido (2026-06-04):** default `None` → regra `paper_table_ix_1_to_1`; override via `--benign-target` apenas para demo PortScan do notebook. |

---

### P7 — Biased classifiers (tier 4) com hold-out interno

| Campo | Valor |
|-------|-------|
| **Arquivo** | `biased_classifiers.py`, `phase11_anomaly_biased.py` |
| **Função** | `pick_best_biased_mode` |
| **Problema** | Nenhum — gate B₁/B₂ usa `train_test_split` **apenas no treino**, não no teste LOAO. |
| **Impacto metodológico** | Sem vazamento do teste zero-day na seleção do modo biased. |
| **Correção** | **Nenhuma necessária.** |

---

## 6. Análise de vazamento (data leakage)

| Etapa | Ajuste usa teste? | Avaliação |
|-------|-------------------|-----------|
| Z-score | Não — `fit_transform(X_train)`, `transform(X_test)` | OK |
| IG (90%) | Não — `mutual_info_classif(X_train, y_train)` | OK |
| FCBF | Não — `fit_fcbf(X_train)`, `transform_fcbf(X_test)` | OK |
| KPCA | Não — `fit_kpca(X_train)`, `transform_kpca(X_test)` | OK |
| SMOTE | Não — apenas `X_train` (fases 9, 11) | OK |
| CL-k-means fit | Não — `km.fit_predict(X_train)` | OK |
| BO-GP HPO | **Sim (avaliação)** — objetivo = acc no teste LOAO | Intencional (artigo) |
| Biased gate | Não — hold-out interno do treino | OK |

---

## 7. Logs esperados pós-correção

```text
=== LOAO [fase 7 (split binário)] ===
  treino: rows=24825 labels={'0': 18163, '1': 6662}
  teste:  rows=1966 labels={'1': 1966}
  zero-day=1966 benign_test=0 regra=fase7_zero_day_only
  rótulos originais no treino (pré-binário): {'0': 18163, '2': 96, ...}
  zero-day label=1 excluído_do_treino=True

=== LOAO [fase 8 (pré-normalização)] ===
  treino: rows=24825 labels={'0': 18163, '1': 6662}
  teste:  rows=3932 labels={'1': 1966, '0': 1966}
  zero-day=1966 benign_test=1966 regra=paper_table_ix_1_to_1

Partição LOAO (fase 9): treino=(24825, 10) labels={'0': 18163, '1': 6662} | teste=(3932, 10) labels={'0': 1966, '1': 1966}
Partição LOAO (fase 10): treino=(36450, 10) labels={...} | teste=(3932, 10) labels={'0': 1966, '1': 1966}
Partição LOAO (fase 11): treino=(36450, 10) labels={...} | teste=(3932, 10) labels={'0': 1966, '1': 1966}
```

Campos em `a06_test_slice.json` (compatíveis com consumidores existentes):

- `n_train_rows`, `n_test_rows`
- `train_binary_label_counts`, `test_binary_label_counts`
- `zero_day_label`, `train_original_label_counts`
- `zero_day_fully_excluded_from_train`
- `benign_pairing_rule`, `benign_overlap_train_test`

---

## 8. Arquivos alterados nesta auditoria (2026-06-05)

| Arquivo | Mudança |
|---------|---------|
| `anomaly_io.py` | `validate_loao_partition` reforçada (exclusão zero-day, 1:1 estrito) |
| `phase07_anomaly_datasets.py` | `log_loao_partition` na fase 7 |
| `phase08_anomaly_features.py` | Revalidação pós-merge de metadados LOAO |
| `phase10_anomaly_cluster_hpo.py` | Logs e campos de partição no relatório |
| `phase11_anomaly_biased.py` | Logs e campos de partição no relatório |
| `phase12_anomaly_loao.py` | `loao_summary.json` com contagens treino/teste por ataque |

Interfaces públicas (`build_anomaly_binary_split`, `build_loao_train_test_split`, assinaturas CLI) **inalteradas**.

---

## 9. Como reexecutar a verificação

```powershell
python -m mth_ids_pipeline.experiment_runner --label-profile merged --from 1 --to 2
python -m mth_ids_pipeline.run_anomaly --label-profile merged --loao
```

Inspecionar por ataque:

- `anomaly/loao/attack_<N>/a00_loao_round.json`
- `anomaly/loao/attack_<N>/a06_test_slice.json`
- `anomaly/loao/loao_summary.json`

---

*Auditoria de reprodutibilidade científica — pipeline MTH-IDS, ramo anômalo LOAO.*
