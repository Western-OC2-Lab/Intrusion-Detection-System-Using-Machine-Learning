# Auditoria de reprodutibilidade — protocolo LOAO (MTH-IDS, Tabela IX)

**Data:** 2026-06-04  
**Escopo:** ramo anômalo (fases 7–12), comparação com artigo Yang et al. (IEEE IoT Journal, 2022) e notebook oficial.

---

## 1. Resumo executivo

| Critério LOAO (Tabela IX) | Status |
|---------------------------|--------|
| Zero-day removido completamente do treino | **Conforme** |
| Todas as amostras zero-day no teste | **Conforme** |
| Teste com mesma quantidade de benignos (1:1) | **Conforme** |
| Demais ataques permanecem no treino (colapsados em classe 1) | **Conforme** |
| IG / FCBF / KPCA sem vazamento do teste | **Conforme** (melhoria vs notebook) |
| Normalização sem vazamento | **Conforme** (diverge do notebook em detalhe) |
| HPO usando informação do teste | **Conforme ao notebook**; diverge de validação estrita |

**Veredicto:** o particionamento LOAO reproduz fielmente a Tabela IX. O pré-processamento de features (fase 8) é **mais rigoroso** que o notebook publicado (evita vazamento). A fase 10 (BO-GP) otimiza no conjunto de teste LOAO, como no notebook — comportamento esperado para reproduzir números do artigo tier 3.

---

## 2. Mapa da pipeline anômala

```text
02_sampled_kmeans.parquet
        │
        ▼
[Fase 7]  build_anomaly_binary_split
          → a01 (treino sem zero-day, binário) + a02 (só zero-day)
        ▼
[Fase 8]  build_loao_train_test_split → IG/FCBF/KPCA (fit treino)
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

---

## 4. Verificação por invariante LOAO

### 4.1 Zero-day excluído do treino

- **Implementação:** `df1 = df[df[label_col] != attack_label]` em `build_anomaly_binary_split`.
- **Verificação empírica** (attack_1 / Bot): treino 24 825 linhas; zero-day (label 1) = 1 966 fluxos ausentes do treino; `zero_day_fully_excluded_from_train: true`.

### 4.2 Todas as amostras zero-day no teste

- **Implementação:** `df2 = df[df[label_col] == attack_label]`, depois concatenação integral em `test_df`.
- **Verificação:** `test.label[1] == zero_day_samples` (1 966 = 1 966).

### 4.3 Emparelhamento benigno 1:1

- **Implementação:** `benign_sample_size_for_zero_day(n_zero_day, available_benign)` → `min(n_zero_day, benignos_disponíveis)`.
- **Verificação:** teste `{0: 1966, 1: 1966}`.

### 4.4 Demais ataques no treino

- Após fase 7, rótulos originais > 0 viram classe binária 1 (ataques conhecidos agregados).
- Exemplo attack_1: treino binário `{0: 18163, 1: 6662}` — 6 662 fluxos de ataques conhecidos (DoS, PortScan, etc., exceto Bot).

### 4.5 Sobreposição benigno treino/teste

- Benignos amostrados para o teste **permanecem no treino** (protocolo notebook: `df = df1.append(df2)` sem remoção).
- **Impacto:** leve otimismo nas métricas de benignos no teste; **fiel ao notebook/artigo**, não é bug de implementação.

---

## 5. Relatório de problemas e correções

### P1 — Logs insuficientes de rótulos e tamanhos

| Campo | Valor |
|-------|-------|
| **Arquivo** | `phase07_anomaly_datasets.py`, `phase08_anomaly_features.py`, `phase09_anomaly_cluster.py`, `anomaly_io.py` |
| **Função** | `main()`, `build_loao_train_test_split`, `load_anomaly_splits` |
| **Problema** | Não havia registro explícito de rótulos originais no treino, contagens binárias treino/teste nem validação automática dos invariantes LOAO. |
| **Impacto metodológico** | Dificulta auditoria e reprodutibilidade; erros de partição passariam despercebidos. |
| **Correção** | **Implementada:** `log_loao_partition`, `validate_loao_partition`, `loao_original_label_report`, `a00_loao_round.json`, campos extras em `a06_test_slice.json` e relatórios JSON. |

---

### P2 — Metadados de rótulo original não persistidos entre fases

| Campo | Valor |
|-------|-------|
| **Arquivo** | `phase07_anomaly_datasets.py`, `phase08_anomaly_features.py` |
| **Função** | `main()` |
| **Problema** | Após colapso binário (fase 7), perdia-se quais rótulos inteiros permaneciam no treino. |
| **Impacto metodológico** | Impossível auditar “quais ataques conhecidos” alimentam cada rodada LOAO. |
| **Correção** | **Implementada:** `a00_loao_round.json` + campos `train_original_label_counts`, `train_attack_labels_present` no relatório fase 7 e meta fase 8. |

---

### P3 — IG / FCBF / KPCA no conjunto combinado (notebook)

| Campo | Valor |
|-------|-------|
| **Arquivo** | Notebook `MTH_IDS_IoTJ.ipynb` (células ~74–82) |
| **Função** | `mutual_info_classif(X, y)`, `fcbf.fit_transform(X_fs, y)`, `KernelPCA.fit` |
| **Problema (notebook)** | Features ajustadas com treino **e** teste concatenados antes do split por índice. |
| **Impacto metodológico** | Vazamento de informação do zero-day e dos benignos de teste na seleção de atributos e KPCA. |
| **Correção no pipeline** | **Já corrigido** em `phase08_anomaly_features.py` + `AnomalyFeaturePipeline` + `fit_kpca`/`transform_kpca`: fit exclusivamente no treino, transform no teste. |

---

### P4 — Normalização Z-score: notebook vs pipeline

| Campo | Valor |
|-------|-------|
| **Arquivo** | Notebook (célula 68); `feature_selection.py` / `phase08` |
| **Função** | Z-score por coluna |
| **Problema** | Notebook normaliza **df1 e df2 separadamente** (cada um com sua média/desvio); pipeline ajusta `StandardScaler` no treino e transforma o teste. |
| **Impacto metodológico** | Para fluxos zero-day, estatísticas de normalização diferem. Pipeline evita usar estatísticas do teste no ajuste; notebook usa estatísticas só do zero-day no df2. Ambos são defensáveis; **não idênticos**. |
| **Correção** | **Mantido pipeline atual** (fit treino → transform teste) por ser mais rigoroso contra vazamento. Documentado como divergência consciente vs notebook. |

---

### P5 — HPO (fase 10) otimiza acurácia no teste LOAO

| Campo | Valor |
|-------|-------|
| **Arquivo** | `phase10_anomaly_cluster_hpo.py` |
| **Função** | `objective()` → `cl_kmeans(..., X_test, y_test)` |
| **Problema** | Seleção de `n_clusters` usa o conjunto de teste da rodada LOAO. |
| **Impacto metodológico** | Infla métricas finais vs hold-out independente; **reproduz notebook e tier 3 do artigo**. |
| **Correção** | **Nenhuma** (comportamento intencional para reproduzir Tabela IX). Para avaliação estrita, usar `--skip-phase10` e k fixo. |

---

### P6 — Documentação desatualizada (LOAO_PROTOCOL_CORRECTION_REPORT §5.1)

| Campo | Valor |
|-------|-------|
| **Arquivo** | `docs/LOAO_PROTOCOL_CORRECTION_REPORT.md` |
| **Problema** | Ainda menciona “IG → FCBF → KPCA (conjunto combinado)”. |
| **Impacto** | Confusão na auditoria. |
| **Correção** | Código já usa `fit_train_transform_test`; este relatório substitui/atualiza a seção de vazamento. |

---

## 6. Estado pós-correção — logs esperados

Exemplo de saída (fase 7 + 8):

```text
LOAO fase 7 — zero-day label=1: treino=24825 (sem zero-day), teste parcial=1966 (só zero-day)
  rótulos originais no treino: {'0': 18163, '2': 96, '3': 3042, '4': 11, '5': 1255, '6': 2258}
  ataques conhecidos no treino: [2, 3, 4, 5, 6]
  zero-day excluído do treino: True

=== LOAO [fase 8 (pré-normalização)] ===
  treino: rows=24825 labels={'0': 18163, '1': 6662}
  teste:  rows=3932 labels={'1': 1966, '0': 1966}
  zero-day=1966 benign_test=1966 regra=paper_table_ix_1_to_1
```

Campos novos em `a06_test_slice.json` (compatíveis com consumidores existentes):

- `train_binary_label_counts`
- `test_binary_label_counts`
- `benign_overlap_train_test`
- `zero_day_label`, `train_original_label_counts` (quando `a00_loao_round.json` presente)

---

## 7. Arquivos alterados nesta auditoria

| Arquivo | Mudança |
|---------|---------|
| `anomaly_io.py` | `loao_original_label_report`, `log_loao_partition`, `validate_loao_partition`, meta enriquecida |
| `phase07_anomaly_datasets.py` | Logs, `a00_loao_round.json`, relatório enriquecido |
| `phase08_anomaly_features.py` | Logs de partição, propagação meta LOAO |
| `phase09_anomaly_cluster.py` | Logs e `test_label_counts` no relatório |
| `config.py` | Constante `A00_LOAO_ROUND` |
| `anomaly_io.load_anomaly_splits` | Log de partição KPCA |

Interfaces públicas (`build_anomaly_binary_split`, `build_loao_train_test_split`, assinaturas CLI) **inalteradas**.

---

## 8. Como reexecutar a verificação

```powershell
python -m mth_ids_pipeline.run_supervised --label-profile merged --from 1 --to 2
python -m mth_ids_pipeline.run_anomaly --loao --from 12 --to 12
```

Inspecionar por ataque:

- `anomaly/loao/attack_<N>/a00_loao_round.json`
- `anomaly/loao/attack_<N>/a06_test_slice.json`
- logs stdout das fases 7–9

---

*Auditoria gerada automaticamente. Referência: Yang et al., MTH-IDS, IEEE IoT Journal, 2022; notebook `paper_and_notebooks/MTH_IDS_IoTJ.ipynb`.*
