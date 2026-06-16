# Relatório de correção — protocolo LOAO / Tabela IX (MTH-IDS)

Data: 2026-06-04  
Objetivo: alinhar o pipeline ao protocolo da **Tabela IX** do artigo, sem preservar atalhos do notebook que contradizem o artigo.

---

## 1. Arquivos modificados

| Arquivo | Tipo de mudança |
|---------|-----------------|
| `mth_ids_pipeline/anomaly_io.py` | Nova função + documentação |
| `mth_ids_pipeline/phase08_anomaly_features.py` | Regra de emparelhamento BENIGN |
| `mth_ids_pipeline/phase12_anomaly_loao.py` | Fluxo LOAO 7→10→11, remoção de k fixo |
| `mth_ids_pipeline/experiment_runner.py` | `anomaly_benign_target=None` por padrão |
| `docs/PIPELINE_PHASES.md` | Documentação do fluxo |
| `docs/GUIA_ARQUITETURA_MTH_IDS.md` | Diagrama textual fase 12 |

---

## 2. Funções / pontos de entrada modificados

| Módulo | Função / `main()` | Alteração |
|--------|-------------------|-----------|
| `anomaly_io.py` | **`benign_sample_size_for_zero_day`** (nova) | `min(n_zero_day, available_benign)` |
| `phase08_anomaly_features.py` | **`main()`** | Default Tabela IX; meta JSON enriquecido |
| `phase12_anomaly_loao.py` | **`main()`** | Fases 7–8–9–**10**–11; sem `--benign-target` / `--n-clusters` fixos |
| `experiment_runner.py` | **`ExperimentConfig`**, **`_phase_extra_args`** | Não passa `--benign-target` salvo override explícito |

**Sem alteração** (já compatíveis com o artigo, conforme solicitado):

- `build_anomaly_binary_split` (fase 7)
- `load_anomaly_splits` + SMOTE só no treino (fases 9–11)
- `cl_kmeans_fit_predict` — `fit` apenas em `X_train` (fase 9+)

---

## 3. Diff resumido (comportamento)

### 3.1 `anomaly_io.py`

```diff
+def benign_sample_size_for_zero_day(n_zero_day: int, available_benign: int) -> int:
+    if n_zero_day <= 0 or available_benign <= 0:
+        return 0
+    return min(int(n_zero_day), int(available_benign))
```

### 3.2 `phase08_anomaly_features.py`

```diff
-    elif args.benign_target is not None:
-        sample_n = int(args.benign_target)
-    else:
-        sample_n = min(len(df2), n_benign)
+    else:
+        sample_n = benign_sample_size_for_zero_day(n_zero_day, available_benign)
+        pairing_rule = "paper_table_ix_1_to_1"
```

Meta `a06_test_slice.json` passa a registrar `zero_day_samples`, `benign_sampled`, `benign_pairing_rule`.

### 3.3 `phase12_anomaly_loao.py`

```diff
-    parser.add_argument("--benign-target", type=int, default=1255)
-    parser.add_argument("--n-clusters", type=int, default=16)
-    # fluxo: 7 → 8 → 9 → 11
+    parser.add_argument("--benign-target", type=int, default=None)  # override opcional
+    # fluxo: 7 → 8 → 9 → 10 → 11
+    _run_phase("mth_ids_pipeline.phase10_anomaly_cluster_hpo", ...)
+    # fase 11 sem --n-clusters → lê phase10_anomaly_cluster_hpo.json
```

### 3.4 `experiment_runner.py`

```diff
-    anomaly_benign_target: int = 1255
+    anomaly_benign_target: int | None = None
-    extra += ["--benign-target", str(cfg.anomaly_benign_target), ...]
+    if cfg.anomaly_benign_target is not None:
+        extra += ["--benign-target", str(cfg.anomaly_benign_target)]
```

---

## 4. Comportamento antigo vs novo

### 4.1 Emparelhamento BENIGN no teste

| | Antigo | Novo |
|---|--------|------|
| **Regra** | `--benign-target 1255` em todas as rodadas LOAO | `sample_n = min(len(df2), benignos_em_df1)` |
| **PortScan (1255 fluxos)** | 1255 + 1255 | 1255 + 1255 (igual) |
| **DoS (3042 fluxos)** | 1255 + 3042 (desbalanceado) | 3042 + 3042 (se houver benignos suficientes) |
| **BruteForce (96 fluxos)** | 1255 + 96 | 96 + 96 |
| **Override** | Sempre 1255 | `--benign-target N` opcional (reprodução notebook) |

### 4.2 Número de clusters

| | Antigo | Novo |
|---|--------|------|
| **LOAO (fase 12)** | `n_clusters=16` fixo em fases 9 e 11 | Fase **10** BO-GP (`gp_minimize`, 20 calls) por ataque |
| **Fase 11** | k=16 da CLI | k = `best_n_clusters` de `phase10_anomaly_cluster_hpo.json` no subdir do ataque |
| **Fase 9** | CL-k-means com k=16 | CL-k-means com default interno (8); relatório exploratório; k final vem da fase 10 |

### 4.3 Fluxo LOAO

| | Antigo | Novo |
|---|--------|------|
| **Sequência** | 7 → 8 → 9 → 11 | **7 → 8 → 9 → 10 → 11** |
| **BO-GP** | Omitido (custo) | Integrado por ataque |

---

## 5. Protocolo final (após correções)

### 5.1 Por rodada LOAO (ataque `a` hold-out)

```text
02_sampled_kmeans.parquet
        │
        ▼
[Fase 7] build_anomaly_binary_split
        │  df1 = tudo exceto ataque a (binário: 0=benigno, 1=outros ataques)
        │  df2 = só ataque a (rótulo 1)
        ▼
[Fase 8] Re-Z-score; amostra BENIGN: n = min(|df2|, |benign em df1|)
        │  df_test_part = df2 ∪ benignos_amostrados
        │  df = df1 ‖ df_test_part
        │  IG → FCBF → KPCA (conjunto combinado; split por índice depois)
        ▼
[Fase 9] Treino = primeiras n_df1 linhas (sem zero-day)
        │  Teste  = restante (todo zero-day + N benignos)
        │  SMOTE classe 1 → 18225 (só treino)
        │  CL-k-means fit em X_train; predict em X_test (baseline k=8)
        ▼
[Fase 10] BO-GP em n_clusters ∈ [2, 50], objetivo = acurácia no teste LOAO
        │  Salva best_n_clusters
        ▼
[Fase 11] CL-k-means com k = best_n_clusters
        │  Biased B1/B2 (modo auto, p*=0.933)
        │  Métricas DR / FAR / F1 no teste
```

### 5.2 Montagem do conjunto de **teste** (cada ataque)

Para ataque zero-day com rótulo inteiro `a`:

1. **Ataque:** todas as linhas de `02_sampled` com `Label == a` → `df2`, rótulo binário 1.
2. **Benignos:** amostra sem reposição de `df1[Label==0]`, tamanho  
   `n_benign = benign_sample_size_for_zero_day(len(df2), len(benign_train))`.
3. **Teste final:** `df2` (N ataques) + N benignos → após KPCA, linhas `y[ n_df1 : ]`.
4. **Balanceamento no teste:** N amostras classe 1 (zero-day) e N classe 0 (benigno), com N = `min(|zero-day|, benignos_disponíveis)`.

Exemplo com contagens da amostra notebook (~27k):

| Ataque | N zero-day | N benignos no teste |
|--------|------------|---------------------|
| PortScan (5) | 1255 | 1255 |
| DoS (3) | 3042 | 3042 |
| Bot (1) | 1966 | 1966 |
| BruteForce (2) | 96 | 96 |

### 5.3 Treino preservado (artigo)

```text
Treino (df1, sem zero-day)
    ↓  [Fase 8] features (parte treino do parquet KPCA)
    ↓  [Fase 9] SMOTE → 18225 amostras classe ataque
    ↓  [Fase 9/10/11] CL-k-means.fit apenas em X_train pós-SMOTE
    ↓  [Fase 10] BO-GP escolhe k
    ↓  [Fase 11] Biased classifiers nos fluxos incertos (p* < 0.933)
```

---

## 6. Como executar (Tabela IX)

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile fine
python -m mth_ids_pipeline.run_supervised --label-profile fine --from 1 --to 2
python -m mth_ids_pipeline.run_anomaly --label-profile fine --loao
```

Override legado (notebook PortScan apenas):

```powershell
python -m mth_ids_pipeline.run_anomaly --loao --from 12 --to 12 `
  --intermediate-dir data/pipeline_mth_ids_merged
# Opcional por ataque via fase 8: --phase8-extra "--benign-target 1255"
```

---

## 7. Diff completo

Gerar localmente:

```powershell
git diff -- mth_ids_pipeline/anomaly_io.py `
  mth_ids_pipeline/phase08_anomaly_features.py `
  mth_ids_pipeline/phase12_anomaly_loao.py `
  mth_ids_pipeline/experiment_runner.py
```

---

## 8. Itens não alterados nesta correção

- IG/FCBF/KPCA ainda no conjunto combinado antes do split por índice (como no notebook; possível divergência metodológica estrita train-only — fora do escopo desta tarefa).
- BO-GP ainda maximiza acurácia no **conjunto de teste LOAO** (como notebook/artigo tier 3).
- Amostragem 0,8% (fase 2) e SMOTE 18225 no treino mantidos.
