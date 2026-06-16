# Relatório — Refatoração BO-GP do CL-k-means (MTH-IDS)

Data: 2026-06-05  
Escopo: fases 9–11 (tier 3 anomaly), sem alterar demais fases da pipeline.

---

## 1. Implementação anterior

### Hiperparâmetros otimizados

| Parâmetro | Otimizado? | Espaço |
|-----------|------------|--------|
| `n_clusters` | **Sim** | `[2, 50]` via `Integer` (skopt) |
| `metric` (distância) | **Não** (fixo `euclidean`) | — |

### Mecanismo

- **Otimizador:** `gp_minimize` (BO-GP, scikit-optimize).
- **Chamadas:** 20 (`--n-calls`).
- **Objetivo:** `1 - accuracy` no conjunto de teste LOAO (mesmo protocolo do notebook).
- **Função:** `optimize_cl_kmeans_clusters(lambda n: objective(n))` — apenas `n_clusters`.
- **Flag `optimize_metric`:** existia em `hyperparameter_optimization.py`, mas **nunca era ativada** na fase 10; quando ativada, limitava-se a `{euclidean, manhattan}` sem `cosine`.

### Métricas de distância

- `clustering.py` tentava passar `metric="manhattan"` ao `MiniBatchKMeans`, mas o `try/except` estava incorreto (o `TypeError` ocorre na instanciação, não na atribuição do dict).
- Em sklearn ≥ 1.6, `MiniBatchKMeans` **não expõe** parâmetro `metric`; portanto **manhattan e cosine não tinham efeito real** — todas as execuções usavam distância euclidiana.

### Registro

- Relatório `phase10_anomaly_cluster_hpo.json` guardava `best_n_clusters`, `best_metric` (sempre `"euclidean"`) e acurácias.
- **Não registrava** o histórico de trials (`hpo_trials`).

### Propagação para fase 11

- `best_n_clusters` era lido automaticamente da fase 10.
- `metric` permanecia fixo em `euclidean` via CLI (`--metric` default).

---

## 2. Implementação nova (alinhada ao artigo MTH-IDS)

### Hiperparâmetros otimizados

| Parâmetro | Otimizado? | Espaço |
|-----------|------------|--------|
| `n_clusters` | **Sim** | `[2, 50]` |
| `metric` | **Sim** | `{euclidean, manhattan, cosine}` |

### Mecanismo

- **BO-GP conjunto:** um único `gp_minimize` sobre espaço misto `(Integer, Categorical)`.
- **Objetivo:** inalterado — acurácia no teste LOAO (`accuracy_score` sobre labels binários mapeados).
- **Validação:** mesma partição `load_anomaly_splits` (SMOTE só no treino, zero-day no teste).

### Implementação das distâncias

| Métrica | Implementação |
|---------|---------------|
| `euclidean` | `MiniBatchKMeans` padrão (comportamento idêntico ao notebook) |
| `cosine` | Normalização L2 por amostra + `MiniBatchKMeans` euclidiano (k-means esférico) |
| `manhattan` | MiniBatch k-means customizado com atribuição L1 (`pairwise_distances_argmin_min`) e atualização de centróides por mediana parcial |

### Registro ampliado (`phase10_anomaly_cluster_hpo.json`)

```json
{
  "best_config": {"n_clusters": 16, "metric": "cosine", "accuracy": 0.91},
  "hpo_trials": [
    {"trial": 0, "n_clusters": 8, "metric": "euclidean", "accuracy": 0.65, "loss": 0.35},
    ...
  ],
  "search_space": {"n_clusters": [2, 50], "metric": ["euclidean", "manhattan", "cosine"]},
  "optimizer": "BO-GP (skopt gp_minimize)",
  "objective": "accuracy on LOAO test split"
}
```

### Propagação para fase 11

- `load_best_metric()` lê `best_metric` do relatório da fase 10.
- Fase 11 usa automaticamente `n_clusters` **e** `metric` da fase 10, salvo override via CLI.
- Relatório da fase 11 inclui `metric_source` (`phase10` | `cli` | `default`).

---

## 3. Comparação com o artigo MTH-IDS

| Aspecto | Artigo (Sec. IV-D, Tier 3) | Notebook publicado | Pipeline anterior | Pipeline nova |
|---------|---------------------------|-------------------|-------------------|---------------|
| Otimizar `n_clusters` | Sim, BO-GP, [2, 50] | Sim | Sim | Sim |
| Otimizar métrica de distância | Sim (euclidean, manhattan, cosine) | Não (só k) | Não (flag inativa) | **Sim** |
| Otimizador | BO-GP | BO-GP | BO-GP | BO-GP |
| n_calls | 20 | 20 | 20 | 20 |
| Objetivo HPO | Acurácia (validação/teste LOAO) | Acurácia no teste | Acurácia no teste | Acurácia no teste |
| Algoritmo base | CL-k-means (MiniBatchKMeans) | MiniBatchKMeans | MiniBatchKMeans | MiniBatchKMeans + variantes L1/cosine |

**Nota:** o notebook IoTJ reproduz apenas a otimização de `k`. A nova implementação segue o **artigo** (otimização conjunta de `k` e distância), mantendo o protocolo de avaliação do notebook/pipeline (acurácia no teste LOAO).

---

## 4. Diferenças metodológicas

### O que mudou

1. **Espaço de busca:** de 1D (`n_clusters`) para 2D (`n_clusters` × `metric`).
2. **Métricas funcionais:** manhattan e cosine passam a alterar de fato o clustering (antes eram no-ops).
3. **Rastreabilidade:** cada trial BO-GP é persistido em `hpo_trials`.
4. **Consistência downstream:** fase 11 herda `best_metric` da fase 10.

### O que não mudou

- Particionamento LOAO (fases 8–12).
- SMOTE apenas no treino (`smote_target=18225`).
- Baseline k=8 euclidiano na fase 10.
- Fases 1–9, 12 e ramo supervisionado.
- Objetivo de otimização (test accuracy, fiel ao notebook).
- `random_state`, `batch_size=100`, rotulagem por maioria no cluster.

### Impacto esperado nos resultados

- Acurácias podem **diferir** da reprodução anterior: o espaço de busca é maior e as métricas não euclidianas agora são efetivas.
- Para comparar com o notebook (k=16 euclidiano), use `--skip-hpo` na fase 10 ou filtre `hpo_trials` onde `metric=euclidean`.

---

## 5. Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `mth_ids_pipeline/clustering.py` | Suporte real a `euclidean`, `manhattan`, `cosine` |
| `mth_ids_pipeline/hyperparameter_optimization.py` | BO-GP conjunto + `CLKmeansHpoResult` com trials |
| `mth_ids_pipeline/phase10_anomaly_cluster_hpo.py` | HPO 2D + relatório expandido |
| `mth_ids_pipeline/biased_classifiers.py` | `load_best_metric()` |
| `mth_ids_pipeline/phase11_anomaly_biased.py` | Herda `best_metric` da fase 10 |
| `mth_ids_pipeline/validate_reproduction.py` | Exibe métrica no resumo |

---

## 6. Uso

```bash
# HPO conjunto (padrão)
python -m mth_ids_pipeline.phase10_anomaly_cluster_hpo --work-dir data/pipeline_mth_ids/anomaly

# Pular HPO (baseline k=8)
python -m mth_ids_pipeline.phase10_anomaly_cluster_hpo --skip-hpo

# Fase 11 usa k e metric da fase 10 automaticamente
python -m mth_ids_pipeline.phase11_anomaly_biased --work-dir data/pipeline_mth_ids/anomaly
```
