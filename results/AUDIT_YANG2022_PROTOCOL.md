# Auditoria de conformidade — Yang et al. (2022) vs pipeline MTH-IDS

Data: 2026-06-08  
Escopo: correções metodológicas pontuais (sem refatoração de arquitetura, APIs ou nomes de arquivos).

---

## 1. Arquivos modificados

| Arquivo | Papel |
|---------|-------|
| `mth_ids_pipeline/phases/phase02_sample_kmeans.py` | Default `--frac` → 0.008 (mantido) |
| `mth_ids_pipeline/config.py` | Constantes `PAPER_*`, `DEFAULT_KMEANS_FRAC`, `NOTEBOOK_KMEANS_FRAC` |
| `mth_ids_pipeline/protocol.py` | Preset `paper`: split 70/30, k-means 10%, `cl_hpo_metric_source` |
| `mth_ids_pipeline/orchestration/experiment_runner.py` | Propaga `cl_hpo_metric_source` à fase 10 |
| `mth_ids_pipeline/core/feature_selection.py` | `fit_fcbf` com modos `k` e `alpha` |
| `mth_ids_pipeline/core/hyperparameter_optimization.py` | `optimize_fcbf_alpha` (BO-GP) |
| `mth_ids_pipeline/phases/phase04_feature_engineering.py` | CLI `--fcbf-mode`, `--fcbf-alpha`, `--optimize-fcbf` |
| `mth_ids_pipeline/phases/phase10_anomaly_cluster_hpo.py` | Relatório `hpo_metric_source` + evidências |
| `tests/test_paper_protocol.py` | **Novo** — 9 testes de conformidade |

**Sem alteração necessária (já conformes):**

- `mth_ids_pipeline/phases/phase06_supervised_models.py` — `best-base` + HPO em CV já implementados.

---

## 2. Linhas alteradas (diff desta tarefa)

| Arquivo | Linhas (+/-) |
|---------|--------------|
| `phase02_sample_kmeans.py` | ~21 |
| `config.py` | ~19 |
| `protocol.py` | ~66 |
| `experiment_runner.py` | ~11 |
| `feature_selection.py` | ~32 |
| `hyperparameter_optimization.py` | ~68 |
| `phase04_feature_engineering.py` | ~60 |
| `phase10_anomaly_cluster_hpo.py` | ~31 |
| `tests/test_paper_protocol.py` | ~75 (novo) |

---

## 3. Divergências encontradas — código vs artigo (antes das correções)

| Item | Código anterior | Artigo Yang et al. (2022) | Evidência |
|------|-----------------|---------------------------|-----------|
| Split treino/teste (`paper`) | **80/20** (`DEFAULT_TEST_SIZE`) | **70/30** (Sec. IV-F, Tabela X) | `NOTA_REPRODUCAO.md` §1 |
| FCBF | `FCBFK(k=20)` apenas | Texto descreve FCBF com limiar; auditoria confirma **k=20** no artigo | `METHODOLOGICAL_AUDIT.md` §5; notebook `FCBFK(k=20)` |
| CL-k-means HPO métrica | F1 (preset paper) sem documentação de fonte | Métrica de objetivo do BO-GP **não nomeada**; Tabela IX reporta F1/DR/FAR | `METHODOLOGICAL_AUDIT.md` §7 (ambiguidade alta) |
| Meta-learner stacking | Já `best-base` no preset paper | Clone do melhor classificador base (maior F1) | Artigo tier 2; `phase06` `_pick_best_base_name` |

**Já conformes antes desta tarefa:**

- `PAPER_HPO_ON_VALIDATION=True` → Stratified K-Fold CV (10 folds) via `hpo_objective_on_validation`.
- `PAPER_META_LEARNER="best-base"` definido em `config.py`; lógica em `phase06` linhas 494–502.

---

## 4. Divergências encontradas — notebook vs artigo

| Item | Notebook `MTH_IDS_IoTJ.ipynb` | Artigo |
|------|-------------------------------|--------|
| K-means sampling | **0.8%** (`frac=0.008`) | **10%** (CAN); CICIDS auditado como 0.8% no notebook |
| Split | **80/20** | **70/30** (trechos Sec. IV-F) |
| FCBF | `FCBFK(k=20)` no dataset completo | k=20 só treino (preset paper) |
| HPO supervisionado | `accuracy_score` no **hold-out teste** | CV 10-fold no treino |
| Stacking meta | **XGBoost + HPO** | Melhor base (F1) |
| CL-k-means HPO | `gp_minimize` com **accuracy** no teste LOAO | Métrica não especificada; avaliação Tabela IX em F1 |
| α IG | **0.9 fixo** | BO-GP otimizado |
| SMOTE | alvo **1000** | texto cita até **100 000** |

Fonte notebook CL-k-means: célula `gp_minimize` → `return (1-cm)` onde `cm = metrics.accuracy_score(y_test, result2)`.

---

## 5. Mudanças realizadas

### 5.1 K-means sampling
- `phase02_sample_kmeans.py`: `--frac` default **0.008** (`DEFAULT_KMEANS_FRAC`) — mantido conforme notebook/CICIDS.
- Preset `paper` (CICIDS): `kmeans_frac=0.008`.
- Preset `can_paper`: mantém `CAN_PAPER_KMEANS_FRAC=0.10` (Tabela VI CAN).

### 5.2 Protocolo PAPER
- `protocol.PAPER.test_size` → `PAPER_TEST_SIZE` (0.3).
- `protocol.PAPER.kmeans_frac` → `PAPER_KMEANS_FRAC` (0.10).
- Confirmados: `cv_folds=10`, `hpo_on_validation=True`, `meta_learner="best-base"`.
- `ExperimentConfig.from_protocol("paper")` propaga todos os valores às fases via `experiment_runner`.

### 5.3 Meta-learner stacking
- Sem alteração de código: `phase06` já seleciona o melhor base por `f1_weighted` e clona com `sklearn.base.clone` quando `--meta-learner best-base`.

### 5.4 HPO supervisionado
- Sem alteração: `PAPER_HPO_ON_VALIDATION=True` ativa `StratifiedKFold(n_splits=10)` em `_hyperopt_objective` → `hpo_objective_on_validation`.
- Hold-out só quando `--no-hpo-on-validation` (preset `notebook`).

### 5.5 FCBF
- **Auditoria:** artigo e notebook usam `FCBFK(k=20)`; não há evidência de BO-GP para limiar FCBF no artigo (α IG sim).
- **Implementado (compatibilidade):**
  - `--fcbf-mode k|alpha` (default `k`).
  - `--fcbf-alpha` (default 0.01) + `--optimize-fcbf` (BO-GP, mesmo mecanismo que IG).
  - `optimize_fcbf_alpha()` em `hyperparameter_optimization.py`.
- **Preset `paper`:** mantém `fcbf_k=20`, modo `k` (evidência artigo/notebook).

### 5.6 CL-KMeans HPO
- Campo `cl_hpo_metric_source` em `ProtocolSettings` (`article` | `notebook`).
- Fase 10: `--hpo-metric-source` + relatório `hpo_metric_provenance` com evidências explícitas.
- Preset `paper`: `cl_hpo_metric="f1"`, `source="article"` (inferido de Tabela IX; objetivo HPO não nomeado no PDF).
- Preset `notebook`: `cl_hpo_metric="accuracy"`, `source="notebook"` (célula confirmada).

### 5.7 Testes
- `tests/test_paper_protocol.py`: 9 testes — **todos passaram** (`pytest -v`).

---

## 6. Mudanças NÃO realizadas e motivo

| Item | Motivo |
|------|--------|
| Ativar `--optimize-fcbf` no preset `paper` | Artigo e notebook confirmam `FCBFK(k=20)`; sem evidência de otimização de limiar FCBF |
| Alterar SMOTE para 100k (artigo) | Notebook usa 1000; mudança alteraria escala e métricas sem mandato explícito nesta tarefa |
| Mudar preset `paper` CL-k-means para `accuracy` | Notebook usa accuracy, mas artigo não nomeia métrica de HPO; mantido F1 com documentação de ambiguidade |
| Refatorar arquitetura / APIs / nomes de arquivos | Restrição explícita do escopo |
| K-means 10% no preset `paper` (CICIDS) | Mantido **0.008** (notebook/IoTJ); 10% só em `can_paper` via `CAN_PAPER_KMEANS_FRAC` |
| Habilitar FCBF α como padrão global | Retrocompatibilidade: default permanece `k=20` |

---

## 7. Validação

```powershell
python -m pytest tests/test_paper_protocol.py -v
```

Resultado: **9 passed**.

Para reexecutar o protocolo paper após as mudanças:

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper --from 1 --to 6
```

Verificar em `phase_reports/experiment_runner_config.json`:
- `test_size`: 0.3
- `kmeans_frac`: 0.008
- `cv_folds`: 10
- `hpo_on_validation`: true
- `meta_learner`: "best-base"
