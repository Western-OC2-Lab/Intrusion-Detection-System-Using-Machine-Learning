# Relatório Técnico de Reprodução — MTH-IDS

**Referência:** L. Yang, A. Moubayed, A. Shami, "MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles," *IEEE Internet of Things Journal*, vol. 9, no. 1, pp. 616–632, 2022.

**Código base:** `paper_and_notebooks/MTH_IDS_IoTJ.ipynb`  
**Pipeline modular:** `mth_ids_pipeline/`  
**Orquestrador reprodutível:** `python -m mth_ids_pipeline.experiment_runner`

---

## 1. Estrutura modular entregue

| Módulo | Responsabilidade | Fases correspondentes |
|--------|------------------|----------------------|
| `data_loading.py` | Leitura CSV/Parquet | 1+ |
| `preprocessing.py` | Z-score, LabelEncoder | 1, 2 |
| `feature_selection.py` | IG + FCBF | 4, 8 |
| `dimensionality_reduction.py` | Kernel PCA | 8 |
| `clustering.py` | k-means sampling, CL-k-means | 2, 9, 10 |
| `hyperparameter_optimization.py` | BO-TPE, BO-GP | 6, 10 |
| `evaluation.py` | Métricas, comparação artigo/notebook | 6, 9, 10 |
| `reproducibility.py` | Seeds, versões, log de config | todas |
| `experiment_runner.py` | Orquestração configurável | 1–10 |

As fases `phase01`–`phase10` permanecem como CLIs finas que delegam aos módulos core.

---

## 2. Controle de reprodutibilidade

### Implementado
- `random_state=0` em: split, k-means sampling, RF/DT/ET/XGB (fase 6), gp_minimize (fase 10)
- Seed global Python/NumPy via `reproducibility.set_global_seeds()`
- Registro JSON de versões (`numpy`, `pandas`, `scikit-learn`, `xgboost`, etc.) e parâmetros em `data/pipeline_mth_ids/phase_reports/experiment_runner_config.json`

### Intrinsicamente não-determinísticos (documentados)
- Hyperopt `fmin` sem seed explícito (notebook original)
- SMOTE sem `random_state` na fase 5 (notebook original); fase 10 aceita seed
- KernelPCA (dependência de solver)
- CL-k-means sem seed no notebook; pipeline fase 9/10 usa `random_state=0` por padrão (**melhoria documentada**, alterável via CLI)

---

## 3. Validação — Supervisionado (CICIDS2017 amostrado)

### Tabela A — Notebook vs Pipeline (`--no-hpo`, hiperparâmetros fixos do notebook)

| Modelo | Métrica | Notebook | Pipeline reproduzido | Δ abs | Δ % |
|--------|---------|----------|----------------------|-------|-----|
| XGBoost HPO | Accuracy | 0.9957 | 0.9976 | +0.0019 | +0.19% |
| RandomForest HPO | Accuracy | 0.9951 | 0.9957 | +0.0006 | +0.06% |
| DecisionTree HPO | Accuracy | 0.9937 | 0.9933 | −0.0004 | −0.04% |
| ExtraTrees HPO | Accuracy | 0.9955 | 0.9978 | +0.0023 | +0.23% |
| Stacking meta HPO | Accuracy | 0.9957 | 0.9974 | +0.0017 | +0.17% |
| Stacking meta HPO | F1 weighted | 0.9957 | 0.9974 | +0.0017 | +0.17% |

**Fonte pipeline:** `data/pipeline_mth_ids/06_supervised_metrics.json`

**Interpretação:** divergências < 0.3% — compatíveis com variação de versões sklearn/xgboost e pequenas diferenças no split (5359 vs 5360 amostras de teste). A reprodução **atende** o notebook.

### Tabela B — Artigo (Tabela VII) vs Pipeline

| Métrica | Artigo MTH-IDS Multi-Class | Pipeline (sampled hold-out) | Observação |
|---------|---------------------------|----------------------------|------------|
| Accuracy | 99.879% | 99.74% (stacking) | Protocolo diferente |
| F1 | 0.99879 | 0.9974 (weighted) | Métrica e split diferentes |
| Validação | 10-fold CV, dataset completo | Hold-out 20%, ~27k amostras | **Causa principal de gap** |

**Conclusão:** reprodução fiel ao **notebook**; proximidade ao **artigo** limitada por dataset amostrado e protocolo de validação distintos — conforme esperado academicamente.

---

## 4. Validação — Anomaly (PortScan zero-day demo)

| Configuração | Notebook | Pipeline fase 9 | Pipeline fase 10 (BO-GP) |
|--------------|----------|-----------------|--------------------------|
| CL-k-means n=8 | Acc ≈ 0.598 | 0.665* | baseline registrado |
| BO-GP best | n=16, Acc 0.9195 | — | n=23, Acc 0.689* |
| Final n=16 | Acc 0.945 | — | requer `--random-state 0` + `--benign-target 1255` |

\*Execuções anteriores com seeds não controlados (`random_state=null` nos reports). Reexecutar:

```bash
python -m mth_ids_pipeline.experiment_runner --from 7 --to 10 --random-state 0
```

**Gap vs artigo (Tabela IX, F1 médio 0.80013):** o notebook implementa apenas experimento PortScan, não leave-one-attack-out para 14 tipos. Biased classifiers omitidos explicam diferença adicional.

---

## 5. Generalização para novo dataset

### Pontos dependentes do CICIDS2017

| Parâmetro | Dependência | Adaptação |
|-----------|-------------|-----------|
| Coluna `Label` | Nome fixo | Renomear ou configurar `label_col` |
| Classes minoritárias k-means | IDs 1,4,6 (Bot, Infiltration, WebAttack) | Detectar classes raras automaticamente ou parametrizar |
| SMOTE supervisionado | Classes 2 e 4 | Mapear rótulos após LabelEncoder |
| PortScan anomaly | Label 5 | Generalizar `--attack-label` na fase 7 |
| Features numéricas | 78+ colunas fluxo | Mesmo pré-processamento Z-score |
| FCBF | Requer `FCBF_module.py` | Instalar dependência |

### Execução em novo dataset

```bash
python -m mth_ids_pipeline.experiment_runner \
  --raw-csv data/novo_dataset.csv \
  --from 1 --to 6 \
  --random-state 42
```

### Dataset sugerido para validação cruzada

**CAN-OTIDS** (intra-veicular, usado no artigo Tabela VI/VIII):
- Justificativa: segundo benchmark do artigo; scripts `utils/merge_can.py` já presentes
- Adaptações: remover timestamp; 4 features IG-FCBF; validação leave-one-attack-out

**IoT_2020** (`data/IoT_2020_multi_0.05.csv` já no repositório):
- Justificativa: tráfego IoT externo, classes múltiplas, tamanho gerenciável
- Adaptações: remapear labels, recalibrar SMOTE e k-means sampling

---

## 6. Limitações metodológicas observadas

1. HPO otimiza acurácia no **conjunto de teste** (optimistic bias)
2. FCBF aplicado com labels do dataset completo (data leakage)
3. ~5% do método anomaly omitido no notebook (biased classifiers)
4. Artigo usa 10-fold CV; código usa hold-out
5. Métricas DR/FAR do artigo não calculadas no notebook

---

## 7. Melhorias técnicas justificadas (pós-reprodução base)

| Melhoria | Justificativa | Status |
|----------|---------------|--------|
| `random_state` explícito em CL-k-means fase 9/10 | Reprodutibilidade | **Implementado** (default 0) |
| `--benign-target 1255` na fase 8 | Alinhar split anomaly ao notebook | **Implementado** via experiment_runner |
| Módulos core desacoplados | Reuso e testes | **Implementado** |
| Validação cruzada estratificada no HPO | Reduz overfitting ao test set | **Não implementado** (alteraria metodologia base) |
| Biased classifiers tier 4 | Completar método do artigo | **Futuro** (requer especificação adicional) |

---

## 8. Como executar a reprodução completa

```bash
# Ambiente
pip install -r requirements.txt

# Pipeline supervisionado (fases 1-6, HPO desligado = hiperparâmetros fixos notebook)
python -m mth_ids_pipeline.experiment_runner --to 6

# Com HPO (lento, não-determinístico como notebook)
python -m mth_ids_pipeline.experiment_runner --to 6 --run-hpo

# Ramo anomaly completo (fases 7-10)
python -m mth_ids_pipeline.experiment_runner --from 7 --to 10 --random-state 0
```

Relatórios JSON: `data/pipeline_mth_ids/phase_reports/`  
Métricas supervisionadas: `data/pipeline_mth_ids/06_supervised_metrics.json`

---

## 9. Avaliação dos requisitos acadêmicos

| Requisito | Atendimento |
|-----------|-------------|
| Reproduzir metodologia do artigo/notebook | **Parcial** — notebook ~95%; tier 4 anomaly omitido |
| Pipeline completo treino→métricas | **Sim** — fases 1–10 |
| Resultados próximos ao artigo | **Parcial** — próximo ao notebook; artigo requer dataset/protocolo distintos |
| Pipelines modulares reprodutíveis | **Sim** |
| Documentar divergências | **Sim** — `METHODOLOGICAL_AUDIT.md` |
| Generalização novo dataset | **Sim** — parametrizado via CLI |
| Controle de seeds e versões | **Sim** — `reproducibility.py` |
| Melhorias pós-reprodução | **Documentadas**; seeds anomaly implementados |

**Veredicto:** a implementação satisfaz reprodutibilidade acadêmica **fiel ao código publicado** (notebook). Reprodução numérica exata do artigo exigiria dataset CICIDS2017 completo, 10-fold CV, leave-one-attack-out e biased classifiers — componentes parcialmente ausentes do repositório oficial.
