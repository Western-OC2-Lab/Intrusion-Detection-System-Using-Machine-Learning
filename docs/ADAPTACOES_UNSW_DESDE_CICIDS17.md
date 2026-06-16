# Adaptações do UNSW-NB15 a partir do CICIDS2017

Este documento resume o que foi reaproveitado do fluxo do **CICIDS2017** e o que precisou ser adaptado para o **UNSW-NB15** no ramo `unsw` / `unsw_nb15`.

## O que foi herdado do CICIDS2017

O pipeline continua usando a mesma estrutura de fases do ramo anomaly:

1. fase 7: montagem do binário LOAO
2. fase 8: Z-score, IG, FCBF e KPCA
3. fase 9: SMOTE no treino binário
4. fase 10: BO-GP para `n_clusters`
5. fase 11: CL-k-means + biased learners

A lógica geral também permaneceu igual:

- treino e teste são derivados do mesmo conjunto amostrado da fase 2
- o zero-day é removido do treino da fase 7
- as fases 9–11 operam sobre o treino binário da fase 8
- a avaliação final usa a cascata supervisionado + anomaly da fase 13

## O que mudou no UNSW-NB15

### 1. O rótulo benigno não é o mesmo do CICIDS2017

No CICIDS2017 o atalho do código podia assumir `BENIGN = 0`. No UNSW-NB15 isso não vale no workspace atual: a versão processada em `data/pipeline_unsw_nb15_*` usa **`Benign = 3`**.

Isso afeta diretamente:

- a seleção do benigno na fase 7
- a contagem de benignos amostrados na fase 8
- a descoberta das labels de ataque na fase 12
- a definição do benigno na fase 13

### 2. A binarização deixou de depender de `label > 0`

No CICIDS2017, `label > 0` era suficiente para distinguir ataques de benigno. No UNSW-NB15 isso quebrava o LOAO porque algumas classes de ataque têm ids menores que o benigno.

A adaptação passou a usar comparação explícita com o `benign_label` resolvido para o dataset atual.

### 3. O LOAO agora respeita o encodamento do UNSW

Antes da correção, a fase 7 do ramo anomaly tratava classe 0 como benigna por herança do CICIDS2017.

Agora o fluxo faz o seguinte:

- resolve o benigno do UNSW via configuração do dataset
- remove apenas o zero-day do treino
- colapsa qualquer classe diferente do benigno para `1`
- mantém o benigno como `0` no binário gerado pela fase 7

### 4. A fase 12 não herda a tabela de nomes do CICIDS2017

O orquestrador LOAO do CICIDS2017 usa tabelas de nomes específicas do CICIDS. Para o UNSW isso não serve, porque o conjunto de ids e os nomes das classes são diferentes.

O ramo `unsw` passou a usar a tabela de nomes do UNSW e a descobrir as rodadas LOAO excluindo o benigno correto.

### 5. A avaliação final passou a usar benigno dataset-aware

A fase 13 também foi ajustada para não assumir que o benigno é sempre o menor id.

Isso corrige:

- roteamento da cascata supervisionado → anomaly
- métricas binárias de detecção e falso alarme

## Resultado prático

O comando abaixo continua sendo o ponto de entrada, mas agora a execução respeita o encodamento real do UNSW-NB15 neste workspace:

```powershell
python -m mth_ids_pipeline.run_anomaly --protocol unsw --loao
```

Se você quiser rodar um ataque específico, o valor do zero-day deve ser escolhido entre as classes de ataque codificadas no UNSW, excluindo o benigno.

## Resumo curto

O que foi reaproveitado do CICIDS2017 foi a **arquitetura do pipeline**. O que precisou mudar foi a **semântica dos rótulos**, principalmente a classe benigna, que no UNSW-NB15 deste workspace é `3` e não `0`.