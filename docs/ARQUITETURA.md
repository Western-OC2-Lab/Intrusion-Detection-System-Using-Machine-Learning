# Arquitetura MTH-IDS

Referência: Yang et al., *MTH-IDS*, IEEE IoT Journal 2022.

Documentos relacionados: [EXECUCAO.md](EXECUCAO.md) (comandos) · [PIPELINE_PHASES.md](PIPELINE_PHASES.md) (fases 1–13) · [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) · [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) · [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md)

---

## 4 tiers do método

| Tier | Componente | Fases |
|------|------------|-------|
| **1** | Z-score / normalização | 1 (+ scaler na 4/8) |
| **2** | IG + FCBF + DT/RF/ET/XGB + stacking | 4–6 |
| **3** | IG + FCBF + KPCA + CL-k-means | 8–10 |
| **4** | Biased B₁/B₂ + threshold p* | 11 |

Diagramas Mermaid: [`docs/figures/`](figures/) (`01_quatro_tiers.txt` … `04_experimento3_cascata_tabela_x.txt`).

---

## 3 experimentos

| Experimento | Tiers | Tabela CICIDS | Tabela CAN | Script |
|-------------|-------|---------------|------------|--------|
| **Supervisionado** | 1–2 | VII | VI | `run_supervised` |
| **LOAO / zero-day** | 3–4 (+ família tier 2 p/ biased) | IX | VIII | `run_anomaly --loao` |
| **Sistema completo** | 1–2–3–4 (cascata) | X | X | `run_global_anomaly` + `run_eval` |

| | Supervisionado | LOAO | Sistema completo |
|---|----------------|------|------------------|
| Classificação | Multi-classe | Binária | Multi-classe + fallback anomaly |
| Modelos anomaly | — | **1 por zero-day** | **1 global** + stacking |
| Teste | Hold-out 20% | Slice do ataque zero-day | Hold-out 20% (cascata) |

**LOAO:** cada `attack_N/` roda fases 7–11 de forma independente — **não** é a cascata da Tabela X.

**Tabela X:** stacking classifica; se BENIGN → anomaly global (`core/inference.py`). Treino anomaly em `anomaly/global/`; avaliação na fase 13.

---

## Perfis de rótulo e pastas

| Perfil | CSV CICIDS | Classes | Pasta | Usado em |
|--------|------------|---------|-------|----------|
| `merged` | `CICIDS2017.csv` | 7 famílias | `pipeline_mth_ids_merged/` | VII, X |
| `fine` | `CICIDS2017_fine.csv` | ~15 subtipos | `pipeline_mth_ids_fine/` | IX (LOAO) |
| `can_merged` | `CAN_OTIDS_Dataset.csv` (Car-Hacking ou OTIDS) | 4 | `pipeline_can_otids_merged/` | VI, X |
| `can_fine` | mesmo CSV | 4 LOAO (intrusion) / 3 (OTIDS) | `pipeline_can_otids_fine/` | VIII |

**Regra:** VII/VI e X → **merged** (`can_merged`). IX/VIII → **fine** (`can_fine`).

LOAO e global são ramos **diferentes** — rodar um não substitui o outro. Detalhes e comandos: [EXECUCAO.md](EXECUCAO.md).

---

## Layout do pacote

```
mth_ids_pipeline/
├── config.py, protocol.py, label_profiles.py, cli.py
├── run_supervised.py      # fases 1–6
├── run_anomaly.py         # fases 7–12 (LOAO)
├── run_global_anomaly.py  # fases 7–11 global
├── run_eval.py            # fase 13
├── report_paper_tables.py
├── core/                  # ML (preprocessing, clustering, HPO, inference, …)
├── io/                    # anomaly_io, loao_reporting, run_log, …
├── phases/                # phase01 … phase13
├── orchestration/experiment_runner.py
└── utils/                 # merge_cicids, merge_can, bootstrap, FCBF
```

### Ramos de execução

| Ramo | Fases | Perfil | Pasta padrão |
|------|-------|--------|--------------|
| Supervisionado CICIDS | 1–6 | `merged` | `pipeline_mth_ids_merged/` |
| LOAO CICIDS | 7–12 | `fine` | `pipeline_mth_ids_fine/` |
| Global + eval CICIDS | 7–11, 13 | `merged` | `…/anomaly/global/` |
| Supervisionado CAN | 1–6 | `can_merged` | `pipeline_can_otids_merged/` |
| LOAO CAN | 7–12 | `can_fine` | `pipeline_can_otids_fine/` |

Logs timestampados: `results/logs/`. LOAO: espelho em `results/logs/loao/attack_<N>.log`.

---

## CAN vs CICIDS

| Aspecto | CICIDS2017 | CAN |
|---------|------------|-----|
| Protocolo | `--protocol paper` | `--protocol can` |
| LOAO rodadas | ~14 | 3 (DoS, Fuzzy, Impersonation) |
| SMOTE | Sim (supervisionado + anomaly) | Não |
| Amostragem fase 2 | k-means 0,8% + minoritárias preservadas | k-means 0,8% em **todas** as classes |
