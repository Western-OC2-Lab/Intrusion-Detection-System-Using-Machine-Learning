# Protocolo MTH-IDS — presets `paper` vs `notebook`

Comparativo rápido dos presets. Detalhes por dataset:

- **CICIDS2017:** [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md)
- **CAN (pipeline):** [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md)
- **CAN Car-Hacking / OTIDS:** [PROTOCOLO_CAN_INTRUSION.md](PROTOCOLO_CAN_INTRUSION.md) · [PROTOCOLO_CAN_OTIDS.md](PROTOCOLO_CAN_OTIDS.md)
- **UNSW-NB15:** [PROTOCOLO_UNSW_NB15.md](PROTOCOLO_UNSW_NB15.md)

Comandos: [EXECUCAO.md](EXECUCAO.md) · Arquitetura: [ARQUITETURA.md](ARQUITETURA.md)

---

## CICIDS2017

| Item | `paper` (padrão) | `notebook` |
|------|------------------|------------|
| Split | 80/20 | 80/20 |
| α IG | BO-GP | 0,9 fixo |
| FCBF | só treino | dataset completo |
| HPO fase 6 | 10-fold CV | hold-out / teste |
| Stacking | `best-base` | XGBoost + HPO |
| Perfil LOAO | `fine` | `merged` |
| SMOTE | sim (sup. + anomaly) | sim |

Preset completo: [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md).

---

## CAN (`can_paper` / `can_notebook`)

Válido para **ambos** os datasets após `merge_can` (Car-Hacking ou OTIDS). Dados: [PROTOCOLO_CAN_INTRUSION.md](PROTOCOLO_CAN_INTRUSION.md) · [PROTOCOLO_CAN_OTIDS.md](PROTOCOLO_CAN_OTIDS.md).

| Item | `can` (`can_paper`) | `can_notebook` |
|------|---------------------|----------------|
| Amostragem | k-means 0,8% em todas as classes | igual |
| SMOTE | desligado | desligado |
| α IG | BO-GP | 0,9 dinâmico |
| Stacking | `best-base` + CV | XGBoost + hold-out |

Preset completo: [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md).

---

## Dependências

`requirements.txt` · Python 3.10+. `imbalanced-learn` ≥ 0.12 remove `n_jobs` do SMOTE (tratado no código). BO-GP: `n_calls` mínimo 10; presets `paper` usam **15** (`PAPER_HPO_N_CALLS`).

Exportação: `report_paper_tables` → `results/` (`--results-dir`, `--no-save`).
