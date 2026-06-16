# Documentação MTH-IDS

Índice do pacote `mth_ids_pipeline` — reprodução Yang et al. (IEEE IoT Journal 2022).

## Documentos

| Documento | Conteúdo |
|-----------|----------|
| [ARQUITETURA.md](ARQUITETURA.md) | 4 tiers, 3 experimentos, layout do código, perfis merged/fine |
| [EXECUCAO.md](EXECUCAO.md) | Comandos por tabela, pastas, bootstrap, troubleshooting |
| [PIPELINE_PHASES.md](PIPELINE_PHASES.md) | Referência das fases 1–13 + [CLI manual](PIPELINE_PHASES.md#rodar-cada-fase-manualmente) |
| [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) | CICIDS2017 (Tabelas VII/IX/X) |
| [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) | CAN — índice pipeline (Tabelas VI/VIII/X) |
| [PROTOCOLO_CAN_INTRUSION.md](PROTOCOLO_CAN_INTRUSION.md) | Car-Hacking original (artigo, `merge_can --source original`) |
| [PROTOCOLO_CAN_OTIDS.md](PROTOCOLO_CAN_OTIDS.md) | Repack OTIDS (`merge_can --source otids`) |
| [PROTOCOLO_UNSW_NB15.md](PROTOCOLO_UNSW_NB15.md) | UNSW-NB15 (`--protocol unsw`) |
| [ADAPTACOES_UNSW_DESDE_CICIDS17.md](ADAPTACOES_UNSW_DESDE_CICIDS17.md) | Diferenças práticas entre CICIDS2017 e UNSW-NB15 |
| [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) | Comparativo `paper` vs `notebook` |

Cada protocolo inclui a seção **Tabelas — `report_paper_tables`** (`--table vii` / `ix` / `x` / `all`).
| [NOTA_REPRODUCAO.md](NOTA_REPRODUCAO.md) | **Limitações artigo × notebook** (texto para relatório) |
| [REPRODUCAO_CICIDS2017_VALIDACAO.md](REPRODUCAO_CICIDS2017_VALIDACAO.md) | Validação vs artigo e notebook |

Histórico de auditorias: [archive/](archive/README.md).

---

## Início rápido (CICIDS2017)

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.utils.merge_cicids --profile fine

python -m mth_ids_pipeline.run_supervised --protocol paper          # Tabela VII
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao     # Tabela IX
python -m mth_ids_pipeline.run_global_anomaly --protocol paper       # Tabela X (treino)
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_mth_ids_merged `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global

python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

## Início rápido (CAN)

Escolha **uma** fonte de dados (não misturar):

```powershell
# Car-Hacking (artigo) — pastas pipeline_can_intrusion_*
python -m mth_ids_pipeline.utils.merge_can --source original
python -m mth_ids_pipeline.run_supervised --protocol can
python -m mth_ids_pipeline.run_anomaly --protocol can --loao

# OTIDS — pastas pipeline_can_otids_* (não sobrescreve intrusion)
python -m mth_ids_pipeline.utils.merge_can --source otids
python -m mth_ids_pipeline.run_supervised --protocol can_otids
python -m mth_ids_pipeline.run_anomaly --protocol can_otids --loao
```

Detalhes: [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) · [PROTOCOLO_CAN_INTRUSION.md](PROTOCOLO_CAN_INTRUSION.md) · [PROTOCOLO_CAN_OTIDS.md](PROTOCOLO_CAN_OTIDS.md) · [EXECUCAO.md](EXECUCAO.md).

## Início rápido (UNSW-NB15)

```powershell
# Pré-requisito: data/UNSW-NB15_merged.csv
python -m mth_ids_pipeline.run_supervised --protocol unsw
python -m mth_ids_pipeline.run_anomaly --protocol unsw --loao
```

Detalhes: [PROTOCOLO_UNSW_NB15.md](PROTOCOLO_UNSW_NB15.md).
