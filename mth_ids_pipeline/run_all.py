"""Atalho para experiment_runner.

Exemplos:
  python -m mth_ids_pipeline.run_all --protocol paper --from 1 --to 6
  python -m mth_ids_pipeline.run_all --protocol paper --loao --from 7 --to 12
  python -m mth_ids_pipeline.run_all --protocol notebook --from 1 --to 6

  # LOAO — um ataque (fine, paper):
  python -m mth_ids_pipeline.run_all --label-profile fine \\
    --protocol paper --from 12 --to 12 --skip-bootstrap \\
    --attack-labels 14

  # Tabela X (global + fase 13):
  python -m mth_ids_pipeline.run_global_anomaly --protocol paper
  python -m mth_ids_pipeline.run_eval \\
    --intermediate-dir data/pipeline_mth_ids_merged \\
    --work-dir data/pipeline_mth_ids_merged/anomaly/global
"""

from __future__ import annotations

from mth_ids_pipeline.orchestration.experiment_runner import main

if __name__ == "__main__":
    main()
