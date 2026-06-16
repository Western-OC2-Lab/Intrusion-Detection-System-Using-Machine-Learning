"""Ramo anomaly — fases 7–12.

Paper/CICIDS: ``data/pipeline_mth_ids_fine`` + LOAO
CAN intrusion: ``--protocol can`` → ``data/pipeline_can_intrusion_fine``
CAN OTIDS: ``--protocol can_otids`` → ``data/pipeline_can_otids_fine``
"""

from __future__ import annotations

from mth_ids_pipeline.orchestration.experiment_runner import (
    build_arg_parser,
    config_from_args,
    run_experiment,
)
from mth_ids_pipeline.protocol import get_protocol_settings, is_can_protocol, is_unsw_protocol


def main() -> None:
    parser = build_arg_parser(
        "MTH-IDS anomaly (fases 7–12)"
    )
    parser.set_defaults(from_phase=7, to=11, label_profile="fine")
    args = parser.parse_args()
    if args.loao:
        args.to = 12
    if is_can_protocol(args.protocol) or is_unsw_protocol(args.protocol):
        ps = get_protocol_settings(args.protocol)
        if args.label_profile in (None, "fine"):
            args.label_profile = ps.anomaly_profile
    run_experiment(config_from_args(args, branch="anomaly"))


if __name__ == "__main__":
    main()
